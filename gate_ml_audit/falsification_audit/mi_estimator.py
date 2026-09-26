"""
Mutual-information-with-step-count estimator, for the Falsification-Audit
Benchmark (§7.2 of the experiment design).

Purpose
-------
Given a monitored training-time signal (e.g. GATE-ML's per-language
EMA cosine-divergence, or any other "adaptive mechanism" trigger signal)
recorded alongside the training step at which it was measured, estimate
how much information the signal carries about raw step count. A signal
that is really just tracking elapsed training time will show HIGH mutual
information with step count. A signal carrying genuine, step-independent
information will show LOW mutual information with step count.

This module implements:
  - a binned plug-in mutual information estimator,
  - a normalized-MI variant (0-1 scale, comparable across signals with
    different entropies),
  - a permutation test that estimates significance against a null
    distribution built from the SAME estimator applied to shuffled data
    (this is what makes the test robust to the plug-in estimator's own
    bias -- both the observed MI and the null MI share the same bias,
    so the null-referenced significance is meaningful even though the
    raw MI value on its own is a biased estimate of true MI),
  - synthetic ground-truth signal generators used to validate the
    estimator BEFORE trusting it on any real mechanism (per the
    experiment design's explicit requirement in §7.2).

Dependencies: numpy only. No GPU required. Runs on a laptop CPU.

Usage
-----
    python mi_estimator.py

runs the built-in synthetic validation (a deterministic-vs-step signal
and an independent-noise signal) and asserts the estimator correctly
separates the two. Import the functions directly to audit a real
mechanism's logged signal:

    from mi_estimator import normalized_mutual_information, permutation_test_mi

    nmi = normalized_mutual_information(divergence_values, step_indices)
    result = permutation_test_mi(divergence_values, step_indices)
    print(result.p_value, result.observed_mi, result.null_mean, result.null_std)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------------------------------
# Core estimator
# --------------------------------------------------------------------------


def _validate_inputs(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs shared by all estimator functions.

    Parameters
    ----------
    x, y : array-like
        The two variables to estimate mutual information between (e.g.
        signal value and step index). Must be 1-D, equal length, and
        contain at least `bins` distinct-enough values to be binned
        meaningfully.
    bins : int
        Number of histogram bins requested per dimension.

    Returns
    -------
    (x, y) as float64 numpy arrays.

    Raises
    ------
    ValueError
        If shapes mismatch, arrays are empty, too short for the
        requested bin count, or contain non-finite values.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have the same length, got {x.shape[0]} and {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError("x and y must be non-empty")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    if x.shape[0] < bins * 2:
        raise ValueError(
            f"Need at least {bins * 2} samples for {bins} bins per dimension "
            f"(got {x.shape[0]}); reduce `bins` or provide more samples."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values (no NaN/Inf)")

    return x, y


def _joint_histogram_probs(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    """Compute the joint probability mass function via equal-width binning.

    Degenerate (constant) inputs are handled by nudging the bin edges so
    `np.histogram2d` does not raise on a zero-width range.
    """
    x_range = (float(np.min(x)), float(np.max(x)))
    y_range = (float(np.min(y)), float(np.max(y)))

    if x_range[0] == x_range[1]:
        x_range = (x_range[0] - 0.5, x_range[1] + 0.5)
    if y_range[0] == y_range[1]:
        y_range = (y_range[0] - 0.5, y_range[1] + 0.5)

    joint_counts, _, _ = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    total = joint_counts.sum()
    if total == 0:
        raise ValueError("Joint histogram is empty; this should not happen for non-empty inputs")
    return joint_counts / total


def binned_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Plug-in mutual information estimate I(X;Y) via equal-width histogram binning.

    Returns MI in bits (log base 2). This is a biased estimator (plug-in
    MI estimators are known to overestimate true MI at finite sample
    sizes, more so with more bins and fewer samples) -- use
    `permutation_test_mi` for a significance-tested result rather than
    trusting the raw value in isolation.

    Parameters
    ----------
    x, y : array-like, shape (n,)
    bins : int, default 16
        Number of equal-width bins per dimension.

    Returns
    -------
    float
        Estimated mutual information in bits. Always >= 0 (up to floating
        point rounding, which is clipped at 0).
    """
    x, y = _validate_inputs(x, y, bins)
    p_xy = _joint_histogram_probs(x, y, bins)

    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x * p_y  # shape (bins, bins) via broadcasting

    # Restrict to strictly positive joint-probability cells: wherever
    # p_xy[i, j] > 0, both marginals p_x[i] and p_y[j] are also > 0 (a
    # marginal is a sum that includes that positive joint cell), so
    # denom is guaranteed nonzero on this mask -- no division by zero,
    # no log(0), no nan/inf ever produced, and no errstate suppression
    # needed.
    mask = p_xy > 0
    if not np.any(mask):
        return 0.0

    ratio = p_xy[mask] / denom[mask]
    log2_ratio = np.log2(ratio)
    mi = float(np.sum(p_xy[mask] * log2_ratio))
    return max(mi, 0.0)


def _entropy_from_probs(p: np.ndarray) -> float:
    """Shannon entropy in bits from a probability vector (zeros are skipped)."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def normalized_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Normalized mutual information, scaled to [0, 1].

    NMI = MI(X;Y) / min(H(X), H(Y))

    Dividing by the smaller marginal entropy bounds the result at 1
    (achieved when one variable is a deterministic function of the
    other) and makes scores comparable across signals with different
    intrinsic variability -- unlike raw MI (in bits), which is not
    directly comparable across signals with different marginal entropy.

    Returns 0.0 if either marginal entropy is 0 (a constant signal or a
    constant step sequence carries no information to share).
    """
    x, y = _validate_inputs(x, y, bins)
    p_xy = _joint_histogram_probs(x, y, bins)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    h_x = _entropy_from_probs(p_x)
    h_y = _entropy_from_probs(p_y)
    min_h = min(h_x, h_y)

    if min_h <= 0.0:
        logger.warning(
            "normalized_mutual_information: a marginal entropy is 0 "
            "(constant signal or constant step sequence) -- returning 0.0"
        )
        return 0.0

    mi = binned_mutual_information(x, y, bins=bins)
    return float(np.clip(mi / min_h, 0.0, 1.0))


# --------------------------------------------------------------------------
# Permutation significance test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a permutation-based significance test for MI(signal; step).

    Attributes
    ----------
    observed_mi : float
        Raw plug-in MI (bits) computed on the real (unshuffled) data.
    observed_nmi : float
        Normalized MI (0-1) computed on the real (unshuffled) data.
    null_mean : float
        Mean raw MI (bits) across permutations (the null baseline).
    null_std : float
        Standard deviation of raw MI across permutations.
    p_value : float
        Fraction of permutations whose MI is >= the observed MI
        (one-sided test: is the signal MORE informative about step than
        chance would produce). Computed with a +1 correction in both
        numerator and denominator (Davison & Hinkley, 1997) so it is
        never exactly zero, which would misleadingly imply certainty.
    n_permutations : int
        Number of permutations used.
    null_method : str
        Which null-construction method produced this result -- either
        "circular_shift" (default) or "full_shuffle". See
        `permutation_test_mi`'s docstring for why this choice matters.
    """

    observed_mi: float
    observed_nmi: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    null_method: str = "circular_shift"


def permutation_test_mi(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 16,
    n_permutations: int = 200,
    seed: int = 0,
    null_method: str = "circular_shift",
) -> PermutationTestResult:
    """Test whether MI(x; y) is significantly greater than chance.

    Builds a null distribution and compares the observed MI against it.
    Because the same plug-in estimator (with the same finite-sample bias)
    is applied to both the real data and every null draw, the comparison
    is far more trustworthy than the raw MI value alone -- the bias is
    shared and largely cancels out in the comparison, even though it does
    not cancel out of the raw number.

    Null construction -- read this before choosing `null_method`
    --------------------------------------------------------------
    Two null methods are available, and the choice is NOT cosmetic --
    discovered via AuditBench's synthetic calibration sweep
    (`auditbench.py`), which found that the wrong choice produces a real
    false-positive failure mode on exactly the kind of signal this
    estimator exists to audit:

    - "full_shuffle" (the original implementation): randomly permutes all
      of x. This destroys x's own autocorrelation structure along with
      any real alignment to y. For an i.i.d. signal this is harmless. But
      every real adaptive-mechanism trigger signal this project audits
      (EMA-smoothed cosine divergence, gradient magnitude, etc.) is
      smooth and autocorrelated BY DESIGN -- that's what the EMA is for.
      A full-shuffle null strips that autocorrelation out of the null
      while the observed statistic keeps it, so the observed MI ends up
      far above the null purely because of serial correlation within a
      single trajectory, NOT because of any genuine dependence on step.
      AuditBench's alpha=1.0 case (a signal built to be genuinely
      independent of step, but autocorrelated like a real signal would
      be) was WRONGLY flagged as step-dependent 100% of the time under
      this null (observed NMI ~0.20, p ~0.005) -- a serious false-positive
      rate on precisely the population of signals this tool will see in
      practice.
    - "circular_shift" (default as of this fix): each null draw circularly
      rotates x by a random offset (`np.roll`) instead of fully shuffling
      it. This preserves x's exact autocorrelation structure and marginal
      distribution -- the null differs from the observed data only in
      WHERE the (otherwise identical) trajectory sits relative to step
      index, which is exactly the thing a "does this line up with step
      count more than chance" test should be asking. Re-running
      AuditBench with this null: the alpha=1.0 case is correctly
      classified as NOT step-dependent, while the alpha=0.0 case (a pure
      deterministic function of step) is still correctly flagged as
      significant, and the original two-point sanity check in
      `run_synthetic_validation` still passes. This is the standard fix
      for permutation testing on autocorrelated single time series (the
      "circular shift" / "phase randomization" family of tests used
      broadly in time-series and neuroscience statistics) applied here
      for the first time to this specific problem, as far as this
      project's literature search found.

    Use "full_shuffle" only if you have a specific reason to believe the
    signal under test is i.i.d. (no serial correlation) -- for anything
    resembling a real training-time monitored signal, "circular_shift" is
    the correct default and the only one validated against AuditBench's
    graded synthetic family end-to-end.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Typically x = the monitored signal's values, y = the step index
        (or step count) at which each value was recorded.
    bins : int, default 16
    n_permutations : int, default 200
        More permutations give a finer-grained p-value floor
        (1 / (n_permutations + 1)) at the cost of compute; 200 is a
        reasonable balance for the scale of signals in this project
        (thousands of logged points, not millions).
    seed : int, default 0
        Seed for the permutation RNG, for reproducibility.
    null_method : str, default "circular_shift"
        Either "circular_shift" or "full_shuffle". See above.

    Returns
    -------
    PermutationTestResult
    """
    x, y = _validate_inputs(x, y, bins)
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    if null_method not in ("circular_shift", "full_shuffle"):
        raise ValueError(f"null_method must be 'circular_shift' or 'full_shuffle', got {null_method!r}")

    observed_mi = binned_mutual_information(x, y, bins=bins)
    observed_nmi = normalized_mutual_information(x, y, bins=bins)

    rng = np.random.default_rng(seed)
    null_mis = np.empty(n_permutations, dtype=np.float64)
    n = x.shape[0]

    if null_method == "full_shuffle":
        x_shuffled = x.copy()
        for i in range(n_permutations):
            rng.shuffle(x_shuffled)
            null_mis[i] = binned_mutual_information(x_shuffled, y, bins=bins)
    else:  # circular_shift
        for i in range(n_permutations):
            offset = int(rng.integers(1, n)) if n > 1 else 0
            x_rolled = np.roll(x, offset)
            null_mis[i] = binned_mutual_information(x_rolled, y, bins=bins)

    # +1 correction in numerator and denominator: avoids a p-value of
    # exactly 0.0, which would overstate certainty given a finite number
    # of permutations (Davison & Hinkley, 1997, "Bootstrap Methods and
    # their Application", standard practice for permutation tests).
    n_as_extreme = int(np.sum(null_mis >= observed_mi))
    p_value = (n_as_extreme + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_mi=observed_mi,
        observed_nmi=observed_nmi,
        null_mean=float(np.mean(null_mis)),
        null_std=float(np.std(null_mis)),
        p_value=p_value,
        n_permutations=n_permutations,
        null_method=null_method,
    )


# --------------------------------------------------------------------------
# Falsification-battery control construction
# --------------------------------------------------------------------------


def scrambled_signal_control(signal: np.ndarray, seed: int = 0) -> np.ndarray:
    """Construct the 'scrambled-signal control' version of a real signal.

    Shuffles the signal's values across the (fixed) sequence of steps at
    which they were recorded, preserving the signal's own marginal
    distribution while destroying any real temporal/causal relationship
    it had to training progress. This is the array to feed into a
    reproduction run in place of the real signal for the scrambled-signal
    condition described in the experiment design (§7.1).

    Parameters
    ----------
    signal : array-like, shape (n,)
        The real logged signal values, in step order.
    seed : int, default 0

    Returns
    -------
    np.ndarray
        A shuffled copy of `signal`, same shape and same set of values,
        different order.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.shape[0] == 0:
        raise ValueError("signal must be non-empty")
    rng = np.random.default_rng(seed)
    shuffled = signal.copy()
    rng.shuffle(shuffled)
    return shuffled


# --------------------------------------------------------------------------
# Synthetic ground-truth signals, for validating the estimator itself
# --------------------------------------------------------------------------


def deterministic_step_signal(n_steps: int, noise_std: float = 0.02, seed: int = 0) -> np.ndarray:
    """A signal that IS just a (noisy) deterministic function of step count.

    Modeled as a saturating ramp (like an EMA-smoothed quantity climbing
    toward a ceiling), which is qualitatively similar in shape to the
    kind of divergence trajectory these mechanisms actually log, plus
    small Gaussian noise. A correct MI estimator MUST report high
    MI/NMI and a low permutation p-value for this signal against step
    count, since it is constructed to have no information content beyond
    step count.

    Parameters
    ----------
    n_steps : int
        Number of (signal, step) pairs to generate.
    noise_std : float, default 0.02
        Standard deviation of additive Gaussian noise.
    seed : int, default 0

    Returns
    -------
    np.ndarray, shape (n_steps,)
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps, dtype=np.float64)
    normalized_steps = steps / max(n_steps - 1, 1)
    # Saturating ramp: 1 - exp(-k * t), qualitatively similar to an
    # EMA-smoothed divergence signal rising toward a ceiling.
    ramp = 1.0 - np.exp(-5.0 * normalized_steps)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_steps)
    return ramp + noise


def independent_noise_signal(n_steps: int, seed: int = 1) -> np.ndarray:
    """A signal that is i.i.d. noise, independent of step count.

    A correct MI estimator MUST report low MI/NMI (near the noise floor
    set by finite-sample bias) and a high permutation p-value for this
    signal against step count, since by construction it carries no
    information about step.

    Parameters
    ----------
    n_steps : int
    seed : int, default 1

    Returns
    -------
    np.ndarray, shape (n_steps,)
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.5, scale=0.2, size=n_steps)


def autocorrelated_independent_signal(n_steps: int, mean_reversion: float = 0.02, seed: int = 2) -> np.ndarray:
    """A SMOOTH, autocorrelated signal that is nonetheless independent of step count.

    A mean-reverting AR(1)-style random walk: each value depends on its
    own previous value plus fresh noise, never on the step index itself.
    Real monitored training signals (EMA-smoothed divergence, gradient
    magnitude, etc.) look like this -- smooth and autocorrelated by
    design -- not like i.i.d. noise. This case exists because a
    full-shuffle permutation null gets this one WRONG: shuffling destroys
    the signal's own autocorrelation along with any real step-alignment,
    so the observed MI (inflated by autocorrelation alone) ends up far
    above a full-shuffle null even though the signal carries no true
    information about step. Discovered via AuditBench
    (`auditbench.py`)'s graded synthetic sweep, not by inspection -- this
    case is why `permutation_test_mi` defaults to `null_method=
    "circular_shift"`, which handles it correctly.

    A correct estimator+null combination MUST report a high permutation
    p-value for this signal against step count, same as
    `independent_noise_signal`, DESPITE this signal's raw MI being much
    higher than i.i.d. noise's (autocorrelation alone inflates the raw
    plug-in MI -- that inflation is exactly what the null must also
    capture, which only the circular-shift null does).

    Parameters
    ----------
    n_steps : int
    mean_reversion : float, default 0.02
        Pull-back-to-zero strength per step, in [0, 1).
    seed : int, default 2

    Returns
    -------
    np.ndarray, shape (n_steps,)
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    rng = np.random.default_rng(seed)
    x = np.empty(n_steps, dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    innovations = rng.normal(0.0, 1.0, size=n_steps - 1)
    for t in range(1, n_steps):
        x[t] = (1.0 - mean_reversion) * x[t - 1] + innovations[t - 1]
    return x


# --------------------------------------------------------------------------
# Self-validation entry point
# --------------------------------------------------------------------------


def run_synthetic_validation(
    n_steps: int = 2000,
    bins: int = 16,
    n_permutations: int = 200,
    nmi_pass_threshold: float = 0.3,
    nmi_fail_threshold: float = 0.1,
    p_value_significant: float = 0.05,
) -> bool:
    """Run the estimator on both synthetic ground-truth cases and check separation.

    This is the check required by §7.2 of the experiment design before
    the estimator is trusted on any real mechanism's logged signal.

    Parameters
    ----------
    n_steps : int, default 2000
        Length of each synthetic signal (comparable order of magnitude
        to a real per-language divergence log at the check interval
        described in the original proposal, e.g. every 250 steps over a
        few hundred thousand total steps would give roughly this many
        points; adjust to match your actual logging interval).
    bins : int, default 16
    n_permutations : int, default 200
    nmi_pass_threshold : float, default 0.3
        The deterministic-vs-step case must exceed this NMI to count as
        correctly detected as step-dependent.
    nmi_fail_threshold : float, default 0.1
        The independent-noise case must fall below this NMI to count as
        correctly detected as step-independent.
    p_value_significant : float, default 0.05
        Significance threshold for the permutation test.

    Returns
    -------
    bool
        True if the estimator correctly separates both synthetic cases.

    Raises
    ------
    AssertionError
        If the estimator fails to separate the two cases -- this means
        the estimator implementation (or its bin count / sample size)
        needs to be fixed before it is applied to any real mechanism.
    """
    steps = np.arange(n_steps, dtype=np.float64)

    logger.info("Case A: deterministic function of step (expect HIGH MI, LOW p-value)")
    det_signal = deterministic_step_signal(n_steps, seed=0)
    det_result = permutation_test_mi(det_signal, steps, bins=bins, n_permutations=n_permutations, seed=0)
    logger.info(
        "  observed_mi=%.4f bits, observed_nmi=%.4f, null_mean=%.4f, null_std=%.4f, p_value=%.4f",
        det_result.observed_mi,
        det_result.observed_nmi,
        det_result.null_mean,
        det_result.null_std,
        det_result.p_value,
    )

    logger.info("Case B: independent noise (expect LOW MI, HIGH p-value)")
    noise_signal = independent_noise_signal(n_steps, seed=1)
    noise_result = permutation_test_mi(noise_signal, steps, bins=bins, n_permutations=n_permutations, seed=1)
    logger.info(
        "  observed_mi=%.4f bits, observed_nmi=%.4f, null_mean=%.4f, null_std=%.4f, p_value=%.4f",
        noise_result.observed_mi,
        noise_result.observed_nmi,
        noise_result.null_mean,
        noise_result.null_std,
        noise_result.p_value,
    )

    logger.info(
        "Case C: smooth, autocorrelated, but step-INDEPENDENT signal "
        "(expect HIGH raw NMI but HIGH p-value on most draws -- this is the case "
        "that broke the original full_shuffle null; see "
        "autocorrelated_independent_signal's docstring). This is a rate property, "
        "not a single-draw one: a well-calibrated p<0.05 test is EXPECTED to "
        "false-positive on roughly 5%% of truly-null draws by construction, so this "
        "case is checked across multiple seeds, not one."
    )
    n_case_c_seeds = 20
    autocorr_p_values = []
    for seed in range(n_case_c_seeds):
        autocorr_signal = autocorrelated_independent_signal(n_steps, seed=100 + seed)
        r = permutation_test_mi(autocorr_signal, steps, bins=bins, n_permutations=n_permutations, seed=seed)
        autocorr_p_values.append(r.p_value)
    autocorr_p_values = np.array(autocorr_p_values)
    autocorr_nonsignificant_rate = float(np.mean(autocorr_p_values >= p_value_significant))
    logger.info(
        "  across %d seeds: median_p=%.4f, fraction correctly non-significant=%.2f",
        n_case_c_seeds,
        float(np.median(autocorr_p_values)),
        autocorr_nonsignificant_rate,
    )

    det_correct = (det_result.observed_nmi >= nmi_pass_threshold) and (
        det_result.p_value < p_value_significant
    )
    noise_correct = (noise_result.observed_nmi < nmi_fail_threshold) or (
        noise_result.p_value >= p_value_significant
    )
    # Case C is a rate property (see above): a correctly-calibrated alpha=0.05
    # test WILL false-positive on ~5% of truly-null draws by design, so the bar
    # here is "well above chance-level calibration", not "zero false positives".
    # The original full_shuffle null failed this at essentially 0% correct
    # across repeated draws (a systematic bias, not Monte Carlo noise); the
    # circular_shift null clears it comfortably.
    autocorr_min_nonsignificant_rate = 0.80
    autocorr_correct = autocorr_nonsignificant_rate >= autocorr_min_nonsignificant_rate

    assert det_correct, (
        f"Estimator FAILED to detect the deterministic step-signal as step-dependent "
        f"(nmi={det_result.observed_nmi:.4f}, p={det_result.p_value:.4f}). "
        f"Do not trust this estimator on real mechanism signals until this is fixed "
        f"-- try more bins, more samples, or check the binning implementation."
    )
    assert noise_correct, (
        f"Estimator INCORRECTLY flagged independent noise as step-dependent "
        f"(nmi={noise_result.observed_nmi:.4f}, p={noise_result.p_value:.4f}). "
        f"This means the estimator has a false-positive problem -- likely too few "
        f"samples for the chosen bin count. Increase n_steps or decrease bins."
    )
    assert autocorr_correct, (
        f"Estimator's false-positive rate on smooth, step-independent, autocorrelated "
        f"signals is too high: only {100 * autocorr_nonsignificant_rate:.0f}% of {n_case_c_seeds} "
        f"seeds were correctly non-significant (need >= {100 * autocorr_min_nonsignificant_rate:.0f}%). "
        f"This is the autocorrelation false-positive AuditBench discovered -- do not "
        f"trust this estimator on real (smooth, autocorrelated) mechanism signals "
        f"until it passes. Check that null_method='circular_shift' is in use; a "
        f"full_shuffle null is known to fail this case at close to a 0% pass rate."
    )

    logger.info(
        "PASS: estimator correctly separates step-dependent, step-independent, and "
        "autocorrelated-but-step-independent signals."
    )

    # Sanity-check the scrambled_signal_control utility too, since it will
    # be used to build the actual scrambled-signal condition for the real
    # audit runs: scrambling the deterministic signal should destroy its
    # MI with step, converging it toward the noise case's behavior.
    scrambled_det = scrambled_signal_control(det_signal, seed=42)
    scrambled_result = permutation_test_mi(
        scrambled_det, steps, bins=bins, n_permutations=n_permutations, seed=2
    )
    logger.info(
        "Scrambled version of the deterministic signal: observed_nmi=%.4f, p_value=%.4f "
        "(expected to resemble Case B, confirming scrambled_signal_control works as intended)",
        scrambled_result.observed_nmi,
        scrambled_result.p_value,
    )
    assert scrambled_result.observed_nmi < nmi_pass_threshold, (
        "scrambled_signal_control did not destroy the step-relationship as expected -- "
        "check the shuffle implementation."
    )

    return True


if __name__ == "__main__":
    run_synthetic_validation()
