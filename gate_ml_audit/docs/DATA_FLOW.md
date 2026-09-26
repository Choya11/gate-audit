# Data Flow Diagrams

[← back to README](../README.md)

All diagrams are [Mermaid](https://mermaid.js.org/) — they render natively in GitHub's file viewer, no extra tooling needed.

## End-to-end data flow, one training run

```mermaid
flowchart LR
    CFG[RunConfig] --> DS[DataSource\nreal or synthetic]
    DS -->|token batches| LOOP[Training loop]
    CFG --> MECH[condition-wrapped\nMechanism]
    MECH --> LOOP
    LOOP -->|every step| STATE[TrainingState]
    STATE --> SIG[mechanism.signal]
    SIG --> HIST[signal_history list]
    HIST --> INT[mechanism.should_intervene]
    INT -->|fires| APPLY[mechanism.apply]
    LOOP --> LOG[(JSONL log\none line per step)]
    LOOP -->|trigger event OR periodic interval| CKPT[(CheckpointState)]
    LOOP -.periodic cadence.-> EVAL[eval/harness.py]
    EVAL --> EVALRESULT[EvalResult]
```

## The Real → Step-matched sequencing dependency

```mermaid
sequenceDiagram
    participant R as Real run (mechanism, seed)
    participant L as R's JSONL log
    participant T as battery/trigger_log.py
    participant S as Step-matched run (same mechanism, seed)

    R->>L: writes intervention_fired per step
    Note over R,L: Real must run to COMPLETION first
    T->>L: extract_trigger_steps()
    T->>T: to_fixed_schedule()
    T->>S: fixed schedule (Callable[[int], bool])
    Note over S: should_intervene() now ignores\nthe live signal entirely
```

This is the only cross-run dependency in the whole pipeline. Scrambled has no such dependency and can run whenever.

## From a finished run to the report — including the gap

```mermaid
flowchart TD
    A[Completed run: JSONL log + CheckpointState] --> B[EvalResult\nvia eval/harness.py]
    A --> C["⚠ NOT YET BUILT:\nsignal column → mi_estimator.permutation_test_mi"]
    B --> D["⚠ NOT YET BUILT:\nBatteryResult assembly"]
    C --> D
    D -->|"stats/result.py schema\nalready exists + tested"| E[(BatteryResult JSON files)]
    E --> F[orchestration/report.py]
    F -->|filters to data_source==real| G{any real results?}
    G -->|no| H[raise DataProvenanceError]
    G -->|yes| I[stats/: bootstrap CI per condition]
    I --> J[stats/failure_classifier.py\n4-bucket a/b/c/d]
    I --> K[stats/c3_correlation.py\n3 mechanisms, Spearman]
    J --> L[viz/pass_fail_heatmap.py]
    K --> M[viz/c3_scatter.py]

    style C fill:#4a1f1d,stroke:#c0392b,color:#fff
    style D fill:#4a1f1d,stroke:#c0392b,color:#fff
```

The schema (`BatteryResult`), the consumer (`report.py`), and the ingredient (`mi_estimator.permutation_test_mi`, pre-existing and verified) all exist and are tested. The glue connecting them does not exist yet — see [`docs/ADVANCED.md`](ADVANCED.md).

## Checkpoint retention over one run's lifetime

```mermaid
flowchart LR
    A[step N] -->|trigger fires| B[trigger-event checkpoint\nKEPT FOREVER]
    A -->|periodic interval| C[periodic checkpoint]
    C --> D{more than\nperiodic_retain\non disk?}
    D -->|yes| E[delete oldest]
    D -->|no| F[keep]
    G[run completes] --> H[delete ALL periodic\nkeep only trigger-events + final]
```
