"""
Anomaly detection utilities.

Layout:
    base.py       — shared AnomalyModel ABC + PerMachineNormaliser
    cyclical/     — opinionated cyclical pipeline (Daniel + Jaymon)
    non_cyclical/ — opinionated non-cyclical pipeline (Zi Hin)

Each subpackage exposes:
    score(df, bundle) -> DataFrame    — entry point the orchestrator calls
    a model class inheriting AnomalyModel
    a {subpackage}_config.yaml         — recipe for that pipeline

The orchestrator dispatches /anomaly/score to the matching subpackage by
`model_type`. Each subpackage is internally free to use whatever slicing,
features, and model it wants — only the public interface is shared.
"""
