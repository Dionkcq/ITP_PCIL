# Irregular anomaly pipeline

Anomaly detection for **irregularly-sampled** time-series data: event
logs, error flags, on-change sensor reports — any stream where the gap
between consecutive timestamps is not constant.

This completes the three-data-type architecture from the Week-2 design
(cyclical / non-cyclical / irregular). The first two were built against
real inkjet datasets; this one is a **pipeline definition** in Winardi's
sense — the deliverable is the reusable slicing + features + training
procedure, demonstrated on synthetic event data in `tests/`, ready to
instantiate per machine when real irregular data arrives.

## Why the other two pipelines don't cover this case

| | cyclical | non_cyclical | irregular |
|---|---|---|---|
| Slicing unit | one signal cycle (peak detection) | fixed **row count** (12,800 rows = 0.5 s at 25.6 kHz) | fixed **duration** (wall-clock seconds) |
| Assumes uniform sample rate | yes | yes | **no** |
| Features | resampled 100-pt waveform | RMS / kurtosis / FFT bands | event rate + inter-arrival gaps (+ value stats) |
| Model | 1D CNN autoencoder | Random Forest (supervised) | Isolation Forest (unsupervised) |

Row-count windows break on irregular data because N rows can span 2
seconds or 2 hours. Waveform/spectral features break because there is
no continuous curve to resample — interpolating 3 events to 100 points
manufactures structure that was never measured.

## Pipeline (Winardi's four-step structure)

1. **Slice** (`slice.py`) — fixed-duration tumbling windows over
   wall-clock time. **Empty windows are kept**: for event data,
   silence (a machine that stopped reporting) is often the anomaly.
2. **Features** (`features.py`) — per window: `event_count`,
   `mean_interval`, `std_interval`, `max_interval`; plus
   `value_mean/std/min/max` when a numeric value column exists.
   Sparse windows saturate gap features at the window length.
3. **Normalise** — shared `base.PerMachineNormaliser` (z-score per
   machine_id), fit on clean training data only.
4. **Model** (`model.py`) — Isolation Forest, unsupervised (event logs
   rarely have labels). `.fit(X)` / `.score(X)` per the `AnomalyModel`
   ABC; higher score = more anomalous.

`train.py` glues 1-4 into a bundle `.pkl` (model + normaliser +
metadata + a 95th-percentile starting threshold, same convention as
cyclical). `score.py` reapplies the bundle to new data and returns
per-window `anomaly_score` + `is_anomaly`.

## Usage

```bash
# Train on normal-operation data
python -m pcil.utils.anomaly.irregular.train \
    --input data/irregular_dataset.csv \
    --output data/irregular_inkjet_01.pkl \
    --window-seconds 1.0 --value-column signal_value

# Score new data
python -m pcil.utils.anomaly.irregular.score \
    --input data/irregular_eval.csv \
    --model data/irregular_inkjet_01.pkl \
    --output data/irregular_eval_scored.csv
```

Or via the orchestrator: `POST /anomaly/train` with
`model_type=irregular, training_mode=normal_only`, then
`POST /anomaly/score` with `model_type=irregular`.

Input CSV needs: a machine id column, a parseable timestamp column,
and (optionally) one numeric value column. The bundle records which
columns were used at train time.

## Tuning

- `window_seconds` is the main knob. Set it to the time-scale of the
  anomalies you care about: bursts/stalls shorter than the window get
  averaged away; much longer than the typical event gap and most
  windows look identical.
- Long idle periods (machine off overnight) produce one empty window
  per stride and will dominate training. Slice the input to the
  operating period first.

## Contract (same as the other subpackages)

Input -> output only. Receives a time series, returns scores. Never
reads or writes the shop-floor DB — the engineer writes scores into
the correct rows themselves.
