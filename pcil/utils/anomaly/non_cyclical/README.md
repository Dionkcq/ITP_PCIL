# Non-cyclical anomaly pipeline

Anomaly detection for continuous, non-repeating sensor streams
(vibration, acoustic emission, temperature, pressure).

The deliverable is the **pipeline definition** — not one trained model.
Each machine runs the same pipeline on its own data and produces its own
fitted `.pkl` bundle. Same recipe, different weights per machine.

---

## File overview

| File | Purpose |
|---|---|
| `slice.py` | Fixed-window slicing |
| `features.py` | Per-window feature extraction |
| `model.py` | RandomForest anomaly classifier |
| `normalise.py` | Per-machine z-score normaliser |
| `score.py` | Score new data using a trained `.pkl` |
| `run.py` | End-to-end runner (train + evaluate) |
| `non_cyclical_config.yaml` | All settings — edit this to deploy to a new machine |
---

## How to run

```bash
# From ITP_PCIL/
python pcil/utils/anomaly/non_cyclical/run.py
```

Edit `non_cyclical_config.yaml` to change data paths, machine ID,
window size, or train/test ratio. No Python changes needed.

---

## Four-step design

### Step 1 — Slicing (`slice.py`)
**Method:** Fixed-length windows  
**Window size:** 0.5 s = 12,800 rows at 25.6 kHz  
**Why:** Non-cyclical data has no natural repeat boundary. Fixed windows
are predictable and give enough samples for reliable statistics without
averaging out short transient faults.

### Step 2 — Feature extraction (`features.py`)
**Method:** Time-domain statistics + FFT band energies  
**Features:** RMS, peak, std, kurtosis, crest factor, fft_band_low, fft_band_mid, fft_band_high  
**Channels:** Acceleration 0/1/2 (g) + AE (V)(V) = 4 channels  
**Feature vector size:** 4 × 8 = **32 features per window**

| Feature | Physical meaning |
|---|---|
| RMS | Overall vibration energy |
| Peak | Maximum single spike; catches impact events |
| Std | Signal spread; unstable motion → high std |
| Kurtosis | Impulsiveness; mechanical faults cause non-Gaussian spikes |
| Crest factor | Peak/RMS; dimensionless ratio for impulsive events |
| FFT band low | Energy in 0–800 Hz range |
| FFT band mid | Energy in 800 Hz–3.2 kHz range |
| FFT band high | Energy above 3.2 kHz |

### Step 3 — Per-machine normalisation (`normalise.py`)
**Method:** Z-score per machine, fitted on clean training data only  
**Why:** Different machines have different absolute vibration levels.
Z-scoring against each machine's own baseline makes the same pipeline
redeployable to new machines without code changes.

### Step 4 — Model (`model.py`)
**Default:** Random Forest (supervised)  
**Why:** Since labelled clean/anomaly files are available, supervised
learning gives a much cleaner decision boundary than unsupervised methods.

| Model | Type | When to use |
|---|---|---|
| `RandomForestModel` | Supervised | **Default.** Use when anomaly file is available. |

---

## Evaluation results

Train/test split: 80% train, 20% test (time-series order preserved, no shuffle).  
Threshold: q=0.50 (95th percentile of clean test scores).

| Model | Precision | Recall | Notes |
|---|---|---|---|
| Isolation Forest (unsupervised) | 0.615 | 0.618 | No labels needed |
| **Random Forest (supervised)** | **0.586** | **0.680** | **Current default** |

Random Forest achieves higher recall (catches more real anomalies) at
the cost of slightly lower precision (slightly more false alarms).

---

## Known limitations

- **One machine only.** Per-machine normalisation is implemented and
  validated as a code path. Cross-machine validation will be possible
  when data from additional machines is collected.
- **Single fault type.** The anomaly files were recorded by physically
  knocking on the robotic arm. Real factory faults (bearing wear,
  misalignment, overheating) may have different signatures.
- **Batch only.** The current pipeline processes files. Streaming
  window inference is a future extension.

## Generalisation note

`FEATURE_REGISTRY` in `features.py` also contains `mean`,
`gradient_mean`, and `integrated_area` for slow-moving signals such as
temperature or pressure. Override the `feature_names` argument in
`extract_features()` to use a different subset for non-vibration sensors.