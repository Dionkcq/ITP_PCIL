# trigger.py — Usage

Generic time-slicer for tabular machine data. Reads a CSV, returns a
slice of its rows, and writes the slice to a new CSV. Intended to sit
between the (future) shop-floor database and `preprocess.py`.

## Setup

Run from the `PCIL_dev/` root:

```
cd C:\Users\dionk\Desktop\University\Trimesters\Year2Trimester3\ITP\PCIL_dev
```

Dependencies are already pinned by the project's `pip install` line
(`pandas` + `numpy`). No extras needed.

## Slice modes

Pick **one** of these per call:

| Mode | Flags | Returns |
|---|---|---|
| Time range | `--start <ISO> --end <ISO>` | Rows where timestamp is between `start` and `end` (inclusive). |
| Last N rows | `--last <N>` | The most recent `N` rows. |

The CLI rejects calls that use both modes or neither.

## CLI flags

| Flag | Purpose | Default |
|---|---|---|
| `--input` | Path to the source CSV. **Required.** | — |
| `--output` | Path for the saved slice. | `triggered_slices/slice_<run_time>.csv` under the current directory |
| `--start` | ISO 8601 start timestamp (with `--end`) | — |
| `--end` | ISO 8601 end timestamp (with `--start`) | — |
| `--last` | Number of most-recent rows to keep | — |
| `--timestamp-column` | Name of the timestamp column | `timestamp` |
| `--sep` | CSV delimiter | auto-detected (comma, semicolon, etc.) |
| `--preview-rows` | How many rows to preview on stdout | `10` (set `0` to skip) |

## Commands

**1. Slice by time range — Clean_Data.csv (its timestamp column is `_time`):**

```
python pcil/trigger.py --input ../data/Clean_Data.csv --timestamp-column _time --start "2025-08-08 10:08:18+00:00" --end "2025-08-08 10:08:19+00:00"
```

**2. Slice the last 1000 rows:**

```
python pcil/trigger.py --input ../data/Clean_Data.csv --last 1000
```

**3. Slice by time range, save to a named file:**

```
python pcil/trigger.py --input ../data/Clean_Data.csv --timestamp-column _time --start "2025-08-08 10:08:18+00:00" --end "2025-08-08 10:08:19+00:00" --output triggered_slices/morning_window.csv
```

**4. Slice last N from a comma-delimited shop-floor sample (default `timestamp` column works):**

```
python pcil/trigger.py --input machines/inkjet_printer/output/sample_shop_floor_slice.csv --last 200
```

**5. Force a delimiter (skip auto-detect):**

```
python pcil/trigger.py --input ../data/Clean_Data.csv --sep ";" --timestamp-column _time --last 500
```

**6. Skip the stdout preview (useful in scripts):**

```
python pcil/trigger.py --input ../data/Clean_Data.csv --last 100 --preview-rows 0
```

## Output

Slices land in a new CSV file. By default:

```
PCIL_dev/triggered_slices/slice_<UTC timestamp>.csv
```

Pass `--output path/to/file.csv` to override. Parent directories are
created automatically.

The output preserves the input's columns exactly — same column names,
same dtypes — just with fewer rows.

## What to do with the slice

Feed it into Pipeline #1:

```
python pcil/preprocess.py --input triggered_slices/<slice file>.csv
```

Then adapter and context-model training as normal:

```
python pcil/adapter.py
python pcil/train_context_model.py
```

## Errors and what they mean

| Message | Meaning | Fix |
|---|---|---|
| `Pass either --start AND --end, or --last.` | No slice mode given. | Add one. |
| `Pass either --start/--end or --last, not both.` | Conflicting flags. | Drop one. |
| `KeyError: ... no column 'timestamp'` (and lists available columns) | The timestamp column is named differently. | Pass `--timestamp-column <actual name>` (e.g. `_time` for `Clean_Data.csv`). |
| `ValueError: time data ... doesn't match format` | Mixed timestamp precisions; `format="ISO8601"` should already handle this — if you see it, the source isn't ISO 8601. | Convert the source to ISO 8601 first, or extend `slice_by_time` to accept a custom format. |

## When the shop-floor DB goes live

The CLI shape doesn't change. The internal `pd.read_csv(...)` call gets
replaced by a Postgres pull, but `slice_by_time` and `slice_last_n_rows`
keep their signatures so existing callers still work.
