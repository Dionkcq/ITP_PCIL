# Instructional Prompt — Minimal RAG Integration into ITP/PCIL

## Purpose

This document is a precise, step-by-step integration guide for adding a
minimal Retrieval-Augmented Generation (RAG) system into the existing
ITP/PCIL codebase. It is written for a developer (or an AI coding agent)
who has read the full codebase and must follow the conventions already
established there.

Read this entire document before writing a single line of code.

---

## Codebase snapshot — what already exists
PCIL_dev/
├── pcil/
│   ├── orchestrator.py          ← FastAPI app; THIS is where RAG output plugs in
│   ├── preprocess.py
│   ├── adapter.py
│   ├── train_context_model.py
│   ├── trigger.py
│   └── rag/
│       ├── __init__.py          ← empty
│       ├── loader.py            ← implemented: load_docx(), load_all_recovery_docs()
│       ├── lookup.py            ← implemented: lookup_keywords()
│       ├── composer.py          ← Gemini composer
│       ├── prototype.py         ← WIRED CLI — do not break its interface
│       └── README.md            ← design spec, read it
├── data/
│   └── RAG/
│       └── Screen Printer.docx  ← source document, DO NOT EDIT
├── requirements.txt
├── rag_frontend/
│   ├── app.py
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── machines/inkjet_printer/config.yaml

### The impacts dict schema (live-generated, used by composer)

`train_context_model.py` generates this structure on every request.
Do NOT read from the stored `machines/inkjet_printer/output/context_model_impacts.json`
— that file uses an older schema and is only kept for reference.

```json
{
  "system": "inkjet_printer",
  "model": "linear_regression",
  "fitted_at": "...",
  "context_window": {
    "start_time": "...", "end_time": "...",
    "row_count": 625, "feature_count": 8, "target_count": 4
  },
  "context": [
        {
            "target": "availability",
            "intercept": 0.849,
            "ranked_feature_impacts": [
                {
                    "feature": "air_pressure_low_ratio",
                    "description": "Proportion of operating time during which air pressure falls below the defined threshold.",
                    "raw_impact_score": 0.157,
                    "rank": 1
                }
            ]
        }
  ]
}
```

### The two placeholders you are replacing

In `pcil/orchestrator.py`, inside `_run_pipeline_on_df()`:

```python
return {
    ...
    "recovery_records": [],                          # ← REPLACE THIS
    "operator_recommendation": (
        "<LLM composer not wired yet — see deliverables/Week3/todo.md>"
    ),                                               # ← REPLACE THIS
    ...
}
```

### The TypedDict contract (do not change its shape)

```python
# pcil/rag/loader.py
class RecoveryRecord(TypedDict):
    error:      str
    cause:      str
    recovery:   str
    source_doc: str   # DOCX filename, for traceability
```

`recovery_records` in the API response must be a list of these dicts.

---

## Constraints — follow every one

1. **Do not edit anything inside `data/RAG/`.**
2. **Do not change the function signatures** of `load_docx`,
   `load_all_recovery_docs`, or `lookup_keywords` — `prototype.py` calls
   them directly and its CLI must keep working.
3. **Do not introduce a database** for this iteration. Add a one-line
   comment where persistence is relevant:
   `# TODO: when containerising, replace in-process cache with pgvector on PostgreSQL`
4. **Use Google AI Studio (Gemini)** for LLM generation. Store the API key
   in an environment variable named `GEMINI_API_KEY`. Never hard-code it.
   Import the key with `os.environ.get("GEMINI_API_KEY")` and raise a
   clear `RuntimeError` if it is missing at call time (not at import time).
5. **Keyword matching only** for retrieval in this iteration (v1).
   Vector similarity search is a future extension; do not implement it now.
6. **Follow the existing code style**: `from __future__ import annotations`,
   full docstrings, type hints, no bare `except`, raise `ValueError` on bad
   input just as `preprocess.py` and `adapter.py` do.
7. **Add new dependencies to `requirements.txt`** using `>=` version floors,
   consistent with the existing entries. Do not pin to exact versions.
8. **No separate Dockerfile** for the RAG layer. The existing Dockerfile
   already copies `pcil/rag/` into the image.
9. The Flask frontend is a **separate process** from the FastAPI orchestrator.
   It must not import from `pcil` directly; it communicates only via HTTP.
10. **All imports in `orchestrator.py` must remain at module level.** Do not
    place `import` statements inside function bodies.
11. **The existing pytest suite must continue to pass** after your changes.
    Verify with `python -m pytest tests/ -v` before considering the work done.

---

## Environment files and templates (repo-local additions)

- `.envexample` is the canonical template for local environment values.
- Each developer should copy `.envexample` to `.env` and fill in their own
    secrets (do not commit `.env`).
- `data/schema/schema.sql` is a future-facing template for a PostgreSQL +
    pgvector deployment. It is **not** used by the current runtime and should
    be overridden during containerisation.
- Frontend dependencies are installed via the root `requirements.txt`.

---

## Run the application (local)

1) Create your local environment file (do not commit it):

```bash
copy .envexample .env
```

Fill in these keys in `.env`:
- `GEMINI_API_KEY` (required)
- `ORCHESTRATOR_URL` (for the Flask UI, default: http://localhost:8000)
- `PCIL_CONFIG_PATH` (default: machines/inkjet_printer/config.yaml)
- `PCIL_PROJECT_ROOT` (optional override for Docker/dev)

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Start the orchestrator (terminal 1):

```bash
uvicorn pcil.orchestrator:app --host 0.0.0.0 --port 8000
```

4) Start the Flask frontend (terminal 2):

```bash
python rag_frontend/app.py
```

5) Open `http://localhost:5000` and click "Run Pipeline".

---

## Step 1 — Implement `pcil/rag/loader.py`

Fill in the two stub functions. Do not change the function signatures or
the `RecoveryRecord` TypedDict.

### Module-level caches (add at the top of the file)

```python
# Per-file parse cache: path string → list of records parsed from that file.
_RECORD_CACHE: dict[str, list[RecoveryRecord]] = {}

# Aggregate cache: str(rag_dir) → full concatenated list across all docs.
# Populated by load_all_recovery_docs on first call per directory.
# TODO: when containerising, replace in-process cache with pgvector on PostgreSQL
_ALL_RECORDS_CACHE: dict[str, list[RecoveryRecord]] = {}
```

### `load_docx(docx_path: Path) -> list[RecoveryRecord]`

- Check `_RECORD_CACHE` first; return the cached list if present.
- Open the DOCX with `from docx import Document`.
- Walk `doc.paragraphs` (and tables if needed — inspect the file first).
- Screen Printer.docx uses a repeating heading pattern:
  **Error Message** / **Root Cause** / **Recovery Steps**.
  Detect these headings case-insensitively (strip and lower before comparing)
  and group each trio into one `RecoveryRecord`.
- Set `source_doc` to `docx_path.name` (just the filename, not the full path).
- Store the result in `_RECORD_CACHE[str(docx_path)]` before returning.
- Return an empty list (not an exception) if the file has no parseable
  blocks, so callers handle a missing/malformed doc gracefully.
- Log a warning to stderr if fewer than 5 records are found, because
  Screen Printer.docx should have 19.

```python
import sys

def load_docx(docx_path: Path) -> list[RecoveryRecord]:
    """Parse one DOCX into a list of RecoveryRecord dicts. Results are
    cached in _RECORD_CACHE to avoid re-parsing on subsequent calls."""
    cache_key = str(docx_path)
    if cache_key in _RECORD_CACHE:
        return _RECORD_CACHE[cache_key]

    from docx import Document  # noqa: PLC0415 — keep docx as optional dep
    doc = Document(docx_path)
    records: list[RecoveryRecord] = []

    current: dict[str, str] = {}
    heading_map = {
        "error message": "error",
        "root cause":    "cause",
        "recovery steps": "recovery",
    }

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        key = heading_map.get(text.lower())
        if key:
            current[key] = ""
        elif current:
            # Accumulate body text under the most recently seen heading
            last_key = list(current)[-1]
            current[last_key] = (current[last_key] + " " + text).strip()

        if all(k in current for k in ("error", "cause", "recovery")):
            records.append(RecoveryRecord(
                error=current["error"],
                cause=current["cause"],
                recovery=current["recovery"],
                source_doc=docx_path.name,
            ))
            current = {}

    if len(records) < 5:
        print(
            f"[loader] WARNING: only {len(records)} records found in "
            f"{docx_path.name}; expected ≥ 5.",
            file=sys.stderr,
        )

    _RECORD_CACHE[cache_key] = records
    return records
```

### `load_all_recovery_docs(rag_dir: Path) -> list[RecoveryRecord]`

- Check `_ALL_RECORDS_CACHE` first; return immediately if present.
- **Defensive guard**: if `rag_dir` does not exist or is not a directory,
  return `[]` — do not raise. Callers (including `prototype.py`) must be
  able to call this function with a bad path without crashing.
- Glob `*.docx` from `rag_dir`.
- Skip any file whose name contains "e-scentz" (case-insensitive).
- Call `load_docx` on each file, extend the result list.
- Store the aggregated list in `_ALL_RECORDS_CACHE[str(rag_dir)]` and return it.

```python
def load_all_recovery_docs(rag_dir: Path) -> list[RecoveryRecord]:
    """Load all recovery DOCX files in rag_dir (skipping E-Scentz.docx).
    Results are cached at the directory level after the first call."""
    cache_key = str(rag_dir)
    if cache_key in _ALL_RECORDS_CACHE:
        return _ALL_RECORDS_CACHE[cache_key]

    if not rag_dir.is_dir():
        return []

    all_records: list[RecoveryRecord] = []
    for docx_path in sorted(rag_dir.glob("*.docx")):
        if "e-scentz" in docx_path.name.lower():
            continue
        all_records.extend(load_docx(docx_path))

    _ALL_RECORDS_CACHE[cache_key] = all_records
    return all_records
```

---

## Step 2 — Implement `pcil/rag/lookup.py`

Fill in `lookup_keywords(query, records, top_k=3)`.

Do not change the function signature.

### Algorithm (bag-of-words, case-insensitive substring match)

1. Lowercase the query and split on whitespace.
2. Drop stopwords:
{"the", "is", "a", "an", "of", "in", "and", "or", "to", "for",
"with", "on", "at", "from", "by", "this", "that", "it", "be",
"was", "has", "have", "not", "are", "as", "but", "if", "so",
"defined", "below", "which", "during", "time", "proportion"}
3. For each record, count how many query tokens appear as substrings in (record["error"] + " " + record["cause"]).lower().
4. Sort records by count descending.
5. Return the top_k records whose count > 0. If nothing matches, return an empty list.

Return type: `list[RecoveryRecord]`.

```python
from __future__ import annotations
from pcil.rag.loader import RecoveryRecord

_STOPWORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "of", "in", "and", "or", "to", "for",
    "with", "on", "at", "from", "by", "this", "that", "it", "be",
    "was", "has", "have", "not", "are", "as", "but", "if", "so",
    "defined", "below", "which", "during", "time", "proportion",
})


def lookup_keywords(
    query: str,
    records: list[RecoveryRecord],
    *,
    top_k: int = 3,
) -> list[RecoveryRecord]:
    """Return the top_k records best matching query by token count.
    Uses case-insensitive substring matching against error + cause fields."""
    if not query or not records:
        return []

    tokens = [
        t for t in query.lower().split()
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return []

    def score(record: RecoveryRecord) -> int:
        haystack = (record["error"] + " " + record["cause"]).lower()
        return sum(1 for t in tokens if t in haystack)

    scored = [(score(r), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for count, r in scored[:top_k] if count > 0]
```

---

## Step 3 — Add `pcil/rag/composer.py` (new file)

This is the only new file inside `pcil/rag/`. Create it from scratch.

```python
"""
RAG LLM composer — turns impacts + recovery records into an operator
recommendation using the Gemini API.

# TODO: when containerising, replace in-process cache with pgvector on PostgreSQL
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcil.rag.loader import RecoveryRecord


def compose_recommendation(
    impacts: dict,
    records: list["RecoveryRecord"],
    *,
    model: str = "gemini-2.0-flash",
) -> str:
    """Generate a plain-English operator recommendation.

    Parameters
    ----------
    impacts:
        The live-generated impacts dict from train_context_model_from_df().
        Must use the current schema: top-level "context" key with
        "ranked_feature_impacts" lists (not the legacy "blocks" schema).
    records:
        Recovery records retrieved by lookup_keywords(). May be empty.
    model:
        Gemini model name. Defaults to "gemini-2.0-flash".

    Returns
    -------
    str
        One concise paragraph for the operator, or a fallback string
        if records are empty or the API call fails.
    """
    if not records:
        return (
            "No matching recovery records found. "
            "Review the feature impacts data manually."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before starting the orchestrator."
        )

    prompt = _build_prompt(impacts, records)

    try:
        import google.generativeai as genai  # noqa: PLC0415
        genai.configure(api_key=api_key)
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        return (
            f"LLM composition failed ({type(exc).__name__}): {exc}. "
            "Review the recovery records below manually."
        )


def _build_prompt(impacts: dict, records: list["RecoveryRecord"]) -> str:
    """Construct the Gemini prompt from impacts and retrieved records.

    Keeps total length under ~1 800 characters by truncating long
    recovery texts.
    """
    system_name = impacts.get("system", "unknown system")

    # Identify the two worst-performing targets by lowest intercept.
    # In this linear model the intercept is the baseline predicted value
    # when all normalised features are 0 — a lower intercept indicates
    # a weaker performance baseline for that target.
    context_blocks = impacts.get("context", [])
    sorted_blocks = sorted(context_blocks, key=lambda b: b["intercept"])
    worst_two = sorted_blocks[:2]
    target_lines = "\n".join(
        f"  - {b['target']} (baseline: {b['intercept']:.3f})"
        for b in worst_two
    )

    # Collect top-ranked feature impact descriptions (deduplicated).
    seen: set[str] = set()
    impact_lines: list[str] = []
    for block in context_blocks:
        for fi in block.get("ranked_feature_impacts", [])[:1]:
            feat = fi["feature"]
            if feat not in seen:
                seen.add(feat)
                desc = fi.get("description") or feat.replace("_", " ")
                impact_lines.append(
                    f"  - {feat} (score {fi['raw_impact_score']:+.3f}): {desc}"
                )
            if len(impact_lines) >= 3:
                break
        if len(impact_lines) >= 3:
            break

    # Format recovery records; truncate long recovery text.
    record_blocks: list[str] = []
    for i, rec in enumerate(records, 1):
        recovery_text = rec["recovery"]
        if len(recovery_text) > 300:
            recovery_text = recovery_text[:297] + "..."
        record_blocks.append(
            f"Record {i} (source: {rec['source_doc']})\n"
            f"  Error:    {rec['error']}\n"
            f"  Cause:    {rec['cause']}\n"
            f"  Recovery: {recovery_text}"
        )

    records_text = "\n\n".join(record_blocks)
    impacts_text = "\n".join(impact_lines) or "  (none ranked)"

    return (
        f"Machine: {system_name}\n\n"
        f"Worst-performing targets:\n{target_lines}\n\n"
        f"Top contributing features:\n{impacts_text}\n\n"
        f"Relevant recovery records:\n{records_text}\n\n"
        "Task: Write one concise paragraph (3–5 sentences) for a factory "
        "floor operator. State what the data suggests is wrong, which "
        "physical component to inspect first, and the most important "
        "recovery step from the records above. Use plain language; avoid "
        "technical jargon or variable names."
    )
```

---

## Step 4 — Wire RAG into `pcil/orchestrator.py`

Make these changes in order. Do not place any imports inside function bodies.

### 4a. Add module-level imports (alongside the existing imports at the top of the file)

```python
# RAG pipeline — imported at module level; guarded so missing
# google-generativeai does not break non-RAG endpoints or the test suite.
from pcil.rag.loader import load_all_recovery_docs
from pcil.rag.lookup import lookup_keywords
from pcil.rag.composer import compose_recommendation
```

If `google-generativeai` is not yet installed, the `compose_recommendation`
import itself will succeed (the SDK import is deferred inside the function).
No import guard is needed at this level.

### 4b. Add `RAG_DIR` constant (directly below the `PROJECT_ROOT` definition)

```python
RAG_DIR = PROJECT_ROOT / "data" / "RAG"
```

### 4c. Add `_build_rag_query` as a module-level helper function

Place this below the helper `_context_window_filename` function, before
any endpoint definitions.

```python
def _build_rag_query(impacts: dict) -> str:
    """Build a keyword query string from the impacts dict for RAG retrieval.

    Extracts vocabulary from feature *descriptions* (not raw column names)
    so that tokens like "air", "pressure", "vibration" match human-readable
    DOCX error text. Falls back to splitting the column name on underscores
    when no description is available.
    """
    tokens: set[str] = set()
    for block in impacts.get("context", []):
        tokens.add(block["target"])
        for fi in block.get("ranked_feature_impacts", [])[:2]:
            description = fi.get("description", "")
            if description:
                tokens.update(description.lower().split())
            else:
                # Fallback: underscores → individual words
                tokens.update(fi["feature"].lower().split("_"))
    return " ".join(tokens)
```

### 4d. Replace the placeholder values in `_run_pipeline_on_df`

Find the `return` statement at the end of `_run_pipeline_on_df` and add
the RAG block immediately before it. The entire block must be wrapped in
`try/except` so a DOCX parse error or unexpected exception in the RAG
layer never converts a successful pipeline response into a 500.

```python
    # ── RAG retrieval + LLM composition ─────────────────────────────
    if RAG_DIR.is_dir():
        try:
            rag_query = _build_rag_query(impacts)
            all_records = load_all_recovery_docs(RAG_DIR)
            recovery_records = lookup_keywords(rag_query, all_records, top_k=3)
            operator_recommendation = compose_recommendation(impacts, recovery_records)
        except Exception as exc:  # noqa: BLE001
            recovery_records = []
            operator_recommendation = (
                f"RAG retrieval failed ({type(exc).__name__}): {exc}. "
                "Review the impacts data manually."
            )
    else:
        recovery_records = []
        operator_recommendation = (
            "RAG document directory not found. "
            f"Expected: {RAG_DIR}. "
            "Mount the data/RAG/ folder and restart the orchestrator."
        )
    # ─────────────────────────────────────────────────────────────────

    return {
        "status": "ok",
        "input_rows": int(len(slice_df)),
        "golden_rows": int(len(golden_df)),
        "impacts": impacts,
        "recovery_records": recovery_records,           # was []
        "operator_recommendation": operator_recommendation,  # was placeholder
        "artifacts": artifact_paths,
    }
```

---

## Step 5 — Build the Flask frontend (`rag_frontend/`)

Create a new top-level directory `rag_frontend/` alongside `pcil/`.
This is intentionally outside the `pcil` package — it is a separate
process that talks to the orchestrator over HTTP.

rag_frontend/
├── app.py
├── index.html
├── styles.css
└── app.js

### `rag_frontend/app.py`

```python
"""
RAG frontend — minimal Flask UI that calls the PCIL orchestrator.
All retrieval and generation logic lives in the orchestrator.
This file only serves static assets and proxies a run request.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
DEFAULT_CONFIG = os.environ.get(
    "PCIL_CONFIG_PATH", "machines/inkjet_printer/config.yaml"
)
FRONTEND_DIR = Path(__file__).resolve().parent


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/app.js", methods=["GET"])
def app_js():
    return send_from_directory(FRONTEND_DIR, "app.js")


@app.route("/styles.css", methods=["GET"])
def styles_css():
    return send_from_directory(FRONTEND_DIR, "styles.css")


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Proxy request to the orchestrator to avoid browser CORS issues."""
    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/pipeline/run",
            json={"config_path": DEFAULT_CONFIG, "persist": False},
            timeout=60,
        )
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return jsonify({"error": detail}), resp.status_code
        return jsonify(resp.json())
    except requests.ConnectionError:
        return jsonify({
            "error": f"Cannot reach orchestrator at {ORCHESTRATOR_URL}. Is it running?",
        }), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

### Frontend dependencies
Installed from the root `requirements.txt` alongside backend packages.

---

## Step 6 — Update `requirements.txt` in PCIL_dev

Add to the existing `requirements.txt` (do not remove any existing entry):

RAG — LLM composer (google-generativeai package; import as google.generativeai)
google-generativeai>=0.7

Frontend — Flask UI
flask>=3.0
requests>=2.31

---

## Step 7 — Verify `prototype.py` still works

After all changes, the existing CLI must still function:

```bash
python -m pcil.rag.prototype \
    --doc "data/RAG/Screen Printer.docx" \
    --query "pressure low valve"
```

- It calls `load_docx` then `lookup_keywords`.
- It does **not** call `compose_recommendation`.
- If the implementation is correct, at least one match should print.

---

## Step 8 — Verify the existing test suite passes

Run the full pytest suite before and after your changes:

```bash
python -m pytest tests/ -v
```

All 21 existing tests must still pass. The RAG imports added to
`orchestrator.py` must not cause `ImportError` in the test environment
(they won't, because `compose_recommendation` defers the
`google.generativeai` import to call time).

---

## Step 9 — Smoke test the full flow

Start the orchestrator with the Gemini key set:

```bash
GEMINI_API_KEY=<your-key> uvicorn pcil.orchestrator:app \
    --host 0.0.0.0 --port 8000
```

Run a curl to `/pipeline/run`:

```bash
curl -X POST http://localhost:8000/pipeline/run \
     -H "Content-Type: application/json" \
     -d '{"config_path": "machines/inkjet_printer/config.yaml", "persist": false}'
```

Check the response:
- `recovery_records` must be a list (may be empty if `data/RAG/` is absent).
- `operator_recommendation` must not contain the old placeholder text.

Start the Flask frontend:

```bash
python rag_frontend/app.py
```

Open `http://localhost:5000` and click "Run Pipeline". You should see
the recommendation and records on the page, or a clear error message
if `data/RAG/` or `mock_shop_floor.csv` is absent.

---

## Execution order summary

| Step | File(s) touched | Status |
|---|---|---|
| 1 | `pcil/rag/loader.py` | DONE — implemented caches + parser (paragraph-based) |
| 2 | `pcil/rag/lookup.py` | DONE — keyword scoring implemented |
| 3 | `pcil/rag/composer.py` | DONE — Gemini composer added |
| 4 | `pcil/orchestrator.py` | DONE — RAG wired with helper + imports |
| 5 | `rag_frontend/app.py`, `index.html`, `styles.css`, `app.js` | DONE — Flask UI using static assets |
| 6 | `requirements.txt` (PCIL_dev root) | DONE — `google-generativeai` added |
| 7 | — | NOT VERIFIED — prototype CLI not run |
| 8 | — | NOT VERIFIED — pytest not run |
| 9 | — | NOT VERIFIED — smoke test not run |

---

## What NOT to do

- Do not add a database, ORM, or migration tool.
- Do not add authentication.
- Do not add a `/rag/query` endpoint; the RAG result flows through
  the existing `/pipeline/run` and `/pipeline/run_csv` responses.
- Do not edit `prototype.py`; it is already wired.
- Do not edit anything in `data/RAG/`.
- Do not hard-code the Gemini API key.
- Do not change `RecoveryRecord`'s TypedDict fields.
- Do not change the return shape of `_run_pipeline_on_df` beyond
  replacing the two placeholder values.
- Do not place import statements inside function bodies.
- Do not read from `machines/inkjet_printer/output/context_model_impacts.json`
  inside the composer — always use the live-generated `impacts` dict.