# pcil/rag/

Pipeline #3 — recovery-document retrieval. After the context model
identifies *what's wrong* (as numbers), this package looks up *what to
do about it* from Winardi's reference DOCX files.

---

## Where this sits in the full pipeline

```
shop-floor DB
   |
trigger        (pcil/trigger.py — done, by Dion)
   |
preprocess     (pcil/preprocess.py — done, by Dion)
   |
adapter        (pcil/adapter.py — done, by Dion)
   |
context model  (pcil/train_context_model.py — LinearRegression baseline; Pipeline #2 will swap in SHAP/XGBoost later)
   |
   v
+---------------------------------------------+
|  RAG retrieval         <-- THIS PACKAGE     |
|  (look up matching recovery records)        |
|                                             |
|  LLM composer          <-- stretch goal     |
|  (turn signals + records into a sentence)   |
+---------------------------------------------+
   |
   v
operator dashboard
```

The adapter sits **before** the model, not between model and RAG.
Don't get those confused.

---

## What the context model actually outputs

**Important — the model does NOT emit an English sentence.** It emits
numbers. Concretely, something like:

```python
{
    "prediction":     {"OEE": 0.62},
    "feature_impacts": {
        "pressure_variance": +0.31,
        "cycle_time_std":    -0.18,
        "set_velocity":      +0.04,
    },
    "anomaly_flags":  ["pressure"],
    "error_codes":    [],
}
```

The English sentence ("OEE dropped 30%, check valve V-12...") is what
an LLM composer produces *downstream of this package*, by combining
the model's numbers with the recovery records this package retrieves.

For v1 you don't need a real LLM — printing the retrieved records is
enough. The LLM wrap is a stretch goal.

---

## Your contract

**Input:** a query string (or, later, the dict above)

**Output:** top-k matching `RecoveryRecord` dicts:

```python
{
    "error":      "Pressure low",
    "cause":      "Valve V-12 partially blocked",
    "recovery":   "Shut off line, remove valve, clean and reseat...",
    "source_doc": "Screen Printer.docx",
}
```

You do **not** need to touch the ML model, the adapter, the
preprocessing pipeline, or the anomaly detector. Develop standalone.
You can test by hardcoding a query like `"pressure spike vibration"`
and checking the right record comes back.

---

## Source documents

`data/RAG/` (gitignored — sits alongside the repo):

| File | Useful? |
|---|---|
| `Infrared Oven.docx` | yes — `Error Message` / `Root Cause` / `Recovery Steps` blocks |
| `Injection Molding.docx` | yes — 24 error blocks |
| `Laser Trimming.docx` | yes — 18 error blocks |
| `Laser Welding.docx` | yes — 16 error blocks |
| `Membrane Assembly.docx` | yes — 22 error blocks |
| `Screen Printer.docx` | yes — 19 error blocks (Dion confirmed; **start here**) |
| `E-Scentz.docx` | **skip** — product overview only, no error/recovery content |

For v1, pick **one** doc and prototype against it.

---

## Files to fill in

| File | What it does | Status |
|---|---|---|
| `loader.py` | parse a DOCX into a list of `RecoveryRecord` dicts | **TODO** |
| `lookup.py` | given a query string, return top-k matching records | **TODO** |
| `prototype.py` | end-to-end CLI demo glueing loader + lookup | Already wired — runs once the above two exist |

---

## Recommended order

1. **Install the parser:**

   ```bash
   pip install python-docx
   ```

2. **Fill in `loader.py`** — implement `load_docx(path)`. Use
   `from docx import Document`, walk the paragraphs/tables, detect the
   heading pattern (`Error Message` / `Root Cause` / `Recovery Steps`),
   and group each trio into one `RecoveryRecord`. Start with
   `Screen Printer.docx` — Dion read it and confirmed it has 19 clean
   structured blocks.

   Heads up: some docs use tables, some use plain paragraphs. Inspect
   the doc structure first (open it in Word, or print the first 50
   paragraphs/tables). Don't try to handle all 6 docs on day 1.

3. **Fill in `lookup.py`** — implement `lookup_keywords(query, records, top_k=3)`:

   - Lowercase + split the query on whitespace, drop stopwords
     (`the`, `is`, `a`, `of`, etc.).
   - For each record, count how many query tokens appear in
     `record["error"] + " " + record["cause"]`.
   - Sort records by count descending, return the top `top_k`.
   - Empty list if nothing matches.

   Plain substring/bag-of-words match is fine for v1. Vector
   embeddings is a stretch goal.

4. **Run the prototype:**

   ```bash
   python -m pcil.rag.prototype \
       --doc "../data/RAG/Screen Printer.docx" \
       --query "pressure low valve"
   ```

   You should see the matching error/cause/recovery printed. That's
   "done" for v1.

---

## Done condition for v1

- `loader.py` loads `Screen Printer.docx` and returns ≥ 15 records
  with non-empty `error`/`cause`/`recovery` fields.
- `lookup.py` with query `"pressure"` returns at least one
  pressure-related record at rank 1.
- `prototype.py` CLI runs without errors.

---

## Stretch goals

- **Multi-doc:** make `loader.py` handle all 6 useful docs, then run
  `prototype.py` with `--rag-dir ../data/RAG/` to retrieve across the
  full set.
- **Vector search:** swap the keyword count for
  `sentence-transformers` embeddings + cosine similarity.
- **LLM composer:** wrap the retrieved records with a system prompt
  and call an LLM (Anthropic / OpenAI / local) to write the final
  operator-facing sentence.
- **Re-ranking:** after embedding retrieval, ask the LLM to re-rank
  the top results.

---

## If you get stuck

- The skeleton functions in `loader.py` / `lookup.py` have TODOs in
  their docstrings that mirror this README.
- `pcil/utils/anomaly/normalise.py` is a working reference for what a
  "done" skeleton-fill looks like in this codebase.
- Ask Dion before changing the function signatures — `prototype.py`
  and the downstream LLM composer (future) depend on them.
