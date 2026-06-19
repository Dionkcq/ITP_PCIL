"""
run_llm_e2e.py
==============
End-to-end LIVE test of /pipeline/run with the real Gemini LLM.

Loads GEMINI_API_KEY from PCIL_dev/.env, boots the orchestrator in-process
via TestClient, POSTs the inkjet_printer recipe, and prints the retrieved
recovery records plus the operator recommendation so we can see whether the
LLM actually fired (vs. returning one of the fallback strings).

Run from PCIL_dev/:
    python scripts/run_llm_e2e.py

The .env file is never printed; only the resulting recommendation text is.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PCIL_DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PCIL_DEV))

# Load .env (GEMINI_API_KEY) into os.environ before the app calls the LLM.
try:
    from dotenv import load_dotenv

    load_dotenv(PCIL_DEV / ".env")
except ModuleNotFoundError:
    print("[warn] python-dotenv not installed; relying on the existing environment.")

if not os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY is not set.")
    print(f"Create {PCIL_DEV / '.env'} containing a single line:")
    print("    GEMINI_API_KEY=your_real_key_here")
    sys.exit(2)

from fastapi.testclient import TestClient  # noqa: E402

from pcil.orchestrator import RAG_DIR, app  # noqa: E402

# Absolute path so the run works regardless of the current working directory.
# (_resolve_config resolves a relative path against cwd; trigger.source is
# resolved against the config file's own directory, so an absolute config
# path makes the whole run cwd-independent.)
CONFIG_PATH = str(PCIL_DEV / "systems" / "inkjet_printer" / "config.yaml")

FALLBACK_MARKERS = (
    "RAG document directory not found",
    "RAG retrieval failed",
    "LLM composition failed",
    "No matching recovery records",
    "GEMINI_API_KEY environment variable is not set",
)


def main() -> int:
    print(f"RAG_DIR: {RAG_DIR}")
    print(f"RAG_DIR exists: {RAG_DIR.is_dir()}")
    print(f"GEMINI_API_KEY: set (length {len(os.environ['GEMINI_API_KEY'])})")

    client = TestClient(app)
    r = client.post(
        "/pipeline/run", json={"config_path": CONFIG_PATH, "persist": False}
    )
    print(f"\nHTTP {r.status_code}")
    body = r.json()
    if r.status_code != 200:
        print(json.dumps(body, indent=2))
        return 1

    print(f"input_rows={body['input_rows']}  golden_rows={body['golden_rows']}")

    # Top OEE driver from the context model (the "X, Y, Z" half of the diagnosis).
    oee = next(
        (b for b in body["impacts"]["context"] if b["target"] == "oee"), None
    )
    if oee and oee["ranked_feature_impacts"]:
        top = oee["ranked_feature_impacts"][0]
        print(
            f"top OEE driver: {top['feature']} "
            f"(standardised {top['standardized_impact_score']:+.3f})"
        )

    print("\nRetrieved recovery records (from data/RAG/*.docx):")
    if body["recovery_records"]:
        for rec in body["recovery_records"]:
            print(f"  - [{rec['source_doc']}] {rec['error']}")
    else:
        print("  (none)")

    rec_text = body["operator_recommendation"]
    is_fallback = any(m in rec_text for m in FALLBACK_MARKERS)
    label = "[FALLBACK - LLM did NOT fire]" if is_fallback else "[LIVE LLM OUTPUT]"
    print("\n" + "=" * 70)
    print(f"OPERATOR RECOMMENDATION  {label}")
    print("=" * 70)
    print(rec_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
