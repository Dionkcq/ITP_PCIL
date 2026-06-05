"""
run_llm_only.py
===============
Isolate the Gemini call: feed the composer a hand-made recovery record plus a
realistic impacts dict and see whether it returns a real paragraph. This proves
the LLM path works independently of the DOCX loader.

Named run_* (not test_*) so pytest does not auto-collect it.

Run from PCIL_dev/:  python scripts/run_llm_only.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PCIL_DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PCIL_DEV))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PCIL_DEV / ".env")

from pcil.rag.composer import compose_recommendation  # noqa: E402

impacts = {
    "system": "inkjet_printer",
    "context": [
        {
            "target": "oee",
            "intercept": 0.62,
            "ranked_feature_impacts": [
                {
                    "feature": "air_pressure_low_ratio",
                    "raw_impact_score": -0.40,
                    "standardized_impact_score": -0.98,
                    "rank": 1,
                    "description": (
                        "Proportion of operating time during which air "
                        "pressure falls below the defined threshold."
                    ),
                }
            ],
        }
    ],
}

records = [
    {
        "error": "Low air pressure detected",
        "cause": "Air supply below threshold; possible leak in the pneumatic line.",
        "recovery": (
            "Check the air compressor output and inspect the pneumatic lines "
            "for leaks. Confirm the regulator is set to specification before "
            "resuming the job."
        ),
        "source_doc": "MOCK.docx",
    }
]

if __name__ == "__main__":
    print("Calling Gemini via compose_recommendation()...\n")
    result = compose_recommendation(impacts, records)
    print("=" * 70)
    print(result)
    print("=" * 70)
