"""
Smoke test for pcil/orchestrator.py.

Boots the FastAPI app in-process via TestClient and exercises every
endpoint against the inkjet_printer config + mock shop-floor CSV.

Run from PCIL_dev/:
    python scripts/smoke_test_orchestrator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add PCIL_dev/ to sys.path so `from pcil.orchestrator` resolves whether
# this script is invoked as `python scripts/smoke_test_orchestrator.py`
# or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from pcil.orchestrator import app

CONFIG_PATH = "machines/inkjet_printer/config.yaml"


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> int:
    client = TestClient(app)

    banner("GET /")
    r = client.get("/")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    banner("POST /pipeline/run")
    r = client.post("/pipeline/run", json={"config_path": CONFIG_PATH, "persist": False})
    print(f"status: {r.status_code}")
    body = r.json()
    if r.status_code != 200:
        print(json.dumps(body, indent=2))
        return 1
    print(f"input_rows: {body['input_rows']}")
    print(f"golden_rows: {body['golden_rows']}")
    print(f"impacts.system: {body['impacts']['system']}")
    print(f"impacts.model: {body['impacts']['model']}")
    print(f"impacts.context_window: {json.dumps(body['impacts']['context_window'], indent=2)}")
    print(f"impacts.context[0] (target={body['impacts']['context'][0]['target']}):")
    for fi in body['impacts']['context'][0]['ranked_feature_impacts']:
        print(f"  [{fi['rank']}] {fi['feature']:<28s} raw={fi['raw_impact_score']:+.4f}  std={fi['standardized_impact_score']:+.4f}")
        print(f"       \"{fi['description']}\"")

    banner("POST /pipeline/save_csv")
    r = client.post("/pipeline/save_csv", json={"config_path": CONFIG_PATH})
    print(f"status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    if r.status_code != 200:
        return 1

    banner("POST /anomaly/score (cyclical — needs cyclical_inkjet_01.pkl)")
    import pandas as pd
    cyclical_csv = Path("../data/cyclical_eval.csv")
    if cyclical_csv.is_file():
        df = pd.read_csv(cyclical_csv).head(20000)
        r = client.post("/anomaly/score", json={
            "data": df.to_dict("records"),
            "model_type": "cyclical",
            "model_id": "inkjet_01",
        })
        print(f"status: {r.status_code}")
        body = r.json()
        if r.status_code != 200:
            print(json.dumps(body, indent=2))
            return 1
        print(f"  cycles_scored: {body['cycles_scored']}")
        print(f"  score range:   {min(body['anomaly_scores']):.3f} .. {max(body['anomaly_scores']):.3f}")
    else:
        print(f"Skipping — {cyclical_csv} not found. Run prepare_data + train first.")

    banner("POST /anomaly/score (non_cyclical — needs non_cyclical_inkjet_01.pkl)")
    acoustic_csv = Path("../data/Inkjet Printer Data Collection/Acoustic Sensor Data/machine_on_anomaly.csv")
    if acoustic_csv.is_file():
        df = pd.read_csv(acoustic_csv, skiprows=5).head(38400)
        rows = df[["Acceleration 0 (g)", "Acceleration 1 (g)",
                    "Acceleration 2 (g)", "AE (V) (V)"]].to_dict("records")
        r = client.post("/anomaly/score", json={
            "data": rows, "model_type": "non_cyclical", "model_id": "inkjet_01",
        })
        print(f"status: {r.status_code}")
        body = r.json()
        if r.status_code != 200:
            print(json.dumps(body, indent=2))
            return 1
        print(f"  windows_scored: {body['windows_scored']}")
        print(f"  anomaly_scores: {body['anomaly_scores']}")
    else:
        print(f"Skipping — {acoustic_csv} not found.")

    banner("All endpoints responded OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
