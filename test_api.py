"""HTTP smoke checks. Start the API before running this script."""
import csv
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API = os.getenv("FRAUDGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")


def post(path, payload):
    # Send raw JSON to exercise the server's handling of NaN and Infinity.
    return requests.post(
        API + path, data=json.dumps(payload),
        headers={"Content-Type": "application/json"}, timeout=30,
    )


def main():
    health = requests.get(API + "/health", timeout=10)
    assert health.status_code == 200 and health.json()["status"] == "healthy"
    config = requests.get(API + "/features", timeout=10)
    config.raise_for_status()
    features = config.json()["required_features"]
    assert set(features) == {"Time", "Amount", *(f"V{i}" for i in range(1, 29))}
    with (ROOT / "examples/demo_transactions.csv").open(newline="") as file:
        records = [
            {"features": {key: float(row[key]) for key in features}}
            for row in csv.DictReader(file)
        ]

    singles = []
    for record in records:
        response = post("/predict", record)
        response.raise_for_status()
        result = response.json()
        assert result["status"] == "ok" and 0 <= result["probability"] <= 1
        singles.append(result)
    assert {result["flag"] for result in singles} == {0, 1}

    for bad_value in (True, float("nan"), float("inf"), -1, 1e100, "invalid", None):
        response = post("/predict", {"features": records[0]["features"] | {"Amount": bad_value}})
        assert response.status_code == 422, response.text
    assert post("/predict", {"features": {"Amount": 25}}).status_code == 422
    assert post("/predict", records[0] | {"extra": float("nan")}).status_code == 422

    mixed = post("/predict/batch", [records[0], {"features": {"Amount": 25}}, records[1]])
    mixed.raise_for_status()
    results = mixed.json()["results"]
    assert results[0] == singles[0] and results[2] == singles[1]
    assert results[1]["status"] == "error"
    assert post("/predict/batch", []).status_code == 422
    assert post("/predict/batch", [records[0]] * 5001).status_code == 422
    full = post("/predict/batch", [records[0]] * 5000)
    full.raise_for_status()
    assert full.json()["results"] == [singles[0]] * 5000
    print("Passed API health, schema, single scoring, invalid inputs and 5,000-row batch checks.")


if __name__ == "__main__":
    main()
