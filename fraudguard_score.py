"""FraudGuard scoring: validate one transaction and score it with the saved model.

Usage (with the supplied artifacts in models/):
    python fraudguard_score.py                      # scores a demo transaction
    python fraudguard_score.py transaction.json     # scores a transaction from a JSON file

Or in Python:
    from fraudguard_score import score_transaction
    result = score_transaction({"Time": 100.0, "Amount": 25.5, "V1": 0.1, ..., "V28": 0.0})
"""
import json
import os
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MODEL_DIR = Path(os.getenv("FRAUDGUARD_MODEL_DIR", "models"))
MODEL_DIR = ROOT / MODEL_DIR
MODEL_PATH = MODEL_DIR / "fraudguard_model.joblib"
META_PATH = MODEL_DIR / "fraudguard_meta.json"

_model = joblib.load(MODEL_PATH)
# Render instances have limited CPU. Use one native inference thread per request.
_model.set_params(n_jobs=1)
_meta = json.loads(META_PATH.read_text(encoding="utf-8"))
FEATURES = _meta["features"]
THRESHOLD = _meta["threshold"]
BOUNDS = _meta["bounds"]
MAX_FEATURE_VALUE = float(np.finfo(np.float32).max)


def _validate_transaction(record):
    out = {"status": "error", "probability": None, "flag": None, "warnings": [], "message": ""}
    if not isinstance(record, dict):
        out["message"] = "Input must be a dictionary of feature names to numbers."
        return out, None
    missing = [f for f in FEATURES if f not in record or record[f] is None]
    if missing:
        out["message"] = f"Missing required fields: {missing}"
        return out, None
    values = {}
    for f in FEATURES:
        v = record[f]
        if isinstance(v, bool):
            out["message"] = f"Field '{f}' must be a number, not a boolean."
            return out, None
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                out["message"] = f"Field '{f}' must be numeric, got '{record[f]}'."
                return out, None
        if not isinstance(v, (int, float, np.integer, np.floating)):
            out["message"] = f"Field '{f}' must be numeric, got {type(v).__name__}."
            return out, None
        v = float(v)
        if not math.isfinite(v):
            out["message"] = f"Field '{f}' must be a finite number (no NaN/inf)."
            return out, None
        if abs(v) > MAX_FEATURE_VALUE:
            out["message"] = f"Field '{f}' exceeds the model's numeric range."
            return out, None
        if f in ("Amount", "Time") and v < 0:
            out["message"] = f"Field '{f}' cannot be negative."
            return out, None
        lo, hi = BOUNDS[f]
        if v < lo or v > hi:
            out["warnings"].append(
                f"'{f}'={v:g} is outside the training range [{lo:g}, {hi:g}]; the score may be unreliable.")
        values[f] = v
    extra = [k for k in record if k not in FEATURES]
    if extra:
        out["warnings"].append(f"Ignored unexpected fields: {extra}")
    return out, values


def score_transactions(records):
    """Validate each row and run one prediction call for all valid rows."""
    results = []
    valid_results = []
    valid_records = []
    for record in records:
        result, values = _validate_transaction(record)
        results.append(result)
        if values is not None:
            valid_results.append(result)
            valid_records.append(values)
    if valid_records:
        frame = pd.DataFrame(valid_records, columns=FEATURES)
        for result, probability in zip(valid_results, _model.predict_proba(frame)[:, 1], strict=True):
            probability = float(probability)
            flagged = probability >= THRESHOLD
            result.update(
                status="ok", probability=round(probability, 6), flag=int(flagged),
                message="Flag for review" if flagged else "Approve",
            )
    return results


def score_transaction(record):
    """Return a dict with status, probability, flag, warnings and message."""
    return score_transactions([record])[0]


def score_batch(df, max_rows=5000):
    """Score a DataFrame of transactions row by row. Returns the input plus status columns."""
    df = df.head(max_rows).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c in FEATURES]
    rows = []
    for r in score_transactions(df[feature_cols].to_dict("records")):
        rows.append({"status": r["status"], "probability": r["probability"], "flag": r["flag"],
                     "message": r["message"], "warnings": "; ".join(r["warnings"])})
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:  # demo: an all-zero transaction with Amount 25
        record = {f: 0.0 for f in FEATURES}
        record["Amount"] = 25.0
        print("No file given; scoring a demo transaction.")
    print(json.dumps(score_transaction(record), indent=2))
