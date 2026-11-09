"""FraudGuard Streamlit frontend backed by FastAPI.

Start backend: python -m uvicorn api:app --reload --port 8000
Start frontend: python -m streamlit run app.py
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="centered")
API_URL = os.getenv("FRAUDGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
DATA_PATH = ROOT / Path(os.getenv("FRAUDGUARD_DATA", "examples/demo_transactions.csv"))
META_PATH = ROOT / "models/fraudguard_meta.json"
DEFAULT_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
MAX_ROWS = 5000
RESULT_FIELDS = ["status", "probability", "flag", "message", "warnings"]


def api_request(method, path, *, timeout=30, **kwargs):
    try:
        response = requests.request(method, API_URL + path, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot connect to FastAPI at {API_URL}. Check the backend and FRAUDGUARD_API_URL.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("FastAPI request timed out.") from exc
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise RuntimeError(f"FastAPI HTTP {exc.response.status_code}: {detail}") from exc
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise RuntimeError(f"FastAPI request failed: {exc}") from exc


def score_one(record):
    return api_request("POST", "/predict", json={"features": record})


def score_file(frame, features):
    """Call /predict/batch with a JSON array, one entry per row.

    The current strict FastAPI schema rejects the whole request if any feature
    is null or nonnumeric. Detect those rows locally, retain them as rejected,
    and send only valid rows to the backend. Other backend errors are surfaced.
    """
    scored = frame.reset_index(drop=True).copy()
    for field in RESULT_FIELDS:
        scored[field] = None
    valid_payload = []
    valid_positions = []
    for pos, row in scored.iterrows():
        record = {}
        invalid = []
        for name in features:
            try:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError("not finite")
                record[name] = value
            except (ValueError, TypeError):
                invalid.append(name)
        if invalid:
            scored.at[pos, "status"] = "error"
            scored.at[pos, "message"] = "Invalid numeric value: " + ", ".join(invalid)
            scored.at[pos, "warnings"] = []
        else:
            valid_positions.append(pos)
            valid_payload.append({"features": record})

    if valid_payload:
        response = api_request("POST", "/predict/batch", json=valid_payload, timeout=120)
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(valid_positions):
            raise RuntimeError("FastAPI returned an unexpected batch result count.")
        for pos, result in zip(valid_positions, results):
            if not isinstance(result, dict):
                raise RuntimeError("FastAPI returned an invalid batch result.")
            for field in RESULT_FIELDS:
                scored.at[pos, field] = result.get(field)
    return scored


@st.cache_data(show_spinner=False)
def load_examples(path):
    file = Path(path)
    if not file.exists():
        return None
    return pd.read_csv(file).drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metadata(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


st.title("🛡️ FraudGuard")
st.caption("Credit card fraud scoring | FUSE AI-201, Group 7 | Student prototype")
st.info("Risk scores support human review; they must not automatically decline transactions.")

try:
    health = api_request("GET", "/health", timeout=75)
    config = api_request("GET", "/features", timeout=5)
    if health.get("status") != "healthy":
        raise RuntimeError("FastAPI health endpoint did not report 'healthy'.")
    features = config.get("required_features")
    if not isinstance(features, list) or set(features) != set(DEFAULT_FEATURES) or len(features) != 30:
        raise RuntimeError("Unexpected feature list from FastAPI; expected Time, Amount, V1–V28.")
    threshold = config.get("threshold")
    st.success("FastAPI backend connected")
except RuntimeError as exc:
    st.error(str(exc))
    st.caption("Check FRAUDGUARD_API_URL in the deployment settings.")
    st.button("Retry backend connection")
    st.stop()


def load_example(kind):
    try:
        df = load_examples(str(DATA_PATH))
        if df is None:
            st.session_state["example_msg"] = f"Dataset not found: {DATA_PATH}"
            return
        missing = [f for f in features + ["Class"] if f not in df.columns]
        if missing:
            st.session_state["example_msg"] = f"Example dataset missing columns: {missing}"
            return
        subset = df.loc[df["Class"] == (1 if kind == "fraud" else 0)]
        if subset.empty:
            st.session_state["example_msg"] = "No matching example found."
            return
        row = subset.sample(1).iloc[0]
        for name in features:
            st.session_state[name] = float(row[name])
        st.session_state["example_msg"] = f"Loaded a {kind} example. The model can still misclassify it."
    except (OSError, ValueError, KeyError) as exc:
        st.session_state["example_msg"] = f"Could not load example: {exc}"


def reset_form():
    for name in features:
        st.session_state[name] = 0.0
    st.session_state["example_msg"] = ""


for name in features:
    st.session_state.setdefault(name, 0.0)
st.session_state.setdefault("example_msg", "")

tab_one, tab_file, tab_about = st.tabs(["Score one transaction", "Score a file", "About the model"])

with tab_one:
    c1, c2, c3 = st.columns(3)
    c1.button("Load legitimate example", on_click=load_example, args=("legit",))
    c2.button("Load fraud example", on_click=load_example, args=("fraud",))
    c3.button("Reset", on_click=reset_form)
    if st.session_state["example_msg"]:
        st.info(st.session_state["example_msg"])
    a, b = st.columns(2)
    a.number_input("Time (seconds since first transaction)", key="Time", format="%.2f", step=1.0)
    b.number_input("Amount", key="Amount", format="%.2f", step=1.0)
    with st.expander("Anonymised features V1 to V28"):
        left, right = st.columns(2)
        for i in range(1, 29):
            (left if i <= 14 else right).number_input(f"V{i}", key=f"V{i}", format="%.4f", step=0.01)
    if st.button("Score transaction", type="primary"):
        record = {name: float(st.session_state[name]) for name in features}
        try:
            with st.spinner("Scoring through FastAPI..."):
                result = score_one(record)
            if result.get("status") != "ok":
                st.error(f"Input rejected: {result.get('message', 'Unknown error')}")
            else:
                probability = float(result["probability"])
                flag = int(result["flag"])
                st.metric("Fraud probability", f"{probability * 100:.2f}%")
                st.progress(min(max(probability, 0.0), 1.0))
                if threshold is not None:
                    st.caption(f"Decision threshold: {float(threshold):.4f}")
                if flag:
                    st.error("⚠️ Flag for human review")
                else:
                    st.success("✅ Transaction not flagged")
                for warning in result.get("warnings") or []:
                    st.warning(str(warning))
                st.caption("A fraud flag is not proof of fraud and must not trigger automatic decline.")
        except (RuntimeError, KeyError, TypeError, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")

with tab_file:
    st.write("Upload a CSV with Time, Amount, and V1–V28. Extra columns are preserved. Maximum 5,000 rows.")
    st.download_button("Download empty CSV template", ",".join(features) + "\n", file_name="fraudguard_template.csv", mime="text/csv")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            data = None
        if data is not None:
            missing = [name for name in features if name not in data.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif data.empty:
                st.warning("The CSV has no transactions.")
            else:
                if len(data) > MAX_ROWS:
                    st.warning(f"Only the first {MAX_ROWS:,} of {len(data):,} rows will be scored.")
                if st.button("Score CSV transactions", type="primary"):
                    try:
                        with st.spinner("Scoring CSV through FastAPI..."):
                            scored = score_file(data.head(MAX_ROWS), features)
                        ok = scored.loc[scored["status"] == "ok"]
                        flagged = pd.to_numeric(ok["flag"], errors="coerce").fillna(0).sum()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Rows scored", f"{len(ok):,}")
                        m2.metric("Flagged for review", f"{int(flagged):,}")
                        m3.metric("Rejected rows", f"{int((scored['status'] != 'ok').sum()):,}")
                        st.dataframe(scored[RESULT_FIELDS].sort_values("probability", ascending=False, na_position="last"), use_container_width=True)
                        st.download_button("Download results CSV", scored.to_csv(index=False), file_name="fraudguard_results.csv", mime="text/csv")
                    except (RuntimeError, ValueError, TypeError) as exc:
                        st.error(f"Batch scoring failed: {exc}")

with tab_about:
    meta = load_metadata(str(META_PATH))
    st.subheader("Model")
    st.write(f"**Model:** {meta.get('model_name', 'XGBoost (see backend metadata)')}")
    if threshold is not None:
        st.write(f"**Decision threshold:** {float(threshold):.4f}")
    st.write(f"**Backend:** FastAPI ({API_URL})")
    st.write("**Frontend:** Streamlit")
    metrics = meta.get("test_metrics", {})
    if metrics:
        keys = ["AUPRC", "ROC_AUC", "Precision", "Recall", "F1", "FPR", "TP", "FP", "FN", "TN"]
        shown = [k for k in keys if k in metrics]
        st.write("**Held-out test-set results:**")
        st.dataframe(pd.DataFrame({"Metric": shown, "Value": [metrics[k] for k in shown]}), hide_index=True)
    else:
        st.caption("Optional local model metrics not found in models/fraudguard_meta.json.")
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- Historical 2013 data may not reflect current transactions.\n"
        "- Fraud cases are rare and estimates are uncertain.\n"
        "- V1–V28 are anonymised and difficult to explain in business terms.\n"
        "- Distribution shifts can reduce model accuracy.\n"
        "- Flags require human review, not automatic decline.\n"
        "- Revalidate the threshold before any real-world use."
    )
    st.code("Streamlit → FastAPI → fraudguard_score.py → XGBoost → FastAPI → Streamlit", language="text")
