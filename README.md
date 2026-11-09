# FraudGuard · FUSE AI-201 Module 6.4

Group 7’s deployment package runs a FastAPI scoring service and a Streamlit interface. The included XGBoost model uses 30 transaction features and returns a fraud probability and a flag for human review. The public demo is intended for sample data.

## Deploy to Render

1. Extract the ZIP. Commit the contents of this project folder, including the hidden config files and the two files in `models/`, to the root of a Git repository connected to Render.
2. In Render, choose **New → Blueprint**, select the repository, and apply `render.yaml`. It creates two free Docker web services. The Streamlit service gets the API’s public URL through `FRAUDGUARD_API_URL`.
3. Wait for both services to become live. Open the API’s `/health` and `/docs` pages, then open the Streamlit URL. Choose **Load legitimate example** or **Load fraud example**, then **Score transaction**.

If creating the services manually, use these settings.

| Setting | FastAPI | Streamlit |
|---|---|---|
| Service type | Web Service | Web Service |
| Runtime | Docker | Docker |
| Dockerfile path | `./Dockerfile.api` | `./Dockerfile.streamlit` |
| Docker build context | Repository root | Repository root |
| Health check path | `/health` | `/_stcore/health` |
| Environment variables | Render provides `PORT` | Set `FRAUDGUARD_API_URL` to the API service's full HTTPS URL. Render provides `PORT`. |

Leave Docker Command blank to use each Dockerfile’s start command. When a free service wakes from idle, its first request can take longer. If the UI shows a connection error, wait for the API's `/health` endpoint to respond, then select **Retry backend connection**. See [Docker on Render](https://render.com/docs/docker) and the [Blueprint reference](https://render.com/docs/blueprint-spec).

## Run locally

With Docker Compose:

```bash
docker compose up --build
```

Open [Streamlit](http://127.0.0.1:8502) and [API docs](http://127.0.0.1:8000/docs).

With Python 3.12, create a virtual environment and install the dependencies. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, from the same folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS or Linux, use `python3.12 -m venv .venv` and substitute `.venv/bin/python` for the Windows executable in the commands above. For local Python runs, Streamlit opens at [port 8501](http://127.0.0.1:8501). Copy `.env.example` to `.env` if you need to change local defaults. Variables set by the process take precedence over `.env`.

## API and verification

| Method and path | Purpose |
|---|---|
| `GET /health` | Confirms the API is running after loading the model artifacts |
| `GET /features` | Returns the required feature names and threshold |
| `POST /predict` | Accepts one object with a `features` mapping |
| `POST /predict/batch` | Accepts an array of 1 to 5,000 such objects and returns results in the same order |

Supply `Time`, `Amount`, and `V1` through `V28`. All required values must be finite numbers, and `Time` and `Amount` must be nonnegative. A single invalid transaction returns HTTP 422. In a structurally valid batch, transactions with missing required features or invalid numeric values such as negative amounts receive per-row error results. Malformed request bodies return HTTP 422.

With the API running, execute `python test_api.py` using the project environment. Set `FRAUDGUARD_API_URL` in the shell to test a different base URL. The checks exercise the included demo rows, invalid input, and the 5,000-row batch limit. Upload `examples/demo_transactions.csv` in the Streamlit **Score a file** tab to try batch scoring.

Local checks passed on Linux with Python 3.12. These covered dependency installation, API endpoints, both service health URLs, batch scoring, and Streamlit's two example predictions. Docker image builds and a live Render deployment still need to be run.

## Model and training files

- `models/` includes the original trained XGBoost model and metadata. The threshold and feature order come from the supplied metadata.
- `examples/demo_transactions.csv` contains two labeled sample rows from the original training dataset, one from each class. `examples/fraudguard_30_expected_results.csv` is the separately supplied reference output. Its matching inputs were unavailable, so the smoke test uses the two included demo rows.
- The original Module 5 retraining notebook and figures are included for reference. For retraining, install `requirements-training.txt` alongside the API requirements and provide the full dataset at `data/creditcard.csv`. Save or commit a copy of the supplied model artifacts first because the notebook regenerates them. Deployment uses the included artifacts and needs no retraining.
