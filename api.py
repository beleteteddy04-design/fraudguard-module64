"""FastAPI interface for the FraudGuard model."""
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from fraudguard_score import FEATURES, THRESHOLD, score_transaction, score_transactions

app = FastAPI(
    title="FraudGuard API",
    description="Credit card fraud scoring using XGBoost",
    version="1.0.0",
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Preserve boolean values so they cannot be coerced into 0 or 1.
    features: dict[str, StrictBool | float] = Field(
        description="Time, Amount and V1 through V28"
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_request, exc):
    # Raw invalid input can contain NaN or Infinity, which JSON cannot encode.
    detail = [
        {key: error[key] for key in ("type", "loc", "msg")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/")
def home():
    return {"message": "FraudGuard API is running", "version": app.version}


@app.get("/health")
def health():
    # The model and metadata must load successfully before the server starts.
    return {"status": "healthy"}


@app.get("/features")
def features():
    return {"required_features": FEATURES, "threshold": THRESHOLD}


@app.post("/predict")
def predict(transaction: Transaction):
    result = score_transaction(transaction.features)
    if result["status"] != "ok":
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.post("/predict/batch")
def predict_batch(transactions: Annotated[list[Transaction], Field(min_length=1, max_length=5000)]):
    results = score_transactions([transaction.features for transaction in transactions])
    return {"total": len(results), "results": results}
