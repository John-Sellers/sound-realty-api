import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union

import pandas as pd
from fastapi import FastAPI, HTTPException

# Load artifacts once
ART_DIR = Path("model")
MODEL_PATH = ART_DIR / "model.pkl"
FEATURES_PATH = ART_DIR / "model_features.json"
DEMOS_PATH = Path("data") / "zipcode_demographics.csv"

model = pickle.load(open(MODEL_PATH, "rb"))
feature_order: List[str] = json.load(open(FEATURES_PATH))
demos = pd.read_csv(DEMOS_PATH, dtype={"zipcode": str}).set_index("zipcode")

# Columns the client must send to the "required" endpoint
# We compute them as the model's features that do NOT come from demographics
DEMO_COLS = set(demos.columns) - {"zipcode"}
HOUSE_REQUIRED = sorted([c for c in feature_order if c not in DEMO_COLS])
REQUIRED_SET = set(HOUSE_REQUIRED + ["zipcode"])

app = FastAPI(title="Sound Realty Price API")


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/predict", "/docs", "/healthz"]}


@app.get("/healthz")
def healthz():
    return "ok"


def prepare_features(house: Dict[str, Any]) -> pd.DataFrame:
    # Must have zipcode to join demographics
    if "zipcode" not in house:
        raise HTTPException(status_code=400, detail="zipcode is required")

    # demographics are indexed by string zipcode
    z = str(house["zipcode"])
    if z not in demos.index:
        raise HTTPException(
            status_code=400, detail=f"zipcode {z} not found in demographics"
        )

    # merge request fields with demographic fields
    merged: Dict[str, Any] = {**house, **demos.loc[z].to_dict()}

    # select and order the exact features used in training
    row = {k: merged.get(k, 0) for k in feature_order}

    return pd.DataFrame([row], columns=feature_order)


@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    # accept either a single object or a list of objects
    rows = payload if isinstance(payload, list) else [payload]

    X = pd.concat([prepare_features(r) for r in rows], ignore_index=True)
    yhat = model.predict(X).tolist()

    return {
        "predictions": yhat,
        "count": len(yhat),
        "model_meta": {
            "feature_count": len(feature_order),
            "feature_order_sha": __import__("hashlib")
            .sha1(",".join(feature_order).encode())
            .hexdigest()[:12],
        },
    }


@app.post("/predict_required")
def predict_required(payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    items = payload if isinstance(payload, list) else [payload]

    # Validate that each item has at least the minimal fields
    def _validate(item: Dict[str, Any]) -> Dict[str, Any]:
        missing = [k for k in REQUIRED_SET if k not in item]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing required keys: {missing}"
            )
        return item

    validated = [_validate(x) for x in items]
    # Reuse the main predictor so the behavior stays identical
    return predict(validated)


@app.get("/predict_required/required_keys")
def required_keys():
    return {"required_keys": HOUSE_REQUIRED + ["zipcode"]}
