import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st
import requests

# ---------- Page setup ----------
st.set_page_config(page_title="Sound Realty Demo", page_icon="🏠", layout="centered")

PRIMARY = "#2563eb"
ACCENT = "#0ea5e9"
st.markdown(
    f"""
    <style>
      .title {{ font-size: 2.0rem; font-weight: 700; color: {PRIMARY}; margin-bottom: .25rem; }}
      .subtitle {{ color: #64748b; margin-bottom: 1rem; }}
      .card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 16px rgba(0,0,0,.06);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
      }}
      .metric-big span {{ font-size: 2rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">Sound Realty Price Estimator</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Enter basic home details. We will fetch demographics by zipcode on the backend.</div>',
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
DATA_DIR = Path("data")
ART_DIR = Path("model")
MODEL_PATH = ART_DIR / "model.pkl"
FEATURES_PATH = ART_DIR / "model_features.json"
DEMOS_PATH = DATA_DIR / "zipcode_demographics.csv"
FUTURE_PATH = DATA_DIR / "future_unseen_examples.csv"


def load_defaults() -> Dict[str, Any]:
    try:
        row = pd.read_csv(FUTURE_PATH).iloc[0].to_dict()
        return {
            "bedrooms": int(row.get("bedrooms", 3)),
            "bathrooms": float(row.get("bathrooms", 2.0)),
            "sqft_living": int(row.get("sqft_living", 1800)),
            "sqft_lot": int(row.get("sqft_lot", 5000)),
            "floors": float(row.get("floors", 1.0)),
            "waterfront": int(row.get("waterfront", 0)),
            "view": int(row.get("view", 0)),
            "condition": int(row.get("condition", 3)),
            "grade": int(row.get("grade", 7)),
            "sqft_above": int(row.get("sqft_above", 1500)),
            "sqft_basement": int(row.get("sqft_basement", 0)),
            "yr_built": int(row.get("yr_built", 1975)),
            "yr_renovated": int(row.get("yr_renovated", 0)),
            "zipcode": int(row.get("zipcode", 98118)),
            "lat": float(row.get("lat", 47.55)),
            "long": float(row.get("long", -122.28)),
            "sqft_living15": int(row.get("sqft_living15", 1600)),
            "sqft_lot15": int(row.get("sqft_lot15", 6000)),
        }
    except Exception:
        return {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 1.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 7,
            "sqft_above": 1500,
            "sqft_basement": 0,
            "yr_built": 1975,
            "yr_renovated": 0,
            "zipcode": 98118,
            "lat": 47.55,
            "long": -122.28,
            "sqft_living15": 1600,
            "sqft_lot15": 6000,
        }


def local_predict(house: Dict[str, Any]) -> Dict[str, Any]:
    # Load artifacts lazily so the app can still open without them
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        # Try to build model once if missing
        if Path("create_model.py").exists():
            import subprocess

            subprocess.run(["python", "create_model.py"], check=True)
        else:
            raise RuntimeError(
                "Model artifacts not found and create_model.py is missing"
            )

    model = pickle.load(open(MODEL_PATH, "rb"))
    feature_order = json.load(open(FEATURES_PATH))
    demos = pd.read_csv(DEMOS_PATH, dtype={"zipcode": str}).set_index("zipcode")

    z = str(house["zipcode"])
    if z not in demos.index:
        raise ValueError(f"zipcode {z} not found in demographics")

    merged = {**house, **demos.loc[z].to_dict()}
    row = {k: merged.get(k, 0) for k in feature_order}
    X = pd.DataFrame([row], columns=feature_order)
    yhat = model.predict(X).tolist()
    return {"predictions": yhat, "count": 1}


# ---------- Sidebar ----------
st.sidebar.header("Run mode")
mode = st.sidebar.radio(
    "Choose where to run predictions", ["Use Modal API", "Run locally"]
)
default_api = os.getenv("API_BASE_URL", "")
api_base = st.sidebar.text_input(
    "Modal API base URL",
    value=default_api,
    help="Example: https://john-sellers--sound-realty-api-fastapi-app.modal.run",
)
st.sidebar.caption(
    "Switch to local mode if you are offline or want to demo without the API"
)

# ---------- Input form ----------
defaults = load_defaults()
with st.form("house-form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    bedrooms = c1.number_input(
        "Bedrooms", min_value=0, value=defaults["bedrooms"], step=1
    )
    bathrooms = c2.number_input(
        "Bathrooms",
        min_value=0.0,
        value=float(defaults["bathrooms"]),
        step=0.25,
        format="%.2f",
    )
    floors = c3.number_input(
        "Floors",
        min_value=0.0,
        value=float(defaults["floors"]),
        step=0.5,
        format="%.1f",
    )

    c4, c5, c6 = st.columns(3)
    sqft_living = c4.number_input(
        "Sqft living", min_value=0, value=defaults["sqft_living"], step=10
    )
    sqft_lot = c5.number_input(
        "Sqft lot", min_value=0, value=defaults["sqft_lot"], step=10
    )
    sqft_above = c6.number_input(
        "Sqft above", min_value=0, value=defaults["sqft_above"], step=10
    )

    c7, c8, c9 = st.columns(3)
    sqft_basement = c7.number_input(
        "Sqft basement", min_value=0, value=defaults["sqft_basement"], step=10
    )
    waterfront = c8.selectbox(
        "Waterfront", options=[0, 1], index=defaults["waterfront"]
    )
    view = c9.number_input(
        "View", min_value=0, max_value=4, value=defaults["view"], step=1
    )

    c10, c11, c12 = st.columns(3)
    condition = c10.number_input(
        "Condition", min_value=1, max_value=5, value=defaults["condition"], step=1
    )
    grade = c11.number_input(
        "Grade", min_value=1, max_value=13, value=defaults["grade"], step=1
    )
    yr_built = c12.number_input(
        "Year built", min_value=1800, max_value=2025, value=defaults["yr_built"], step=1
    )

    c13, c14, c15 = st.columns(3)
    yr_renovated = c13.number_input(
        "Year renovated",
        min_value=0,
        max_value=2025,
        value=defaults["yr_renovated"],
        step=1,
    )
    zipcode = c14.number_input(
        "Zipcode", min_value=1, value=defaults["zipcode"], step=1
    )
    lat = c15.number_input(
        "Latitude", value=float(defaults["lat"]), step=0.0001, format="%.6f"
    )

    c16, c17 = st.columns(2)
    long = c16.number_input(
        "Longitude", value=float(defaults["long"]), step=0.0001, format="%.6f"
    )
    sqft_living15 = c17.number_input(
        "Sqft living 15", min_value=0, value=defaults["sqft_living15"], step=10
    )

    c18 = st.columns(1)[0]
    sqft_lot15 = c18.number_input(
        "Sqft lot 15", min_value=0, value=defaults["sqft_lot15"], step=10
    )

    submitted = st.form_submit_button("Estimate price", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    payload = {
        "bedrooms": bedrooms,
        "bathrooms": float(bathrooms),
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "grade": int(grade),
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "zipcode": int(zipcode),
        "lat": float(lat),
        "long": float(long),
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
    }

    try:
        if mode == "Use Modal API":
            if not api_base:
                st.error("Please enter your Modal API base URL in the sidebar")
            else:
                url = api_base.rstrip("/") + "/predict_required"
                r = requests.post(url, json=payload, timeout=30)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    res = r.json()
                    price = float(res["predictions"][0])
                    st.success("Prediction complete")
                    st.metric(
                        "Estimated price",
                        f"${price:,.0f}",
                        help="Single prediction using /predict_required",
                    )
        else:
            res = local_predict(payload)
            price = float(res["predictions"][0])
            st.success("Prediction complete")
            st.metric(
                "Estimated price",
                f"${price:,.0f}",
                help="Single prediction using local model",
            )
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
