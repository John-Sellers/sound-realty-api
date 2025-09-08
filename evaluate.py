import json
from pathlib import Path
import json, pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

DATA_DIR = Path("data")
sales = pd.read_csv(DATA_DIR / "kc_house_data.csv", dtype={"zipcode": str})
demos = pd.read_csv(DATA_DIR / "zipcode_demographics.csv", dtype={"zipcode": str})

# Join demographics on zipcode
df = sales.merge(demos, on="zipcode", how="left")

# Target and features
y = df["price"].values

# Use the model's feature list for a fair comparison
feature_order = json.load(open("model/model_features.json"))
X = df[feature_order].copy()

# Simple train/test split (time-based split would be even better, but this is fine for a quick check)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13
)


def report(model_name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name}: RMSE={rmse:,.0f}  MAE={mae:,.0f}")


# Baseline-ish pipeline: RobustScaler + LinearRegression (quick to train, interpretable)
numeric_cols = feature_order
pre = ColumnTransformer([("num", RobustScaler(), numeric_cols)], remainder="drop")
lin = Pipeline([("pre", pre), ("lr", LinearRegression(n_jobs=None))])

lin.fit(X_train, y_train)
pred_lin = lin.predict(X_test)
report("LinearRegression", y_test, pred_lin)

# Optional: a small-tree alternative to show nonlinearity capture (very light)
from sklearn.ensemble import RandomForestRegressor

rf = Pipeline(
    [
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=13, n_jobs=-1)),
    ]
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
report("RandomForestRegressor", y_test, pred_rf)

metrics = {
    "LinearRegression": {
        "rmse": float(mean_squared_error(y_test, pred_lin)),
        "mae": float(mean_absolute_error(y_test, pred_lin)),
    },
    "RandomForestRegressor": {
        "rmse": float(mean_squared_error(y_test, pred_rf)),
        "mae": float(mean_absolute_error(y_test, pred_rf)),
    },
}
pathlib.Path("model").mkdir(exist_ok=True)
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved metrics to model/metrics.json")
