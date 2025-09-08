# test.py
import sys, json, time
import pandas as pd
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

df = pd.read_csv("data/future_unseen_examples.csv")
rows = df.iloc[:N].to_dict(orient="records")

print(f"Sending {N} example(s) to {BASE}/predict")
resp = requests.post(f"{BASE}/predict", json=rows, timeout=30)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:1500])
