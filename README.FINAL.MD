# Sound Realty Price API

A tiny service that returns price estimates for Seattle homes.  
Clients send normal house details. The service looks up zipcode demographics on the server and never asks the client for them.

**Live demo**  
`https://john-sellers--sound-realty-api-fastapi-app.modal.run`

**Interactive docs**  
Append `/docs` to the base URL.

---

## Quick start

### Option A. Run locally with Docker
```bash
docker build -t sound-realty-api .
docker run -p 8000:8000 sound-realty-api
# open http://localhost:8000/docs
```
### Option B. Deploy to Modal
modal_app.py builds from the provided Dockerfile

```bash
modal deploy modal_app.py
# copy the printed URL
Optional. Streamlit demo app
Run locally
```

### API Base URL

Local: http://localhost:8000

Modal: https://john-sellers--sound-realty-api-fastapi-app.modal.run

### Endpoints
GET /: returns status and route pointers

GET /healthz: returns ok

POST /predict: request body matches a row from data/future_unseen_examples.csv. Demographics are added on the server by zipcode

POST /predict_required: accepts only the minimal set of house fields plus zipcode. The server fills in demographics and any unused feature columns

#### Minimal request example for /predict_required
```json
{
  "bedrooms": 4,
  "bathrooms": 1.0,
  "sqft_living": 1680,
  "sqft_lot": 5043,
  "floors": 1.5,
  "waterfront": 0,
  "view": 0,
  "condition": 4,
  "grade": 6,
  "sqft_above": 1680,
  "sqft_basement": 0,
  "yr_built": 1911,
  "yr_renovated": 0,
  "zipcode": 98118,
  "lat": 47.5354,
  "long": -122.273,
  "sqft_living15": 1560,
  "sqft_lot15": 5765
}
```
### Response example
```json
{
  "predictions": [512345.67],
  "count": 1,
  "model_meta": {
    "feature_count": 33,
    "feature_order_sha": "ab12cd34ef56"
  }
}
```

### Postman testing
A ready to import Postman collection is included at
postman/Sound-Realty-API.postman_collection.json

#### How to use it

In Postman Web:
1. import the collection file.

2. Set the collection variable base_url to one of:
    - http://localhost:8000 for local Docker
    - https://john-sellers--sound-realty-api-fastapi-app.modal.run for Modal

3. Use the included requests:

    - GET {{base_url}}/ root check

    - GET {{base_url}}/healthz health check

    - POST {{base_url}}/predict full record

    - POST {{base_url}}/predict_required minimal record

For POST requests, choose Body → raw → JSON and paste the example from this README or the example embedded in the collection.

#### Model artifacts and data
Training produces two artifacts saved in model/ directory:

- model.pkl trained pipeline

- model_features.json feature list and order used at train time

#### Data folder

- kc_house_data.csv training data, price is the target

- zipcode_demographics.csv census features joined by zipcode on the server

- future_unseen_examples.csv request shape for the API

#### Evaluation
Use evaluate.py to report RMSE and MAE and compare a simple linear model and a small tree based model.
```json
{
  "LinearRegression": {
    "rmse": 33961612011.61378,
    "mae": 112834.24818784947
  },
  "RandomForestRegressor": {
    "rmse": 27343372897.39475,
    "mae": 89226.13338424596
  }
}
```

### Design notes
#### Feature integrity
The service builds a single row in the exact training order using model_features.json. This prevents accidental column drift.

#### Demographic join
Clients send only the normal house attributes. The service reads zipcode_demographics.csv and joins by zipcode. The client never sends demographic columns.

#### Portability
One Dockerfile builds the model in the image and serves the same FastAPI app. The same image runs locally and on Modal.

#### Scaling and updates
##### Local

Run multiple containers and front them with a simple reverse proxy

##### Modal

`modal deploy` creates an ASGI app that scales containers on demand.

Deploying a new image swaps to the new version while old containers drain

#### Versioning and traceability

The response includes a short hash of the feature order

Image tags tie a specific model.pkl to a deployment

#### Security options

Keep endpoints open for review

For production, enable a simple bearer token in FastAPI or Modal proxy auth tokens

### Project structure
```pgsql
.
├─ app.py
├─ create_model.py
├─ evaluate.py
├─ test.py
├─ modal_app.py
├─ Dockerfile
├─ streamlit/
├─ streamlit_app.py
├─ requirements.txt
├─ data/
│  ├─ kc_house_data.csv
│  ├─ zipcode_demographics.csv
│  └─ future_unseen_examples.csv
├─ model/
│  ├─ model.pkl
│  ├─ model_features.json
│  └─ metrics.json  optional
└─ postman/
   └─ Sound-Realty-API.postman_collection.json
```

### How this meets the assignment
- A REST endpoint that receives JSON with fields from future_unseen_examples.csv, returns a prediction and metadata, and never asks the client to send demographics. The service joins demographics on the server by zipcode. Instructions Instructions Instructions

- Considerations for scale and for updating models without stopping the service are described, and Modal deploy supports version swaps while traffic continues. Instructions

- A second endpoint accepts only the minimum required features plus zipcode, meeting the bonus requirement. Instructions

- A simple test script is included in the repo as requested by the assignment. The README focuses on Postman for testing to match our review flow. Instructions

- An evaluation script is provided and used to report expected performance and generalization. Instructions

- Docker is used to package and run the service, which is explicitly recommended. 