# 🏠 TruValue Dynamic Property Valuation API

A **production-ready, containerized FastAPI service** providing **property price valuations** with **dynamic retraining capabilities**.  
Supports **data preprocessing**, **model inference**, and **safe model updates without downtime**.

---
![image alt](https://github.com/AshaSharm/TruValue_property_price/blob/9b637c2afaca3eff8fdb42166a0ce2c163139110/truValue_flow.png)

## 📁 Project Structure

ml-challenge/
├── app/
│   ├── main.py                 # FastAPI app with endpoints
│   ├── model_manager.py        # Model loading, predicting, retraining logic
│   ├── schemas.py              # Pydantic request/response models
│   ├── utils.py                # Utilities like prediction sanitization
│   └── models/                 # Saved model artifacts (empty initially)
├── data/
│   └── property_data.csv       # Raw data for training
├── train_model.py              # Script to train initial model
├── sample_request.json         # Sample request for prediction input
├── requirements.txt            # Project dependencies
├── Dockerfile                  # Docker image build recipe
├── docker-compose.yml          # Docker compose to run container
└── README.md                   # README to understand the project
## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone <repository-url>
cd ml-challenge

2️⃣ Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate
Unix/macOS:

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Train the initial model
python train_model.py
This loads data from data/property_data.csv, trains a pipeline, and saves model artifacts to app/models/.

You should see output like:
Training complete.
Saved model pipeline to app/models/truvalue_model.joblib
Saved preprocessor to app/models/preprocessor.joblib

🐳 Running the API using Docker
Open Docker Desktop from your Start menu or taskbar.
🏗️ Build the Docker image:
docker build -t truvalue-api .
▶️ Or start using Docker Compose:
docker-compose up --build
Once running, the FastAPI server will be accessible at:
👉 http://localhost:8000

⚡ Running the API without Docker
You can also run locally with:
uvicorn app.main:app --reload --port 8000
Then open your browser:
👉 http://127.0.0.1:8000/docs

You’ll see the automatically generated Swagger UI.

🔮 API Usage
🔹 Predict Endpoint
Use sample_request.json as example input for prediction.

Request:
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_request.json
or curl request for predict    curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Area_sqft": 1800,
  "Bedrooms": 3,
  "Bathrooms": 3,
  "Location": "Downtown Dubai",
  "Age_years": 4
}'

Response:
{
  "prediction": 1854320.75,
  "model_version": "truvalue_model.joblib"
}


🔹 Retrain Endpoints
Option A — Upload CSV:
curl -X POST "http://127.0.0.1:8000/retrain_csv" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/property_data.csv;type=text/csv"
Option B — Send JSON:
Curl command to retrain curl -X 'POST' \
  'http://127.0.0.1:8000/retrain_json' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    { "Area_sqft": 1200, "Bedrooms": 2, "Bathrooms": 2, "Location": "Downtown Dubai", "Age_years": 3, "Price_AED": 1500000 },
  ],
  "version_name": "v2_test_retrain"
}'

Expected Response:
{
  "status": "retraining_started_from_csv",
  "message": "Running in background."
}


🔹 Health Check
Check the health of the running service:
curl http://127.0.0.1:8000/health

Response:
{
  "status": "ok",
  "model_version": "truvalue_1729308921.joblib"
}

🧠 Design Notes
🚀 Zero Downtime Retraining: Retraining runs asynchronously and atomically swaps in the new model using thread locks.

🔒 Robust Validation: Input validation and output sanitization ensure stable and sensible predictions.

⚙️ Preprocessing Pipeline: Automatically handles missing values, categorical encoding, and normalization.

📦 Tech Stack: FastAPI, scikit-learn, pandas, XGBoost.

🐳 Containerized Deployment: Easily deployable using Docker or Docker Compose.

🧩 Dynamic Updates: Supports graceful model updates via /retrain endpoints.

🏁 Summary
TruValue Dynamic Property Valuation API offers an end-to-end, retrainable ML service designed for real-world deployment.
It ensures scalability, maintainability, and reliability — combining clean MLOps practices with a modern FastAPI architecture.

📧 Author
Asha Sharma
💼 Designed as part of a technical assessment challenge.
