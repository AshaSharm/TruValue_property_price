# app/model_manager.py
import joblib
import os
import threading
import time
import pandas as pd
from typing import Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "truvalue_model.joblib")
DEFAULT_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

_lock = threading.Lock()
_model = None
_model_version = "initial"

def load_model(path: str = DEFAULT_MODEL_PATH):
    global _model, _model_version
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    _model = joblib.load(path)
    _model_version = os.path.basename(path)
    return _model

def predict_row(features: dict) -> Tuple[float, str]:
    global _model
    if _model is None:
        load_model()
    # convert to DataFrame to keep pipeline consistent
    df = pd.DataFrame([features])
    preds = _model.predict(df)
    pred = float(preds[0])
    return pred, _model_version

def retrain_from_dataframe(df: pd.DataFrame, version_name: str = None, target_col: str = "Price_AED"):
    """
    Retrain a new pipeline on provided DataFrame (which must contain target_col),
    save to a temporary file, then atomically replace the production model and swap in memory.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    if target_col not in df.columns:
        raise ValueError(f"Retrain data must include target column '{target_col}'")

    # build pipeline dynamically based on df
    features = [c for c in df.columns if c != target_col]
    cat_features = [c for c in features if df[c].dtype == "object"]
    num_features = [c for c in features if c not in cat_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    new_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train new model (keep it local until saved and swapped)
    new_pipeline.fit(X, y)

    # decide filenames
    timestamp = int(time.time())
    version_name = version_name or f"truvalue_{timestamp}"
    new_model_filename = f"{version_name}.joblib"
    temp_path = os.path.join(MODEL_DIR, new_model_filename + ".temp")
    final_path = os.path.join(MODEL_DIR, new_model_filename)

    # Save to temp, then atomic replace
    joblib.dump(new_pipeline, temp_path)
    os.replace(temp_path, final_path)

    # atomically set in-memory model under lock
    global _model, _model_version
    with _lock:
        _model = new_pipeline
        _model_version = new_model_filename

    return new_model_filename

def retrain_from_csv_file(file_path: str, version_name: str = None, target_col: str = "Price_AED"):
    df = pd.read_csv(file_path)
    return retrain_from_dataframe(df, version_name=version_name, target_col=target_col)
