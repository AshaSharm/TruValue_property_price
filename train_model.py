# train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

MODEL_DIR = os.path.join("app", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path="data/property_data.csv"):
    df = pd.read_csv(path)
    return df

def build_pipeline(df, target_col="Price_AED"):
    # Identify features
    features = [c for c in df.columns if c != target_col]
    # heuristics
    cat_features = [c for c in features if df[c].dtype == "object"]
    num_features = [c for c in features if c not in cat_features]

    # numeric transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # categorical transformer
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    return model, preprocessor, num_features, cat_features

def train_and_save(data_path="data/property_data.csv", model_path=None, preprocessor_path=None, target_col="Price_AED"):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "truvalue_model.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")

    df = load_data(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    model, preprocessor, num_features, cat_features = build_pipeline(df, target_col=target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Save the full pipeline (includes preprocessor)
    # Save first to temp file then replace for atomicity
    temp_model_path = model_path + ".temp"
    joblib.dump(model, temp_model_path)
    os.replace(temp_model_path, model_path)
    print(f"Saved model pipeline to {model_path}")

    # Save only the preprocessor too (useful for separate transformations)
    temp_prep_path = preprocessor_path + ".temp"
    joblib.dump(model.named_steps["preprocessor"], temp_prep_path)
    os.replace(temp_prep_path, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    # Evaluate
    # preds = model.predict(X_val)
    # mae = mean_absolute_error(y_val, preds)
    # rmse = mean_squared_error(y_val, preds, squared=False)
    # print(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save metadata
    metadata = {
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "num_features": num_features,
        "cat_features": cat_features
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return model

if __name__ == "__main__":
    train_and_save()
