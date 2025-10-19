# app/utils.py
from typing import List, Dict, Any
import pandas as pd

def json_rows_to_df(rows: List[Dict[str, Any]]):
    return pd.DataFrame(rows)

def sanitize_prediction(value: float):
    # business rule: price can't be negative, clamp to 0
    try:
        val = float(value)
    except Exception:
        raise ValueError("Prediction is not numeric.")
    if val < 0:
        return 0.0
    return val
