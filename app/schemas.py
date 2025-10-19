from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    Area_sqft: float = Field(..., description="Area in square feet")
    Bedrooms: int = Field(..., description="Number of bedrooms")
    Bathrooms: int = Field(..., description="Number of bathrooms")
    Location: str = Field(..., description="Property location")
    Age_years: int = Field(..., description="Age of the property in years")


class PredictResponse(BaseModel):
    prediction: float
    model_version: str


class RetrainJsonRequest(BaseModel):
    data: list[dict] = Field(..., description="List of rows (dicts) including target column 'Price_AED'")
    version_name: Optional[str] = Field(None, description="Optional version name for the saved model")
