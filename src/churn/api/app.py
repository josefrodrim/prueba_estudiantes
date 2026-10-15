"""API REST para servir el modelo. Ejecutar: uvicorn churn.api.app:app --reload"""
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from churn.models.predict import predict

app = FastAPI(title="Telco Churn API", version="0.1.0")

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: YesNo
    Dependents: YesNo
    tenure: int
    PhoneService: YesNo
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: YesNo
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_one(customer: Customer):
    row = predict(pd.DataFrame([customer.model_dump()])).iloc[0]
    return {"churn_probability": float(row["churn_probability"]),
            "churn_prediction": int(row["churn_prediction"])}
