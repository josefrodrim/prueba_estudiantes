import pandas as pd
import pytest

from churn.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def sample_df():
    """Muestra sintética con el mismo esquema que el dataset Telco."""
    base = {
        "customerID": "0000-AAAA", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5, "TotalCharges": "846.0", "Churn": "Yes",
    }
    rows = []
    for i in range(40):
        r = dict(base, customerID=f"{i:04d}-X", tenure=i, Churn="Yes" if i % 3 == 0 else "No",
                 Contract="Two year" if i % 2 else "Month-to-month")
        rows.append(r)
    rows[0]["TotalCharges"] = " "  # caso real: cliente nuevo sin cargos
    return pd.DataFrame(rows)
