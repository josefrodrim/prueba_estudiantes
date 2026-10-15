"""Ingeniería de variables y preprocesamiento (ColumnTransformer)."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea variables derivadas. Es stateless, así que puede ir dentro del Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["avg_charge_per_month"] = X["TotalCharges"] / X["tenure"].replace(0, 1)
        X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        X["is_long_contract"] = (X["Contract"] != "Month-to-month").astype(int)
        return X


def build_preprocessor(features_cfg: dict) -> Pipeline:
    numeric = features_cfg["numeric"] + features_cfg["engineered"]
    categorical = features_cfg["categorical"]

    column_tf = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("engineer", FeatureEngineer()), ("columns", column_tf)])
