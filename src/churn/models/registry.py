"""Fábrica de modelos candidatos a partir del config."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_model(name: str, params: dict, random_state: int):
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if name == "xgboost":
        return XGBClassifier(random_state=random_state, eval_metric="logloss", **params)
    raise ValueError(f"Modelo desconocido: {name}")
