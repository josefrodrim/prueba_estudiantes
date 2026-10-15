"""Inferencia: carga el modelo entrenado y predice probabilidad de fuga."""
import argparse
from functools import lru_cache

import joblib
import pandas as pd

from churn.config import load_config, resolve
from churn.data.make_dataset import clean


@lru_cache
def load_model() -> dict:
    return joblib.load(resolve(load_config()["artifacts"]["model_path"]))


def predict(df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model()
    X = clean(df).drop(columns=["customerID", "Churn"], errors="ignore")
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame({"churn_probability": proba,
                        "churn_prediction": (proba >= bundle["threshold"]).astype(int)})
    if "customerID" in df.columns:
        out.insert(0, "customerID", df["customerID"].values)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción batch de churn")
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default="reports/predictions.csv")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.input_csv))
    result.to_csv(args.output, index=False)
    print(f"{len(result)} predicciones -> {args.output}")
