"""Limpieza del dataset crudo -> data/processed/."""
import pandas as pd

from churn.config import load_config, resolve


def clean(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    # TotalCharges viene como texto con espacios vacíos para clientes nuevos (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)
    if target in df.columns:
        df[target] = df[target].map({"Yes": 1, "No": 0}).astype(int)
    return df.drop_duplicates()


def load_raw() -> pd.DataFrame:
    return pd.read_csv(resolve(load_config()["data"]["raw_path"]))


def main() -> None:
    cfg = load_config()["data"]
    df = clean(load_raw(), cfg["target"])
    out = resolve(cfg["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Dataset limpio: {df.shape} -> {out}")
    print(f"Tasa de churn: {df[cfg['target']].mean():.2%}")


if __name__ == "__main__":
    main()
