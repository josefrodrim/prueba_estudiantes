"""Entrena los modelos candidatos, selecciona el mejor por CV y lo guarda."""
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn.config import load_config, resolve
from churn.features.build_features import build_preprocessor
from churn.models.evaluate import evaluate
from churn.models.registry import get_model


def split(df: pd.DataFrame, cfg: dict):
    data_cfg = cfg["data"]
    X = df.drop(columns=[data_cfg["target"], data_cfg["id_column"]])
    y = df[data_cfg["target"]]
    return train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        stratify=y,
        random_state=cfg["project"]["random_state"],
    )


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    model_cfg = cfg["model"]

    df = pd.read_parquet(resolve(cfg["data"]["processed_path"]))
    X_train, X_test, y_train, y_test = split(df, cfg)

    # XGBoost no tiene class_weight: compensamos el desbalance con scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=model_cfg["cv_folds"], shuffle=True, random_state=seed)

    results = {}
    for name, params in model_cfg["candidates"].items():
        params = dict(params)
        if name == "xgboost":
            params["scale_pos_weight"] = pos_weight
        pipe = Pipeline([
            ("preprocess", build_preprocessor(cfg["features"])),
            ("model", get_model(name, params, seed)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=model_cfg["scoring"])
        results[name] = (scores.mean(), pipe)
        print(f"{name:<22} {model_cfg['scoring']}={scores.mean():.4f} ± {scores.std():.4f}")

    best_name, (best_score, best_pipe) = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\nMejor modelo: {best_name} ({best_score:.4f})")

    best_pipe.fit(X_train, y_train)
    model_path = resolve(cfg["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "name": best_name,
                 "threshold": model_cfg["threshold"]}, model_path)
    print(f"Modelo guardado en {model_path}")

    evaluate(best_pipe, X_test, y_test, cfg, model_name=best_name)


if __name__ == "__main__":
    main()
