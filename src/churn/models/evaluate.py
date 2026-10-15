"""Métricas y gráficos sobre el conjunto de test."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from churn.config import resolve


def evaluate(pipeline, X_test, y_test, cfg: dict, model_name: str = "model") -> dict:
    threshold = cfg["model"]["threshold"]
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print("\n=== Evaluación en test ===")
    print(classification_report(y_test, pred, target_names=["No churn", "Churn"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    metrics_path = resolve(cfg["artifacts"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    fig_dir = resolve(cfg["artifacts"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No churn", "Churn"])
    plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight"); plt.close()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.savefig(fig_dir / "roc_curve.png", bbox_inches="tight"); plt.close()

    return metrics
