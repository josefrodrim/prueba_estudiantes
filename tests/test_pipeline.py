from sklearn.pipeline import Pipeline

from churn.data.make_dataset import clean
from churn.features.build_features import build_preprocessor
from churn.models.registry import get_model


def test_end_to_end_pipeline_predicts_probabilities(sample_df, cfg):
    df = clean(sample_df)
    X, y = df.drop(columns=["customerID", "Churn"]), df["Churn"]
    pipe = Pipeline([
        ("preprocess", build_preprocessor(cfg["features"])),
        ("model", get_model("logistic_regression", {"max_iter": 500}, 42)),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
