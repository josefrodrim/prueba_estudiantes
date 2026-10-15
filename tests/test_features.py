from churn.data.make_dataset import clean
from churn.features.build_features import FeatureEngineer, build_preprocessor


def test_feature_engineer_adds_columns(sample_df):
    out = FeatureEngineer().transform(clean(sample_df))
    for col in ["avg_charge_per_month", "num_services", "is_long_contract"]:
        assert col in out.columns
    assert out["avg_charge_per_month"].notna().all()


def test_preprocessor_output_has_no_nans(sample_df, cfg):
    X = clean(sample_df).drop(columns=["customerID", "Churn"])
    Xt = build_preprocessor(cfg["features"]).fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not (Xt != Xt).any()
