from churn.data.make_dataset import clean


def test_clean_handles_blank_total_charges(sample_df):
    df = clean(sample_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0


def test_clean_encodes_target(sample_df):
    df = clean(sample_df)
    assert set(df["Churn"].unique()) <= {0, 1}
