import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES
from src.features.build_features import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureBuilder,
    add_age_bin,
    add_payment_history_features,
    build_features,
    normalize_categories,
)


def client(**overrides) -> pd.DataFrame:
    """Один клиент с «ровной» историей, поля можно переопределить."""
    row = {col: 0 for col in RAW_FEATURES}
    row.update(LIMIT_BAL=100_000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=30)
    row.update(overrides)
    return pd.DataFrame([row])


def test_normalize_categories_maps_undocumented_codes():
    df = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6], "MARRIAGE": [0, 1, 2, 3, 0, 1, 2]})
    out = normalize_categories(df)
    assert out["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3, 3, 1, 2]


def test_delay_features():
    df = client(PAY_0=2, PAY_2=1, PAY_3=-1, PAY_4=-2, PAY_5=-2, PAY_6=0)
    out = add_payment_history_features(df).iloc[0]
    assert out["PAY_DELAY_MAX"] == 2
    assert out["PAY_DELAY_COUNT"] == 2
    assert out["PAY_NO_USE_COUNT"] == 2
    assert out["PAY_FULL_COUNT"] == 1


def test_delay_max_is_zero_without_delays():
    df = client(PAY_0=-2, PAY_2=-1, PAY_3=-1, PAY_4=0, PAY_5=0, PAY_6=-2)
    assert add_payment_history_features(df).iloc[0]["PAY_DELAY_MAX"] == 0


def test_utilization_features():
    bills = dict(zip([f"BILL_AMT{i}" for i in range(1, 7)], [80_000, 60_000, 40_000, 20_000, 0, 0]))
    out = add_payment_history_features(client(**bills)).iloc[0]
    assert out["UTIL_LAST"] == pytest.approx(0.8)
    assert out["UTIL_MAX"] == pytest.approx(0.8)
    assert out["UTIL_MEAN"] == pytest.approx(200_000 / 6 / 100_000)
    assert out["BILL_TREND"] == pytest.approx(0.8)


def test_pay_to_bill_ratio_matches_payment_with_previous_bill():
    # Платёж в сентябре (PAY_AMT1) гасит августовский счёт (BILL_AMT2).
    df = client(BILL_AMT2=10_000, PAY_AMT1=5_000)
    assert add_payment_history_features(df).iloc[0]["PAY_TO_BILL_RATIO"] == pytest.approx(0.5)


def test_pay_to_bill_ratio_without_debt_is_one_and_clipped():
    assert add_payment_history_features(client()).iloc[0]["PAY_TO_BILL_RATIO"] == 1.0
    overpaid = client(BILL_AMT2=100, PAY_AMT1=50_000)
    assert add_payment_history_features(overpaid).iloc[0]["PAY_TO_BILL_RATIO"] == 2.0


@pytest.mark.parametrize(
    "age, expected", [(21, "21-25"), (25, "21-25"), (26, "26-30"), (75, "61-100")]
)
def test_age_bins(age, expected):
    assert add_age_bin(client(AGE=age)).iloc[0]["AGE_BIN"] == expected


def test_build_features_without_fe_keeps_only_raw_columns(clean_df):
    out = build_features(clean_df, add_features=False)
    assert list(out.columns) == RAW_FEATURES


def test_feature_builder_output_matches_declared_names(clean_df):
    builder = FeatureBuilder().fit(clean_df)
    out = builder.transform(clean_df)
    assert list(out.columns) == list(builder.get_feature_names_out())
    assert set(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL) <= set(out.columns)
    assert not out[ENGINEERED_NUMERIC].isna().any().any()
    assert np.isfinite(out[ENGINEERED_NUMERIC].to_numpy(dtype=float)).all()


def test_feature_builder_ignores_extra_columns(clean_df):
    out = FeatureBuilder().fit_transform(clean_df.assign(EXTRA=1))
    assert "EXTRA" not in out.columns


def test_feature_builder_fails_on_missing_column(clean_df):
    with pytest.raises(ValueError, match="AGE"):
        FeatureBuilder().fit(clean_df.drop(columns="AGE"))
