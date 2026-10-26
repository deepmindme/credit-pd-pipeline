"""Проверяем, что suite Great Expectations пропускает нормальные данные
и ловит типичные аномалии, которые могут прийти в новой выгрузке."""

import numpy as np
import pytest

from src.config import RAW_TARGET, ROOT_DIR
from src.data.make_dataset import load_raw
from src.data.validation import build_expectation_suite, suite_to_json, validate_dataframe

RAW_PATH = ROOT_DIR / "data/raw/UCI_Credit_Card.csv"

# колонка, «плохое» значение, правило, которое должно сработать
OUT_OF_DOMAIN = {
    "age_150": ("AGE", 150, "expect_column_values_to_be_between"),
    "age_minor": ("AGE", 16, "expect_column_values_to_be_between"),
    "negative_limit": ("LIMIT_BAL", -5_000, "expect_column_values_to_be_between"),
    "negative_payment": ("PAY_AMT3", -100, "expect_column_values_to_be_between"),
    "unknown_sex_code": ("SEX", 3, "expect_column_values_to_be_in_set"),
    "unknown_education": ("EDUCATION", 9, "expect_column_values_to_be_in_set"),
    "pay_status_12": ("PAY_0", 12, "expect_column_values_to_be_in_set"),
    "target_2": (RAW_TARGET, 2, "expect_column_values_to_be_in_set"),
}


def failed_rules(report) -> set[tuple[str, str | None]]:
    return {(f["expectation"], f["column"]) for f in report.failed}


def test_clean_data_passes(raw_df):
    report = validate_dataframe(raw_df)
    assert report.success, report.failed
    assert report.n_expectations > 50


@pytest.mark.parametrize("column, value, rule", OUT_OF_DOMAIN.values(), ids=list(OUT_OF_DOMAIN))
def test_out_of_domain_value_is_detected(raw_df, column, value, rule):
    raw_df.loc[0, column] = value
    report = validate_dataframe(raw_df)
    assert not report.success
    assert (rule, column) in failed_rules(report)


def test_null_is_detected(raw_df):
    raw_df["LIMIT_BAL"] = raw_df["LIMIT_BAL"].astype(float)
    raw_df.loc[3, "LIMIT_BAL"] = np.nan
    report = validate_dataframe(raw_df)
    assert ("expect_column_values_to_not_be_null", "LIMIT_BAL") in failed_rules(report)


def test_wrong_type_is_detected(raw_df):
    raw_df["BILL_AMT1"] = raw_df["BILL_AMT1"].astype(str)
    report = validate_dataframe(raw_df)
    assert ("expect_column_values_to_be_in_type_list", "BILL_AMT1") in failed_rules(report)


def test_duplicated_id_is_detected(raw_df):
    raw_df.loc[1, "ID"] = raw_df.loc[0, "ID"]
    report = validate_dataframe(raw_df)
    assert ("expect_column_values_to_be_unique", "ID") in failed_rules(report)


def test_default_rate_shift_is_detected(raw_df):
    raw_df[RAW_TARGET] = 1
    report = validate_dataframe(raw_df)
    assert ("expect_column_mean_to_be_between", RAW_TARGET) in failed_rules(report)


def test_missing_column_is_detected(raw_df):
    report = validate_dataframe(raw_df.drop(columns="PAY_AMT6"))
    assert ("expect_table_columns_to_match_set", None) in failed_rules(report)


def test_unexpected_column_is_detected(raw_df):
    report = validate_dataframe(raw_df.assign(EXTRA=0))
    assert ("expect_table_columns_to_match_set", None) in failed_rules(report)


def test_scoring_data_without_target(raw_df):
    report = validate_dataframe(raw_df.drop(columns=RAW_TARGET), with_target=False)
    assert report.success, report.failed


def test_exported_suite_has_no_random_ids():
    data = suite_to_json(build_expectation_suite())
    assert "id" not in data
    assert all("id" not in exp for exp in data["expectations"])


@pytest.mark.skipif(not RAW_PATH.exists(), reason="нет данных: выполните dvc pull")
def test_real_raw_data_passes_suite():
    """Падает, если в актуальной выгрузке data/raw появились аномалии."""
    report = validate_dataframe(load_raw(RAW_PATH))
    assert report.success, report.failed
