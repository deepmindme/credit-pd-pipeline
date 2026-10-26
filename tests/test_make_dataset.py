import pandas as pd

from src.config import RAW_FEATURES, TARGET
from src.data.make_dataset import clean_data, split_data


def test_clean_data_drops_id_and_renames_target(raw_df):
    cleaned = clean_data(raw_df)
    assert "ID" not in cleaned.columns
    assert TARGET in cleaned.columns
    assert list(cleaned.columns) == RAW_FEATURES + [TARGET]


def test_clean_data_removes_full_duplicates(raw_df):
    # Строка-дубль с другим ID: после удаления ID она становится полной копией.
    duplicate = raw_df.iloc[[0]].assign(ID=10_000)
    cleaned = clean_data(pd.concat([raw_df, duplicate], ignore_index=True))
    assert len(cleaned) == len(raw_df)


def test_clean_data_casts_float_amounts_to_int(raw_df):
    raw_df["LIMIT_BAL"] = raw_df["LIMIT_BAL"].astype(float)
    cleaned = clean_data(raw_df)
    assert (cleaned.dtypes == "int64").all()


def test_split_is_stratified_and_reproducible(raw_df):
    df = clean_data(raw_df)
    train, test = split_data(df, test_size=0.25, seed=42)
    train_again, _ = split_data(df, test_size=0.25, seed=42)

    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.25) < 0.01
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.02
    assert train.index.equals(train_again.index)
