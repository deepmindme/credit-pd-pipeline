"""Стадия prepare: загрузка сырых данных, очистка и разбиение на train/test.

Здесь только то, что относится к подготовке обучающей выборки. Всё, что
нужно повторять на инференсе (чистка категорий, новые признаки), живёт в
src/features/build_features.py и входит в sklearn Pipeline.

Запуск: python -m src.data.make_dataset [raw_csv] [out_dir]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMN, RAW_FEATURES, RAW_TARGET, TARGET, load_params, resolve

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Загружено %d строк, %d колонок из %s", len(df), df.shape[1], path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает ID и полные дубликаты, приводит типы, переименовывает таргет."""
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    df = df.rename(columns={RAW_TARGET: TARGET})

    # Без ID в данных есть одинаковые строки. Если такая пара попадёт и в train,
    # и в test, это мелкая, но утечка, поэтому оставляем одну копию.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Удалено дубликатов: %d", n_before - len(df))

    # В исходном CSV суммы записаны как float (20000.0), хотя дробных частей нет.
    df[RAW_FEATURES + [TARGET]] = df[RAW_FEATURES + [TARGET]].astype("int64")
    return df[RAW_FEATURES + [TARGET]]


def split_data(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Стратификация по таргету: доля дефолтов ~22%, хотим одинаковую в обеих частях.
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", nargs="?", default=params["data"]["raw_path"])
    parser.add_argument("out_dir", nargs="?", default=params["data"]["processed_dir"])
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_raw(resolve(args.raw_path)))
    train, test = split_data(df, params["data"]["test_size"], params["seed"])
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    summary = {
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
        "default_rate_train": round(float(train[TARGET].mean()), 4),
        "default_rate_test": round(float(test[TARGET].mean()), 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Готово: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
