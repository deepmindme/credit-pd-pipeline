"""Общие пути, имена колонок и загрузка params.yaml."""

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"

RAW_TARGET = "default.payment.next.month"
TARGET = "default"
ID_COLUMN = "ID"

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Признаки клиента в том виде, в каком они приходят из источника (и в API).
RAW_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_COLUMNS,
    *PAY_AMT_COLUMNS,
]

CATEGORICAL_RAW = ["SEX", "EDUCATION", "MARRIAGE"]


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Относительные пути из params.yaml считаем от корня репозитория."""
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path
