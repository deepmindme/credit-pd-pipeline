"""Загрузка обученной модели и предсказание PD.

Запуск из консоли: python -m src.models.predict data/processed/test.csv out.csv
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config import RAW_FEATURES, resolve

DEFAULT_MODEL_PATH = resolve("models/model.joblib")


@dataclass
class LoadedModel:
    model: object
    threshold: float
    meta: dict

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X[RAW_FEATURES])[:, 1]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                "default_probability": proba,
                "default_prediction": (proba >= self.threshold).astype(int),
            },
            index=X.index,
        )


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> LoadedModel:
    path = Path(path)
    # joblib = pickle, поэтому грузим только собственный артефакт из стадии
    # train (версионируется DVC). Файлы из непроверенных источников сюда
    # передавать нельзя.
    model = joblib.load(path)
    meta_path = path.with_name("model_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return LoadedModel(model=model, threshold=float(model.best_threshold_), meta=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model)
    df = pd.read_csv(args.input_csv)
    model.predict(df).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
