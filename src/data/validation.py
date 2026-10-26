"""Валидация входящих данных с помощью Great Expectations.

Набор правил (suite) описан в коде, чтобы его было удобно читать и ревьюить.
Его JSON-копия сохраняется в data/expectations/ как артефакт.

Запуск: python -m src.data.validation data/raw/UCI_Credit_Card.csv
Код возврата 1, если хотя бы одно правило нарушено. На этом держится шаг в CI.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.config import (
    BILL_COLUMNS,
    ID_COLUMN,
    PAY_AMT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURES,
    RAW_TARGET,
)

logger = logging.getLogger(__name__)

SUITE_NAME = "credit_data_suite"
NUMERIC_TYPES = ["int64", "float64", "Int64", "Float64"]

# Допустимые значения взяты из описания датасета на UCI плюс коды, которые
# реально встречаются в данных, хотя в описании их нет: EDUCATION 0/5/6,
# MARRIAGE 0, PAY = -2. Если в источнике появится новый код, это повод
# разобраться, а не молча обучаться.
CATEGORY_SETS = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
    **{col: list(range(-2, 9)) for col in PAY_STATUS_COLUMNS},
}

# Диапазоны для сумм заданы с запасом относительно наблюдаемых значений
# (лимит 10k–1M, счета от -340k до 1.7M). Цель правил — ловить явный мусор
# вроде отрицательного лимита или выплат, а не отсекать редких клиентов.
VALUE_RANGES = {
    "LIMIT_BAL": (1_000, 5_000_000),
    "AGE": (18, 100),
    **{col: (-1_000_000, 5_000_000) for col in BILL_COLUMNS},
    **{col: (0, 5_000_000) for col in PAY_AMT_COLUMNS},
}


@dataclass
class ValidationReport:
    success: bool
    n_rows: int
    n_expectations: int
    failed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_expectation_suite(with_target: bool = True) -> gx.ExpectationSuite:
    columns = [ID_COLUMN, *RAW_FEATURES] + ([RAW_TARGET] if with_target else [])
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=columns, exact_match=True),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        gxe.ExpectColumnValuesToBeUnique(column=ID_COLUMN),
    ]
    for col in columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
        expectations.append(
            gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=NUMERIC_TYPES)
        )
    for col, values in CATEGORY_SETS.items():
        expectations.append(gxe.ExpectColumnValuesToBeInSet(column=col, value_set=values))
    for col, (low, high) in VALUE_RANGES.items():
        expectations.append(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=low, max_value=high)
        )
    if with_target:
        expectations += [
            gxe.ExpectColumnValuesToBeInSet(column=RAW_TARGET, value_set=[0, 1]),
            # Доля дефолтов в портфеле ~22%. Если она вдруг 2% или 60%, скорее
            # всего сломалась выгрузка, а не изменились клиенты.
            gxe.ExpectColumnMeanToBeBetween(column=RAW_TARGET, min_value=0.05, max_value=0.5),
        ]
    return gx.ExpectationSuite(name=SUITE_NAME, expectations=expectations)


def suite_to_json(suite: gx.ExpectationSuite) -> dict:
    """JSON без случайных id, чтобы файл не менялся от запуска к запуску."""
    data = suite.to_json_dict()
    data.pop("id", None)
    for exp in data["expectations"]:
        exp.pop("id", None)
    return data


def _short_result(res) -> dict:
    cfg = res.expectation_config
    info = {"expectation": cfg.type, "column": cfg.kwargs.get("column")}
    for key in ("unexpected_count", "unexpected_percent", "observed_value"):
        if key in res.result:
            info[key] = res.result[key]
    if res.result.get("partial_unexpected_list"):
        info["examples"] = [str(v) for v in res.result["partial_unexpected_list"][:5]]
    return info


def validate_dataframe(df: pd.DataFrame, with_target: bool = True) -> ValidationReport:
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    suite = context.suites.add(build_expectation_suite(with_target))
    batch = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("credit_data")
        .add_batch_definition_whole_dataframe("whole")
        .get_batch(batch_parameters={"dataframe": df})
    )
    result = batch.validate(suite)
    failed = [_short_result(r) for r in result.results if not r.success]
    return ValidationReport(
        success=bool(result.success),
        n_rows=len(df),
        n_expectations=len(result.results),
        failed=failed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--report", type=Path, help="куда сохранить JSON-отчёт")
    parser.add_argument("--suite-out", type=Path, help="куда выгрузить suite в JSON")
    parser.add_argument("--no-target", action="store_true", help="данные без колонки таргета")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    report = validate_dataframe(df, with_target=not args.no_target)

    if args.suite_out:
        args.suite_out.parent.mkdir(parents=True, exist_ok=True)
        suite = build_expectation_suite(with_target=not args.no_target)
        args.suite_out.write_text(json.dumps(suite_to_json(suite), indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2, default=str))

    status = "OK" if report.success else "FAILED"
    logger.info(
        "%s: %s — проверено %d правил на %d строках",
        args.csv_path,
        status,
        report.n_expectations,
        report.n_rows,
    )
    for item in report.failed:
        logger.error("Нарушено правило: %s", item)
    return 0 if report.success else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    sys.exit(main())
