"""Pydantic-схемы запросов и ответов API.

Ограничения на поля повторяют допустимые значения и диапазоны из Great
Expectations suite, так что запрос с явно некорректными данными отклоняется
ещё до модели (HTTP 422).
"""

from pydantic import BaseModel, ConfigDict, Field

BILL = dict(ge=-1_000_000, le=5_000_000)
AMOUNT = dict(ge=0, le=5_000_000)
PAY_STATUS = dict(ge=-2, le=8, description="-2/-1/0 — без просрочки, 1..8 — месяцев просрочки")

EXAMPLE_CLIENT = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 34,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 48210,
    "BILL_AMT2": 47120,
    "BILL_AMT3": 46000,
    "BILL_AMT4": 29500,
    "BILL_AMT5": 28100,
    "BILL_AMT6": 27400,
    "PAY_AMT1": 0,
    "PAY_AMT2": 1800,
    "PAY_AMT3": 1500,
    "PAY_AMT4": 1100,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
}


class ClientFeatures(BaseModel):
    """Данные клиента в формате исходного датасета (суммы в NT$)."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CLIENT})

    LIMIT_BAL: float = Field(ge=1_000, le=5_000_000, description="Кредитный лимит")
    SEX: int = Field(ge=1, le=2, description="1 — мужчина, 2 — женщина")
    EDUCATION: int = Field(ge=0, le=6, description="1 магистратура, 2 вуз, 3 школа, 4 другое")
    MARRIAGE: int = Field(ge=0, le=3, description="1 в браке, 2 холост, 3 другое")
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(**PAY_STATUS)
    PAY_2: int = Field(**PAY_STATUS)
    PAY_3: int = Field(**PAY_STATUS)
    PAY_4: int = Field(**PAY_STATUS)
    PAY_5: int = Field(**PAY_STATUS)
    PAY_6: int = Field(**PAY_STATUS)
    BILL_AMT1: float = Field(**BILL, description="Сумма счёта, сентябрь")
    BILL_AMT2: float = Field(**BILL)
    BILL_AMT3: float = Field(**BILL)
    BILL_AMT4: float = Field(**BILL)
    BILL_AMT5: float = Field(**BILL)
    BILL_AMT6: float = Field(**BILL, description="Сумма счёта, апрель")
    PAY_AMT1: float = Field(**AMOUNT, description="Сумма платежа, сентябрь")
    PAY_AMT2: float = Field(**AMOUNT)
    PAY_AMT3: float = Field(**AMOUNT)
    PAY_AMT4: float = Field(**AMOUNT)
    PAY_AMT5: float = Field(**AMOUNT)
    PAY_AMT6: float = Field(**AMOUNT, description="Сумма платежа, апрель")


class PredictionResponse(BaseModel):
    default_prediction: int = Field(description="1 — ожидается дефолт в следующем месяце")
    default_probability: float = Field(description="Оценка вероятности дефолта (PD)")
    threshold: float = Field(description="Порог, по которому PD переводится в класс")


class BatchRequest(BaseModel):
    clients: list[ClientFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    threshold: float | None = None
