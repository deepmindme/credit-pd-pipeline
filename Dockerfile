# Образ для инференса: FastAPI + обученная модель из стадии train.
# Перед сборкой модель должна лежать в models/ (dvc repro или dvc pull).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Зависимости ставим отдельным слоем: при изменении кода он берётся из кэша.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY params.yaml .
COPY src/ src/
COPY models/model.joblib models/model_meta.json models/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
