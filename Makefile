# Основные команды проекта. Предполагается активированное виртуальное окружение.
.PHONY: install data validate repro test lint format api docker docker-stop monitor monitor-drift mlflow-ui

install:  ## зависимости для разработки
	pip install -r requirements-dev.txt

data:  ## скачать датасет и модели из DVC remote
	dvc pull

validate:  ## проверка сырых данных правилами Great Expectations
	python -m src.data.validation data/raw/UCI_Credit_Card.csv --report reports/validation_raw.json

repro:  ## весь пайплайн: validate -> prepare -> train -> evaluate
	dvc repro

test:
	pytest --cov

lint:
	black --check .
	flake8 .

format:
	black .

api:  ## API локально, без Docker
	uvicorn src.api.app:app --reload --port 8000

docker:  ## собрать образ, запустить контейнер, проверить /predict
	./scripts/docker_build_run.sh

docker-stop:
	./scripts/docker_build_run.sh --stop

monitor:  ## поток заявок без дрифта -> PSI (API должен быть запущен)
	python -m src.monitoring.simulate_traffic --scenario baseline

monitor-drift:  ## поток со смещённым портфелем -> PSI
	python -m src.monitoring.simulate_traffic --scenario drift

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
