#!/usr/bin/env bash
# Сборка Docker-образа, запуск контейнера и проверка, что API отвечает.
#
#   ./scripts/docker_build_run.sh          # собрать, запустить, проверить
#   ./scripts/docker_build_run.sh --stop   # остановить и удалить контейнер
set -euo pipefail

IMAGE="${IMAGE:-credit-pd-api:latest}"
CONTAINER="${CONTAINER:-credit-pd-api}"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--stop" ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "Контейнер $CONTAINER удалён"
    exit 0
fi

if [[ ! -f models/model.joblib ]]; then
    echo "Нет models/model.joblib — пробую скачать из DVC remote"
    dvc pull models/model.joblib models/model_meta.json
fi

echo "==> Сборка образа $IMAGE"
docker build -t "$IMAGE" .

echo "==> Запуск контейнера $CONTAINER на порту $PORT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p "${PORT}:8000" "$IMAGE" >/dev/null

echo -n "==> Жду готовности API"
for _ in $(seq 1 30); do
    if curl -sf "$URL/health" >/dev/null; then
        echo " — готово"
        break
    fi
    echo -n "."
    sleep 1
done

curl -sf "$URL/health" || { echo; echo "API не поднялся, логи:"; docker logs "$CONTAINER"; exit 1; }
echo

echo "==> Пробный запрос к /predict"
curl -sf -X POST "$URL/predict" \
    -H "Content-Type: application/json" \
    -d @scripts/sample_request.json
echo
echo
echo "Swagger UI: $URL/docs"
echo "Остановить: ./scripts/docker_build_run.sh --stop"
