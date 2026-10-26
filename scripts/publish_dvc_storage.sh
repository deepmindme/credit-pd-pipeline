#!/usr/bin/env bash
# Публикация кэша DVC (датасет и модели) в ветку dvc-storage.
#
# DVC remote проекта — ветка dvc-storage этого же репозитория. DVC читает её
# по HTTP через raw.githubusercontent.com, поэтому `dvc pull` работает сразу
# после клонирования, без ключей и регистрации в облаках. Писать по HTTP
# нельзя, так что публикация идёт через git:
#   dvc push в локальную копию ветки -> git commit -> git push.
#
# Запускать после `dvc repro`, перед пушем основной ветки.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
STORAGE_DIR="${STORAGE_DIR:-$(pwd)/../.$(basename "$(pwd)")-dvc-storage}"

if [[ ! -d "$STORAGE_DIR" ]]; then
    if git ls-remote --exit-code --heads origin dvc-storage >/dev/null 2>&1; then
        git fetch origin dvc-storage:dvc-storage
        git worktree add "$STORAGE_DIR" dvc-storage
    else
        git worktree add --orphan -b dvc-storage "$STORAGE_DIR"
    fi
fi

# Локальный remote, указывающий на копию ветки (в .dvc/config.local, не в git).
dvc remote add --local --force publish "$STORAGE_DIR"
dvc push -r publish

cd "$STORAGE_DIR"
cat > README.md <<'EOF'
# dvc-storage

Служебная ветка: хранилище DVC (content-addressable кэш `files/md5/...`)
для датасета и моделей из основной ветки. Вручную здесь ничего не меняется,
ветка обновляется скриптом `scripts/publish_dvc_storage.sh`.
EOF
git add -A
if git diff --cached --quiet; then
    echo "Новых файлов в кэше нет"
else
    git commit -m "Обновление кэша DVC"
fi
git push -u origin dvc-storage
