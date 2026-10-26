"""Имитация потока новых клиентов и расчёт дрифта (PSI).

Скрипт берёт случайную часть отложенной выборки, отправляет её в работающий
API пачками, получает PD и сравнивает распределения скоров и признаков с
эталоном из train (models/reference_profile.json).

Сценарии:
  baseline — данные как есть, PSI должен быть небольшим;
  drift    — искусственный сдвиг (портфель «помолодел», лимиты урезаны,
             просрочек стало больше), PSI должен это поймать.

Пример:
  python -m src.monitoring.simulate_traffic --api-url http://127.0.0.1:8000 --scenario drift
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import PAY_STATUS_COLUMNS, RAW_FEATURES, TARGET, resolve  # noqa: E402
from src.monitoring.drift import drift_report, psi_status  # noqa: E402

logger = logging.getLogger(__name__)


def apply_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Сценарий «ухудшение портфеля»: так могли бы выглядеть данные после
    запуска маркетинговой акции для молодых клиентов в период спада."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["AGE"] = np.clip(df["AGE"] - rng.integers(5, 12, len(df)), 21, 100)
    df["LIMIT_BAL"] = np.maximum((df["LIMIT_BAL"] * 0.6).round(-4), 10_000)
    worse = rng.random(len(df)) < 0.35
    for col in PAY_STATUS_COLUMNS:
        df.loc[worse, col] = np.clip(df.loc[worse, col] + 2, -2, 8)
    return df


def send_batches(api_url: str, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    results = []
    with requests.Session() as session:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            payload = {"clients": chunk[RAW_FEATURES].to_dict(orient="records")}
            response = session.post(f"{api_url}/predict/batch", json=payload, timeout=30)
            response.raise_for_status()
            results.extend(response.json()["predictions"])
    return pd.DataFrame(results, index=df.index)


def plot_score_shift(reference: dict, scores, path: Path) -> None:
    cuts = reference["cuts"]
    labels = [f"≤{cuts[0]:.2f}"] + [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(cuts[:-1], cuts[1:])]
    labels += [f">{cuts[-1]:.2f}"]
    idx = np.searchsorted(cuts, scores, side="left")
    actual = np.bincount(idx, minlength=len(cuts) + 1) / len(scores)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, reference["proportions"], width=0.4, label="train (эталон)")
    ax.bar(x + 0.2, actual, width=0.4, label="новые данные")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set(ylabel="Доля клиентов", title="Распределение PD по бинам эталона")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data", type=Path, default=resolve("data/processed/test.csv"))
    parser.add_argument("--profile", type=Path, default=resolve("models/reference_profile.json"))
    parser.add_argument("--scenario", choices=["baseline", "drift"], default="baseline")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=resolve("reports/monitoring"))
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="код возврата 2, если PSI скоров ≥ 0.25 (для CD-гейта)",
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    data = pd.read_csv(args.data)
    data = data.sample(n=min(args.n_samples, len(data)), random_state=args.seed)
    if args.scenario == "drift":
        data = apply_drift(data, args.seed)

    logger.info("Отправляем %d клиентов в %s (сценарий %s)", len(data), args.api_url, args.scenario)
    predictions = send_batches(args.api_url, data, args.batch_size)
    scores = predictions["default_probability"].to_numpy()

    report = drift_report(profile, data[RAW_FEATURES], scores)
    score_psi = float(report.loc[report["variable"] == "score", "psi"].iloc[0])
    summary = {
        "scenario": args.scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_clients": int(len(data)),
        "score_psi": round(score_psi, 4),
        "score_status": psi_status(score_psi),
        "drifted_features": report.query("variable != 'score' and status != 'stable'")[
            "variable"
        ].tolist(),
        "mean_pd": round(float(scores.mean()), 4),
        "predicted_default_share": round(float(predictions["default_prediction"].mean()), 4),
        # Метки в отложенной выборке известны, поэтому можно заодно проверить,
        # не просело ли качество. В реальности они приходят с задержкой.
        "actual_default_share": round(float(data[TARGET].mean()), 4),
        "roc_auc": round(float(roc_auc_score(data[TARGET], scores)), 4),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.scenario
    report.round(4).to_csv(f"{prefix}_psi.csv", index=False)
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_score_shift(profile["score"], scores, Path(f"{prefix}_score_shift.png"))

    print(report.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.fail_on_drift and summary["score_status"] == "significant":
        logger.error("PSI скоров %.3f ≥ 0.25 — значимый дрифт", score_psi)
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
