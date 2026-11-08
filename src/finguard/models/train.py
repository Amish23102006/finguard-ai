"""Train, evaluate and persist the baseline credit and fraud models.

Run ``python -m finguard.models.train``. Metrics are written to ``<model_dir>/metrics.json``;
copy only *measured* values into model cards and the README.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from finguard.config import get_settings
from finguard.features import CREDIT_FEATURES, FRAUD_FEATURES
from finguard.logging_config import configure_logging
from finguard.models.bundle import ModelBundle, compute_stats, make_pipeline
from finguard.synthetic.generator import generate_credit_training, generate_fraud_training

logger = logging.getLogger(__name__)

CREDIT_MODEL = "credit-logreg"
FRAUD_MODEL = "fraud-logreg"
MODEL_VERSION = "1.0.0"


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width-bin expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _temporal_split(
    df: pd.DataFrame, order_col: str, frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier records, test on later ones (no random shuffling -> less leakage)."""
    df = df.sort_values(order_col)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def _fit(
    name: str, df: pd.DataFrame, features: list[str], label: str, order: str, seed: int
) -> ModelBundle:
    train, test = _temporal_split(df, order)
    pipe = make_pipeline(seed)
    pipe.fit(train[features].to_numpy(), train[label].to_numpy())
    p = pipe.predict_proba(test[features].to_numpy())[:, 1]
    y = test[label].to_numpy()
    metrics: dict[str, Any] = {
        "n_train": len(train),
        "n_test": len(test),
        "positive_rate_test": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "split": "temporal 80/20 on synthetic order column",
        "data": "synthetic (see data/README.md)",
    }
    return ModelBundle(
        name=name,
        version=MODEL_VERSION,
        features=features,
        pipeline=pipe,
        stats=compute_stats(train[features].to_numpy(), features),
        metrics=metrics,
        seed=seed,
    )


def train_all(seed: int = 42) -> dict[str, ModelBundle]:
    """Train both baseline models deterministically."""
    credit = _fit(
        CREDIT_MODEL,
        generate_credit_training(seed=seed),
        CREDIT_FEATURES,
        "default",
        "application_order",
        seed,
    )
    fraud = _fit(
        FRAUD_MODEL,
        generate_fraud_training(seed=seed),
        FRAUD_FEATURES,
        "fraud",
        "event_order",
        seed,
    )
    return {credit.name: credit, fraud.name: fraud}


def save_bundles(bundles: dict[str, ModelBundle], model_dir: Path) -> None:
    """Persist bundles and a metrics summary."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        joblib.dump(bundle, model_dir / f"{name}.joblib")
    summary = {n: {"version": b.version, **b.metrics} for n, b in bundles.items()}
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=settings.resolve(settings.model_dir))
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()
    configure_logging(settings.log_level)
    bundles = train_all(args.seed)
    save_bundles(bundles, args.out)
    for name, b in bundles.items():
        m = b.metrics
        logger.info(
            "%s v%s ROC-AUC=%.3f PR-AUC=%.3f Brier=%.4f ECE=%.4f",
            name,
            b.version,
            m["roc_auc"],
            m["pr_auc"],
            m["brier"],
            m["ece"],
        )
    logger.info("saved to %s", args.out)


if __name__ == "__main__":
    main()
