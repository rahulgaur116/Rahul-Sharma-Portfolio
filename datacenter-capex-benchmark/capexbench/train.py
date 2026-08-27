"""Train quantile ML models for cost-per-MW estimation.

Three GradientBoostingRegressor models (quantile loss at 0.1 / 0.5 / 0.9)
predict log cost-per-MW from project parameters, giving calibrated
P10/P50/P90 estimate bands. Trained artifacts are saved to models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .dataset import load

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

CATEGORICAL = ["facility_type", "region", "country", "market", "cooling",
               "redundancy", "build_type"]
NUMERIC = ["delivery_year", "it_load_mw", "kw_per_rack"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "cost_per_mw_musd"
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def _make_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    # Tail quantiles get stronger regularization - deep quantile GBMs
    # overfit the tails and undercover the nominal P10-P90 band.
    is_tail = alpha != 0.50
    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=alpha,
        n_estimators=300 if is_tail else 500,
        learning_rate=0.05,
        max_depth=3 if is_tail else 4,
        subsample=0.85,
        min_samples_leaf=25 if is_tail else 8,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("gbr", gbr)])


def train(save: bool = True) -> dict:
    df = load()
    X = df[FEATURES]
    y = np.log(df[TARGET])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models: dict[str, Pipeline] = {}
    metrics: dict[str, float] = {}
    for name, alpha in QUANTILES.items():
        pipe = _make_pipeline(alpha)
        pipe.fit(X_tr, y_tr)
        models[name] = pipe

    # Evaluate on holdout in original units
    pred_p50 = np.exp(models["p50"].predict(X_te))
    actual = np.exp(y_te)
    metrics["mape_p50"] = round(float(mean_absolute_percentage_error(actual, pred_p50)), 4)

    p10 = np.exp(models["p10"].predict(X_te))
    p90 = np.exp(models["p90"].predict(X_te))
    metrics["coverage_p10_p90"] = round(float(np.mean((actual >= p10) & (actual <= p90))), 4)
    metrics["n_train"] = int(len(X_tr))
    metrics["n_test"] = int(len(X_te))

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, pipe in models.items():
            joblib.dump(pipe, MODELS_DIR / f"cost_per_mw_{name}.joblib")
        (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
