"""Parametric capex estimation from trained quantile models.

`estimate_capex` takes a project spec, returns P10/P50/P90 cost-per-MW,
total capex, cost-per-SF (via the density->area model), and an indicative
system-level breakdown drawn from comparable projects.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from . import config as C
from .benchmark import query_benchmarks
from .dataset import load, sf_per_mw
from .train import FEATURES, MODELS_DIR, QUANTILES


@lru_cache(maxsize=1)
def _models() -> dict:
    models = {}
    for name in QUANTILES:
        path = MODELS_DIR / f"cost_per_mw_{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python -m capexbench.train` first"
            )
        models[name] = joblib.load(path)
    return models


def _validate(spec: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "facility_type": "wholesale_colo",
        "market": "Dallas-Fort Worth",
        "delivery_year": 2026,
        "it_load_mw": 24.0,
        "kw_per_rack": 20.0,
        "cooling": "air",
        "redundancy": "N+1",
        "build_type": "greenfield",
    }
    spec = {**defaults, **{k: v for k, v in spec.items() if v is not None}}

    for key, allowed in [
        ("facility_type", C.FACILITY_TYPES), ("market", C.MARKETS),
        ("cooling", C.COOLING), ("redundancy", C.REDUNDANCY),
        ("build_type", C.BUILD_TYPES),
    ]:
        if spec[key] not in allowed:
            raise ValueError(f"{key}={spec[key]!r} not recognized; options: {sorted(allowed)}")
    spec["delivery_year"] = int(min(max(spec["delivery_year"], min(C.ESCALATION)), max(C.ESCALATION)))
    spec["it_load_mw"] = float(spec["it_load_mw"])
    spec["kw_per_rack"] = float(spec["kw_per_rack"])
    if not (0.1 <= spec["it_load_mw"] <= 1000):
        raise ValueError("it_load_mw must be between 0.1 and 1000")
    if not (2 <= spec["kw_per_rack"] <= 250):
        raise ValueError("kw_per_rack must be between 2 and 250")
    return spec


def estimate_capex(spec: dict[str, Any]) -> dict:
    """Estimate capex for a project spec. Returns quantile bands + breakdown."""
    spec = _validate(spec)
    region, country, _ = C.MARKETS[spec["market"]]
    row = pd.DataFrame([{
        "facility_type": spec["facility_type"],
        "region": region,
        "country": country,
        "market": spec["market"],
        "cooling": spec["cooling"],
        "redundancy": spec["redundancy"],
        "build_type": spec["build_type"],
        "delivery_year": spec["delivery_year"],
        "it_load_mw": spec["it_load_mw"],
        "kw_per_rack": spec["kw_per_rack"],
    }])[FEATURES]

    import numpy as np
    models = _models()
    per_mw = {name: float(np.exp(m.predict(row)[0])) for name, m in models.items()}
    # Quantile crossing guard
    p10, p50, p90 = sorted([per_mw["p10"], per_mw["p50"], per_mw["p90"]])

    mw = spec["it_load_mw"]
    total_sf = sf_per_mw(spec["kw_per_rack"]) * mw

    # Indicative breakdown from comparable projects (median shares)
    peers = query_benchmarks(load(), {
        "facility_type": spec["facility_type"],
        "cooling": spec["cooling"],
        "build_type": spec["build_type"],
    })
    shares = peers.get("breakdown_median_share", {})

    return {
        "spec": spec,
        "cost_per_mw_musd": {"p10": round(p10, 2), "p50": round(p50, 2), "p90": round(p90, 2)},
        "total_capex_musd": {
            "p10": round(p10 * mw, 1), "p50": round(p50 * mw, 1), "p90": round(p90 * mw, 1),
        },
        "cost_per_kw_usd": {"p50": round(p50 * 1e6 / 1000, 0)},
        "estimated_total_sf": round(total_sf, 0),
        "cost_per_sf_usd": {
            "p10": round(p10 * mw * 1e6 / total_sf, 0),
            "p50": round(p50 * mw * 1e6 / total_sf, 0),
            "p90": round(p90 * mw * 1e6 / total_sf, 0),
        },
        "breakdown_p50_musd": {
            system: round(share * p50 * mw, 1) for system, share in shares.items()
        },
        "breakdown_shares": shares,
        "n_comparables_for_breakdown": peers.get("n_comparables", 0),
        "notes": "Construction cost excluding land. P10-P90 band reflects "
                 "market, escalation, and project execution variance.",
    }
