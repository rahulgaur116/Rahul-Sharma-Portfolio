"""Comparables benchmark query engine.

Filters the benchmark set down to peer projects and returns distribution
statistics (P10/P25/P50/P75/P90) for cost per MW, cost per SF, and the
system-level cost breakdown. If a filter combination yields too few
comparables, constraints are relaxed in a defined order and the response
reports which filters were dropped.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BREAKDOWN_SYSTEMS = ["shell_core", "electrical", "mechanical", "fitout_network", "gc_soft"]

# Order in which filters are dropped when there are too few comparables
RELAXATION_ORDER = [
    "operator_type", "build_type", "redundancy", "cooling",
    "kw_per_rack_min", "kw_per_rack_max", "delivery_year_min", "delivery_year_max",
    "it_load_mw_min", "it_load_mw_max", "market", "country", "region",
]

CATEGORICAL_FILTERS = [
    "facility_type", "region", "country", "market", "cooling",
    "redundancy", "tier", "build_type", "operator_type",
]
RANGE_FILTERS = {
    "it_load_mw_min": ("it_load_mw", "ge"),
    "it_load_mw_max": ("it_load_mw", "le"),
    "kw_per_rack_min": ("kw_per_rack", "ge"),
    "kw_per_rack_max": ("kw_per_rack", "le"),
    "delivery_year_min": ("delivery_year", "ge"),
    "delivery_year_max": ("delivery_year", "le"),
}

MIN_COMPARABLES = 8


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df
    for key in CATEGORICAL_FILTERS:
        if key in filters and filters[key] is not None:
            val = filters[key]
            vals = val if isinstance(val, list) else [val]
            out = out[out[key].isin(vals)]
    for key, (col, op) in RANGE_FILTERS.items():
        if key in filters and filters[key] is not None:
            out = out[out[col] >= filters[key]] if op == "ge" else out[out[col] <= filters[key]]
    return out


def _quantiles(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": round(float(q.loc[0.10]), 3),
        "p25": round(float(q.loc[0.25]), 3),
        "p50": round(float(q.loc[0.50]), 3),
        "p75": round(float(q.loc[0.75]), 3),
        "p90": round(float(q.loc[0.90]), 3),
        "mean": round(float(s.mean()), 3),
    }


def query_benchmarks(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> dict:
    """Return benchmark stats for projects matching `filters`.

    Relaxes filters (least-important first) until at least MIN_COMPARABLES
    peers match; the response lists any filters that were dropped.
    """
    filters = dict(filters or {})
    unknown = set(filters) - set(CATEGORICAL_FILTERS) - set(RANGE_FILTERS)
    if unknown:
        raise ValueError(f"Unknown filter(s): {sorted(unknown)}")

    relaxed: list[str] = []
    peers = _apply_filters(df, filters)
    for key in RELAXATION_ORDER:
        if len(peers) >= MIN_COMPARABLES:
            break
        if key in filters:
            filters.pop(key)
            relaxed.append(key)
            peers = _apply_filters(df, filters)

    if peers.empty:
        return {"n_comparables": 0, "filters_applied": filters, "filters_relaxed": relaxed}

    result = {
        "n_comparables": int(len(peers)),
        "filters_applied": filters,
        "filters_relaxed": relaxed,
        "cost_per_mw_musd": _quantiles(peers["cost_per_mw_musd"]),
        "cost_per_sf_usd": _quantiles(peers["cost_per_sf_usd"]),
        "cost_per_kw_usd": _quantiles(peers["cost_per_kw_usd"]),
        "it_load_mw": _quantiles(peers["it_load_mw"]),
        "breakdown_median_share": {
            s: round(float(peers[f"share_{s}"].median()), 4) for s in BREAKDOWN_SYSTEMS
        },
        "breakdown_median_musd_per_mw": {
            s: round(float((peers[f"capex_{s}_musd"] / peers["it_load_mw"]).median()), 3)
            for s in BREAKDOWN_SYSTEMS
        },
    }
    return result


def dimensions(df: pd.DataFrame) -> dict:
    """Enumerate queryable dimensions and their values/ranges."""
    return {
        "categorical": {k: sorted(df[k].unique().tolist()) for k in CATEGORICAL_FILTERS},
        "ranges": {
            "it_load_mw": [float(df["it_load_mw"].min()), float(df["it_load_mw"].max())],
            "kw_per_rack": [float(df["kw_per_rack"].min()), float(df["kw_per_rack"].max())],
            "delivery_year": [int(df["delivery_year"].min()), int(df["delivery_year"].max())],
        },
        "metrics": ["cost_per_mw_musd", "cost_per_sf_usd", "cost_per_kw_usd",
                    "total_capex_musd", "breakdown by system"],
        "n_projects": int(len(df)),
    }
