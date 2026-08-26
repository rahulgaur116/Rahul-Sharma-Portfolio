"""Generate the structured capex benchmark dataset from the parametric model.

Each row is one delivered project with full parametric metadata, a total
capex, cost-per-MW / cost-per-SF metrics, and a system-level breakdown.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "benchmarks.csv"


def _cost_per_mw_musd(
    facility_type: str,
    market: str,
    delivery_year: int,
    it_load_mw: float,
    kw_per_rack: float,
    cooling: str,
    redundancy: str,
    build_type: str,
    schedule_premium: float = 1.0,
) -> float:
    """Deterministic parametric cost per MW (no project noise)."""
    _, _, cost_index = C.MARKETS[market]
    scale = (it_load_mw / C.BASELINE_MW) ** C.SCALE_ELASTICITY
    density_premium = 1.0 + C.DENSITY_COST_PER_DOUBLING * max(
        0.0, math.log2(kw_per_rack / C.DENSITY_COST_REF_KW)
    )
    return (
        C.BASELINE_COST_PER_MW_MUSD
        * C.FACILITY_TYPES[facility_type]
        * C.REDUNDANCY[redundancy]
        * C.COOLING[cooling]
        * C.BUILD_TYPES[build_type]
        * cost_index
        * C.ESCALATION[delivery_year]
        * scale
        * density_premium
        * schedule_premium
    )


def sf_per_mw(kw_per_rack: float) -> float:
    return C.WHITE_SPACE_COEF / kw_per_rack + C.SUPPORT_SF_FLOOR


def _breakdown_shares(cooling: str, build_type: str, rng: np.random.Generator) -> dict:
    shares = dict(C.BASE_SHARES)
    for k, v in C.COOLING_SHARE_SHIFT[cooling].items():
        shares[k] += v
    for k, v in C.BUILD_SHARE_SHIFT[build_type].items():
        shares[k] += v
    # Project-level jitter, then renormalize
    jitter = rng.normal(1.0, 0.05, size=len(shares))
    vals = np.clip(np.array(list(shares.values())) * jitter, 0.01, None)
    vals = vals / vals.sum()
    return dict(zip(shares.keys(), vals))


def _sample_project(rng: np.random.Generator, project_id: int) -> dict:
    facility_type = rng.choice(
        list(C.FACILITY_TYPES), p=[0.24, 0.14, 0.28, 0.20, 0.14]
    )
    market = rng.choice(list(C.MARKETS))
    region, country, cost_index = C.MARKETS[market]
    delivery_year = int(rng.choice(list(C.ESCALATION), p=[0.04, 0.05, 0.08, 0.11, 0.14, 0.19, 0.21, 0.18]))

    # Size distribution by facility type (MW IT load)
    size_ranges = {
        "hyperscale_ai": (20, 300),
        "hyperscale_cloud": (24, 200),
        "wholesale_colo": (6, 96),
        "retail_colo": (1, 24),
        "edge": (0.3, 5),
    }
    lo, hi = size_ranges[facility_type]
    it_load_mw = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), 1))

    dlo, dhi = C.DENSITY_RANGES[facility_type]
    kw_per_rack = float(np.round(rng.uniform(dlo, dhi), 0))

    if facility_type == "hyperscale_ai":
        cooling = rng.choice(
            ["liquid_dtc", "hybrid_air_liquid", "immersion", "air"], p=[0.5, 0.3, 0.08, 0.12]
        )
        redundancy = rng.choice(["N", "N+1", "N+2", "2N"], p=[0.18, 0.5, 0.2, 0.12])
    elif facility_type == "retail_colo":
        cooling = rng.choice(["air", "free_air_evap", "hybrid_air_liquid"], p=[0.72, 0.14, 0.14])
        redundancy = rng.choice(["N+1", "N+2", "2N", "2N+1"], p=[0.42, 0.18, 0.3, 0.1])
    else:
        cooling = rng.choice(
            ["air", "free_air_evap", "hybrid_air_liquid", "liquid_dtc"], p=[0.5, 0.16, 0.2, 0.14]
        )
        redundancy = rng.choice(["N", "N+1", "N+2", "2N", "2N+1"], p=[0.08, 0.46, 0.16, 0.24, 0.06])

    build_type = rng.choice(list(C.BUILD_TYPES), p=[0.72, 0.15, 0.13])
    schedule_premium = float(rng.uniform(*C.SCHEDULE_PREMIUM_RANGE))
    operator_type = rng.choice(C.OPERATOR_TYPES, p=[0.34, 0.2, 0.2, 0.1, 0.16])

    cost_mw = _cost_per_mw_musd(
        facility_type, market, delivery_year, it_load_mw, kw_per_rack,
        cooling, redundancy, build_type, schedule_premium,
    )
    # Idiosyncratic project noise
    cost_mw *= float(np.exp(rng.normal(0.0, C.PROJECT_NOISE_SIGMA)))

    construction_capex_musd = cost_mw * it_load_mw
    total_sf = sf_per_mw(kw_per_rack) * it_load_mw * float(rng.normal(1.0, 0.06))
    total_sf = max(total_sf, 1500.0)

    # Land (greenfield only)
    if build_type == "greenfield":
        band = "high" if cost_index >= 1.1 else ("low" if cost_index < 0.9 else "mid")
        acres = C.ACRES_PER_MW * it_load_mw * float(rng.uniform(0.7, 1.5))
        land_musd = C.LAND_MUSD_PER_ACRE[band] * acres
    else:
        land_musd = 0.0

    shares = _breakdown_shares(cooling, build_type, rng)
    row = {
        "project_id": f"P{project_id:05d}",
        "operator_type": operator_type,
        "facility_type": facility_type,
        "region": region,
        "country": country,
        "market": market,
        "delivery_year": delivery_year,
        "tier": C.TIER_FOR_REDUNDANCY[redundancy],
        "redundancy": redundancy,
        "cooling": cooling,
        "build_type": build_type,
        "it_load_mw": it_load_mw,
        "kw_per_rack": kw_per_rack,
        "total_sf": round(total_sf, 0),
        "schedule_premium": round(schedule_premium, 3),
        "land_capex_musd": round(land_musd, 2),
        "construction_capex_musd": round(construction_capex_musd, 2),
        "total_capex_musd": round(construction_capex_musd + land_musd, 2),
        "cost_per_mw_musd": round(construction_capex_musd / it_load_mw, 3),
        "cost_per_sf_usd": round(construction_capex_musd * 1e6 / total_sf, 0),
        "cost_per_kw_usd": round(construction_capex_musd * 1e6 / (it_load_mw * 1000), 0),
    }
    for system, share in shares.items():
        row[f"capex_{system}_musd"] = round(construction_capex_musd * share, 2)
        row[f"share_{system}"] = round(float(share), 4)
    return row


def generate(n_projects: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_sample_project(rng, i + 1) for i in range(n_projects)]
    return pd.DataFrame(rows)


def load(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m capexbench.dataset` to generate it"
        )
    return pd.read_csv(path)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Wrote {len(df)} projects to {DATASET_PATH}")
    print(df[["cost_per_mw_musd", "cost_per_sf_usd"]].describe().round(1))


if __name__ == "__main__":
    main()
