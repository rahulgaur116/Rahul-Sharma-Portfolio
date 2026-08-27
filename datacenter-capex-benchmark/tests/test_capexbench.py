"""End-to-end sanity tests for dataset, benchmark engine, models, and API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from capexbench import benchmark as bench
from capexbench.dataset import generate, load
from capexbench.predict import estimate_capex


@pytest.fixture(scope="session")
def df():
    return load()


def test_generator_is_deterministic():
    a = generate(n_projects=50, seed=3)
    b = generate(n_projects=50, seed=3)
    assert a.equals(b)


def test_dataset_ranges(df):
    assert len(df) >= 1000
    # Plausible industry ranges for construction cost per MW ($M)
    assert df["cost_per_mw_musd"].between(2, 40).all()
    assert df["cost_per_sf_usd"].between(200, 12000).all()
    # Breakdown shares sum to ~1
    share_cols = [c for c in df.columns if c.startswith("share_")]
    assert np.allclose(df[share_cols].sum(axis=1), 1.0, atol=0.02)
    # Breakdown $ sums to construction capex
    capex_cols = [f"capex_{s}_musd" for s in bench.BREAKDOWN_SYSTEMS]
    assert np.allclose(df[capex_cols].sum(axis=1), df["construction_capex_musd"], rtol=0.02)


def test_benchmark_query(df):
    r = bench.query_benchmarks(df, {"facility_type": "hyperscale_ai", "region": "North America"})
    assert r["n_comparables"] >= 8
    q = r["cost_per_mw_musd"]
    assert q["p10"] < q["p50"] < q["p90"]


def test_benchmark_relaxation(df):
    # Impossibly narrow filters must relax rather than return nothing
    r = bench.query_benchmarks(df, {
        "facility_type": "edge", "market": "Reykjavik", "cooling": "immersion",
        "redundancy": "2N+1", "delivery_year_min": 2026,
    })
    assert r["n_comparables"] >= 8
    assert r["filters_relaxed"]


def test_benchmark_rejects_unknown_filter(df):
    with pytest.raises(ValueError):
        bench.query_benchmarks(df, {"vibe": "immaculate"})


def test_estimate_quantiles_ordered():
    r = estimate_capex({"facility_type": "hyperscale_ai", "market": "Phoenix",
                        "it_load_mw": 60, "kw_per_rack": 100,
                        "cooling": "liquid_dtc", "delivery_year": 2026})
    q = r["cost_per_mw_musd"]
    assert q["p10"] <= q["p50"] <= q["p90"]
    assert 5 < q["p50"] < 25
    assert abs(sum(r["breakdown_shares"].values()) - 1.0) < 0.02


def test_estimate_direction_liquid_vs_air():
    base = {"facility_type": "hyperscale_ai", "market": "Dallas-Fort Worth",
            "it_load_mw": 60, "kw_per_rack": 80, "delivery_year": 2026}
    air = estimate_capex({**base, "cooling": "air"})
    liq = estimate_capex({**base, "cooling": "liquid_dtc"})
    assert liq["cost_per_mw_musd"]["p50"] > air["cost_per_mw_musd"]["p50"]


def test_estimate_rejects_bad_market():
    with pytest.raises(ValueError):
        estimate_capex({"market": "Atlantis"})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    from api.main import app
    return TestClient(app)


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_dimensions(client):
    r = client.get("/api/dimensions")
    assert "hyperscale_ai" in r.json()["categorical"]["facility_type"]


def test_api_benchmark(client):
    r = client.post("/api/benchmark", json={"filters": {"facility_type": "wholesale_colo"}})
    assert r.status_code == 200
    assert r.json()["n_comparables"] > 0


def test_api_benchmark_bad_filter(client):
    r = client.post("/api/benchmark", json={"filters": {"nope": 1}})
    assert r.status_code == 422


def test_api_estimate(client):
    r = client.post("/api/estimate", json={"facility_type": "retail_colo", "market": "London",
                                           "it_load_mw": 8, "kw_per_rack": 12})
    assert r.status_code == 200
    assert r.json()["cost_per_mw_musd"]["p50"] > 0


def test_api_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CapexBench" in r.text


def test_api_ask_without_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = client.post("/api/ask", json={"question": "How much per MW?"})
    assert r.status_code == 503
