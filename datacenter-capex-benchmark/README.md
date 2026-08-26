# CapexBench — Parametric Benchmarking for Data Center Capex

A structured **cost-per-MW / cost-per-SF benchmark set with an AI query layer**,
aimed at colocation operators and AI-infrastructure scaleups (the buyers least
served by traditional cost consultancies and the fastest growing).

Ask it things like:

> *"What should a 60 MW liquid-cooled AI build in Phoenix delivering in 2026
> cost per MW — and how does that compare to Dallas?"*

and it answers from real distribution statistics over comparable projects plus
an ML parametric estimate, with the underlying tool calls exposed for audit.

## What's inside

| Layer | Where | What it does |
|---|---|---|
| Parametric cost model | `capexbench/config.py` | Calibrated multipliers: facility type, market cost index (38 markets), cooling, redundancy, density, build type, scale elasticity, year escalation |
| Benchmark dataset | `capexbench/dataset.py` → `data/benchmarks.csv` | 1,200 structured project records: $/MW, $/SF, $/kW + 5-system cost breakdown (shell/core, electrical, mechanical, fit-out, GC & soft) |
| Benchmark query engine | `capexbench/benchmark.py` | Peer filtering with graceful constraint relaxation; P10/P25/P50/P75/P90 distributions |
| ML estimator | `capexbench/train.py`, `predict.py` | Gradient-boosted **quantile models** (P10/P50/P90) on log cost-per-MW. Holdout: **7.9% MAPE** at P50; P10–P90 empirical coverage 71% (nominal 80% — quantile GBMs undercover; documented, not hidden) |
| AI query layer | `capexbench/ai_query.py` | Claude (tool use, `claude-opus-5`) translates natural language into benchmark queries + estimates and writes a grounded analyst answer |
| API | `api/main.py` | FastAPI: `/api/estimate`, `/api/benchmark`, `/api/dimensions`, `/api/ask` |
| Web app | `app/index.html` | Spec form → P10/P50/P90 tiles, cost-breakdown chart, peer benchmark table, AI chat |

## Quickstart

```bash
cd datacenter-capex-benchmark
pip install -r requirements.txt

python -m capexbench.dataset   # generate data/benchmarks.csv (deterministic, seeded)
python -m capexbench.train     # train quantile models -> models/

uvicorn api.main:app --port 8000
# open http://localhost:8000
```

The AI query layer needs Anthropic credentials:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m capexbench.ai_query "Compare 2N vs N+1 cost per MW for wholesale colo in Europe"
```

Everything else (estimates, benchmarks, the app) works without a key.

Tests: `pytest` (15 tests covering dataset invariants, query relaxation,
quantile ordering, directional effects, and the API surface).

## Example: 100 MW AI build, Abilene TX, 2026, 130 kW/rack, direct-to-chip liquid

```json
{
  "cost_per_mw_musd": {"p10": 10.66, "p50": 12.10, "p90": 13.63},
  "total_capex_musd": {"p10": 1066, "p50": 1210, "p90": 1363},
  "breakdown_p50_musd": {
    "shell_core": 168, "electrical": 490, "mechanical": 289,
    "fitout_network": 96, "gc_soft": 168
  }
}
```

## Data provenance & methodology

The seed dataset is **synthetic but calibrated**: generated from the parametric
model in `config.py`, whose baseline (~$9M per MW for a 2024 Tier III US
wholesale build) and multipliers are anchored to public industry sources
(Turner & Townsend Data Centre Cost Index, CBRE/C&W/JLL market reports, Uptime
Institute). It exists so the schema, models, API, and AI layer are exercised
end-to-end. The commercial product swaps in contributed project actuals under
the same schema — the query layer, estimator, and app don't change.

Conventions: costs are **construction cost excluding land**, USD, per **MW of
IT load**. Land is tracked separately per record.

## Productization path

1. **Data**: replace synthetic records with contributed actuals (anonymized),
   with a data-contribution agreement per operator — the Turner & Townsend /
   RSMeans model, but for this asset class.
2. **Models**: retrain per release; add conformal calibration for honest bands;
   add a schedule (months-to-RFS) model alongside cost.
3. **App**: auth + per-org data partitions, saved benchmarks, PDF export,
   scenario comparison; move sessions out of process memory.
4. **AI layer**: add citations to individual (anonymized) comparables,
   escalation forecasting, and "what moved my estimate" attributions.
