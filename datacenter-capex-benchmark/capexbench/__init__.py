"""CapexBench: parametric benchmarking for data center capex.

A structured cost-per-MW / cost-per-SF benchmark set with:
  - a calibrated parametric cost engine (config.py)
  - a synthetic-but-grounded benchmark dataset generator (dataset.py)
  - ML quantile models for P10/P50/P90 estimates (train.py / predict.py)
  - a comparables benchmark query engine (benchmark.py)
  - an AI natural-language query layer over both (ai_query.py)
"""

__version__ = "0.1.0"
