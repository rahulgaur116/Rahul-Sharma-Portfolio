"""Parametric cost-model assumptions for data center capex.

All multipliers are applied to a calibrated baseline:
a ~24 MW IT-load, Tier III (N+1), air-cooled, greenfield wholesale colo
delivered in a mid-cost US market (cost index 1.00) in 2024, at roughly
$9.0M per MW of IT load, all-in construction cost excluding land.

Anchors are drawn from public industry sources (Turner & Townsend Data
Centre Cost Index, CBRE / Cushman & Wakefield market reports, JLL, Uptime
Institute) as of 2024-2025:
  - US wholesale/hyperscale builds: ~$8-12 per watt all-in construction
  - Electrical systems: ~40-48% of hard cost on high-density builds
  - Mechanical/cooling: ~18-25% of hard cost, higher for liquid cooling
  - Cost escalation 2021-2025 well above CPI (equipment lead times,
    switchgear/generator shortages, skilled-labor scarcity)

The seed dataset produced from these assumptions is synthetic. It exists so
that the schema, models, and query layer are fully exercised end-to-end;
in production the same schema is populated with contributed project actuals.
"""

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

BASELINE_COST_PER_MW_MUSD = 9.0     # $M per MW IT load, 2024, index-1.00 market
BASELINE_YEAR = 2024
BASELINE_MW = 24.0
SCALE_ELASTICITY = -0.055           # cost/MW ~ (MW / 24) ** elasticity

# ---------------------------------------------------------------------------
# Categorical dimensions and multipliers (applied to cost per MW)
# ---------------------------------------------------------------------------

FACILITY_TYPES = {
    # AI/HPC hyperscale: dense electrical + liquid plant, but scale efficiencies
    "hyperscale_ai": 1.10,
    # Traditional cloud hyperscale
    "hyperscale_cloud": 0.97,
    "wholesale_colo": 1.00,
    # Retail colo: more fit-out, meet-me rooms, office per MW
    "retail_colo": 1.12,
    # Edge: small footprints lose all scale economies
    "edge": 1.28,
}

REDUNDANCY = {
    "N": 0.87,
    "N+1": 1.00,
    "N+2": 1.08,
    "2N": 1.18,
    "2N+1": 1.27,
}

TIER_FOR_REDUNDANCY = {
    "N": "II",
    "N+1": "III",
    "N+2": "III",
    "2N": "IV",
    "2N+1": "IV",
}

COOLING = {
    "air": 1.00,
    # Direct-to-chip liquid: CDUs, manifolds, hybrid plant; more $/MW,
    # far fewer SF/MW
    "liquid_dtc": 1.08,
    "hybrid_air_liquid": 1.04,
    "immersion": 1.13,
    "free_air_evap": 0.96,
}

BUILD_TYPES = {
    "greenfield": 1.00,
    # Existing building, new MEP
    "retrofit": 0.86,
    # Powered shell already exists; fit-out only
    "shell_fitout": 0.68,
}

# Regional construction cost indices (labor + materials + logistics),
# US mid-cost market = 1.00
MARKETS = {
    # market: (region, country, cost_index)
    "Northern Virginia": ("North America", "US", 1.02),
    "Dallas-Fort Worth": ("North America", "US", 0.95),
    "Phoenix": ("North America", "US", 0.97),
    "Chicago": ("North America", "US", 1.03),
    "Silicon Valley": ("North America", "US", 1.18),
    "Atlanta": ("North America", "US", 0.94),
    "Columbus": ("North America", "US", 0.93),
    "Salt Lake City": ("North America", "US", 0.94),
    "Reno": ("North America", "US", 0.96),
    "Abilene": ("North America", "US", 0.92),
    "Toronto": ("North America", "Canada", 1.04),
    "Querétaro": ("Latin America", "Mexico", 0.78),
    "São Paulo": ("Latin America", "Brazil", 0.82),
    "Santiago": ("Latin America", "Chile", 0.84),
    "London": ("Europe", "UK", 1.16),
    "Frankfurt": ("Europe", "Germany", 1.13),
    "Dublin": ("Europe", "Ireland", 1.09),
    "Amsterdam": ("Europe", "Netherlands", 1.11),
    "Paris": ("Europe", "France", 1.10),
    "Madrid": ("Europe", "Spain", 0.95),
    "Milan": ("Europe", "Italy", 0.99),
    "Oslo": ("Nordics", "Norway", 1.03),
    "Stockholm": ("Nordics", "Sweden", 1.02),
    "Helsinki": ("Nordics", "Finland", 1.00),
    "Reykjavik": ("Nordics", "Iceland", 1.05),
    "Singapore": ("APAC", "Singapore", 1.21),
    "Tokyo": ("APAC", "Japan", 1.26),
    "Osaka": ("APAC", "Japan", 1.22),
    "Seoul": ("APAC", "South Korea", 1.12),
    "Sydney": ("APAC", "Australia", 1.09),
    "Melbourne": ("APAC", "Australia", 1.06),
    "Mumbai": ("APAC", "India", 0.64),
    "Chennai": ("APAC", "India", 0.62),
    "Johor": ("APAC", "Malaysia", 0.72),
    "Batam": ("APAC", "Indonesia", 0.74),
    "Riyadh": ("MEA", "Saudi Arabia", 0.98),
    "Dubai": ("MEA", "UAE", 0.96),
    "Johannesburg": ("MEA", "South Africa", 0.80),
}

# Escalation index by delivery year (2024 = 1.00). Reflects the 2021-2023
# supply-chain shock (switchgear, generators, chillers) and AI-era demand.
ESCALATION = {
    2019: 0.80,
    2020: 0.81,
    2021: 0.85,
    2022: 0.93,
    2023: 0.97,
    2024: 1.00,
    2025: 1.05,
    2026: 1.10,
}

# Fast-track schedule premium range (uniform draw within)
SCHEDULE_PREMIUM_RANGE = (1.00, 1.12)

# ---------------------------------------------------------------------------
# Density -> area model
# ---------------------------------------------------------------------------
# Total building SF per MW of IT load. White space shrinks ~1/density;
# electrical/mechanical support space has a floor that does not shrink.
#   total_sf_per_mw ≈ WHITE_SPACE_COEF / kw_per_rack + SUPPORT_SF_FLOOR
# At 10 kW/rack -> ~8,000 SF/MW; at 40 -> ~3,900; at 130 (NVL72-class) -> ~2,950.

WHITE_SPACE_COEF = 55_000.0
SUPPORT_SF_FLOOR = 2_500.0

# Typical rack density (kW/rack) ranges by facility type
DENSITY_RANGES = {
    "hyperscale_ai": (40, 140),
    "hyperscale_cloud": (12, 30),
    "wholesale_colo": (8, 40),
    "retail_colo": (5, 20),
    "edge": (5, 15),
}

# Mild cost premium for extreme density (busway/UPS/distribution upsizing):
# +4% per doubling of density above 20 kW/rack.
DENSITY_COST_REF_KW = 20.0
DENSITY_COST_PER_DOUBLING = 0.04

# ---------------------------------------------------------------------------
# Cost breakdown shares (of construction cost, excluding land)
# ---------------------------------------------------------------------------
# Base shares for the baseline air-cooled facility. Adjusted by cooling and
# build type at generation time, then renormalized.

BASE_SHARES = {
    "shell_core": 0.16,       # site, structure, envelope
    "electrical": 0.42,       # utility, gensets, UPS, switchgear, distribution
    "mechanical": 0.20,       # cooling plant, CRAH/CDU, piping, controls
    "fitout_network": 0.08,   # racks/containment support, security, MMR, BMS
    "gc_soft": 0.14,          # GC fees, design, PM, commissioning, contingency
}

# Share adjustments (additive to the base share before renormalizing)
COOLING_SHARE_SHIFT = {
    "air": {},
    "liquid_dtc": {"mechanical": +0.04, "shell_core": -0.02, "electrical": -0.01},
    "hybrid_air_liquid": {"mechanical": +0.02, "shell_core": -0.01},
    "immersion": {"mechanical": +0.06, "shell_core": -0.03, "electrical": -0.01},
    "free_air_evap": {"mechanical": -0.03, "shell_core": +0.01},
}

BUILD_SHARE_SHIFT = {
    "greenfield": {},
    "retrofit": {"shell_core": -0.08, "electrical": +0.04, "mechanical": +0.02},
    "shell_fitout": {"shell_core": -0.13, "electrical": +0.07, "mechanical": +0.04},
}

# Land cost as $M per acre by cost-index band, and acres per MW
LAND_MUSD_PER_ACRE = {"low": 0.15, "mid": 0.55, "high": 1.8}
ACRES_PER_MW = 0.9

OPERATOR_TYPES = ["colo_operator", "ai_cloud", "hyperscaler", "enterprise", "developer"]

# Log-normal noise on total cost (sigma of ln)
PROJECT_NOISE_SIGMA = 0.075
