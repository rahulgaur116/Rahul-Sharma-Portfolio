"""AI query layer: natural-language questions over the benchmark set.

Uses Claude with tool use. Claude translates a question like
"what should a 60MW liquid-cooled AI build in Phoenix cost next year,
and how does that compare to Dallas?" into calls against the benchmark
query engine and the parametric estimator, then writes an analyst-grade
answer grounded in the returned numbers.

Requires ANTHROPIC_API_KEY (or another credential source the Anthropic SDK
resolves). The rest of CapexBench works without it.
"""

from __future__ import annotations

import json
from functools import lru_cache

import anthropic
from anthropic import beta_tool

from .benchmark import dimensions as _dimensions
from .benchmark import query_benchmarks as _query_benchmarks
from .dataset import load
from .predict import estimate_capex as _estimate_capex

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are CapexBench, a benchmarking analyst for data center construction costs, \
serving colocation operators and AI-infrastructure companies.

You answer questions about data center capital expenditure using two tools:
- query_benchmarks: distribution statistics (P10-P90) over comparable delivered \
projects, filterable by facility type, geography, cooling, redundancy, size, \
density, and delivery year.
- estimate_capex: a parametric ML estimate (P10/P50/P90) for a specific project \
spec, with a system-level cost breakdown.

Call list_dimensions first if you are unsure which filter values exist. \
Ground every number you state in tool results - never invent benchmark figures. \
Costs are construction cost excluding land unless stated otherwise, in USD. \
Cost per MW refers to MW of IT load. When a user's question is comparative, \
run the tools for each side of the comparison. State the number of comparables \
behind a benchmark and flag when filters had to be relaxed. Be concise and \
quantitative: lead with the answer, then the supporting numbers.

The current benchmark set is a modeled/synthetic calibration set - say so if \
asked about data provenance.\
"""


@beta_tool
def list_dimensions() -> str:
    """List the queryable dimensions of the benchmark set: categorical filter values (facility types, markets, cooling types, redundancy levels, build types), numeric ranges, and available metrics."""
    return json.dumps(_dimensions(load()))


@beta_tool
def query_benchmarks(filters_json: str) -> str:
    """Query benchmark statistics (P10/P25/P50/P75/P90 of cost per MW, cost per SF, cost per kW, and system-level breakdowns) over comparable projects.

    Args:
        filters_json: JSON object of filters. Categorical (string or list of strings): facility_type, region, country, market, cooling, redundancy, tier, build_type, operator_type. Numeric: it_load_mw_min, it_load_mw_max, kw_per_rack_min, kw_per_rack_max, delivery_year_min, delivery_year_max. Example: {"facility_type": "hyperscale_ai", "region": "North America", "delivery_year_min": 2024}
    """
    filters = json.loads(filters_json) if filters_json.strip() else {}
    try:
        return json.dumps(query_benchmarks_impl(filters))
    except ValueError as e:
        return json.dumps({"error": str(e)})


def query_benchmarks_impl(filters: dict) -> dict:
    return _query_benchmarks(load(), filters)


@beta_tool
def estimate_capex(spec_json: str) -> str:
    """Parametric ML estimate of construction capex for a specific project spec. Returns P10/P50/P90 cost per MW, total capex, cost per SF, and a system-level breakdown (shell/core, electrical, mechanical, fit-out, GC & soft costs).

    Args:
        spec_json: JSON object with any of: facility_type (hyperscale_ai | hyperscale_cloud | wholesale_colo | retail_colo | edge), market (e.g. "Northern Virginia", "Phoenix", "London", "Singapore"), delivery_year (2019-2026), it_load_mw, kw_per_rack, cooling (air | liquid_dtc | hybrid_air_liquid | immersion | free_air_evap), redundancy (N | N+1 | N+2 | 2N | 2N+1), build_type (greenfield | retrofit | shell_fitout). Missing fields use sensible defaults. Example: {"facility_type": "hyperscale_ai", "market": "Phoenix", "it_load_mw": 60, "kw_per_rack": 100, "cooling": "liquid_dtc", "delivery_year": 2026}
    """
    spec = json.loads(spec_json) if spec_json.strip() else {}
    try:
        return json.dumps(_estimate_capex(spec))
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


TOOLS = [list_dimensions, query_benchmarks, estimate_capex]


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def ask(question: str, history: list[dict] | None = None) -> dict:
    """Answer a natural-language capex question. Returns the answer text plus
    the tool calls made (for auditability) and updated conversation history."""
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    tool_calls: list[dict] = []
    runner = _client().beta.messages.tool_runner(
        model=MODEL,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        tools=TOOLS,
        messages=messages,
    )

    last = None
    for message in runner:
        last = message
        for block in message.content:
            if block.type == "tool_use":
                tool_calls.append({"tool": block.name, "input": block.input})
        # Mirror history so multi-turn conversations can continue
        messages.append({"role": "assistant", "content": message.content})
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            messages.append(tool_response)

    answer = ""
    if last is not None:
        answer = "".join(b.text for b in last.content if b.type == "text")
        if last.stop_reason == "refusal":
            answer = answer or "The model declined to answer this question."

    return {"answer": answer, "tool_calls": tool_calls, "history": messages}


def main() -> None:
    import sys
    question = " ".join(sys.argv[1:]) or (
        "What does a 60 MW liquid-cooled AI data center in Phoenix delivering "
        "in 2026 cost per MW, and how does that compare to Dallas?"
    )
    result = ask(question)
    print(result["answer"])
    print("\n--- tool calls ---")
    for call in result["tool_calls"]:
        print(json.dumps(call))


if __name__ == "__main__":
    main()
