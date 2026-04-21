"""Single source of truth for LLM per-token pricing (USD / 1M tokens).

Every caller that computes LLM costs MUST import MODEL_PRICING from this
module. Duplicating the table elsewhere causes silent drift: e.g. prior
to this consolidation, `classifier.py` had gpt-5-mini at $1.00 output
while `providers/openai.py` had $2.00 — every gpt-5-mini cost row
computed through the classifier was undercounted by 50% for months.

Prices are Azure Standard Global tier (verified against the Azure OpenAI
pricing page, April 2026). Where Azure publishes only "cached input",
that value is stored in the `cached` field; otherwise `cached` equals
`input` (callers fall back to input pricing if cache hits aren't
discounted for that model).

Entries for `gpt-5.4` / `gpt-5.4-mini` track the user's private Azure
Foundry deployment names — they have no public Azure SKU and are
approximated to the closest base model (see inline comments).
"""

from typing import Dict, TypedDict, Optional


class ModelPricing(TypedDict, total=False):
    input: float      # USD per 1M prompt tokens (uncached)
    output: float     # USD per 1M completion tokens
    cached: float     # USD per 1M cached prompt tokens (if supported)


# Each entry: model_name -> {input, output, cached?}  in USD per 1M tokens.
MODEL_PRICING: Dict[str, ModelPricing] = {
    # ---------------- OpenAI GPT-4 family ----------------
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached": 1.25},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00, "cached": 1.25},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600, "cached": 0.075},

    # GPT-4.1 series (75% cache discount)
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached": 0.025},

    # ---------------- OpenAI GPT-5 family (Azure Standard Global, Apr 2026) ----------------
    "gpt-5": {"input": 1.25, "output": 10.00, "cached": 0.13},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached": 0.03},
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "cached": 0.01},
    "gpt-5.1": {"input": 1.25, "output": 10.00, "cached": 0.13},
    "gpt-5.1-instant": {"input": 1.00, "output": 8.00, "cached": 0.10},
    "gpt-5.1-thinking": {"input": 1.50, "output": 12.00, "cached": 0.15},

    # gpt-5.4 / gpt-5.4-mini: Azure Foundry Sweden deployment names, no public SKU.
    # Priced to mirror closest base models — verify against the actual Azure
    # Foundry deployment base model and correct if needed.
    #   gpt-5.4      ≈ gpt-5.3 Chat Global ($1.75 in / $14 out / $0.18 cached)
    #   gpt-5.4-mini ≈ gpt-5-mini Global   ($0.25 in / $2 out / $0.03 cached)
    "gpt-5.4": {"input": 1.75, "output": 14.00, "cached": 0.18},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00, "cached": 0.03},

    # o-series reasoning models
    "o1-preview": {"input": 15.00, "output": 60.00, "cached": 7.50},
    "o1-mini": {"input": 3.00, "output": 12.00, "cached": 1.50},
    "o3": {"input": 2.00, "output": 8.00, "cached": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached": 0.28},

    # Legacy
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "cached": 0.25},

    # ---------------- Anthropic Claude ----------------
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


_FALLBACK = "gpt-5-mini"  # cheap, well-specified model used when an unknown name appears


def get_pricing(model_name: str) -> ModelPricing:
    """Return pricing for `model_name`; falls back to a well-defined model.

    Callers should not mutate the returned dict (TypedDict is not frozen).
    """
    return MODEL_PRICING.get(model_name, MODEL_PRICING[_FALLBACK])


def compute_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Compute USD cost given token counts.

    `cached_tokens` are billed at the `cached` rate if defined for the
    model; otherwise at the full `input` rate (no cache discount).
    """
    p = get_pricing(model_name)
    uncached = max(prompt_tokens - cached_tokens, 0)
    cached_rate = p.get("cached", p["input"])
    return (
        uncached / 1_000_000 * p["input"]
        + cached_tokens / 1_000_000 * cached_rate
        + completion_tokens / 1_000_000 * p["output"]
    )
