"""Test dataset synthesis for the LLM Routing Benchmark Suite.

Implements section 4 of ``routing_benchmark_spec.md``: synthesizing
``TaskCase``s across the four dimensions of variation (intent complexity,
simulated context depth, injected tool failures, task domain), entirely
offline and reproducibly given a seed.

Generation is template-based, not an LLM call: every domain/complexity
combination has a small set of hand-written templates with placeholders
(company, metric, quarter, file, number) filled in by a per-task
``random.Random`` instance. ``random.Random`` seeded with a string hashes
that string via a stable algorithm independent of ``PYTHONHASHSEED``, so
the same ``(seed, domain, complexity, depth, failure_profile)`` combination
always produces byte-identical output across processes and machines.
"""

from __future__ import annotations

import random

from routing_benchmark.environment import DEFAULT_CONTEXT_WINDOW_LIMIT
from routing_benchmark.models import (
    ContextDepthLevel,
    IntentComplexity,
    SyntheticTurn,
    TaskCase,
    ToolFailureProfile,
)

__all__ = [
    "DOMAINS",
    "COMPLEXITY_LEVELS",
    "CONTEXT_DEPTHS",
    "FAILURE_PROFILES",
    "MAX_TURNS_BY_COMPLEXITY",
    "expected_tool_count",
    "deterministic_id",
    "render_prompt",
    "generate_padding_turns",
    "synthesize_dataset",
]

DOMAINS: tuple[str, ...] = (
    "data_lookup",
    "multi_step_calculation",
    "file_edit_simulation",
    "ambiguous_intent",
)

COMPLEXITY_LEVELS: tuple[IntentComplexity, ...] = tuple(IntentComplexity)
CONTEXT_DEPTHS: tuple[ContextDepthLevel, ...] = tuple(ContextDepthLevel)
FAILURE_PROFILES: tuple[ToolFailureProfile, ...] = tuple(ToolFailureProfile)

MAX_TURNS_BY_COMPLEXITY: dict[IntentComplexity, int] = {
    IntentComplexity.TRIVIAL: 4,
    IntentComplexity.MODERATE: 8,
    IntentComplexity.COMPLEX_MULTI_TOOL: 14,
}

_EXPECTED_TOOL_CALLS: dict[str, dict[IntentComplexity, int]] = {
    "data_lookup": {
        IntentComplexity.TRIVIAL: 1,
        IntentComplexity.MODERATE: 1,
        IntentComplexity.COMPLEX_MULTI_TOOL: 2,
    },
    "multi_step_calculation": {
        IntentComplexity.TRIVIAL: 1,
        IntentComplexity.MODERATE: 2,
        IntentComplexity.COMPLEX_MULTI_TOOL: 4,
    },
    "file_edit_simulation": {
        IntentComplexity.TRIVIAL: 1,
        IntentComplexity.MODERATE: 2,
        IntentComplexity.COMPLEX_MULTI_TOOL: 3,
    },
    "ambiguous_intent": {
        IntentComplexity.TRIVIAL: 0,
        IntentComplexity.MODERATE: 1,
        IntentComplexity.COMPLEX_MULTI_TOOL: 1,
    },
}

# Fraction of context_window_limit that each ContextDepthLevel should
# occupy before the run's first real turn even happens. OVER_WALL
# intentionally exceeds 1.0 to simulate a task that starts already past
# the local model's nominal context capacity.
_DEPTH_TARGET_RATIOS: dict[ContextDepthLevel, float] = {
    ContextDepthLevel.SHALLOW: 0.10,
    ContextDepthLevel.MID: 0.50,
    ContextDepthLevel.NEAR_WALL: 0.85,
    ContextDepthLevel.OVER_WALL: 1.10,
}

_COMPANIES = ("Acme Corp", "Globex", "Initech", "Umbrella Inc", "Soylent Co")
_QUARTERS = ("Q1", "Q2", "Q3", "Q4")
_FILES = ("report.csv", "ledger.xlsx", "notes.md", "config.json", "sales_data.csv")
_METRICS = ("revenue", "churn rate", "active users", "gross margin", "operating expenses")

_PROMPT_TEMPLATES: dict[str, dict[IntentComplexity, tuple[str, ...]]] = {
    "data_lookup": {
        IntentComplexity.TRIVIAL: (
            "What is the {metric} for {company} in {quarter}?",
        ),
        IntentComplexity.MODERATE: (
            "Look up the {metric} for {company} in {quarter} and compare it "
            "to the prior quarter.",
        ),
        IntentComplexity.COMPLEX_MULTI_TOOL: (
            "Look up the {metric} for {company} across {quarter} and the "
            "prior quarter, cross-reference it against the figure in "
            "{file}, and flag any discrepancy.",
        ),
    },
    "multi_step_calculation": {
        IntentComplexity.TRIVIAL: (
            "Calculate the monthly payment for a ${amount} loan at {rate}% "
            "over {years} years.",
        ),
        IntentComplexity.MODERATE: (
            "Calculate the compound interest on ${amount} at {rate}% for "
            "{years} years, then convert the result to euros.",
        ),
        IntentComplexity.COMPLEX_MULTI_TOOL: (
            "Reconcile the totals in {file} against the {metric} reported "
            "for {company}, recompute the variance for {quarter}, and "
            "produce a corrected summary.",
        ),
    },
    "file_edit_simulation": {
        IntentComplexity.TRIVIAL: (
            "Open {file} and change the title on the first line.",
        ),
        IntentComplexity.MODERATE: (
            "Open {file}, update the {metric} column for {quarter}, and "
            "save the result.",
        ),
        IntentComplexity.COMPLEX_MULTI_TOOL: (
            "Merge the data from {file} and a second export, deduplicate "
            "rows, update the {metric} field for {company}, and save the "
            "merged file.",
        ),
    },
    "ambiguous_intent": {
        IntentComplexity.TRIVIAL: (
            "Can you take a look at this for me?",
        ),
        IntentComplexity.MODERATE: (
            "I think something's off with the {file} numbers, can you "
            "check it out?",
        ),
        IntentComplexity.COMPLEX_MULTI_TOOL: (
            "Something about {company}'s {metric} doesn't add up across "
            "{file} and the dashboard -- can you figure out what's going "
            "on and fix it?",
        ),
    },
}

_PADDING_TEMPLATES: dict[str, tuple[str, ...]] = {
    "data_lookup": (
        "Can you look up the {metric} for {company}?",
        "Sure, checking {file} now for {company}'s {quarter} numbers.",
        "Found it: {metric} for {company} in {quarter} was {number}.",
    ),
    "multi_step_calculation": (
        "I need the {metric} recalculated for {quarter}.",
        "Pulling the figures from {file} to recompute that.",
        "The recalculated {metric} for {company} comes to {number}.",
    ),
    "file_edit_simulation": (
        "Please update {file} with the latest {metric} figures.",
        "Opening {file} now to make that change for {company}.",
        "Updated {file}; the new {metric} value is {number}.",
    ),
    "ambiguous_intent": (
        "Something seems off with {company}'s {file}.",
        "Taking a closer look at {file} to see what's going on.",
        "Found a discrepancy of {number} in the {metric} column.",
    ),
}


def expected_tool_count(domain: str, complexity: IntentComplexity) -> int:
    """Expected number of tool calls for a "complete" run of this domain/complexity."""
    return _EXPECTED_TOOL_CALLS[domain][complexity]


def deterministic_id(
    domain: str,
    complexity: IntentComplexity,
    depth: ContextDepthLevel,
    failure_profile: ToolFailureProfile,
    seed: int,
) -> str:
    """Stable, human-readable identifier for one matrix cell at a given seed."""
    return f"{domain}:{complexity.value}:{depth.value}:{failure_profile.value}:seed{seed}"


def _fill_placeholders(template: str, rng: random.Random) -> str:
    return template.format(
        metric=rng.choice(_METRICS),
        company=rng.choice(_COMPANIES),
        quarter=rng.choice(_QUARTERS),
        file=rng.choice(_FILES),
        number=rng.randint(100, 99_999),
        amount=rng.randint(1_000, 500_000),
        rate=round(rng.uniform(1.5, 9.5), 2),
        years=rng.randint(2, 30),
    )


def render_prompt(domain: str, complexity: IntentComplexity, rng: random.Random) -> str:
    """Render one initial-prompt string for the given domain/complexity."""
    templates = _PROMPT_TEMPLATES[domain][complexity]
    template = rng.choice(templates)
    return _fill_placeholders(template, rng)


def generate_padding_turns(
    depth: ContextDepthLevel,
    domain: str,
    rng: random.Random,
    context_window_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT,
) -> list[SyntheticTurn]:
    """Synthesize plausible prior turns whose combined token weight
    approximates ``depth``'s target fraction of ``context_window_limit``.
    """
    target_tokens = round(_DEPTH_TARGET_RATIOS[depth] * context_window_limit)
    templates = _PADDING_TEMPLATES.get(domain, _PADDING_TEMPLATES["data_lookup"])
    speakers = ("user", "assistant", "tool")

    turns: list[SyntheticTurn] = []
    total_tokens = 0
    index = 0
    # Safety valve: target_tokens is bounded (max ratio 1.10), and each
    # turn contributes at least one token, so this cannot loop forever --
    # the cap just guards against a pathological future template change.
    while total_tokens < target_tokens and index < 5000:
        speaker = speakers[index % len(speakers)]
        template = templates[index % len(templates)]
        text = _fill_placeholders(template, rng)
        approx_tokens = max(1, len(text.split()))

        turns.append(SyntheticTurn(speaker=speaker, text=text, approx_tokens=approx_tokens))
        total_tokens += approx_tokens
        index += 1

    return turns


def synthesize_dataset(seed: int) -> list[TaskCase]:
    """Synthesize the full benchmark matrix of TaskCases for one seed.

    Iterates every (domain, complexity, depth, failure_profile) combination
    -- 4 x 3 x 4 x 4 = 192 TaskCases per seed, per spec section 4.1 (this
    count, multiplied by 3 routers, is the 576-run matrix). Each task gets
    its own ``random.Random`` seeded from its deterministic id, so content
    varies with ``seed`` (for the benchmark's N_REPEATS) while the matrix's
    structural dimensions stay fixed.
    """
    tasks: list[TaskCase] = []
    for domain in DOMAINS:
        for complexity in COMPLEXITY_LEVELS:
            for depth in CONTEXT_DEPTHS:
                for failure_profile in FAILURE_PROFILES:
                    task_id = deterministic_id(domain, complexity, depth, failure_profile, seed)
                    rng = random.Random(task_id)

                    initial_prompt = render_prompt(domain, complexity, rng)
                    synthetic_history = generate_padding_turns(depth, domain, rng)

                    tasks.append(
                        TaskCase(
                            id=task_id,
                            domain=domain,
                            complexity=complexity,
                            initial_prompt=initial_prompt,
                            synthetic_history=synthetic_history,
                            failure_profile=failure_profile,
                            max_turns=MAX_TURNS_BY_COMPLEXITY[complexity],
                            expected_tool_calls=expected_tool_count(domain, complexity),
                            context_depth=depth,
                        )
                    )
    return tasks
