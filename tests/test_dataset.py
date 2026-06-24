import random

import pytest

from routing_benchmark.dataset import (
    COMPLEXITY_LEVELS,
    CONTEXT_DEPTHS,
    DOMAINS,
    FAILURE_PROFILES,
    MAX_TURNS_BY_COMPLEXITY,
    deterministic_id,
    expected_tool_count,
    generate_padding_turns,
    render_prompt,
    synthesize_dataset,
)
from routing_benchmark.environment import DEFAULT_CONTEXT_WINDOW_LIMIT
from routing_benchmark.models import ContextDepthLevel, IntentComplexity, ToolFailureProfile


def test_dataset_matrix_has_expected_size():
    tasks = synthesize_dataset(seed=0)
    expected_count = len(DOMAINS) * len(COMPLEXITY_LEVELS) * len(CONTEXT_DEPTHS) * len(FAILURE_PROFILES)
    assert expected_count == 192
    assert len(tasks) == expected_count


def test_dataset_task_ids_are_unique():
    tasks = synthesize_dataset(seed=0)
    ids = [task.id for task in tasks]
    assert len(ids) == len(set(ids))


def test_dataset_is_deterministic_for_same_seed():
    first = synthesize_dataset(seed=42)
    second = synthesize_dataset(seed=42)

    assert [t.id for t in first] == [t.id for t in second]
    assert [t.initial_prompt for t in first] == [t.initial_prompt for t in second]
    assert [len(t.synthetic_history) for t in first] == [len(t.synthetic_history) for t in second]


def test_dataset_differs_across_seeds_but_keeps_structure():
    a = synthesize_dataset(seed=1)
    b = synthesize_dataset(seed=2)

    assert len(a) == len(b)
    assert [t.id for t in a] != [t.id for t in b]
    # Structural dimensions (domain/complexity/depth/failure_profile) are
    # identical across seeds, only content and ids vary.
    structural_a = [(t.domain, t.complexity, t.failure_profile, t.max_turns) for t in a]
    structural_b = [(t.domain, t.complexity, t.failure_profile, t.max_turns) for t in b]
    assert structural_a == structural_b


def test_every_task_has_valid_max_turns_and_expected_tool_calls():
    tasks = synthesize_dataset(seed=0)
    for task in tasks:
        assert task.max_turns == MAX_TURNS_BY_COMPLEXITY[task.complexity]
        assert task.expected_tool_calls == expected_tool_count(task.domain, task.complexity)
        assert task.expected_tool_calls >= 0
        assert task.max_turns > 0


def test_deterministic_id_is_stable_and_distinguishes_dimensions():
    id_a = deterministic_id("data_lookup", IntentComplexity.TRIVIAL, ContextDepthLevel.SHALLOW, ToolFailureProfile.NONE, 0)
    id_b = deterministic_id("data_lookup", IntentComplexity.TRIVIAL, ContextDepthLevel.SHALLOW, ToolFailureProfile.NONE, 0)
    id_c = deterministic_id("data_lookup", IntentComplexity.MODERATE, ContextDepthLevel.SHALLOW, ToolFailureProfile.NONE, 0)

    assert id_a == id_b
    assert id_a != id_c


def test_render_prompt_is_deterministic_given_same_rng_seed():
    rng_a = random.Random("fixed-seed")
    rng_b = random.Random("fixed-seed")

    prompt_a = render_prompt("data_lookup", IntentComplexity.MODERATE, rng_a)
    prompt_b = render_prompt("data_lookup", IntentComplexity.MODERATE, rng_b)

    assert prompt_a == prompt_b
    assert isinstance(prompt_a, str) and len(prompt_a) > 0


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
def test_render_prompt_covers_every_domain_complexity_pair(domain, complexity):
    rng = random.Random(f"{domain}-{complexity.value}")
    prompt = render_prompt(domain, complexity, rng)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "{" not in prompt  # no unfilled placeholders


def test_generate_padding_turns_reaches_target_budget_without_overshooting_far():
    rng = random.Random("padding-test")
    turns = generate_padding_turns(
        ContextDepthLevel.MID, "data_lookup", rng, context_window_limit=8000
    )
    total_tokens = sum(t.approx_tokens for t in turns)

    target = round(0.50 * 8000)
    assert total_tokens >= target
    # Each turn template is short; overshoot should stay within one turn's worth.
    assert total_tokens < target + 50


def test_generate_padding_turns_scales_monotonically_with_depth():
    totals = {}
    for depth in CONTEXT_DEPTHS:
        rng = random.Random(f"monotonic-{depth.value}")
        turns = generate_padding_turns(depth, "data_lookup", rng, context_window_limit=8000)
        totals[depth] = sum(t.approx_tokens for t in turns)

    assert (
        totals[ContextDepthLevel.SHALLOW]
        < totals[ContextDepthLevel.MID]
        < totals[ContextDepthLevel.NEAR_WALL]
        < totals[ContextDepthLevel.OVER_WALL]
    )


def test_generate_padding_turns_over_wall_exceeds_context_window():
    rng = random.Random("over-wall-test")
    turns = generate_padding_turns(
        ContextDepthLevel.OVER_WALL, "data_lookup", rng, context_window_limit=8000
    )
    total_tokens = sum(t.approx_tokens for t in turns)
    assert total_tokens > 8000


def test_dataset_uses_default_context_window_limit_for_padding():
    tasks = synthesize_dataset(seed=0)
    shallow_tasks = [t for t in tasks if "shallow" in t.id]
    assert shallow_tasks
    total_tokens = sum(turn.approx_tokens for turn in shallow_tasks[0].synthetic_history)
    target = round(0.10 * DEFAULT_CONTEXT_WINDOW_LIMIT)
    assert total_tokens >= target
