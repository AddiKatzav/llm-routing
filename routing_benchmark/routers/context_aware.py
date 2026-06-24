"""Router 2 -- Context-Aware Dynamic Routing (Local LLM-as-Judge).

Implements section 3.3 of ``routing_benchmark_spec.md``. Unlike
``StaticSemanticRouter``, this router is re-evaluated every turn and reads
``RoutingRequest.features`` (computed fresh per turn by the Agent
Environment) rather than caching a single decision.

Decision order, per the spec pseudo-code:

1. If consecutive tool failures meet or exceed ``escalation_threshold``,
   escalate to cloud immediately -- no judge call.
2. If context occupancy meets or exceeds ``wall_proximity_threshold``,
   escalate to cloud immediately -- no judge call.
3. Otherwise, consult a small local judge model (any ``BaseModelProvider``)
   and parse its LOCAL/CLOUD verdict.

Two deviations from the literal spec pseudo-code, both documented because
the spec underspecified them:

- The spec's ``extract_features`` is the Agent Environment's job (see
  ``RoutingRequest.features`` in ``models.py``), not this router's --
  duplicating that computation here would couple the router to
  ``AgentState`` internals it has no other reason to know about.
- ``BaseRouter.route()`` is documented to raise ``RouterTimeoutError`` if
  the judge exceeds its timeout budget. A synchronous provider call cannot
  be preempted without threads/async, so this implementation measures
  elapsed wall-clock time around the (already-completed) call and raises
  *after the fact* -- sufficient to flag slow judges in metrics, not to
  cancel them mid-flight.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from routing_benchmark.models import ModelTarget, RoutingDecision, RoutingRequest, TaskCase, RoutingFeatures
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError
from routing_benchmark.router import BaseRouter, RouterTimeoutError

__all__ = ["ContextAwareRouter"]

_LABEL_PATTERN = re.compile(r"\b(LOCAL|CLOUD)\b", re.IGNORECASE)


class ContextAwareRouter(BaseRouter):
    """LLM-as-judge router that escalates on failure/context signals first."""

    def __init__(
        self,
        local_judge_model: BaseModelProvider,
        escalation_threshold: int = 2,
        wall_proximity_threshold: float = 0.85,
        local_model_id: str = "llama3.1:8b",
        cloud_model_id: str = "claude-sonnet-4-6",
        judge_timeout_ms: float = 150.0,
        ambiguous_parse_fallback: ModelTarget = ModelTarget.CLOUD,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if escalation_threshold < 1:
            raise ValueError("escalation_threshold must be >= 1")
        if wall_proximity_threshold <= 0.0:
            raise ValueError("wall_proximity_threshold must be positive")
        if judge_timeout_ms <= 0.0:
            raise ValueError("judge_timeout_ms must be positive")

        self.local_judge_model = local_judge_model
        self.escalation_threshold = escalation_threshold
        self.wall_proximity_threshold = wall_proximity_threshold
        self.local_model_id = local_model_id
        self.cloud_model_id = cloud_model_id
        self.judge_timeout_ms = judge_timeout_ms
        self.ambiguous_parse_fallback = ambiguous_parse_fallback
        self._clock = clock

    def route(self, request: RoutingRequest) -> RoutingDecision:
        features = request.features

        if features.consecutive_tool_failures >= self.escalation_threshold:
            return RoutingDecision(
                target=ModelTarget.CLOUD,
                model_id=self.cloud_model_id,
                reason=f"failure_escalation:{features.consecutive_tool_failures}",
            )

        occupancy = features.context_occupancy_ratio
        if occupancy >= self.wall_proximity_threshold:
            return RoutingDecision(
                target=ModelTarget.CLOUD,
                model_id=self.cloud_model_id,
                reason=f"context_proximity_to_wall:{occupancy:.4f}",
            )

        return self._consult_judge(features, request.task)

    def _consult_judge(self, features: RoutingFeatures, task: TaskCase) -> RoutingDecision:
        judge_prompt = self._render_judge_prompt(features, task)

        start = self._clock()
        try:
            completion = self.local_judge_model.generate(judge_prompt, {})
        except ProviderUnavailableError as exc:
            return RoutingDecision(
                target=ModelTarget.CLOUD,
                model_id=self.cloud_model_id,
                reason=f"judge_unavailable_escalation:{exc}",
            )
        elapsed_ms = (self._clock() - start) * 1000.0

        if elapsed_ms > self.judge_timeout_ms:
            raise RouterTimeoutError(
                f"local judge model exceeded timeout budget: "
                f"{elapsed_ms:.1f}ms > {self.judge_timeout_ms:.1f}ms"
            )

        label, ambiguous = self._parse_judge_label(completion.text)
        model_id = self.local_model_id if label is ModelTarget.LOCAL else self.cloud_model_id
        reason = f"judge:{label.value}" + (":ambiguous_fallback" if ambiguous else "")

        return RoutingDecision(
            target=label,
            model_id=model_id,
            reason=reason,
            raw_provider_metadata={"judge_token_usage": completion.token_usage},
        )

    def _render_judge_prompt(self, features: RoutingFeatures, task: TaskCase) -> str:
        return (
            "You are a routing judge deciding whether the NEXT turn of an agent "
            "conversation should be handled by a small local model or escalated "
            "to a larger cloud model.\n"
            f"Task domain: {task.domain}\n"
            f"Intent complexity: {task.complexity.value}\n"
            f"Turn count so far: {features.turn_count}\n"
            f"Context occupancy: {features.context_occupancy_ratio:.2%}\n"
            f"Consecutive tool failures: {features.consecutive_tool_failures}\n"
            f"Cumulative silent failures: {features.cumulative_silent_failure_count}\n"
            f"Rolling wall-hit rate: {features.rolling_wall_hit_rate:.2%}\n"
            "Respond with exactly one word: LOCAL or CLOUD."
        )

    def _parse_judge_label(self, text: str | None) -> tuple[ModelTarget, bool]:
        """Parse the judge's verdict; returns (target, was_ambiguous)."""
        if text is not None:
            match = _LABEL_PATTERN.search(text)
            if match is not None:
                return ModelTarget(match.group(1).lower()), False
        return self.ambiguous_parse_fallback, True

    def reset(self) -> None:
        """No-op: this router holds no per-run state of its own.

        All state it reads (turn count, failure counts, context occupancy)
        comes from RoutingFeatures computed fresh by the Agent Environment
        on every call, so there is nothing to clear between runs.
        """

    @property
    def name(self) -> str:
        return "context_aware"
