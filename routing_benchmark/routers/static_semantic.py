"""Router 1 -- Static / Semantic Routing (Local).

Implements section 3.2 of ``routing_benchmark_spec.md``: a router that
embeds the *initial* prompt, finds the nearest intent exemplar by cosine
similarity, and maps that exemplar to a fixed provider/model. It is
intentionally blind to agent state -- it never looks at context occupancy,
turn count, or tool-failure history -- which is exactly the control
condition the benchmark exists to evaluate against the context-aware
router.

The "computes its decision once" property from the spec is implemented as
a per-task decision cache: ``route()`` embeds and classifies only on the
first call for a given ``task.id``, and returns the cached decision on
every subsequent call for that same task, no matter how the state changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from routing_benchmark.embedding import Embedder, HashingEmbedder, cosine_similarity
from routing_benchmark.models import ModelTarget, RoutingDecision, RoutingRequest
from routing_benchmark.router import BaseRouter

__all__ = ["IntentExemplar", "StaticSemanticRouter"]


@dataclass(frozen=True)
class IntentExemplar:
    """One labeled point in the static router's fixed intent index.

    Attributes:
        label: Human-readable intent label (e.g. "simple_lookup").
        example_text: Representative prompt text embedded to position this
            exemplar in similarity space.
        target: Provider class this intent should be routed to.
        model_id: Concrete model identifier for that provider class.
    """

    label: str
    example_text: str
    target: ModelTarget
    model_id: str


class StaticSemanticRouter(BaseRouter):
    """Embedding nearest-neighbor router over a fixed set of intent exemplars."""

    def __init__(
        self,
        intent_exemplars: list[IntentExemplar],
        embedder: Embedder | None = None,
        default_target: ModelTarget = ModelTarget.LOCAL,
        default_model_id: str = "llama3.1:8b",
    ) -> None:
        if not intent_exemplars:
            raise ValueError("StaticSemanticRouter requires at least one intent exemplar")

        self._exemplars = list(intent_exemplars)
        self._embedder = embedder or HashingEmbedder()
        self._exemplar_embeddings = [
            self._embedder.embed(exemplar.example_text) for exemplar in self._exemplars
        ]
        self._default_target = default_target
        self._default_model_id = default_model_id
        self._decision_cache: dict[str, RoutingDecision] = {}

    def route(self, request: RoutingRequest) -> RoutingDecision:
        task_id = request.task.id
        cached = self._decision_cache.get(task_id)
        if cached is not None:
            return cached

        decision = self._classify(request.task.initial_prompt)
        self._decision_cache[task_id] = decision
        return decision

    def _classify(self, initial_prompt: str) -> RoutingDecision:
        query_embedding = self._embedder.embed(initial_prompt)

        if not any(query_embedding):
            return RoutingDecision(
                target=self._default_target,
                model_id=self._default_model_id,
                reason="semantic:empty_query_fallback",
            )

        best_index = 0
        best_score = -1.0
        for index, exemplar_embedding in enumerate(self._exemplar_embeddings):
            score = cosine_similarity(query_embedding, exemplar_embedding)
            if score > best_score:
                best_score = score
                best_index = index

        exemplar = self._exemplars[best_index]
        return RoutingDecision(
            target=exemplar.target,
            model_id=exemplar.model_id,
            reason=f"semantic:{exemplar.label}:{best_score:.4f}",
        )

    def reset(self) -> None:
        """Clear the per-task decision cache before starting a new run."""
        self._decision_cache.clear()

    @property
    def name(self) -> str:
        return "static_semantic"
