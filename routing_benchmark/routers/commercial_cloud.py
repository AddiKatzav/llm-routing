"""Router 3 -- Commercial Cloud Router Baseline.

Implements section 3.4 of ``routing_benchmark_spec.md``: a thin wrapper
around an external, opaque routing service (e.g. ``openrouter/auto``, Not
Diamond). This router is deliberately a black box -- it makes no
LOCAL/CLOUD decision logic of its own, only:

1. Builds a provider-agnostic payload from the current turn's state.
2. Submits it to whatever ``CloudRouterClient`` implementation is injected.
3. Maps the external service's selected model id back to a
   ``ModelTarget`` (LOCAL/CLOUD) for KPI bucketing, since a commercial
   meta-router *could* select a model the benchmark considers "local"
   even though the router call itself goes over the network.

Section 8 of the spec leaves the exact external service (OpenRouter vs Not
Diamond) as an open question; this module only depends on the
``CloudRouterClient`` protocol, so swapping providers means writing a new
client adapter, not touching this router.

Unlike ``ContextAwareRouter``, this router has no local fallback path: if
the external service is unreachable, ``CloudRouterUnavailableError``
propagates unmodified. A commercial-router baseline whose own service is
down has nothing meaningful to evaluate -- swallowing the error here would
hide that failure rather than surface it as a routing outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from routing_benchmark.models import ModelTarget, RoutingDecision, RoutingRequest
from routing_benchmark.router import BaseRouter

__all__ = [
    "CloudRouterClient",
    "CloudRouterResponse",
    "CloudRouterUnavailableError",
    "CommercialCloudRouter",
]


class CloudRouterUnavailableError(Exception):
    """Raised by a CloudRouterClient implementation when the external
    routing service cannot be reached (network error, auth failure, etc.).
    Not caught by CommercialCloudRouter -- see module docstring.
    """


@dataclass(frozen=True)
class CloudRouterResponse:
    """Normalized response from an external commercial routing service.

    Attributes:
        selected_model: The model id the external service chose to handle
            this turn (e.g. "anthropic/claude-3.5-sonnet").
        raw: Opaque passthrough of the service's full response, persisted
            for debugging but never depended on by this router.
    """

    selected_model: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selected_model:
            raise ValueError("selected_model must be a non-empty string")


class CloudRouterClient(Protocol):
    """Structural type for any external commercial router's client adapter."""

    def submit(self, payload: dict[str, Any]) -> CloudRouterResponse:
        """Submit a routing request to the external service.

        Raises:
            CloudRouterUnavailableError: If the service cannot be reached.
        """
        ...


class CommercialCloudRouter(BaseRouter):
    """Wraps an external commercial routing service behind BaseRouter."""

    def __init__(
        self,
        cloud_router_client: CloudRouterClient,
        model_target_map: dict[str, ModelTarget] | None = None,
        default_target: ModelTarget = ModelTarget.CLOUD,
    ) -> None:
        self.cloud_router_client = cloud_router_client
        self._model_target_map = dict(model_target_map or {})
        self.default_target = default_target

    def route(self, request: RoutingRequest) -> RoutingDecision:
        payload = self._build_payload(request)
        response = self.cloud_router_client.submit(payload)

        target = self._model_target_map.get(response.selected_model, self.default_target)
        return RoutingDecision(
            target=target,
            model_id=response.selected_model,
            reason=f"cloud_router:{response.selected_model}",
            raw_provider_metadata=response.raw,
        )

    def _build_payload(self, request: RoutingRequest) -> dict[str, Any]:
        """Build a provider-agnostic payload describing the current turn.

        Field names are this benchmark's own vocabulary, not any specific
        commercial API's wire format -- a real CloudRouterClient adapter is
        responsible for translating this into whatever shape its service
        expects (e.g. OpenRouter's `models: ["openrouter/auto"]` request).
        """
        return {
            "prompt": request.state.to_prompt(),
            "task_domain": request.task.domain,
            "intent_complexity": request.task.complexity.value,
            "turn_count": request.features.turn_count,
            "context_occupancy_ratio": request.features.context_occupancy_ratio,
        }

    def reset(self) -> None:
        """No-op: this router holds no per-run state of its own.

        Any state-tracking the external service does internally is its own
        concern, opaque to this benchmark.
        """

    @property
    def name(self) -> str:
        return "commercial_cloud"
