from __future__ import annotations

import time
from dataclasses import dataclass

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError, ProviderResponse


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
        cost_budget: float = float("inf"),
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache
        self.cost_budget = cost_budget
        self._cumulative_cost: float = 0.0

    def complete(self, prompt: str) -> GatewayResponse:
        """Return a reliable response or a static fallback.

        Route reasons are specific:
          - "cache_hit:{score:.2f}"          — served from cache
          - "primary:{provider_name}"        — served by first provider
          - "fallback:{provider_name}"       — served by a fallback provider
          - "static_fallback"                — all providers failed
        """
        start_ts = time.perf_counter()

        # --- Cache check ---
        if self.cache is not None:
            cached, score = self.cache.get(prompt)
            if cached is not None:
                latency_ms = (time.perf_counter() - start_ts) * 1000
                return GatewayResponse(
                    text=cached,
                    route=f"cache_hit:{score:.2f}",
                    provider=None,
                    cache_hit=True,
                    latency_ms=latency_ms,
                    estimated_cost=0.0,
                )

        # --- Cost budget check: skip expensive providers if over budget ---
        over_budget = self._cumulative_cost >= self.cost_budget

        last_error: str | None = None
        for idx, provider in enumerate(self.providers):
            # Skip expensive primary if over budget
            if over_budget and idx == 0 and len(self.providers) > 1:
                last_error = "cost_budget_exceeded:skipping_primary"
                continue

            breaker = self.breakers[provider.name]
            is_primary = idx == 0
            # Route label: "primary:{name}" or "fallback:{name}" — starts with standard prefix
            # so downstream checks on "primary" / "fallback" prefix still work
            route_label = f"primary:{provider.name}" if is_primary else f"fallback:{provider.name}"

            try:
                response: ProviderResponse = breaker.call(provider.complete, prompt)
                # Store in cache (only on successful response)
                if self.cache is not None:
                    self.cache.set(prompt, response.text, {"provider": provider.name})
                self._cumulative_cost += response.estimated_cost
                latency_ms = (time.perf_counter() - start_ts) * 1000
                return GatewayResponse(
                    text=response.text,
                    route=route_label,
                    provider=provider.name,
                    cache_hit=False,
                    latency_ms=latency_ms,
                    estimated_cost=response.estimated_cost,
                )
            except (ProviderError, CircuitOpenError) as exc:
                last_error = str(exc)
                continue

        latency_ms = (time.perf_counter() - start_ts) * 1000
        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=latency_ms,
            estimated_cost=0.0,
            error=last_error,
        )
