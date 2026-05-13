from reliability_lab.cache import ResponseCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.providers import FakeLLMProvider


def test_gateway_returns_response_with_route_reason() -> None:
    provider = FakeLLMProvider("primary", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.001)
    breaker = CircuitBreaker("primary", failure_threshold=2, reset_timeout_seconds=1)
    gateway = ReliabilityGateway([provider], {"primary": breaker}, ResponseCache(60, 0.5))
    result = gateway.complete("hello world")
    assert result.text
    # Route reasons are now specific: "primary:{name}", "fallback:{name}", "cache_hit:{score}",
    # or "static_fallback"
    assert (
        result.route.startswith("primary:")
        or result.route.startswith("fallback:")
        or result.route.startswith("cache_hit:")
        or result.route == "static_fallback"
    )
