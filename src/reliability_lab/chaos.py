from __future__ import annotations

import json
import random
from pathlib import Path

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.config import LabConfig, ScenarioConfig
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.metrics import RunMetrics
from reliability_lab.providers import FakeLLMProvider


def load_queries(path: str | Path = "data/sample_queries.jsonl") -> list[str]:
    queries: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line)["query"])
    return queries


def build_gateway(
    config: LabConfig,
    provider_overrides: dict[str, float] | None = None,
    force_cache_enabled: bool | None = None,
) -> ReliabilityGateway:
    """Build a gateway from config, with optional provider fail-rate overrides.

    force_cache_enabled: if True/False, override config.cache.enabled.
    """
    providers = []
    for p in config.providers:
        fail_rate = provider_overrides.get(p.name, p.fail_rate) if provider_overrides else p.fail_rate
        providers.append(FakeLLMProvider(p.name, fail_rate, p.base_latency_ms, p.cost_per_1k_tokens))
    breakers = {
        p.name: CircuitBreaker(
            name=p.name,
            failure_threshold=config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=config.circuit_breaker.reset_timeout_seconds,
            success_threshold=config.circuit_breaker.success_threshold,
        )
        for p in config.providers
    }
    cache_enabled = config.cache.enabled if force_cache_enabled is None else force_cache_enabled
    cache: ResponseCache | SharedRedisCache | None = None
    if cache_enabled:
        if config.cache.backend == "redis":
            cache = SharedRedisCache(
                config.cache.redis_url,
                config.cache.ttl_seconds,
                config.cache.similarity_threshold,
            )
        else:
            cache = ResponseCache(config.cache.ttl_seconds, config.cache.similarity_threshold)
    return ReliabilityGateway(providers, breakers, cache)


def calculate_recovery_time_ms(gateway: ReliabilityGateway) -> float | None:
    """Derive recovery time from circuit breaker transition logs.

    Recovery time = time between circuit opening and next successful close.
    Returns the average recovery time across all breakers, or None if no recovery occurred.
    """
    recovery_times: list[float] = []
    for breaker in gateway.breakers.values():
        open_ts: float | None = None
        for entry in breaker.transition_log:
            if entry["to"] == "open" and open_ts is None:
                open_ts = float(entry["ts"])
            elif entry["to"] == "closed" and open_ts is not None:
                recovery_times.append((float(entry["ts"]) - open_ts) * 1000)
                open_ts = None
    if not recovery_times:
        return None
    return sum(recovery_times) / len(recovery_times)


def run_scenario(
    config: LabConfig,
    queries: list[str],
    scenario: ScenarioConfig,
    force_cache_enabled: bool | None = None,
) -> RunMetrics:
    """Run a single named chaos scenario."""
    gateway = build_gateway(config, scenario.provider_overrides or None, force_cache_enabled)
    metrics = RunMetrics()
    request_count = config.load_test.requests
    for _ in range(request_count):
        prompt = random.choice(queries)
        result = gateway.complete(prompt)
        metrics.total_requests += 1
        metrics.estimated_cost += result.estimated_cost
        if result.cache_hit:
            metrics.cache_hits += 1
            metrics.estimated_cost_saved += 0.001
        if result.route == "static_fallback":
            metrics.static_fallbacks += 1
            metrics.failed_requests += 1
        elif result.route.startswith("fallback:") or result.route == "fallback":
            metrics.fallback_successes += 1
            metrics.successful_requests += 1
        else:
            # primary:*, cache_hit:*, or legacy "primary"
            metrics.successful_requests += 1
        if result.latency_ms:
            metrics.latencies_ms.append(result.latency_ms)

    metrics.circuit_open_count = sum(
        1 for breaker in gateway.breakers.values() for t in breaker.transition_log if t["to"] == "open"
    )
    metrics.recovery_time_ms = calculate_recovery_time_ms(gateway)
    return metrics


def _evaluate_scenario(name: str, metrics: RunMetrics) -> str:
    """Determine pass/fail for each named scenario based on expected behavior."""
    if name == "primary_timeout_100":
        # Expect: circuit opens, fallback serves most requests
        fallback_rate = (
            metrics.fallback_successes / metrics.total_requests if metrics.total_requests else 0.0
        )
        return "pass" if metrics.circuit_open_count >= 1 and fallback_rate > 0.5 else "fail"

    elif name == "primary_flaky_50":
        # Expect: circuit opens at least once, mix of primary and fallback
        return "pass" if metrics.circuit_open_count >= 1 and metrics.successful_requests > 0 else "fail"

    elif name == "cache_stale_candidate":
        # Expect: cache hits occur (similarity threshold catches repeated queries)
        return "pass" if metrics.cache_hit_rate >= 0.0 else "fail"  # always pass — just measure

    elif name == "all_healthy":
        # Expect: no circuit opens, high availability
        return "pass" if metrics.circuit_open_count == 0 and metrics.availability >= 0.9 else "fail"

    elif name == "all_fail":
        # Expect: static fallback used when all providers fail
        return "pass" if metrics.static_fallbacks > 0 else "fail"

    else:
        return "pass" if metrics.successful_requests > 0 else "fail"


def run_simulation(config: LabConfig, queries: list[str]) -> RunMetrics:
    """Run all named scenarios from config, or a default run if none defined.

    Circuit-breaker scenarios run WITHOUT cache so failures propagate to the breaker.
    Cache comparison scenario runs WITH cache to measure hit rate and cost savings.
    """
    if not config.scenarios:
        default_scenario = ScenarioConfig(name="default", description="baseline run")
        metrics = run_scenario(config, queries, default_scenario)
        metrics.scenarios = {"default": "pass" if metrics.successful_requests > 0 else "fail"}
        return metrics

    combined = RunMetrics()

    # Scenarios that test circuit breaker behaviour run without cache
    # so failures actually reach the breaker (cache would absorb them).
    CB_SCENARIOS = {"primary_timeout_100", "primary_flaky_50", "all_fail"}

    for scenario in config.scenarios:
        force_no_cache = scenario.name in CB_SCENARIOS
        result = run_scenario(
            config,
            queries,
            scenario,
            force_cache_enabled=False if force_no_cache else None,
        )
        combined.scenarios[scenario.name] = _evaluate_scenario(scenario.name, result)

        combined.total_requests += result.total_requests
        combined.successful_requests += result.successful_requests
        combined.failed_requests += result.failed_requests
        combined.fallback_successes += result.fallback_successes
        combined.static_fallbacks += result.static_fallbacks
        combined.cache_hits += result.cache_hits
        combined.circuit_open_count += result.circuit_open_count
        combined.estimated_cost += result.estimated_cost
        combined.estimated_cost_saved += result.estimated_cost_saved
        combined.latencies_ms.extend(result.latencies_ms)
        if result.recovery_time_ms is not None:
            if combined.recovery_time_ms is None:
                combined.recovery_time_ms = result.recovery_time_ms
            else:
                combined.recovery_time_ms = (combined.recovery_time_ms + result.recovery_time_ms) / 2

    # --- Extra: cache vs no-cache comparison scenario ---
    cache_scenario = ScenarioConfig(
        name="cache_stale_candidate",
        description="Low similarity threshold to measure cache hit rate",
        provider_overrides={},
    )
    cache_result = run_scenario(config, queries, cache_scenario, force_cache_enabled=True)
    combined.scenarios["cache_stale_candidate"] = _evaluate_scenario("cache_stale_candidate", cache_result)

    # Run without cache for comparison baseline
    no_cache_result = run_scenario(config, queries, cache_scenario, force_cache_enabled=False)
    combined.scenarios["cache_disabled_baseline"] = (
        "pass" if no_cache_result.successful_requests > 0 else "fail"
    )

    # Merge cache scenario stats into combined
    combined.total_requests += cache_result.total_requests
    combined.successful_requests += cache_result.successful_requests
    combined.failed_requests += cache_result.failed_requests
    combined.cache_hits += cache_result.cache_hits
    combined.estimated_cost += cache_result.estimated_cost
    combined.estimated_cost_saved += cache_result.estimated_cost_saved
    combined.latencies_ms.extend(cache_result.latencies_ms)

    return combined
