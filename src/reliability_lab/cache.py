from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — use these in both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache (existing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """Simple in-memory cache with improved semantic similarity and false-hit guardrails."""

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []

    def get(self, query: str) -> tuple[str | None, float]:
        # Privacy guardrail: skip cache for sensitive queries
        if _is_uncacheable(query):
            return None, 0.0

        best_value: str | None = None
        best_score = 0.0
        best_key: str | None = None
        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl_seconds]

        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best_score = score
                best_value = entry.value
                best_key = entry.key

        if best_score >= self.similarity_threshold and best_key is not None:
            # False-hit guardrail: reject if 4-digit numbers differ
            if _looks_like_false_hit(query, best_key):
                return None, best_score
            return best_value, best_score
        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        # Privacy guardrail: don't cache sensitive queries
        if _is_uncacheable(query):
            return
        self._entries.append(CacheEntry(query, value, time.time(), metadata or {}))

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Improved similarity using exact-match fast path + TF-IDF-style token overlap.

        Strategy:
        1. Exact match → 1.0
        2. Character n-gram overlap (trigrams) for better semantic matching
        3. Token Jaccard as fallback
        """
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()

        # Fast path: exact match
        if a_norm == b_norm:
            return 1.0

        # Character trigram overlap (more robust than token overlap)
        def trigrams(s: str) -> set[str]:
            s = " " + s + " "
            return {s[i : i + 3] for i in range(len(s) - 2)}

        tri_a = trigrams(a_norm)
        tri_b = trigrams(b_norm)
        if tri_a or tri_b:
            tri_score = len(tri_a & tri_b) / len(tri_a | tri_b)
        else:
            tri_score = 0.0

        # Token Jaccard
        tokens_a = set(a_norm.split())
        tokens_b = set(b_norm.split())
        if tokens_a or tokens_b:
            token_score = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        else:
            token_score = 0.0

        # Weighted combination: trigrams carry more weight
        return 0.6 * tri_score + 0.4 * token_score


# ---------------------------------------------------------------------------
# Redis shared cache (new)
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed shared cache for multi-instance deployments."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis."""
        # 1. Privacy guardrail
        if _is_uncacheable(query):
            return None, 0.0

        try:
            # 2. Exact-match lookup
            exact_key = f"{self.prefix}{self._query_hash(query)}"
            response = self._redis.hget(exact_key, "response")
            if response is not None:
                return response, 1.0

            # 3. Similarity scan
            best_value: str | None = None
            best_score = 0.0
            best_cached_query: str | None = None

            for key in self._redis.scan_iter(f"{self.prefix}*"):
                cached_query = self._redis.hget(key, "query")
                if cached_query is None:
                    continue
                score = ResponseCache.similarity(query, cached_query)
                if score > best_score:
                    best_score = score
                    best_cached_query = cached_query
                    best_value = self._redis.hget(key, "response")

            if best_score >= self.similarity_threshold and best_cached_query is not None:
                # 4. False-hit guardrail
                if _looks_like_false_hit(query, best_cached_query):
                    self.false_hit_log.append(
                        {
                            "query": query,
                            "cached_query": best_cached_query,
                            "score": best_score,
                            "reason": "different_4digit_numbers",
                        }
                    )
                    return None, best_score
                return best_value, best_score

            return None, best_score

        except Exception:
            # Graceful degradation: if Redis is down, return cache miss
            return None, 0.0

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with TTL."""
        # 1. Privacy guardrail
        if _is_uncacheable(query):
            return

        try:
            # 2. Build key and store
            key = f"{self.prefix}{self._query_hash(query)}"
            self._redis.hset(key, mapping={"query": query, "response": value})
            self._redis.expire(key, self.ttl_seconds)
        except Exception:
            # Graceful degradation: if Redis is down, silently skip caching
            pass

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
