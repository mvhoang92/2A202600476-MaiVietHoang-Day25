from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a circuit is open and calls should fail fast."""


@dataclass(slots=True)
class CircuitBreaker:
    """Production-safe 3-state circuit breaker.

    States:
    - CLOSED: calls pass through; failure_count increments on each error.
    - OPEN: fail fast (CircuitOpenError) until reset_timeout_seconds elapses.
    - HALF_OPEN: allow one probe; close on success, re-open immediately on failure.

    Transition log records every state change with timestamp for recovery-time analysis.
    """

    name: str
    failure_threshold: int
    reset_timeout_seconds: float
    success_threshold: int = 1
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float | None = None
    transition_log: list[dict[str, str | float]] = field(default_factory=list)

    def allow_request(self) -> bool:
        """Return True if a request should be attempted.

        Returns False when OPEN and the reset timeout has not elapsed.
        Transitions to HALF_OPEN when the timeout elapses, allowing one probe.
        """
        if self.state == CircuitState.OPEN:
            if (
                self.opened_at is not None
                and time.monotonic() - self.opened_at >= self.reset_timeout_seconds
            ):
                self._transition(CircuitState.HALF_OPEN, "reset_timeout_elapsed")
                return True
            return False
        return True

    def call(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Call a function through the circuit breaker."""
        if not self.allow_request():
            raise CircuitOpenError(f"circuit {self.name} is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def record_success(self) -> None:
        """Record a successful call.

        Resets failure_count. In HALF_OPEN, closes the circuit once
        success_count reaches success_threshold.
        """
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN and self.success_count >= self.success_threshold:
            self._transition(CircuitState.CLOSED, "probe_success")
            self.success_count = 0

    def record_failure(self) -> None:
        """Record a failed call.

        Increments failure_count and resets success_count.
        In HALF_OPEN: re-opens immediately.
        In CLOSED: opens when failure_count reaches failure_threshold.
        """
        self.failure_count += 1
        self.success_count = 0
        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self._transition(CircuitState.OPEN, "failure_threshold")
            self.opened_at = time.monotonic()

    def _transition(self, new_state: CircuitState, reason: str) -> None:
        if self.state == new_state:
            return
        self.transition_log.append(
            {"from": self.state.value, "to": new_state.value, "reason": reason, "ts": time.time()}
        )
        self.state = new_state
