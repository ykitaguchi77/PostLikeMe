"""Async rate limiter with sliding window.

Provides a reusable, async-safe rate limiter that can be shared across
concurrent collector calls to ensure API rate limits are respected.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Sliding window rate limiter for API calls.

    Tracks request timestamps in a sliding window and blocks callers when
    the limit is reached until the oldest request falls out of the window.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within the window.
    window_seconds:
        Duration of the sliding window in seconds.

    Example
    -------
    >>> limiter = RateLimiter(max_requests=900, window_seconds=900)
    >>> async with limiter:
    ...     await make_api_call()
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made within the rate limit.

        This method is safe to call from multiple concurrent coroutines;
        an internal lock serialises access to the timestamp list.
        """
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps outside the current window
            self._timestamps = [
                t for t in self._timestamps if now - t < self.window_seconds
            ]

            if len(self._timestamps) >= self.max_requests:
                # Wait until the oldest request falls out of the window
                sleep_time = self.window_seconds - (now - self._timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # After sleeping, prune the expired timestamp
                now = time.monotonic()
                self._timestamps = [
                    t for t in self._timestamps if now - t < self.window_seconds
                ]

            self._timestamps.append(time.monotonic())

    @property
    def remaining(self) -> int:
        """Return the approximate number of requests still available in the current window."""
        now = time.monotonic()
        active = [t for t in self._timestamps if now - t < self.window_seconds]
        return max(0, self.max_requests - len(active))

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass
