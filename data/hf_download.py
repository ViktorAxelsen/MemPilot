"""Configure a shared Hugging Face HTTP request budget for preprocessing."""

from __future__ import annotations

import argparse
import logging
import math
import re
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Mapping

import requests


# https://huggingface.co/docs/hub/rate-limits lists anonymous limits of
# 500 API / 3,000 resolver requests per 300 seconds (free: 1,000 / 5,000).
# One request per second leaves room below both anonymous limits.
DEFAULT_HF_DOWNLOAD_RPS = 1.0
_MAX_RATE_LIMIT_RETRIES = 5
_LOGGER = logging.getLogger(__name__)


def _positive_rate(value: str | float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("hf_download_rps must be a finite number greater than zero.")
    return rate


def add_hf_download_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf_download_rps",
        type=_positive_rate,
        default=DEFAULT_HF_DOWNLOAD_RPS,
        help=(
            "Maximum Hugging Face HTTP requests per second across download threads "
            "in this process (default: %(default)s). Applies to dataset and model "
            "downloads during preprocessing; cached files need no download. "
            "Divide the budget between processes if running several at once."
        ),
    )


class _RequestLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / _positive_rate(requests_per_second)
        self._next_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delay = self._next_request - now
                if delay <= 0:
                    self._next_request = now + self._interval
                    return
            # Recheck after sleeping so a 429 cooldown also affects waiting threads.
            time.sleep(delay)

    def defer(self, seconds: float) -> None:
        with self._lock:
            self._next_request = max(self._next_request, time.monotonic() + seconds)


def _retry_delay(headers: Mapping[str, str]) -> float:
    delays = [float(value) for value in re.findall(r"\bt\s*=\s*(\d+(?:\.\d+)?)", headers.get("RateLimit", ""))]
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    delays = [delay for delay in delays if math.isfinite(delay) and delay >= 0]
    # Missing/invalid headers: wait one full HF window. Add a reset-boundary margin.
    return max(delays, default=300.0) + 1.0


class _RateLimitedAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, limiter: _RequestLimiter) -> None:
        super().__init__()
        self._limiter = limiter

    def send(self, request, **kwargs):
        from huggingface_hub import constants
        from huggingface_hub.errors import OfflineModeIsEnabled

        if constants.HF_HUB_OFFLINE:
            raise OfflineModeIsEnabled("Hugging Face network access is disabled by HF_HUB_OFFLINE.")
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            self._limiter.wait()
            response = super().send(request, **kwargs)
            if response.status_code != 429 or request.method not in {"GET", "HEAD"}:
                return response
            delay = _retry_delay(response.headers)
            self._limiter.defer(delay)
            if attempt == _MAX_RATE_LIMIT_RETRIES:
                return response
            response.close()
            _LOGGER.warning(
                "Hugging Face returned HTTP 429; pausing downloads for %.1fs (retry %d/%d).",
                delay,
                attempt + 1,
                _MAX_RATE_LIMIT_RETRIES,
            )


def configure_hf_downloads(requests_per_second: float = DEFAULT_HF_DOWNLOAD_RPS) -> None:
    """Pace Hub HTTP calls, including redirects/retries, across this process's threads.

    Uses the public requests backend hook in the pinned huggingface_hub 0.36.x.
    HF authentication, caching and Xet file transfers continue to use the SDK;
    native Xet storage transfers are separate from this Hub HTTP request budget.
    """
    from huggingface_hub import configure_http_backend

    limiter = _RequestLimiter(requests_per_second)

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.mount("http://", _RateLimitedAdapter(limiter))
        session.mount("https://", _RateLimitedAdapter(limiter))
        return session

    configure_http_backend(backend_factory=backend_factory)
