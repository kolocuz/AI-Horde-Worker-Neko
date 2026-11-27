"""Small network helpers for Horde worker HTTP calls."""

from __future__ import annotations

import time
from typing import Any

import requests

from worker.logger import logger

DEFAULT_TIMEOUT = 20
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 521, 522, 523, 524, 599}


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int = 3,
    timeout: int | float = DEFAULT_TIMEOUT,
    backoff_seconds: float = 2.0,
    retry_status_codes: set[int] | tuple[int, ...] = tuple(RETRYABLE_STATUS_CODES),
    log_prefix: str = "Request",
    **kwargs: Any,
) -> requests.Response | None:
    """Perform an HTTP request with basic retry/backoff.

    Returns the final Response on success or after non-retryable HTTP response.
    Returns None only when all attempts fail due to request exceptions.
    """
    retry_status_codes = set(retry_status_codes)
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as err:
            if attempt >= attempts:
                logger.warning("{} failed after {} attempts: {}", log_prefix, attempt, err)
                return None
            wait = backoff_seconds * attempt
            logger.warning(
                "{} network error (attempt {}/{}): {}. Waiting {}s...",
                log_prefix,
                attempt,
                attempts,
                err,
                wait,
            )
            time.sleep(wait)
            continue

        if response.status_code in retry_status_codes and attempt < attempts:
            wait = backoff_seconds * attempt
            logger.warning(
                "{} got retryable status {} (attempt {}/{}). Waiting {}s...",
                log_prefix,
                response.status_code,
                attempt,
                attempts,
                wait,
            )
            response.close()
            time.sleep(wait)
            continue
        return response
    return None


def response_json(response: requests.Response | None, *, default: Any = None) -> Any:
    """Safely decode JSON response."""
    if response is None:
        return default
    try:
        return response.json()
    except ValueError:
        return default
