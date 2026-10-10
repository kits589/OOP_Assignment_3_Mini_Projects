"""The HTTP layer: one client, one retry policy, one response envelope.

Everything network-facing goes through ApiClient, so the timeout, the
raise_for_status call, the backoff and the User-Agent are configured in one
place. ReplayClient serves recorded responses from disk instead, which is what
lets the tests and the notebooks run with no network.

RetryPolicy is a frozen object rather than four keyword arguments on the
client because delays() is then a pure function of it, and the backoff
schedule can be asserted without issuing a request.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import requests

from src.weather.errors import InvalidRequestError, UpstreamUnavailableError

__all__ = ["ApiClient", "Envelope", "ReplayClient", "RetryPolicy", "SAMPLE_DIR"]

# Where the recorded responses live.
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

# Status codes worth trying again. A 429 or a 5xx may clear; a 400 will not.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Envelope:
    """One HTTP response, with the metadata the brief asks to be reported."""

    url: str
    status: int
    content_type: str
    elapsed_ms: float
    payload: dict[str, Any]
    attempts: int = 1
    replayed: bool = False

    def __repr__(self) -> str:
        source = "replayed" if self.replayed else "live"
        return (f"Envelope({self.status}, {self.elapsed_ms:.0f} ms, "
                f"{source}, attempts={self.attempts})")

    def summary(self) -> dict[str, Any]:
        """The status/content-type/timing line Part A asks to be inspected."""
        return {"url": self.url, "status": self.status,
                "content_type": self.content_type,
                "elapsed_ms": round(self.elapsed_ms, 1),
                "attempts": self.attempts, "replayed": self.replayed}


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts, and how long to wait between them.

    The delay before attempt n is base * factor ** (n - 1), capped at ceiling.
    Frozen, so a policy cannot be edited halfway through a run.
    """

    attempts: int = 3
    base_seconds: float = 0.5
    factor: float = 2.0
    ceiling_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.base_seconds < 0 or self.ceiling_seconds < 0:
            raise ValueError("delays cannot be negative")
        if self.factor < 1:
            raise ValueError("factor must be at least 1, or delays shrink")
        if self.ceiling_seconds < self.base_seconds:
            raise ValueError("ceiling_seconds cannot be below base_seconds")

    def delays(self) -> list[float]:
        """The waits between attempts; one shorter than attempts.

        Pure, so the backoff can be asserted in a test without sleeping.

        >>> RetryPolicy(attempts=4, base_seconds=0.5).delays()
        [0.5, 1.0, 2.0]
        """
        out, wait = [], self.base_seconds
        for _ in range(self.attempts - 1):
            out.append(min(wait, self.ceiling_seconds))
            wait *= self.factor
        return out

    def total_wait(self) -> float:
        """Worst-case time spent sleeping, in seconds."""
        return sum(self.delays())


class ApiClient:
    """A thin, retrying JSON client over requests."""

    # contact goes in the User-Agent, which MET Norway requires. The caller
    # supplies it; it is never written into this file. sleep is injectable so
    # tests can check the backoff without actually waiting.
    def __init__(self, contact: str, timeout: float = 10.0,
                 policy: RetryPolicy | None = None,
                 session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not contact or not str(contact).strip():
            raise ValueError("a contact string is required for the User-Agent")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.contact = str(contact).strip()
        self.timeout = float(timeout)
        self.policy = policy or RetryPolicy()
        self.session = session or requests.Session()
        self._sleep = sleep

    @property
    def user_agent(self) -> str:
        """The User-Agent sent with every request."""
        return f"advent2026-weather-coursework/1.0 ({self.contact})"

    def __repr__(self) -> str:
        return (f"ApiClient(timeout={self.timeout}s, "
                f"attempts={self.policy.attempts}, contact={self.contact!r})")

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Envelope:
        """GET url and return the decoded JSON, retrying transient failures."""
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        waits = self.policy.delays()
        last_error: Exception | None = None

        for attempt in range(1, self.policy.attempts + 1):
            started = time.perf_counter()
            try:
                response = self.session.get(url, params=params, headers=headers,
                                            timeout=self.timeout)
                elapsed = (time.perf_counter() - started) * 1000

                if response.status_code in RETRYABLE_STATUS:
                    last_error = requests.HTTPError(
                        f"{response.status_code} {response.reason}")
                    raise last_error
                if 400 <= response.status_code < 500:
                    raise InvalidRequestError(response.url, response.status_code,
                                              _reason_of(response))
                response.raise_for_status()

                return Envelope(url=response.url, status=response.status_code,
                                content_type=response.headers.get("Content-Type", ""),
                                elapsed_ms=elapsed, payload=response.json(),
                                attempts=attempt)

            except InvalidRequestError:
                raise                       # a bad request stays bad; do not retry
            except (requests.RequestException, ValueError) as problem:
                last_error = problem
                if attempt < self.policy.attempts:
                    self._sleep(waits[attempt - 1])

        raise UpstreamUnavailableError(url, self.policy.attempts, last_error)


def _reason_of(response: requests.Response) -> str | None:
    """Pull Open-Meteo's reason field out of an error body, if present."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("reason") if isinstance(body, dict) else None


class ReplayClient:
    """Serves recorded responses from disk instead of making requests.

    Interchangeable with ApiClient because it exposes the same get_json
    signature. No base class: the two share an interface and no implementation,
    so inheritance would add coupling without removing any duplication.
    """

    def __init__(self, directory: Path = SAMPLE_DIR,
                 index: Mapping[tuple[str, str], str] | None = None) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"no sample directory at {self.directory}")
        self.index = dict(index or {})
        self.contact = "offline-replay"

    @property
    def user_agent(self) -> str:
        return "advent2026-weather-coursework/1.0 (offline replay)"

    def __repr__(self) -> str:
        return f"ReplayClient({len(self.index)} recorded responses)"

    def load(self, stem: str) -> Envelope:
        """Build an Envelope from one recorded file."""
        path = self.directory / f"{stem}.json"
        if not path.exists():
            raise UpstreamUnavailableError(stem, 0,
                                           FileNotFoundError(f"no sample {path.name}"))
        recorded = json.loads(path.read_text(encoding="utf-8"))
        meta, body = recorded["_recorded"], recorded["body"]
        if 400 <= meta["status"] < 500:
            raise InvalidRequestError(meta["url"], meta["status"],
                                      body.get("reason") if isinstance(body, dict)
                                      else None)
        return Envelope(url=meta["url"], status=meta["status"],
                        content_type=meta["content_type"],
                        elapsed_ms=meta["elapsed_ms"], payload=body,
                        attempts=1, replayed=True)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Envelope:
        """Resolve url/params to a recorded file and replay it."""
        stem = self._lookup(url, params or {})
        if stem is None:
            raise UpstreamUnavailableError(
                url, 0, KeyError("no recorded response matches this request"))
        return self.load(stem)

    def _lookup(self, url: str, params: Mapping[str, Any]) -> str | None:
        if "geocoding-api" in url:
            name = str(params.get("name", "")).strip().lower()
            return self.index.get(("geocode", name))
        if "api.met.no" in url:
            key = f"{float(params.get('lat', 0)):.2f},{float(params.get('lon', 0)):.2f}"
            return self.index.get(("metno", key))
        if "open-meteo" in url:
            lat, lon = float(params.get("latitude", 0)), float(params.get("longitude", 0))
            if not -90 <= lat <= 90:
                return self.index.get(("forecast", "invalid"))
            return self.index.get(("forecast", f"{lat:.2f},{lon:.2f}"))
        return None


def available_samples(directory: Path = SAMPLE_DIR) -> list[str]:
    """Stems of every recorded response on disk."""
    return sorted(path.stem for path in Path(directory).glob("*.json"))
