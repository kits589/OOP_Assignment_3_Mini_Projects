"""Exceptions for the weather client.

Every failure the brief asks to be demonstrated gets its own type, so a caller
can tell "no such city" from "the network is down" without matching on a
message string.
"""
from __future__ import annotations

__all__ = ["AmbiguousLocationError", "ForecastUnavailableError", "InvalidRequestError",
           "LocationNotFoundError", "UpstreamUnavailableError", "WeatherError"]


class WeatherError(Exception):
    """Base class for every error this package raises."""


class UpstreamUnavailableError(WeatherError):
    """Raised when a provider could not be reached after every retry.

    Carries the attempt count and the last underlying exception, so a caller
    can say how hard the client tried before giving up.
    """

    def __init__(self, url: str, attempts: int,
                 last_error: Exception | None = None) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
        super().__init__(
            f"{url} did not respond after {attempts} attempt(s){detail}")


class InvalidRequestError(WeatherError):
    """Raised when the provider rejects the request itself (a 4xx).

    Separate from UpstreamUnavailableError because retrying is pointless: an
    out-of-range latitude will be refused just as firmly on the fourth attempt
    as on the first.
    """

    def __init__(self, url: str, status: int, reason: str | None = None) -> None:
        self.url = url
        self.status = status
        self.reason = reason
        super().__init__(f"{url} returned {status}"
                         + (f": {reason}" if reason else ""))


class LocationNotFoundError(WeatherError):
    """Raised when geocoding returns no candidates for a name.

    Open-Meteo signals this by omitting the results key altogether, not by
    sending an empty list.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"no location matched {name!r}")


class AmbiguousLocationError(WeatherError):
    """Raised when a name matches several places and no choice was made.

    Carries the candidates so a caller can present them for selection.
    """

    def __init__(self, name: str, candidates: list) -> None:
        self.name = name
        self.candidates = candidates
        super().__init__(
            f"{name!r} matched {len(candidates)} locations; pick one")


class ForecastUnavailableError(WeatherError):
    """Raised when a response parsed cleanly but lacked the expected fields."""
