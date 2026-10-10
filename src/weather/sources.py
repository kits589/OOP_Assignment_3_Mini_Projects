"""The two providers, behind one abstract interface.

WeatherSource fixes what a provider must do: latitude and longitude in, tidy
daily table out. Open-Meteo publishes daily aggregates already; MET Norway
publishes an hourly timeseries and is rolled up here. Same method either way,
which makes the Part C comparison a loop instead of two code paths.

geocode() and get_forecast() are the functions the brief names, thin enough
that the notebook can call them directly.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from src.weather.errors import (AmbiguousLocationError, ForecastUnavailableError,
                                LocationNotFoundError)
from src.weather.http import ApiClient, Envelope, ReplayClient
from src.weather.wmo import describe

__all__ = ["GEOCODE_URL", "METNO_URL", "MetNoSource", "OPEN_METEO_URL",
           "OpenMeteoSource", "Place", "REPLAY_INDEX", "WeatherSource",
           "choose_place", "geocode", "get_forecast", "parse_places",
           "parse_open_meteo_daily", "parse_metno_daily"]

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

# The daily variables the brief requires.
DAILY_VARIABLES = ("weather_code", "temperature_2m_max", "temperature_2m_min",
                   "precipitation_sum", "wind_speed_10m_max")

# Column names used by every provider, so the two tables line up.
COLUMNS = ("date", "t_max_c", "t_min_c", "precip_mm", "wind_max_kmh",
           "weather_code", "conditions")

# Maps a request to the file that was recorded for it, for offline replay.
REPLAY_INDEX: dict[tuple[str, str], str] = {
    ("geocode", "dresden"): "geocode_dresden",
    ("geocode", "kampala"): "geocode_kampala",
    ("geocode", "nairobi"): "geocode_nairobi",
    ("geocode", "mbarara"): "geocode_mbarara",
    ("geocode", "entebbe"): "geocode_entebbe",
    ("geocode", "springfield"): "geocode_springfield_ambiguous",
    ("geocode", "zzzqqxyznowhere"): "geocode_zero_results",
    ("forecast", "0.32,32.58"): "forecast_kampala",
    ("forecast", "-0.60,30.65"): "forecast_mbarara",
    ("forecast", "0.06,32.48"): "forecast_entebbe",
    ("forecast", "51.05,13.74"): "forecast_dresden",
    ("forecast", "-1.28,36.82"): "forecast_nairobi",
    ("forecast", "invalid"): "forecast_invalid_latitude",
    ("metno", "0.32,32.58"): "metno_kampala",
    ("metno", "-0.60,30.65"): "metno_mbarara",
    ("metno", "0.06,32.48"): "metno_entebbe",
}


@dataclass(frozen=True)
class Place:
    """One geocoding candidate. Frozen: a resolved location is a fact, not
    working state.
    """

    name: str
    country: str
    latitude: float
    longitude: float
    admin1: str = ""
    population: int | None = None
    timezone: str = ""

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude {self.latitude} is outside -90..90")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude {self.longitude} is outside -180..180")

    @property
    def label(self) -> str:
        """A one-line description good enough to choose between candidates."""
        parts = [self.name]
        if self.admin1:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    def __str__(self) -> str:
        where = f"({self.latitude:.4f}, {self.longitude:.4f})"
        people = f", pop. {self.population:,}" if self.population else ""
        return f"{self.label} {where}{people}"


# ----------------------------------------------------------------------
# Parsing: pure functions, exercised by the offline checks
# ----------------------------------------------------------------------
def parse_places(payload: Mapping[str, Any], name: str = "") -> list[Place]:
    """Turn a geocoding payload into Place objects.

    Open-Meteo signals "no matches" by omitting the results key, not by sending
    an empty list, so payload["results"] raises KeyError on a perfectly
    ordinary response. That is the likeliest bug in this project, and why this
    function is separate from the network code.
    """
    if not isinstance(payload, Mapping):
        raise ForecastUnavailableError("geocoding payload was not an object")
    results = payload.get("results")          # .get, never [] -- see above
    if not results:
        raise LocationNotFoundError(name or "the requested name")
    places = []
    for row in results:
        try:
            places.append(Place(
                name=row["name"], country=row.get("country", ""),
                latitude=float(row["latitude"]), longitude=float(row["longitude"]),
                admin1=row.get("admin1", ""), population=row.get("population"),
                timezone=row.get("timezone", "")))
        except (KeyError, TypeError, ValueError):
            continue                          # skip a malformed row, keep the rest
    if not places:
        raise LocationNotFoundError(name or "the requested name")
    return places


def parse_open_meteo_daily(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Turn an Open-Meteo forecast payload into a tidy daily table.

    Column names carry their units. That matters here because the second
    provider reports wind in m/s and this one in km/h.
    """
    daily = payload.get("daily")
    if not isinstance(daily, Mapping) or "time" not in daily:
        raise ForecastUnavailableError("payload has no daily block")
    missing = [v for v in DAILY_VARIABLES if v not in daily]
    if missing:
        raise ForecastUnavailableError(f"daily block is missing {missing}")

    frame = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "t_max_c": daily["temperature_2m_max"],
        "t_min_c": daily["temperature_2m_min"],
        "precip_mm": daily["precipitation_sum"],
        "wind_max_kmh": daily["wind_speed_10m_max"],
        "weather_code": daily["weather_code"],
    })
    frame["conditions"] = [describe(code) for code in frame["weather_code"]]
    return frame[list(COLUMNS)]


def parse_metno_daily(payload: Mapping[str, Any],
                      days: int | None = None) -> pd.DataFrame:
    """Roll MET Norway's hourly timeseries up into the same daily shape.

    MET Norway publishes instantaneous hourly values plus next_6_hours
    precipitation blocks, so the daily figures are built here: max and min of
    hourly air temperature, the sum of every sixth precipitation block to avoid
    double-counting overlapping windows, and the highest hourly wind.

    Wind arrives in m/s and is converted to km/h so the two providers can be
    compared without a unit mistake.
    """
    series = (payload.get("properties") or {}).get("timeseries")
    if not series:
        raise ForecastUnavailableError("MET Norway payload has no timeseries")

    temps: dict[dt.date, list[float]] = defaultdict(list)
    winds: dict[dt.date, list[float]] = defaultdict(list)
    rain: dict[dt.date, float] = defaultdict(float)

    for entry in series:
        stamp = dt.datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
        day = stamp.date()
        details = (entry.get("data", {}).get("instant", {}).get("details", {}))
        if "air_temperature" in details:
            temps[day].append(float(details["air_temperature"]))
        if "wind_speed" in details:
            winds[day].append(float(details["wind_speed"]) * 3.6)   # m/s -> km/h

        # Six-hour blocks overlap on the hourly grid; take one every six hours.
        block = entry.get("data", {}).get("next_6_hours", {}).get("details", {})
        if "precipitation_amount" in block and stamp.hour % 6 == 0:
            rain[day] += float(block["precipitation_amount"])

    ordered = sorted(day for day in temps if temps[day])
    if days:
        ordered = ordered[:days]
    frame = pd.DataFrame({
        "date": ordered,
        "t_max_c": [max(temps[d]) for d in ordered],
        "t_min_c": [min(temps[d]) for d in ordered],
        "precip_mm": [round(rain[d], 2) for d in ordered],
        "wind_max_kmh": [round(max(winds[d]), 1) if winds[d] else float("nan")
                         for d in ordered],
        "weather_code": [None] * len(ordered),
    })
    frame["conditions"] = "not reported by MET Norway"
    return frame[list(COLUMNS)]


# ----------------------------------------------------------------------
# The providers
# ----------------------------------------------------------------------
class WeatherSource(ABC):
    """A provider that can turn coordinates into a daily forecast table."""

    # Human-readable provider name, used in tables and chart legends.
    label: str = "source"

    # Attribution required by the provider's licence.
    attribution: str = ""

    def __init__(self, client: ApiClient | ReplayClient) -> None:
        self.client = client

    @abstractmethod
    def daily(self, latitude: float, longitude: float,
              days: int = 7) -> tuple[pd.DataFrame, Envelope]:
        """Return the daily table and the envelope the data arrived in."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(client={type(self.client).__name__})"


class OpenMeteoSource(WeatherSource):
    """The main forecast source, which already publishes daily aggregates."""

    label = "Open-Meteo"
    attribution = "Weather data by Open-Meteo.com (CC BY 4.0)"

    def daily(self, latitude: float, longitude: float,
              days: int = 7) -> tuple[pd.DataFrame, Envelope]:
        if not 1 <= int(days) <= 16:
            raise ValueError("Open-Meteo serves between 1 and 16 forecast days")
        envelope = self.client.get_json(OPEN_METEO_URL, {
            "latitude": latitude, "longitude": longitude,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "auto", "forecast_days": int(days)})
        return parse_open_meteo_daily(envelope.payload).head(int(days)), envelope


class MetNoSource(WeatherSource):
    """The second source, used to cross-check Open-Meteo.

    Its numbers will never match Open-Meteo exactly even when the two models
    agree, because these are rolled up from hourly values.
    """

    label = "MET Norway"
    attribution = ("Data from MET Norway Locationforecast 2.0, "
                   "licensed CC BY 4.0 / NLOD")

    def daily(self, latitude: float, longitude: float,
              days: int = 7) -> tuple[pd.DataFrame, Envelope]:
        envelope = self.client.get_json(METNO_URL, {
            "lat": round(float(latitude), 4), "lon": round(float(longitude), 4)})
        return parse_metno_daily(envelope.payload, int(days)), envelope


# ----------------------------------------------------------------------
# The functions the brief names
# ----------------------------------------------------------------------
def geocode(name: str, client: ApiClient | ReplayClient | None = None,
            count: int = 5) -> list[Place]:
    """Find candidate locations for a place name."""
    if not name or not str(name).strip():
        raise ValueError("a place name is required")
    # Defaults to replay, so importing this module and calling it cannot
    # make a surprise network request.
    client = client or ReplayClient(index=REPLAY_INDEX)
    envelope = client.get_json(GEOCODE_URL, {
        "name": str(name).strip(), "count": int(count),
        "language": "en", "format": "json"})
    return parse_places(envelope.payload, name)


def get_forecast(lat: float, lon: float, days: int = 7,
                 client: ApiClient | ReplayClient | None = None) -> pd.DataFrame:
    """Daily forecast for one point, as a tidy table with units in the columns."""
    if not -90 <= float(lat) <= 90:
        raise ValueError(f"latitude {lat} is outside -90..90")
    if not -180 <= float(lon) <= 180:
        raise ValueError(f"longitude {lon} is outside -180..180")
    client = client or ReplayClient(index=REPLAY_INDEX)
    table, _ = OpenMeteoSource(client).daily(lat, lon, days)
    return table


def choose_place(candidates: Sequence[Place], pick: int | str | None = None) -> Place:
    """Resolve an ambiguous geocoding result to a single place.

    One candidate needs no choice. Several need an index, a country name or a
    country code; with none supplied the candidates are raised so the caller
    can show them. Taking the first is how "Springfield" quietly becomes
    Illinois.
    """
    if not candidates:
        raise LocationNotFoundError("no candidates to choose from")
    if len(candidates) == 1:
        return candidates[0]
    if pick is None:
        raise AmbiguousLocationError(candidates[0].name, list(candidates))
    if isinstance(pick, int):
        if not 0 <= pick < len(candidates):
            raise IndexError(f"pick {pick} is outside 0..{len(candidates) - 1}")
        return candidates[pick]
    wanted = str(pick).strip().lower()
    for place in candidates:
        if wanted in (place.country.lower(), place.admin1.lower()):
            return place
    raise LocationNotFoundError(f"no candidate in {pick!r}")
