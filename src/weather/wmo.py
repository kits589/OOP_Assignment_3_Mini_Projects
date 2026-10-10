"""WMO 4677 weather codes, decoded into readable text.

Mapping as published in the Open-Meteo documentation. Codes are grouped so an
unrecognised value degrades to its group rather than to "unknown".
"""
from __future__ import annotations

__all__ = ["WMO_CODES", "describe", "group_of", "is_wet"]

# WMO 4677 weather codes, as documented by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Coarse groups, used when a code is outside the published table.
_GROUPS: tuple[tuple[range, str], ...] = (
    (range(0, 4), "Cloud cover"),
    (range(45, 49), "Fog"),
    (range(51, 58), "Drizzle"),
    (range(61, 68), "Rain"),
    (range(71, 78), "Snow"),
    (range(80, 83), "Rain showers"),
    (range(85, 87), "Snow showers"),
    (range(95, 100), "Thunderstorm"),
)

# Codes that mean precipitation reached the ground.
_WET = frozenset(
    list(range(51, 58)) + list(range(61, 68)) + list(range(71, 78))
    + list(range(80, 83)) + list(range(85, 87)) + list(range(95, 100)))


def describe(code: int | float | None) -> str:
    """A readable description of one WMO code.

    Unknown codes fall back to their group, then to a labelled "Unknown".

    >>> describe(95)
    'Thunderstorm'
    >>> describe(None)
    'Not reported'
    """
    if code is None:
        return "Not reported"
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "Not reported"
    if value in WMO_CODES:
        return WMO_CODES[value]
    group = group_of(value)
    return f"{group} (code {value})" if group else f"Unknown (code {value})"


def group_of(code: int) -> str | None:
    """The coarse group a code belongs to, or None if it fits nowhere."""
    for span, label in _GROUPS:
        if code in span:
            return label
    return None


def is_wet(code: int | float | None) -> bool:
    """Whether a code implies precipitation reaching the ground.

    >>> is_wet(61), is_wet(3)
    (True, False)
    """
    if code is None:
        return False
    try:
        return int(code) in _WET
    except (TypeError, ValueError):
        return False
