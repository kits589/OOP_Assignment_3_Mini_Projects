"""Rendering a forecast table as text and charts.

Kept out of sources.py so the parsing can be tested without importing
matplotlib, and restyled without touching the HTTP layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.weather.wmo import is_wet

__all__ = ["FIGURE_DIR", "PALETTE", "compare_sources", "format_table",
           "house_style", "plot_forecast", "save", "summarise"]

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

# Colour-blind-safe series colours.
PALETTE = ("#b5431a", "#1f4e79", "#2f6b2f", "#7b3294")


def house_style() -> None:
    """Readable matplotlib defaults, applied once per notebook."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9, "legend.frameon": False,
    })
    FIGURE_DIR.mkdir(exist_ok=True)


def save(figure: plt.Figure, filename: str) -> Path:
    """Write a figure into figures/ and report where it went."""
    FIGURE_DIR.mkdir(exist_ok=True)
    target = FIGURE_DIR / filename
    figure.savefig(target)
    return target


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    """A display copy with units in the headings and dates spelled out."""
    display = frame.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%a %d %b")
    display = display.rename(columns={
        "date": "Day", "t_max_c": "Max (°C)", "t_min_c": "Min (°C)",
        "precip_mm": "Rain (mm)", "wind_max_kmh": "Wind max (km/h)",
        "conditions": "Conditions"})
    return display.drop(columns=["weather_code"], errors="ignore")


def summarise(frame: pd.DataFrame, place_label: str,
              attribution: str = "") -> str:
    """A short prose summary of the forecast, for the text part of task B4."""
    if frame.empty:
        return f"No forecast data for {place_label}."
    first, last = frame.iloc[0], frame.iloc[-1]
    wet_days = int(sum(is_wet(code) for code in frame["weather_code"]))
    total_rain = float(frame["precip_mm"].sum())
    warmest = frame.loc[frame["t_max_c"].idxmax()]
    coolest = frame.loc[frame["t_min_c"].idxmin()]

    lines = [
        f"Forecast for {place_label}, {first['date']} to {last['date']} "
        f"({len(frame)} days).",
        f"Daily maxima run {frame['t_max_c'].min():.1f}-{frame['t_max_c'].max():.1f} °C "
        f"and minima {frame['t_min_c'].min():.1f}-{frame['t_min_c'].max():.1f} °C.",
        f"The warmest day is {warmest['date']} at {warmest['t_max_c']:.1f} °C; "
        f"the coolest night is {coolest['date']} at {coolest['t_min_c']:.1f} °C.",
        f"Total precipitation is {total_rain:.1f} mm across the period"
        + (f", with rain reaching the ground on {wet_days} of {len(frame)} days."
           if wet_days else ", with no day showing precipitation."),
        f"The strongest wind is {frame['wind_max_kmh'].max():.1f} km/h on "
        f"{frame.loc[frame['wind_max_kmh'].idxmax(), 'date']}.",
    ]
    if "conditions" in frame and frame["conditions"].notna().any():
        spell = frame["conditions"].mode()
        if len(spell):
            lines.append(f"The most common condition is \"{spell.iloc[0]}\".")
    if attribution:
        lines.append(f"Source: {attribution}")
    return "\n".join(lines)


def plot_forecast(frame: pd.DataFrame, place_label: str,
                  figsize: tuple[float, float] = (11, 5)) -> plt.Figure:
    """Temperature range as a shaded band, with precipitation on a second axis.

    Max and min are drawn as a range because what a reader wants from a
    forecast is how far the day swings.
    """
    if frame.empty:
        raise ValueError("nothing to plot")
    figure, ax = plt.subplots(figsize=figsize)
    days = pd.to_datetime(frame["date"])

    ax.fill_between(days, frame["t_min_c"], frame["t_max_c"],
                    color=PALETTE[0], alpha=0.18, label="daily range")
    ax.plot(days, frame["t_max_c"], "o-", color=PALETTE[0], ms=4, label="max (°C)")
    ax.plot(days, frame["t_min_c"], "o--", color=PALETTE[1], ms=4, label="min (°C)")
    ax.set_ylabel("temperature (°C)")
    ax.set_xlabel("date")
    ax.set_title(f"Seven-day forecast for {place_label}")

    rain = ax.twinx()
    rain.bar(days, frame["precip_mm"], width=0.55, color=PALETTE[2], alpha=0.35,
             label="precipitation (mm)")
    rain.set_ylabel("precipitation (mm)")
    rain.set_ylim(0, max(float(frame["precip_mm"].max()) * 2.2, 1.0))
    rain.grid(False)

    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = rain.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc="upper left", fontsize=8, ncol=2)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure


def compare_sources(primary: pd.DataFrame, secondary: pd.DataFrame,
                    primary_label: str = "Open-Meteo",
                    secondary_label: str = "MET Norway") -> pd.DataFrame:
    """Join two providers' tables on date and show where they disagree.

    Inner join: an outer one would report different forecast horizons as
    disagreements.
    """
    left = primary.set_index("date")
    right = secondary.set_index("date")
    shared = left.index.intersection(right.index)
    if not len(shared):
        raise ValueError("the two forecasts share no dates")
    out = pd.DataFrame(index=shared)
    for column, label in (("t_max_c", "max °C"), ("t_min_c", "min °C"),
                          ("precip_mm", "rain mm"), ("wind_max_kmh", "wind km/h")):
        out[f"{label} ({primary_label})"] = left.loc[shared, column]
        out[f"{label} ({secondary_label})"] = right.loc[shared, column]
        out[f"{label} Δ"] = (left.loc[shared, column]
                             - right.loc[shared, column]).round(2)
    out.index.name = "date"
    return out


def plot_agreement(comparison: pd.DataFrame,
                   figsize: tuple[float, float] = (11, 4.4)) -> plt.Figure:
    """Both providers' maximum temperature, with the gap shaded."""
    figure, ax = plt.subplots(figsize=figsize)
    days = pd.to_datetime(comparison.index)
    open_meteo = [c for c in comparison.columns if c.startswith("max °C (Open")][0]
    met_no = [c for c in comparison.columns if c.startswith("max °C (MET")][0]

    ax.plot(days, comparison[open_meteo], "o-", color=PALETTE[0], ms=4,
            label="Open-Meteo")
    ax.plot(days, comparison[met_no], "s--", color=PALETTE[1], ms=4,
            label="MET Norway")
    ax.fill_between(days, comparison[open_meteo], comparison[met_no],
                    color="0.5", alpha=0.2, label="disagreement")
    ax.set_xlabel("date")
    ax.set_ylabel("daily maximum (°C)")
    ax.set_title("Two independent models on the same point")
    ax.legend(fontsize=8)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure
