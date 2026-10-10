"""Tests for Mini-Project 1.

Every one of these runs offline, against the recorded sample responses in
data/samples. The brief asks for at least five checks on the parsing
functions; the parsing section alone has well over that, because parsing is
where the real bugs live.
"""
import datetime as dt
import json

import pandas as pd
import pytest
import requests

from src.weather.display import compare_sources, format_table, plot_forecast, summarise
from src.weather.errors import (AmbiguousLocationError, ForecastUnavailableError,
                                InvalidRequestError, LocationNotFoundError,
                                UpstreamUnavailableError, WeatherError)
from src.weather.http import ApiClient, Envelope, ReplayClient, RetryPolicy
from src.weather.sources import (COLUMNS, MetNoSource, OpenMeteoSource, Place,
                                 REPLAY_INDEX, choose_place, geocode, get_forecast,
                                 parse_metno_daily, parse_open_meteo_daily,
                                 parse_places)
from src.weather.wmo import WMO_CODES, describe, group_of, is_wet


# ======================================================================
# Parsing -- the offline checks the brief asks for
# ======================================================================
def test_parse_places_reads_a_recorded_response(load_sample):
    places = parse_places(load_sample("geocode_kampala"), "Kampala")
    assert len(places) >= 1
    first = places[0]
    assert first.name == "Kampala" and first.country == "Uganda"
    assert 0.2 < first.latitude < 0.4 and 32.4 < first.longitude < 32.7


def test_parse_places_raises_when_results_key_is_absent(load_sample):
    """The trap the brief warns about: no matches omits 'results' entirely."""
    payload = load_sample("geocode_zero_results")
    assert "results" not in payload          # the API really does this
    with pytest.raises(LocationNotFoundError):
        parse_places(payload, "Zzzqqxyznowhere")


def test_parse_places_survives_a_malformed_row():
    """One broken candidate must not lose the others."""
    payload = {"results": [
        {"name": "Good", "latitude": 1.0, "longitude": 2.0, "country": "X"},
        {"name": "Broken"},                                   # no coordinates
        {"name": "Also good", "latitude": 3.0, "longitude": 4.0},
    ]}
    places = parse_places(payload, "test")
    assert [p.name for p in places] == ["Good", "Also good"]


def test_parse_open_meteo_daily_shape_and_units(load_sample):
    frame = parse_open_meteo_daily(load_sample("forecast_kampala"))
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 7
    assert frame["t_max_c"].between(-60, 60).all()
    assert (frame["precip_mm"] >= 0).all()
    assert (frame["wind_max_kmh"] >= 0).all()
    assert (frame["t_max_c"] >= frame["t_min_c"]).all()


def test_parse_open_meteo_decodes_the_weather_code(load_sample):
    frame = parse_open_meteo_daily(load_sample("forecast_kampala"))
    assert frame["conditions"].notna().all()
    assert not frame["conditions"].str.startswith("Unknown").any()
    # the recorded week contains a thunderstorm, code 95
    assert 95 in set(frame["weather_code"])
    assert "Thunderstorm" in set(frame["conditions"])


def test_parse_open_meteo_rejects_a_payload_without_a_daily_block():
    with pytest.raises(ForecastUnavailableError):
        parse_open_meteo_daily({"latitude": 1.0, "longitude": 2.0})


def test_parse_open_meteo_rejects_a_daily_block_missing_variables():
    with pytest.raises(ForecastUnavailableError):
        parse_open_meteo_daily({"daily": {"time": ["2026-01-01"],
                                          "temperature_2m_max": [20.0]}})


def test_parse_metno_rolls_hourly_into_daily(load_sample):
    frame = parse_metno_daily(load_sample("metno_kampala"), days=7)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 7
    assert (frame["t_max_c"] >= frame["t_min_c"]).all()
    assert (frame["precip_mm"] >= 0).all()


def test_metno_wind_is_converted_from_metres_per_second(load_sample):
    """MET Norway reports m/s; the table is in km/h like Open-Meteo."""
    payload = load_sample("metno_kampala")
    raw = [entry["data"]["instant"]["details"]["wind_speed"]
           for entry in payload["properties"]["timeseries"]
           if "wind_speed" in entry["data"]["instant"]["details"]]
    frame = parse_metno_daily(payload, days=7)
    assert max(raw) < 30                      # plausible as m/s
    assert frame["wind_max_kmh"].max() > max(raw)        # converted upward
    assert frame["wind_max_kmh"].max() == pytest.approx(max(raw) * 3.6, rel=0.02)


def test_metno_rejects_a_payload_with_no_timeseries():
    with pytest.raises(ForecastUnavailableError):
        parse_metno_daily({"properties": {}})


def test_the_two_sources_agree_on_temperature_within_a_few_degrees(load_sample):
    """A cross-check: two independent models should not be far apart."""
    open_meteo = parse_open_meteo_daily(load_sample("forecast_kampala"))
    met_no = parse_metno_daily(load_sample("metno_kampala"), days=7)
    merged = compare_sources(open_meteo, met_no)
    assert merged["max °C Δ"].abs().max() < 5.0


# ======================================================================
# Weather-code decoding
# ======================================================================
@pytest.mark.parametrize("code,expected", [
    (0, "Clear sky"), (3, "Overcast"), (61, "Slight rain"),
    (95, "Thunderstorm"), (99, "Thunderstorm with heavy hail")])
def test_known_codes_decode(code, expected):
    assert describe(code) == expected


def test_unknown_codes_degrade_honestly():
    """An unrecognised code must not silently become 'Clear sky'."""
    assert "Unknown" in describe(7) or "code 7" in describe(7)
    assert describe(None) == "Not reported"
    assert describe("rubbish") == "Not reported"


def test_codes_inside_a_known_group_report_the_group():
    assert group_of(62) == "Rain"
    assert "Rain" in describe(62)


@pytest.mark.parametrize("code,wet", [(0, False), (3, False), (51, True),
                                      (61, True), (80, True), (95, True)])
def test_is_wet(code, wet):
    assert is_wet(code) is wet


def test_the_published_table_is_internally_consistent():
    assert all(isinstance(k, int) and isinstance(v, str)
               for k, v in WMO_CODES.items())
    assert len(set(WMO_CODES.values())) == len(WMO_CODES)   # no duplicates


# ======================================================================
# Place and ambiguity
# ======================================================================
def test_place_rejects_impossible_coordinates():
    with pytest.raises(ValueError):
        Place("Nowhere", "X", 999.0, 0.0)
    with pytest.raises(ValueError):
        Place("Nowhere", "X", 0.0, 400.0)


def test_place_label_and_str():
    place = Place("Kampala", "Uganda", 0.3163, 32.5822, admin1="Central",
                  population=1_680_600)
    assert place.label == "Kampala, Central, Uganda"
    assert "1,680,600" in str(place)


def test_a_single_candidate_needs_no_choice(offline):
    places = geocode("Entebbe", offline)
    chosen = choose_place(places) if len(places) == 1 else places[0]
    assert chosen.country == "Uganda"


def test_ambiguity_raises_rather_than_guessing(offline):
    """'Springfield' must not silently become whichever one is listed first."""
    candidates = geocode("Springfield", offline, count=10)
    assert len(candidates) > 1
    with pytest.raises(AmbiguousLocationError) as caught:
        choose_place(candidates)
    assert len(caught.value.candidates) == len(candidates)


def test_ambiguity_resolves_by_index_or_country(offline):
    candidates = geocode("Springfield", offline, count=10)
    assert choose_place(candidates, 0) is candidates[0]
    picked = choose_place(candidates, "United States")
    assert picked.country == "United States"
    with pytest.raises(IndexError):
        choose_place(candidates, 99)
    with pytest.raises(LocationNotFoundError):
        choose_place(candidates, "Atlantis")


# ======================================================================
# The named functions
# ======================================================================
def test_geocode_requires_a_name(offline):
    with pytest.raises(ValueError):
        geocode("", offline)
    with pytest.raises(ValueError):
        geocode("   ", offline)


def test_get_forecast_validates_coordinates_before_requesting(offline):
    with pytest.raises(ValueError):
        get_forecast(999, 32.0, 7, offline)
    with pytest.raises(ValueError):
        get_forecast(0.3, 400.0, 7, offline)


def test_get_forecast_returns_the_requested_number_of_days(offline):
    assert len(get_forecast(0.31628, 32.58219, 3, offline)) == 3
    assert len(get_forecast(0.31628, 32.58219, 7, offline)) == 7


def test_open_meteo_source_rejects_an_impossible_horizon(offline):
    with pytest.raises(ValueError):
        OpenMeteoSource(offline).daily(0.31628, 32.58219, days=0)
    with pytest.raises(ValueError):
        OpenMeteoSource(offline).daily(0.31628, 32.58219, days=40)


def test_sources_share_an_interface(offline):
    for source in (OpenMeteoSource(offline), MetNoSource(offline)):
        frame, envelope = source.daily(0.31628, 32.58219, 5)
        assert list(frame.columns) == list(COLUMNS)
        assert isinstance(envelope, Envelope)
        assert source.attribution        # every provider states its licence


# ======================================================================
# Retry, timeout and failure handling
# ======================================================================
def test_backoff_schedule_is_exponential_and_capped():
    policy = RetryPolicy(attempts=5, base_seconds=1.0, factor=2.0,
                         ceiling_seconds=4.0)
    assert policy.delays() == [1.0, 2.0, 4.0, 4.0]       # capped, not unbounded
    assert policy.total_wait() == 11.0


def test_a_single_attempt_policy_never_waits():
    assert RetryPolicy(attempts=1).delays() == []


@pytest.mark.parametrize("kwargs", [
    {"attempts": 0}, {"base_seconds": -1}, {"factor": 0.5},
    {"base_seconds": 10, "ceiling_seconds": 1}])
def test_impossible_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


class _FlakySession:
    """Fails a given number of times, then succeeds."""

    def __init__(self, failures: int, status: int = 503):
        self.failures = failures
        self.status = status
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        response = requests.Response()
        response.url = url
        if self.calls <= self.failures:
            response.status_code = self.status
            response.reason = "Service Unavailable"
            response._content = b"{}"
        else:
            response.status_code = 200
            response.reason = "OK"
            response._content = b'{"ok": true}'
        response.headers["Content-Type"] = "application/json"
        return response


def test_a_transient_failure_is_retried_and_then_succeeds():
    session = _FlakySession(failures=2)
    waits = []
    client = ApiClient("test@example.ac.ug", policy=RetryPolicy(attempts=3,
                                                                base_seconds=0.01),
                       session=session, sleep=waits.append)
    envelope = client.get_json("https://example.invalid/x")
    assert envelope.payload == {"ok": True}
    assert envelope.attempts == 3
    assert session.calls == 3
    assert len(waits) == 2                  # it really did back off


def test_giving_up_raises_a_clear_custom_exception():
    session = _FlakySession(failures=99)
    client = ApiClient("test@example.ac.ug",
                       policy=RetryPolicy(attempts=3, base_seconds=0.01),
                       session=session, sleep=lambda _: None)
    with pytest.raises(UpstreamUnavailableError) as caught:
        client.get_json("https://example.invalid/x")
    assert caught.value.attempts == 3
    assert isinstance(caught.value, WeatherError)


class _RejectingSession:
    """Always returns a 400 with an Open-Meteo style reason."""

    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        response = requests.Response()
        response.url = url
        response.status_code = 400
        response.reason = "Bad Request"
        response._content = json.dumps(
            {"error": True, "reason": "Latitude must be in range of -90 to 90°."}
        ).encode()
        response.headers["Content-Type"] = "application/json"
        return response


def test_a_bad_request_is_not_retried():
    """Retrying an out-of-range latitude would waste four round trips."""
    session = _RejectingSession()
    client = ApiClient("test@example.ac.ug",
                       policy=RetryPolicy(attempts=4, base_seconds=0.01),
                       session=session, sleep=lambda _: None)
    with pytest.raises(InvalidRequestError) as caught:
        client.get_json("https://example.invalid/x")
    assert session.calls == 1
    assert caught.value.status == 400
    assert "Latitude" in (caught.value.reason or "")


def test_a_recorded_400_replays_as_the_same_error(offline):
    with pytest.raises(InvalidRequestError):
        offline.load("forecast_invalid_latitude")


def test_client_requires_a_contact_and_a_positive_timeout():
    with pytest.raises(ValueError):
        ApiClient("")
    with pytest.raises(ValueError):
        ApiClient("test@example.ac.ug", timeout=0)


def test_the_user_agent_carries_the_contact():
    client = ApiClient("student@university.ac.ug")
    assert "student@university.ac.ug" in client.user_agent


def test_replay_client_reports_an_unknown_request_rather_than_inventing_one(offline):
    with pytest.raises(UpstreamUnavailableError):
        offline.get_json("https://geocoding-api.open-meteo.com/v1/search",
                         {"name": "Atlantis"})


# ======================================================================
# Display
# ======================================================================
def test_summary_mentions_the_place_and_the_range(offline):
    frame = get_forecast(0.31628, 32.58219, 7, offline)
    text = summarise(frame, "Kampala, Uganda", "Open-Meteo")
    assert "Kampala" in text and "°C" in text and "mm" in text
    assert "Open-Meteo" in text


def test_formatted_table_has_units_in_the_headings(offline):
    display = format_table(get_forecast(0.31628, 32.58219, 7, offline))
    assert "Max (°C)" in display.columns
    assert "Rain (mm)" in display.columns
    assert "weather_code" not in display.columns


def test_the_chart_is_labelled(offline):
    frame = get_forecast(0.31628, 32.58219, 7, offline)
    figure = plot_forecast(frame, "Kampala")
    axis = figure.axes[0]
    assert axis.get_xlabel() and axis.get_ylabel()
    assert "Kampala" in axis.get_title()
    assert figure.axes[1].get_ylabel().startswith("precipitation")


def test_plotting_nothing_raises():
    with pytest.raises(ValueError):
        plot_forecast(pd.DataFrame(columns=list(COLUMNS)), "Nowhere")


def test_comparison_only_uses_shared_dates(offline):
    open_meteo = get_forecast(0.31628, 32.58219, 7, offline)
    met_no, _ = MetNoSource(offline).daily(0.31628, 32.58219, 5)
    merged = compare_sources(open_meteo, met_no)
    assert len(merged) <= 5
    assert any(column.endswith("Δ") for column in merged.columns)
