"""Path setup and shared fixtures.

Matplotlib is forced onto the headless Agg backend so the suite runs with
no display, and every weather test uses the recorded sample responses, so the
whole suite runs with no network either.
"""
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weather.http import ReplayClient, SAMPLE_DIR          # noqa: E402
from src.weather.sources import REPLAY_INDEX                    # noqa: E402


@pytest.fixture(scope="session")
def samples() -> Path:
    """Where the recorded API responses live."""
    assert SAMPLE_DIR.exists(), f"missing sample directory {SAMPLE_DIR}"
    return SAMPLE_DIR


@pytest.fixture
def offline() -> ReplayClient:
    """A client that replays recorded responses instead of making requests."""
    return ReplayClient(index=REPLAY_INDEX)


@pytest.fixture
def load_sample(samples):
    """Return the decoded body of one recorded response."""
    import json

    def read(stem: str):
        path = samples / f"{stem}.json"
        assert path.exists(), f"no sample named {stem}"
        return json.loads(path.read_text(encoding="utf-8"))["body"]

    return read
