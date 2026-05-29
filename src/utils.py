from dataclasses import dataclass

import pandas as pd

from src.events import MixEvent


@dataclass
class MixAnalysisResult:
    name: str
    duration: float
    sample_rate: int
    feature_df: pd.DataFrame
    events: list[MixEvent]
    summary: dict
    role: str = "aspiring"  # "goal" or "aspiring"


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def make_time_ticks(duration_seconds: float, max_ticks: int = 12) -> tuple[list[float], list[str]]:
    """Return evenly-spaced tick positions and MM:SS labels for a Plotly time axis."""
    if duration_seconds <= 0:
        return [0], ["0:00"]
    interval = max(60, round(duration_seconds / max_ticks / 60) * 60)
    vals = list(range(0, int(duration_seconds) + 1, interval))
    labels = [format_duration(v) for v in vals]
    return vals, labels
