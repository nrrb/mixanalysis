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


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
