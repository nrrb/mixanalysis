from dataclasses import dataclass

import pandas as pd

from src.summaries import PRESSURE_ORDER


@dataclass
class MixComparisonProfile:
    name: str
    first_full_pressure: float | None
    high_pressure_share: float
    relief_share: float
    possible_vocal_share: float
    transition_count: int
    longest_high_pressure_run: float
    shape_tag: str
    character: str


def build_comparison_profiles(results: list) -> list[MixComparisonProfile]:
    """Summarize each mix into comparable, plain-English profile metrics."""
    return [_profile_result(result) for result in results]


def summarize_comparison(results: list) -> list[str]:
    """Generate a deterministic comparison summary for multiple mixes."""
    profiles = build_comparison_profiles(results)
    if len(profiles) < 2:
        return []

    lines = []
    earliest = min(
        (profile for profile in profiles if profile.first_full_pressure is not None),
        key=lambda profile: profile.first_full_pressure,
        default=None,
    )
    if earliest:
        lines.append(
            f"{earliest.name} reaches full pressure earliest, around {_format_time(earliest.first_full_pressure)}."
        )

    most_pressure = max(profiles, key=lambda profile: profile.high_pressure_share)
    most_relief = max(profiles, key=lambda profile: profile.relief_share)
    most_vocal = max(profiles, key=lambda profile: profile.possible_vocal_share)
    longest_run = max(profiles, key=lambda profile: profile.longest_high_pressure_run)
    transition_heavy = max(profiles, key=lambda profile: profile.transition_count)

    lines.append(
        f"{most_pressure.name} spends the most time in full or peak pressure."
    )
    lines.append(
        f"{most_relief.name} has the clearest relief pattern and more reset pockets."
    )
    if most_vocal.possible_vocal_share > 0:
        lines.append(
            f"{most_vocal.name} has the strongest possible-vocal signal."
        )
    if longest_run.longest_high_pressure_run > 0:
        lines.append(
            f"{longest_run.name} has the longest sustained high-pressure stretch."
        )
    if transition_heavy.transition_count > 0:
        lines.append(
            f"{transition_heavy.name} appears most transition-heavy by the current rules."
        )

    return _dedupe(lines)


def _profile_result(result) -> MixComparisonProfile:
    df = result.feature_df
    if df.empty:
        return MixComparisonProfile(
            name=result.name,
            first_full_pressure=None,
            high_pressure_share=0.0,
            relief_share=0.0,
            possible_vocal_share=0.0,
            transition_count=0,
            longest_high_pressure_run=0.0,
            shape_tag="needs more audio",
            character="Not enough analysis data",
        )

    pressure_values = df["pressure_label"].map(PRESSURE_ORDER).fillna(0)
    high_mask = pressure_values >= 3
    first_full = df.loc[high_mask, "start_time"]
    relief_share = float((df["relief_type"] != "no clear relief").mean())
    vocal_share = float(df["possible_vocal"].mean()) if "possible_vocal" in df else 0.0
    transition_count = sum(1 for event in result.events if event.event_type == "likely transition")
    shape_tag = _first_summary_tag(result)
    high_share = float(high_mask.mean())
    longest_run = _longest_high_run_seconds(df, high_mask)

    return MixComparisonProfile(
        name=result.name,
        first_full_pressure=float(first_full.iloc[0]) if not first_full.empty else None,
        high_pressure_share=high_share,
        relief_share=relief_share,
        possible_vocal_share=vocal_share,
        transition_count=transition_count,
        longest_high_pressure_run=longest_run,
        shape_tag=shape_tag,
        character=_character_label(high_share, relief_share, vocal_share, transition_count, shape_tag),
    )


def _first_summary_tag(result) -> str:
    tags = result.summary.get("tags", [])
    return tags[0] if tags else "unclear shape"


def _character_label(
    high_share: float,
    relief_share: float,
    vocal_share: float,
    transition_count: int,
    shape_tag: str,
) -> str:
    if high_share >= 0.50 and relief_share < 0.18:
        return "Relentless pressure tool"
    if relief_share >= 0.28:
        return "Contrast-heavy reset style"
    if vocal_share >= 0.18:
        return "Vocal-texture led"
    if transition_count >= 4:
        return "Blend and transition focused"
    if "slow-building" in shape_tag:
        return "Gradual pressure builder"
    if "wave-shaped" in shape_tag:
        return "Wave-shaped flow"
    return "Steady groove driver"


def _longest_high_run_seconds(df: pd.DataFrame, high_mask: pd.Series) -> float:
    longest = 0.0
    start_idx = None
    values = high_mask.to_list()
    for idx, high in enumerate(values):
        if high and start_idx is None:
            start_idx = idx
        if start_idx is not None and (not high or idx == len(values) - 1):
            end_idx = idx if high else idx - 1
            seconds = float(df.iloc[end_idx]["end_time"] - df.iloc[start_idx]["start_time"])
            longest = max(longest, seconds)
            start_idx = None
    return longest


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _dedupe(lines: list[str]) -> list[str]:
    seen = set()
    unique = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique
