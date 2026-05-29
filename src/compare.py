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


def suggest_toward_goal(goal, aspiring) -> dict:
    """Compare one aspiring mix to the goal mix and return plain-English suggestions."""
    goal_profile = _profile_result(goal)
    asp_profile = _profile_result(aspiring)

    suggestions: list[str] = []
    matched: list[str] = []

    # Pressure shape.
    goal_shape = goal_profile.shape_tag
    asp_shape = asp_profile.shape_tag
    if goal_shape == asp_shape:
        matched.append(f"Your overall shape already reads as {asp_shape}, just like the goal.")
    elif "wave-shaped" in goal_shape and "relentless" in asp_shape:
        suggestions.append(
            "The goal mix breathes in waves while yours stays at sustained pressure. "
            "Try dropping the energy back periodically so the heavy sections land harder."
        )
    elif "relentless" in goal_shape and "wave-shaped" in asp_shape:
        suggestions.append(
            "The goal mix holds sustained pressure while yours rises and falls more. "
            "Try keeping the energy up longer between your pullbacks."
        )
    elif "slow-building" in goal_shape:
        suggestions.append(
            "The goal mix builds gradually across its length. "
            "Try arranging your tracks so the energy accumulates instead of arriving early."
        )

    # Relief frequency.
    relief_gap = goal_profile.relief_share - asp_profile.relief_share
    if relief_gap > 0.08:
        suggestions.append(
            "The goal mix has noticeably more relief pockets. "
            "Add a breather after your longest high-pressure run to let the energy reset."
        )
    elif relief_gap < -0.08:
        suggestions.append(
            "Your mix pulls back more often than the goal. "
            "Try tightening some relief sections so the momentum carries further."
        )
    else:
        matched.append("Your use of relief pockets is close to the goal.")

    # Peak placement.
    goal_peak = _peak_third(goal)
    asp_peak = _peak_third(aspiring)
    if goal_peak and asp_peak and goal_peak != asp_peak:
        suggestions.append(
            f"The goal mix saves its biggest pressure for the {goal_peak} third of the set, "
            f"but yours peaks in the {asp_peak} third. "
            "Try moving your hardest section to match that arc."
        )
    elif goal_peak and goal_peak == asp_peak:
        matched.append(f"Your climax lands in the {asp_peak} third, just like the goal.")

    # Vocal usage.
    vocal_gap = goal_profile.possible_vocal_share - asp_profile.possible_vocal_share
    if vocal_gap > 0.10:
        suggestions.append(
            "The goal mix leans on possible vocal moments more than yours. "
            "Consider weaving in more vocal-led tracks, often around the relief sections."
        )
    elif vocal_gap < -0.10:
        suggestions.append(
            "Your mix appears more vocal-heavy than the goal. "
            "Try letting some instrumental stretches run to match its texture."
        )

    # Transition style.
    goal_style = _transition_style(goal)
    asp_style = _transition_style(aspiring)
    if goal_style and asp_style and goal_style != asp_style:
        if goal_style == "gradual blend":
            suggestions.append(
                "The goal mix tends to use long, gradual blends while yours cuts more sharply. "
                "Try extending your transitions so tracks overlap longer."
            )
        elif goal_style == "cut-heavy":
            suggestions.append(
                "The goal mix tends to switch tracks decisively while yours blends slowly. "
                "Try tighter, more deliberate cuts at the key changeovers."
            )
    elif goal_style and goal_style == asp_style:
        matched.append(f"Your transition style ({asp_style}) already matches the goal.")

    # Buildup usage.
    buildup_gap = _buildup_count(goal) - _buildup_count(aspiring)
    if buildup_gap >= 2:
        suggestions.append(
            "The goal mix leads into its peaks with clearer buildups. "
            "Try adding deliberate rising sections before your biggest moments."
        )

    headline = _suggestion_headline(suggestions, matched, asp_profile)

    return {
        "headline": headline,
        "suggestions": suggestions,
        "matched": matched,
    }


def _suggestion_headline(suggestions: list[str], matched: list[str], asp_profile) -> str:
    if not suggestions:
        return "Already very close to the goal mix's flow."
    if len(suggestions) <= 2 and matched:
        return "Close to the goal, with a couple of areas to tighten up."
    return "Several differences from the goal worth working on."


def _peak_third(result) -> str | None:
    """Return which third ('early'/'middle'/'late') holds the mix's strongest pressure."""
    df = result.feature_df
    if df.empty or "pressure_score" not in df or result.duration <= 0:
        return None
    peak_idx = df["pressure_score"].idxmax()
    midpoint = float(df.loc[peak_idx, "start_time"])
    fraction = midpoint / result.duration
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def _transition_style(result) -> str | None:
    transitions = [event for event in result.events if event.event_type == "likely transition"]
    if not transitions:
        return None
    long_blends = sum(1 for event in transitions if "blend" in event.title.lower())
    abrupt = sum(1 for event in transitions if "abrupt" in event.title.lower())
    if long_blends > abrupt:
        return "gradual blend"
    if abrupt > long_blends:
        return "cut-heavy"
    return "transition-marked"


def _buildup_count(result) -> int:
    return sum(1 for event in result.events if event.event_type == "buildup candidate")


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
