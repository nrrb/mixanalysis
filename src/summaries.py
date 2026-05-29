import numpy as np
import pandas as pd

from src.events import MixEvent


PRESSURE_ORDER = {
    "low pressure": 0,
    "groove / cruising": 1,
    "building pressure": 2,
    "full pressure": 3,
    "peak pressure": 4,
}


def assign_pressure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add user-facing pressure, relief, and vocal-proxy labels."""
    if df.empty:
        return df.copy()

    labeled = df.copy()
    labeled["energy_norm"] = robust_normalize(labeled["rms_energy"])
    labeled["bass_norm"] = robust_normalize(labeled["bass_pressure_ratio"])
    labeled["brightness_norm"] = robust_normalize(labeled["spectral_centroid"])
    labeled["density_norm"] = robust_normalize(labeled["onset_density"])
    labeled["flatness_norm"] = robust_normalize(labeled["spectral_flatness"])

    labeled["pressure_score"] = (
        labeled["energy_norm"] * 0.35
        + labeled["bass_norm"] * 0.25
        + labeled["density_norm"] * 0.25
        + labeled["brightness_norm"] * 0.15
    ).clip(0, 1)
    labeled["pressure_label"] = labeled["pressure_score"].apply(_pressure_label)
    labeled["relief_type"] = _assign_relief_types(labeled)
    labeled["possible_vocal"] = _possible_vocal_proxy(labeled)
    return labeled


def summarize_mix(
    df: pd.DataFrame,
    events: list[MixEvent],
    duration: float,
    rhythm=None,
) -> dict:
    """Return a deterministic plain-English summary, tags, and learning notes."""
    if df.empty:
        return {
            "headline": "No usable mix analysis yet",
            "summary": "The uploaded audio did not produce enough analysis windows to describe the mix.",
            "tags": ["needs more audio"],
            "learning_notes": ["Try a longer or clearer audio file."],
        }

    labels = df["pressure_label"].tolist()
    pressure_values = df["pressure_label"].map(PRESSURE_ORDER).fillna(0)
    high_share = float((pressure_values >= 3).mean())
    low_share = float((pressure_values <= 1).mean())
    relief_share = float((df["relief_type"] != "no clear relief").mean())
    vocal_share = float(df["possible_vocal"].mean()) if "possible_vocal" in df else 0.0
    changes = int(sum(1 for a, b in zip(labels, labels[1:]) if a != b))

    first_third = pressure_values.iloc[: max(1, len(pressure_values) // 3)].mean()
    final_third = pressure_values.iloc[-max(1, len(pressure_values) // 3) :].mean()
    trend = float(final_third - first_third)

    event_counts = _event_counts(events)
    shape = _shape_tag(high_share, low_share, relief_share, changes, trend)
    vocal_tag = _vocal_tag(vocal_share)
    transition_tag = _transition_tag(events)
    relief_tag = "frequent relief" if relief_share >= 0.22 else "limited relief"

    headline = _headline(shape, high_share, relief_share)
    summary = _summary_text(
        shape=shape,
        high_share=high_share,
        relief_share=relief_share,
        vocal_share=vocal_share,
        event_counts=event_counts,
        duration=duration,
    )
    notes = _learning_notes(df, events, shape, high_share, relief_share)

    tags = [shape, relief_tag, vocal_tag]
    if transition_tag:
        tags.append(transition_tag)
    tags.append(_tempo_tag(rhythm, df))

    return {
        "headline": headline,
        "summary": summary,
        "tags": tags,
        "learning_notes": notes,
    }


def robust_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    low = values.quantile(0.05)
    high = values.quantile(0.95)
    return ((values - low) / (high - low + 1e-9)).clip(0, 1)


def _pressure_label(score: float) -> str:
    if score < 0.20:
        return "low pressure"
    if score < 0.42:
        return "groove / cruising"
    if score < 0.62:
        return "building pressure"
    if score < 0.82:
        return "full pressure"
    return "peak pressure"


def _assign_relief_types(df: pd.DataFrame) -> pd.Series:
    rolling_pressure = df["pressure_score"].rolling(5, center=True, min_periods=1).mean()
    pressure_drop = rolling_pressure - df["pressure_score"]
    bass_drop = df["bass_norm"].rolling(5, center=True, min_periods=1).mean() - df["bass_norm"]
    density_drop = (
        df["density_norm"].rolling(5, center=True, min_periods=1).mean() - df["density_norm"]
    )

    labels = []
    for idx in df.index:
        if pressure_drop.loc[idx] > 0.30 and df.loc[idx, "pressure_score"] < 0.25:
            labels.append("quiet reset")
        elif bass_drop.loc[idx] > 0.28:
            labels.append("kick/bass pullback")
        elif density_drop.loc[idx] > 0.28:
            labels.append("density drop")
        elif pressure_drop.loc[idx] > 0.20:
            labels.append("breathing room")
        else:
            labels.append("no clear relief")
    return pd.Series(labels, index=df.index)


def _possible_vocal_proxy(df: pd.DataFrame) -> pd.Series:
    stable_chroma = robust_normalize(df["chroma_stability"]) > 0.55
    reduced_bass = df["bass_norm"] < 0.55
    mid_bright = df["brightness_norm"].between(0.25, 0.85)
    not_noisy = df["flatness_norm"] < 0.70
    not_peak = df["pressure_score"] < 0.78
    return stable_chroma & reduced_bass & mid_bright & not_noisy & not_peak


def _event_counts(events: list[MixEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts


def _shape_tag(
    high_share: float,
    low_share: float,
    relief_share: float,
    changes: int,
    trend: float,
) -> str:
    if high_share >= 0.55 and relief_share < 0.18:
        return "relentless / sustained pressure"
    if trend >= 0.75:
        return "slow-building"
    if changes >= 5 and relief_share >= 0.16:
        return "wave-shaped"
    if low_share >= 0.50:
        return "restrained / low-pressure"
    return "steady groove-focused"


def _vocal_tag(vocal_share: float) -> str:
    if vocal_share >= 0.25:
        return "vocal-present"
    if vocal_share >= 0.08:
        return "some possible vocals"
    return "vocal-sparse"


def _transition_tag(events: list[MixEvent]) -> str | None:
    transitions = [event for event in events if event.event_type == "likely transition"]
    if not transitions:
        return None
    long_blends = sum(1 for event in transitions if "blend" in event.title.lower())
    abrupt = sum(1 for event in transitions if "abrupt" in event.title.lower())
    if long_blends > abrupt:
        return "gradual blend style"
    if abrupt:
        return "cut-heavy style"
    return "transition-marked"


def _tempo_tag(rhythm, df: pd.DataFrame) -> str:
    """Tempo tag from the rhythm engine, with a df-only fallback.

    Prefers the Phase 13 ``RhythmResult.tempo_tag``. When no rhythm is supplied
    (e.g. older callers), derive a coarse tag from the per-window ``local_tempo``
    so the tag is never the old hardcoded string.
    """
    if rhythm is not None:
        return rhythm.tempo_tag
    if "local_tempo" not in df:
        return "tempo unclear"
    tempo = pd.to_numeric(df["local_tempo"], errors="coerce").dropna()
    if tempo.empty:
        return "tempo unclear"
    spread = float(tempo.quantile(0.9) - tempo.quantile(0.1))
    if spread <= 2.0:
        return f"steady tempo (~{int(round(tempo.median()))} BPM)"
    return "tempo varies"


def _headline(shape: str, high_share: float, relief_share: float) -> str:
    if shape == "wave-shaped":
        return "Wave-shaped mix with recurring pressure and relief"
    if shape == "slow-building":
        return "Slow-building mix that gains pressure over time"
    if shape == "relentless / sustained pressure":
        return "High-pressure mix with few reset pockets"
    if relief_share >= 0.25:
        return "Groove-led mix with clear breathing-room sections"
    if high_share >= 0.35:
        return "Driving mix with several full-pressure runs"
    return "Steady mix with moderate pressure shifts"


def _summary_text(
    shape: str,
    high_share: float,
    relief_share: float,
    vocal_share: float,
    event_counts: dict[str, int],
    duration: float,
) -> str:
    intensity = "full-pressure and peak sections are common" if high_share >= 0.35 else "the mix spends more time in controlled groove sections"
    relief = "with regular pullbacks that create contrast" if relief_share >= 0.20 else "with relatively few obvious reset moments"
    vocals = "Possible vocal moments appear in several sections." if vocal_share >= 0.12 else "Possible vocal moments are sparse."
    drops = event_counts.get("drop candidate", 0)
    transitions = event_counts.get("likely transition", 0)
    event_text = f"The detector found {drops} drop candidate(s) and {transitions} likely transition region(s)."
    minutes = max(1, int(np.ceil(duration / 60)))
    return (
        f"This {minutes}-minute analysis looks {shape}: {intensity}, {relief}. "
        f"{vocals} {event_text}"
    )


def _learning_notes(
    df: pd.DataFrame,
    events: list[MixEvent],
    shape: str,
    high_share: float,
    relief_share: float,
) -> list[str]:
    notes = []
    if events:
        first_event = events[0]
        notes.append(
            f"Start by checking {first_event.title.lower()} around {_format_time(first_event.start)}."
        )
    if shape == "wave-shaped":
        notes.append("Notice where the mix pulls back before pushing into higher-pressure sections.")
    elif shape == "slow-building":
        notes.append("Watch how pressure accumulates across the mix instead of arriving all at once.")
    elif high_share >= 0.45:
        notes.append("Look for the short relief pockets that keep the high-pressure sections from flattening out.")
    elif relief_share >= 0.20:
        notes.append("Study how lower-pressure sections reset the ear before the next heavier moment.")
    else:
        notes.append("Compare the groove sections to see where subtle density and brightness changes carry momentum.")

    longest = _longest_high_pressure_run(df)
    if longest:
        notes.append(f"The longest high-pressure run appears around {_format_time(longest[0])}.")
    return notes[:3]


def _longest_high_pressure_run(df: pd.DataFrame) -> tuple[float, float] | None:
    is_high = df["pressure_label"].map(PRESSURE_ORDER).fillna(0) >= 3
    best_start = None
    best_end = None
    current_start = None
    for idx, high in enumerate(is_high):
        if high and current_start is None:
            current_start = idx
        if current_start is not None and (not high or idx == len(is_high) - 1):
            end_idx = idx if high else idx - 1
            if best_start is None or end_idx - current_start > best_end - best_start:
                best_start = current_start
                best_end = end_idx
            current_start = None
    if best_start is None:
        return None
    return float(df.iloc[best_start]["start_time"]), float(df.iloc[best_end]["end_time"])


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
