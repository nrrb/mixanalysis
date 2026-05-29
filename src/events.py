from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MixEvent:
    start: float
    end: float | None
    event_type: str
    title: str
    description: str
    confidence: str


def detect_events(
    df: pd.DataFrame,
    sensitivity: str = "balanced",
    minimum_spacing_seconds: float = 30.0,
) -> list[MixEvent]:
    """Detect plain-English mix events from labeled window features."""
    if df.empty or "pressure_score" not in df:
        return []

    settings = _settings(sensitivity)
    events: list[MixEvent] = []
    events.extend(_detect_relief_sections(df, settings))
    events.extend(_detect_buildups(df, settings))
    events.extend(_detect_drops(df, settings, minimum_spacing_seconds))
    events.extend(_detect_possible_vocals(df))
    events.extend(_detect_transitions(df, settings, minimum_spacing_seconds))
    events.extend(_detect_sustained_pressure(df))

    events.sort(key=lambda event: (event.start, event.event_type))
    return _limit_clustered_events(events, spacing_seconds=8.0)


def _settings(sensitivity: str) -> dict[str, float]:
    if sensitivity == "conservative":
        return {
            "drop_jump": 0.34,
            "transition_change": 0.55,
            "buildup_gain": 0.32,
            "relief_max": 0.30,
        }
    if sensitivity == "sensitive":
        return {
            "drop_jump": 0.22,
            "transition_change": 0.38,
            "buildup_gain": 0.20,
            "relief_max": 0.45,
        }
    return {
        "drop_jump": 0.28,
        "transition_change": 0.46,
        "buildup_gain": 0.26,
        "relief_max": 0.38,
    }


def _detect_drops(
    df: pd.DataFrame,
    settings: dict[str, float],
    minimum_spacing_seconds: float,
) -> list[MixEvent]:
    events = []
    last_start = -minimum_spacing_seconds
    pressure = df["pressure_score"].to_numpy()
    bass = df["bass_norm"].to_numpy()
    density = df["density_norm"].to_numpy()
    energy = df["energy_norm"].to_numpy()

    for idx in range(2, len(df)):
        previous_pressure = float(np.mean(pressure[max(0, idx - 4) : idx]))
        jump = pressure[idx] - previous_pressure
        if (
            jump >= settings["drop_jump"]
            and pressure[idx] >= 0.62
            and bass[idx] > np.mean(bass[max(0, idx - 4) : idx])
            and density[idx] >= np.mean(density[max(0, idx - 4) : idx])
            and energy[idx] >= np.mean(energy[max(0, idx - 4) : idx])
        ):
            start = float(df.iloc[idx]["start_time"])
            if start - last_start < minimum_spacing_seconds:
                continue
            confidence = "high" if jump >= settings["drop_jump"] + 0.15 else "medium"
            events.append(
                MixEvent(
                    start=start,
                    end=float(df.iloc[idx]["end_time"]),
                    event_type="drop candidate",
                    title="Major drop candidate" if confidence == "high" else "Pressure jump",
                    description=(
                        "The mix appears to move from a reduced-pressure section into a heavier, "
                        "denser section here."
                    ),
                    confidence=confidence,
                )
            )
            last_start = start
    return events


def _detect_relief_sections(
    df: pd.DataFrame,
    settings: dict[str, float],
) -> list[MixEvent]:
    is_relief = (df["relief_type"] != "no clear relief") & (
        df["pressure_score"] <= settings["relief_max"]
    )
    events = []
    for start_idx, end_idx in _runs(is_relief):
        relief_types = df.iloc[start_idx : end_idx + 1]["relief_type"]
        title = relief_types.mode().iloc[0] if not relief_types.mode().empty else "Breathing room"
        events.append(
            MixEvent(
                start=float(df.iloc[start_idx]["start_time"]),
                end=float(df.iloc[end_idx]["end_time"]),
                event_type="relief section",
                title=title.capitalize(),
                description=(
                    "The low end, density, or overall pressure pulls back here, creating a reset "
                    "before the next push."
                ),
                confidence="medium" if end_idx > start_idx else "low",
            )
        )
    return events


def _detect_buildups(
    df: pd.DataFrame,
    settings: dict[str, float],
) -> list[MixEvent]:
    events = []
    pressure = df["pressure_score"].to_numpy()
    for idx in range(0, max(0, len(df) - 3)):
        segment = pressure[idx : idx + 4]
        if len(segment) < 4:
            continue
        gain = segment[-1] - segment[0]
        mostly_rising = int(np.count_nonzero(np.diff(segment) > -0.03)) >= 2
        if gain >= settings["buildup_gain"] and mostly_rising and segment[-1] >= 0.55:
            events.append(
                MixEvent(
                    start=float(df.iloc[idx]["start_time"]),
                    end=float(df.iloc[idx + 3]["end_time"]),
                    event_type="buildup candidate",
                    title="Building pressure",
                    description=(
                        "Several features rise together across this section, suggesting a buildup "
                        "into a stronger moment."
                    ),
                    confidence="medium",
                )
            )
    return _merge_overlapping(events, "buildup candidate")


def _detect_possible_vocals(df: pd.DataFrame) -> list[MixEvent]:
    if "possible_vocal" not in df:
        return []
    events = []
    for start_idx, end_idx in _runs(df["possible_vocal"]):
        events.append(
            MixEvent(
                start=float(df.iloc[start_idx]["start_time"]),
                end=float(df.iloc[end_idx]["end_time"]),
                event_type="possible vocal section",
                title="Possible vocal section",
                description=(
                    "The timbre and harmonic stability look voice-like here, though this is only "
                    "a weak proxy without source separation."
                ),
                confidence="low" if start_idx == end_idx else "medium",
            )
        )
    return events


def _detect_transitions(
    df: pd.DataFrame,
    settings: dict[str, float],
    minimum_spacing_seconds: float,
) -> list[MixEvent]:
    events = []
    last_start = -minimum_spacing_seconds
    chroma_cols = [col for col in df.columns if col.startswith("chroma_")]
    for idx in range(1, len(df)):
        spectral_change = abs(df.iloc[idx]["brightness_norm"] - df.iloc[idx - 1]["brightness_norm"])
        bass_change = abs(df.iloc[idx]["bass_norm"] - df.iloc[idx - 1]["bass_norm"])
        flatness_change = abs(df.iloc[idx]["flatness_norm"] - df.iloc[idx - 1]["flatness_norm"])
        chroma_change = _chroma_distance(df, chroma_cols, idx)
        change_score = spectral_change * 0.25 + bass_change * 0.30 + flatness_change * 0.20 + chroma_change * 0.25
        rhythm_continues = abs(df.iloc[idx]["density_norm"] - df.iloc[idx - 1]["density_norm"]) < 0.35
        start = float(df.iloc[idx]["start_time"])
        if (
            change_score >= settings["transition_change"]
            and rhythm_continues
            and start - last_start >= minimum_spacing_seconds
        ):
            next_change = 0.0
            if idx + 1 < len(df):
                next_change = abs(df.iloc[idx + 1]["bass_norm"] - df.iloc[idx]["bass_norm"]) + abs(
                    df.iloc[idx + 1]["brightness_norm"] - df.iloc[idx]["brightness_norm"]
                )
            is_blend = next_change > 0.25
            events.append(
                MixEvent(
                    start=start,
                    end=float(df.iloc[min(idx + 1, len(df) - 1)]["end_time"]) if is_blend else float(df.iloc[idx]["end_time"]),
                    event_type="likely transition",
                    title="Long blend candidate" if is_blend else "Abrupt switch candidate",
                    description=(
                        "The timbre, bass profile, and harmonic color shift while rhythmic activity continues."
                    ),
                    confidence="medium",
                )
            )
            last_start = start
    return events


def _detect_sustained_pressure(df: pd.DataFrame) -> list[MixEvent]:
    is_high = df["pressure_score"] >= 0.70
    events = []
    for start_idx, end_idx in _runs(is_high):
        if end_idx - start_idx < 2:
            continue
        events.append(
            MixEvent(
                start=float(df.iloc[start_idx]["start_time"]),
                end=float(df.iloc[end_idx]["end_time"]),
                event_type="sustained pressure run",
                title="Sustained pressure run",
                description=(
                    "This stretch holds a heavier pressure level for multiple analysis windows."
                ),
                confidence="medium",
            )
        )
    return events


def _runs(mask: pd.Series | np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(values) - 1):
            end = idx if value else idx - 1
            runs.append((start, end))
            start = None
    return runs


def _merge_overlapping(events: list[MixEvent], event_type: str) -> list[MixEvent]:
    if not events:
        return events
    events = sorted(events, key=lambda event: event.start)
    merged = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        previous_end = previous.end if previous.end is not None else previous.start
        if event.event_type == event_type and event.start <= previous_end:
            previous.end = max(previous_end, event.end or event.start)
        else:
            merged.append(event)
    return merged


def _chroma_distance(df: pd.DataFrame, chroma_cols: list[str], idx: int) -> float:
    if not chroma_cols:
        return 0.0
    current = df.iloc[idx][chroma_cols].to_numpy(dtype=float)
    previous = df.iloc[idx - 1][chroma_cols].to_numpy(dtype=float)
    denominator = np.linalg.norm(current) * np.linalg.norm(previous) + 1e-9
    cosine = float(np.dot(current, previous) / denominator)
    return max(0.0, min(1.0, 1.0 - cosine))


def _limit_clustered_events(events: list[MixEvent], spacing_seconds: float) -> list[MixEvent]:
    priority = {
        "drop candidate": 0,
        "likely transition": 1,
        "buildup candidate": 2,
        "relief section": 3,
        "sustained pressure run": 4,
        "possible vocal section": 5,
    }
    kept: list[MixEvent] = []
    for event in sorted(events, key=lambda item: (item.start, priority.get(item.event_type, 9))):
        if kept and abs(event.start - kept[-1].start) < spacing_seconds:
            if priority.get(event.event_type, 9) < priority.get(kept[-1].event_type, 9):
                kept[-1] = event
            continue
        kept.append(event)
    return kept
