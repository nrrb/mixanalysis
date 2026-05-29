"""Beat, downbeat, and local-tempo estimation (Phase 13).

Two engines behind one interface:

- **Deep mode:** the ``beat_this`` neural tracker (optional torch extra,
  lazy-imported). State-of-the-art beats/downbeats.
- **Fast mode / fallback:** ``librosa.beat.beat_track``, always available.

Both engines produce beat and downbeat times; the local-tempo curve is derived
uniformly from inter-beat intervals, and a coarse stability label drives the
user-facing tempo tag. ``beat_this`` is used only when explicitly requested and
importable - any failure falls back to librosa, so analysis never crashes on a
missing or misbehaving optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Beats per bar assumed for the POC. Downbeat phase from librosa is unknown, so
# we assume 4/4 starting on the first detected beat; beat_this returns real
# downbeats and ignores this.
DEFAULT_METER = 4


@dataclass
class RhythmResult:
    """Beats, downbeats, a local-tempo curve, and a stability summary."""

    beats: np.ndarray  # beat times (seconds)
    downbeats: np.ndarray  # downbeat times (seconds)
    curve_times: np.ndarray  # times (seconds) for the local-tempo curve
    curve_bpm: np.ndarray  # local BPM aligned to curve_times
    global_tempo: float  # representative BPM (median of the curve)
    stability: str  # "steady" | "drift" | "steps" | "unclear"
    meter: int  # assumed beats per bar
    source: str  # "beat_this" | "librosa"

    @property
    def display_tempo(self) -> int | None:
        if not np.isfinite(self.global_tempo) or self.global_tempo <= 0:
            return None
        return int(round(self.global_tempo))

    @property
    def tempo_tag(self) -> str:
        """Plain-English tag for the mix summary (BPM only when steady)."""
        bpm = self.display_tempo
        if self.stability == "steady" and bpm:
            return f"steady tempo (~{bpm} BPM)"
        if self.stability == "drift":
            return "tempo drifts"
        if self.stability == "steps":
            return "tempo steps between tracks"
        return "tempo unclear"


def estimate_rhythm(y: np.ndarray, sr: int, *, use_beat_this: bool) -> RhythmResult:
    """Return beats, downbeats, a local-tempo curve, and a stability summary.

    Uses the ``beat_this`` neural tracker when ``use_beat_this`` is set and the
    library imports/runs cleanly; otherwise (and on any failure) it falls back to
    ``librosa.beat.beat_track``.
    """
    beats = None
    downbeats = None
    source = "librosa"

    if use_beat_this:
        try:
            beats, downbeats = _beat_this_track(y, sr)
            source = "beat_this"
        except Exception:
            # Any import/runtime failure degrades to the librosa path rather than
            # breaking analysis (see module docstring).
            beats = None
            downbeats = None
            source = "librosa"

    if beats is None:
        beats, downbeats = _librosa_track(y, sr)

    return _assemble(beats, downbeats, source=source)


def window_local_tempo(
    rhythm: RhythmResult,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Per-window local tempo = mean of the curve within ``[start, end)``.

    Windows containing no curve points fall back to the global tempo so the
    column never has gaps.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    out = np.full(starts.shape, rhythm.global_tempo, dtype=float)

    times = rhythm.curve_times
    bpm = rhythm.curve_bpm
    if times.size == 0:
        return out

    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, ends, side="left")
    for idx in range(starts.size):
        if hi[idx] > lo[idx]:
            out[idx] = float(np.mean(bpm[lo[idx] : hi[idx]]))
    return out


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def _beat_this_track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the beat_this neural tracker. Lazy-imports torch/beat_this."""
    import torch
    from beat_this.inference import Audio2Beats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "final0" is beat_this's default checkpoint (downloaded on first use);
    # dbn=False uses the lightweight built-in post-processor (no madmom).
    audio2beats = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    signal = np.ascontiguousarray(y, dtype=np.float32)
    beats, downbeats = audio2beats(signal, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def _librosa_track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Fallback beat grid via librosa; downbeats approximated as every bar.

    Computes the onset envelope once and hands it to ``beat_track`` so the
    tracker does not recompute its own (Phase 9's compute-transforms-once rule).
    A 512-sample hop keeps beat resolution fine enough for accurate timing.
    """
    import librosa

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    downbeats = _downbeats_from_beats(beats, meter=DEFAULT_METER)
    return beats, downbeats


def _downbeats_from_beats(beats: np.ndarray, meter: int) -> np.ndarray:
    """Approximate downbeats as every ``meter``-th beat (assumes 4/4, phase 0)."""
    beats = np.asarray(beats, dtype=float)
    if beats.size == 0:
        return np.asarray([], dtype=float)
    return beats[::meter]


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def _assemble(
    beats: np.ndarray,
    downbeats: np.ndarray | None,
    *,
    source: str,
    meter: int = DEFAULT_METER,
) -> RhythmResult:
    beats = np.asarray(beats, dtype=float)
    downbeats = (
        np.asarray(downbeats, dtype=float)
        if downbeats is not None
        else np.asarray([], dtype=float)
    )
    curve_times, curve_bpm = _tempo_curve(beats)
    global_tempo = float(np.median(curve_bpm)) if curve_bpm.size else float("nan")
    stability = _stability_label(curve_bpm)
    return RhythmResult(
        beats=beats,
        downbeats=downbeats,
        curve_times=curve_times,
        curve_bpm=curve_bpm,
        global_tempo=global_tempo,
        stability=stability,
        meter=meter,
        source=source,
    )


def _tempo_curve(beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local BPM from inter-beat intervals, lightly median-smoothed."""
    if beats.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    intervals = np.diff(beats)
    times = (beats[:-1] + beats[1:]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bpm = 60.0 / intervals

    mask = np.isfinite(bpm) & (bpm > 0)
    times, bpm = times[mask], bpm[mask]
    if bpm.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # A 5-beat rolling median tames octave errors and single-beat jitter without
    # smearing genuine track-to-track tempo steps.
    smoothed = pd.Series(bpm).rolling(5, center=True, min_periods=1).median().to_numpy()
    return times, smoothed


def _stability_label(bpm: np.ndarray) -> str:
    """Classify the tempo curve as steady / steps / drift / unclear."""
    if bpm.size < 4:
        return "unclear"

    p10, _, p90 = np.percentile(bpm, [10, 50, 90])
    spread = float(p90 - p10)
    big_jumps = int(np.count_nonzero(np.abs(np.diff(bpm)) > 4.0))

    if spread <= 2.0:
        return "steady"
    # A handful of clear, discrete jumps reads as track-to-track tempo steps;
    # many small changes (or none crossing the threshold) reads as gradual drift.
    if 1 <= big_jumps <= 6:
        return "steps"
    return "drift"
