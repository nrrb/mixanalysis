import numpy as np
import pandas as pd
import librosa


def analyze_audio_windows(
    y: np.ndarray,
    sr: int,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
) -> pd.DataFrame:
    """Return one row per analysis window with internal numeric features."""
    if y.size == 0:
        return pd.DataFrame()
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("Window and hop sizes must be positive.")

    hop_length = 512
    n_fft = 4096
    duration = len(y) / sr

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    low_frequency_energy, bass_pressure = _compute_bass_features(
        y, sr, n_fft=n_fft, hop_length=hop_length
    )
    frame_count = min(
        len(rms),
        len(onset_env),
        len(centroid),
        len(bandwidth),
        len(flatness),
        chroma.shape[1],
        len(low_frequency_energy),
        len(bass_pressure),
    )

    rms = rms[:frame_count]
    onset_env = onset_env[:frame_count]
    centroid = centroid[:frame_count]
    bandwidth = bandwidth[:frame_count]
    flatness = flatness[:frame_count]
    chroma = chroma[:, :frame_count]
    low_frequency_energy = low_frequency_energy[:frame_count]
    bass_pressure = bass_pressure[:frame_count]
    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=sr, hop_length=hop_length
    )

    global_tempo = _estimate_global_tempo(y, sr, hop_length)
    onset_threshold = np.percentile(onset_env, 75) if onset_env.size else 0.0

    starts = np.arange(0.0, max(duration, hop_seconds), hop_seconds)
    rows = []
    for start in starts:
        if start >= duration:
            break
        end = min(start + window_seconds, duration)
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            continue

        window_chroma = chroma[:, mask]
        row = {
            "start_time": float(start),
            "end_time": float(end),
            "rms_energy": _safe_mean(rms[mask]),
            "low_frequency_energy": _safe_mean(low_frequency_energy[mask]),
            "bass_pressure_ratio": _safe_mean(bass_pressure[mask]),
            "onset_strength": _safe_mean(onset_env[mask]),
            "onset_density": _onset_density(onset_env[mask], onset_threshold, end - start),
            "spectral_centroid": _safe_mean(centroid[mask]),
            "spectral_bandwidth": _safe_mean(bandwidth[mask]),
            "spectral_flatness": _safe_mean(flatness[mask]),
            "local_tempo": global_tempo,
        }
        row.update(_summarize_chroma(window_chroma))
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_bass_features(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 20) & (freqs <= 160)

    low_energy = stft[low_mask, :].sum(axis=0)
    total_energy = stft.sum(axis=0) + 1e-9
    return low_energy, low_energy / total_energy


def _estimate_global_tempo(y: np.ndarray, sr: int, hop_length: int) -> float:
    try:
        tempo = librosa.feature.rhythm.tempo(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        return float("nan")

    if np.ndim(tempo) == 0:
        return float(tempo)
    return float(tempo[0]) if len(tempo) else float("nan")


def _summarize_chroma(window_chroma: np.ndarray) -> dict[str, float]:
    means = window_chroma.mean(axis=1) if window_chroma.size else np.zeros(12)
    chroma_sum = float(means.sum())
    if chroma_sum > 0:
        normalized = means / chroma_sum
        stability = float(normalized.max())
    else:
        stability = 0.0

    result = {f"chroma_{index:02d}": float(value) for index, value in enumerate(means)}
    result["chroma_stability"] = stability
    return result


def _onset_density(onset_values: np.ndarray, threshold: float, seconds: float) -> float:
    if seconds <= 0 or onset_values.size == 0:
        return 0.0
    return float(np.count_nonzero(onset_values >= threshold) / seconds)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))
