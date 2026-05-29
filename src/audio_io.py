from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np


def save_uploaded_file(uploaded_file, cache_dir: Path) -> Path:
    """Save Streamlit UploadedFile to a temporary local path and return the path."""
    return save_uploaded_bytes(uploaded_file.name, uploaded_file.getbuffer(), cache_dir)


def save_uploaded_bytes(file_name: str, data: bytes, cache_dir: Path) -> Path:
    """Save uploaded audio bytes to a temporary local path and return the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(file_name).name
    suffix = Path(source_name).suffix
    stem = Path(source_name).stem or "uploaded_mix"
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    path = cache_dir / f"{safe_stem}_{uuid4().hex[:8]}{suffix}"

    path.write_bytes(bytes(data))
    return path


def load_audio(
    path: Path,
    sr: int = 22050,
    mono: bool = True,
    res_type: str = "soxr_lq",
) -> tuple[np.ndarray, int]:
    """Load audio using librosa and return waveform and sample rate.

    Decoding/resampling is the single largest cost per file on a long mix. The
    default target sr=22050 from 44.1kHz sources is an exact 2:1 ratio that the
    "soxr_lq" resampler handles much faster than the librosa default "soxr_hq",
    with no meaningful effect on the coarse features we extract. Keep sr at a
    clean divisor of 44100 (e.g. 22050) — sr=16000 is both slower to resample and
    shifts features. See OPTIMIZATIONS.md.
    """
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono, res_type=res_type)
    return y, int(loaded_sr)
