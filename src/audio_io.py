from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np


def save_uploaded_file(uploaded_file, cache_dir: Path) -> Path:
    """Save Streamlit UploadedFile to a temporary local path and return the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(uploaded_file.name).name
    suffix = Path(source_name).suffix
    stem = Path(source_name).stem or "uploaded_mix"
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    path = cache_dir / f"{safe_stem}_{uuid4().hex[:8]}{suffix}"

    path.write_bytes(uploaded_file.getbuffer())
    return path


def load_audio(path: Path, sr: int = 22050, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio using librosa and return waveform and sample rate."""
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
    return y, int(loaded_sr)
