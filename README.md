# DJ Mix Flow Analyzer

Proof-of-concept Streamlit app for analyzing DJ mix audio as windowed features.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first version supports uploading one or more MP3, WAV, or M4A files and displays a debug feature table for each analyzed mix.

## Current scope

Implemented:

- Streamlit upload UI.
- Local uploaded-file cache.
- Audio loading with `librosa`.
- Windowed feature extraction with `pandas`.
- Debug data tab for inspecting extracted features.

Not implemented yet:

- Plain-English pressure labels.
- Event detection.
- Visual timelines.
- Multi-mix comparison summaries.
