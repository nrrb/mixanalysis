# DJ Mix Flow Analyzer

Proof-of-concept Streamlit app for analyzing DJ mix audio as windowed features.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The current proof of concept supports uploading one or more MP3, WAV, or M4A files, then produces labeled timelines, event candidates, plain-English summaries, and debug feature tables.

## Current scope

Implemented:

- Streamlit upload UI.
- Local uploaded-file cache.
- Audio loading with `librosa`.
- Windowed feature extraction with `pandas`.
- Plain-English pressure labels.
- Relief labels and possible-vocal proxy labels.
- Deterministic mix summaries and learning notes.
- Event detection for drops, relief, buildups, possible vocals, transitions, and sustained pressure.
- Plotly flow maps, pressure silhouettes, and layered presence maps.
- Multi-mix comparison strips and comparison summaries.
- Debug data tab for inspecting extracted features.

Not implemented yet:

- Phase 6 refinements beyond the initial comparison view.
- Polish items such as caching, screenshots, and broader error handling.
