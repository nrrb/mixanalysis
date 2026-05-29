# DJ Mix Flow Analyzer

Proof-of-concept Streamlit app for analyzing DJ mix audio as windowed features.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The current proof of concept supports uploading one or more MP3, WAV, or M4A files, then produces labeled timelines, event candidates, plain-English summaries, and debug feature tables.

On the Upload tab you can label each file as a **goal-level mix** (a reference you want to sound more like) or an **aspiring mix** (one of your own mixes). When exactly one goal mix is labeled, the Compare Mixes tab adds plain-English suggestions for how each aspiring mix could change to flow more like the goal.

Feature extraction has been optimized so a typical 38-minute mix analyzes end-to-end in a few seconds (roughly 7x faster overall, ~12x faster in the analysis stage) without changing the resulting windows or labels. See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for benchmarks and the reasoning behind each change.

## Current scope

Implemented:

- Streamlit upload UI.
- Local uploaded-file cache.
- Audio loading with `librosa`.
- Windowed feature extraction with `pandas`, built on a single shared STFT for speed.
- Plain-English pressure labels.
- Relief labels and possible-vocal proxy labels.
- Deterministic mix summaries and learning notes.
- Event detection for drops, relief, buildups, possible vocals, transitions, and sustained pressure.
- Plotly flow maps, pressure silhouettes, and layered presence maps.
- Multi-mix comparison strips and comparison summaries.
- Goal-level vs. aspiring mix labeling, with plain-English suggestions for closing the gap to a chosen goal mix.
- Mix character cards for comparing pressure timing, relief, possible vocals, transitions, and sustained pressure.
- Cached analysis for repeated runs with the same file and settings.
- Optimized analysis pipeline (shared STFT, coarser internal hop, `chroma_stft`, `soxr_lq` decode).
- Basic unsupported-file and empty-audio error handling.
- Clear warnings that analysis labels are estimates.
- Debug data tab for inspecting extracted features.

Still intentionally limited:

- Event labels are rule-based estimates, not authoritative music metadata.
- Possible-vocal detection uses a weak full-mix proxy rather than source separation.
- Future screenshots and export features are not included in this local POC.
