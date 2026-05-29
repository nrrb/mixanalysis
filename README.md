# DJ Mix Flow Analyzer

Proof-of-concept Streamlit app for analyzing DJ mix audio as windowed features.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The current proof of concept supports uploading one or more MP3, WAV, or M4A files, then produces labeled timelines, event candidates, plain-English summaries, and debug feature tables.

On the Upload tab you can label each file as a **goal-level mix** (a reference you want to sound more like) or an **aspiring mix** (one of your own mixes). When exactly one goal mix is labeled, the Compare Mixes tab adds plain-English suggestions for how each aspiring mix could change to flow more like the goal.

The Visual Flow and Compare Mixes tabs can also export the visuals as **shareable social-media images**. Click **Prepare shareable images** to render each visual flow and comparison view as high-resolution PNGs in both **square (1:1)** and **vertical (9:16)** ratios, then download them with the per-image buttons. The images use a dark, vibrant color theme with a baked-in title, headline, and watermark so they read well on their own in a feed.

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
- Plotly flow maps, pressure silhouettes, and layered presence maps, styled with a consistent dark, vibrant color theme.
- Shareable social-media image export (square 1:1 and vertical 9:16 PNGs) for every visual flow and comparison view, powered by `kaleido`.
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
- Image export covers static PNGs only; animated or video versions of the shareable images are not included in this local POC.
