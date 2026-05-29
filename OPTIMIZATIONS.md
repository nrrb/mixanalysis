# OPTIMIZATIONS.md

Running log of performance investigations for the DJ Mix Flow Analyzer.

The audio analysis is the dominant cost. This document records what was measured,
what the choke points are, what changes are recommended, and what has actually
landed. Append new investigations to the **Investigation Log** at the bottom as
they happen — keep the **Findings** and **Recommendations** sections updated to
reflect the current best understanding.

---

## TL;DR

- All meaningful cost lives in `src/features.py` (feature extraction) and
  `src/audio_io.py` (decode). `app.py` and the visuals are negligible.
- The two structural problems are **redundant spectral transforms** (5–6 full
  STFT/CQT passes over the whole signal) and **massive oversampling** (frame
  features every ~23 ms that get averaged into 5–10 s output windows).
- Tier 1 algorithmic fixes (shared STFT + `chroma_stft` + coarser hop) are
  expected to give **~5–10× faster** feature extraction with no new dependencies
  and no infra. They are contained entirely within `src/features.py`.
- Files are analyzed **sequentially** in `app.py`; each file is independent and
  trivially parallelizable across processes.

---

## Pipeline overview

Per uploaded file, `app.py::_analyze_uploaded_bytes` runs:

1. `save_uploaded_bytes` — write bytes to `outputs/cache/` (cheap).
2. `load_audio` — `librosa.load(path, sr=22050, mono=True)` decodes + resamples
   the **entire** mix.
3. `analyze_audio_windows` — computes frame-level features over the whole signal,
   then aggregates into output windows. **This is the hot path.**
4. `assign_pressure_labels` / `detect_events` / `summarize_mix` — operate on the
   small per-window DataFrame (cheap).

Caching: `@st.cache_data` keys on the file bytes + parameters, so re-analyzing
the same input is instant. Optimizations below matter for the **first** analysis
of each input.

---

## Benchmark methodology

Synthetic signal, 10 minutes @ 22050 Hz mono (`13,230,000` samples), librosa
0.11.0 / NumPy 2.4.6, single thread. Multiply timings by ~6 for a typical 60-min
mix. Each op was warmed once before timing to exclude import/JIT overhead.
Benchmark script lives at `/tmp/bench.py` (synthetic; regenerate as needed —
see Investigation Log 2026-05-28).

> Numbers are relative guides, not guarantees. Real MP3 decode time is not
> included (no sample file was available); see Open Questions.

---

## Findings — choke points (current code, `hop_length=512`)

Measured on the 10-minute signal:

| Operation                       | 10-min | Notes |
|---------------------------------|-------:|-------|
| `chroma_cqt`                    | 1.46s  | **Slowest single op** — Constant-Q transform |
| `bass STFT n_fft=4096`          | 0.46s  | A full extra STFT just for bass band |
| `spectral_bandwidth`            | 0.46s  | Internally computes its **own** STFT |
| `global tempo`                  | 0.41s  | Recomputes the onset envelope from scratch |
| `spectral_flatness`             | 0.36s  | Internally computes its own STFT |
| `spectral_centroid`             | 0.33s  | Internally computes its own STFT |
| `onset_strength`                | 0.27s  | Its own mel-spectrogram |
| `rms`                           | 0.02s  | cheap |
| **Total feature extraction**    | **~3.8s / 10 min → ~23s / 60 min** | per file, single-threaded |

Two root causes:

1. **Redundant transforms.** `spectral_centroid`, `spectral_bandwidth`, and
   `spectral_flatness` each compute a separate STFT internally; `_compute_bass_features`
   computes another (`n_fft=4096`); `onset_strength` builds a mel-spectrogram;
   `chroma_cqt` does a CQT. That is **5–6 full-signal spectral transforms** where
   a single STFT could feed nearly all of them.
   (`src/features.py:22-31`, `src/features.py:88-100`)

2. **Oversampling.** Frame features are computed every `hop_length=512` samples
   (~23 ms) and then averaged into 5–10 s output windows
   (`src/features.py:64-82`). That is ~430 frames per output window — the
   resolution is discarded. A 4× coarser hop has no visible effect on aggregates.

---

## Recommendations (prioritized)

### Tier 1 — algorithmic (no infra, ~5–10×, all in `src/features.py`)

Measured optimized path on the 10-minute signal:

| Optimized op                         | 10-min |
|--------------------------------------|-------:|
| ONE `|STFT|` (`n_fft=2048`)          | 0.20s  |
| `spectral_centroid(S=S)`             | 0.10s  |
| `spectral_bandwidth(S=S)`            | 0.23s  |
| `spectral_flatness(S=S)`             | 0.13s  |
| `rms(S=S)`                           | 0.01s  |
| `chroma_stft(S=S**2)` (replaces CQT) | 0.44s  |
| bass: mask `S` rows + sum            | ~0     |
| `tempo(onset_envelope=onset_env)`    | 0.15s  |

- **[Task #2] One shared STFT.** Compute `S = np.abs(librosa.stft(y, ...))` once
  and pass `S=S` to `spectral_centroid`/`spectral_bandwidth`/`spectral_flatness`/`rms`.
  Derive bass from the same `S` (`S[low_mask].sum(0) / S.sum(0)`), removing the
  dedicated `n_fft=4096` STFT. Decide whether to standardize on `n_fft=2048`
  (faster) or `4096` (better 20–160 Hz bass resolution — bin width 5.4 Hz vs
  10.8 Hz); one shared transform either way.
- **[Task #3] `chroma_stft` instead of `chroma_cqt`.** Kills the single biggest
  cost (1.46s → 0.44s). CQT precision is overkill for the scalar
  `chroma_stability` metric in a POC. Note `chroma_stft` wants a power
  spectrogram (`S**2`).
- **[Task #4] Coarsen `hop_length`** (512 → 1024 or 2048). Cuts frame count ~4×
  and scales nearly every op down with it (e.g. STFT 0.46s → 0.05s; chroma_cqt
  1.46s → 0.90s even without the other changes). Keep the user-facing
  window/hop sliders untouched — this is the **internal** frame resolution only.
- **[Task #5] Reuse the onset envelope for tempo.** `_estimate_global_tempo`
  recomputes it; pass `onset_envelope=onset_env`
  (`src/features.py:103-111`, called at `src/features.py:55`). 0.41s → 0.15s.

Combined expected effect: ~3.8s → well under 1s per 10 min (**~5–10×**).

### Tier 2 — decode (`src/audio_io.py`)

- **[Task #6] Faster resample / lower sr.** `librosa.load(..., sr=22050)`
  resamples from 44.1k with the high-quality `soxr_hq` resampler, slow on
  hour-long files. Use `res_type="soxr_lq"` or `"kaiser_fast"`. Consider
  `sr=16000` — the highest feature of interest (spectral centroid, bass) sits
  well under 8 kHz. (`src/audio_io.py:27-30`)
- **Non-Python worker:** for long MP3s, decoding via an `ffmpeg` subprocess to a
  raw PCM stream is typically faster than librosa's audioread path.

### Tier 3 — parallelism

- **[Task #7] Per-file parallelism.** `app.py:209-224` analyzes uploaded files
  sequentially; each file is independent. Wrap `_analyze_uploaded_bytes` in a
  `concurrent.futures.ProcessPoolExecutor` (processes, not threads — avoids
  Streamlit threading quirks; NumPy/FFT release the GIL but processes are
  simpler). N files → ~N× on multicore.
- **Single-file chunking (optional).** For one very long mix, chunk the waveform
  and analyze chunks in parallel. Frame features are window-local; the only truly
  global values are the onset percentile threshold (`src/features.py:56`) and
  `global_tempo` — compute those once cheaply, then fan out the windows. More
  wiring than per-file parallelism; do it only if single huge files dominate.

> Caveat: `@st.cache_data` already makes re-analysis instant, so parallelism
> mainly helps the **first** analysis of a batch.

### Suggested order

Tier 1 first (highest leverage, lowest risk, no new deps) → Tier 2 decode →
Tier 3 only if batching many/large files is common.

---

## Status tracker

Mirrors the in-session task list. Update as work lands.

| # | Item | Tier | Status |
|---|------|------|--------|
| 1 | Profile pipeline / find choke points          | —  | ✅ Done (2026-05-28) |
| 2 | One shared STFT, derive spectral features      | 1  | ✅ Done (2026-05-28) |
| 3 | Replace `chroma_cqt` with `chroma_stft`        | 1  | ✅ Done (2026-05-28) |
| 4 | Coarsen `hop_length` (512 → 1024)              | 1  | ✅ Done (2026-05-28) |
| 5 | Reuse onset envelope for tempo                  | 1  | ✅ Done (2026-05-28) |
| 6 | Faster decode/resample (`soxr_lq` @ 22050)     | 2  | ✅ Done (2026-05-28) |
| 7 | Parallelize across files / chunks (processes)  | 3  | ⬜ Pending |
| 8 | Vectorize per-window aggregation (searchsorted)| 1  | ✅ Done (2026-05-28) |
| 9 | Pin `chroma_stft(tuning=0.0)`                   | 1  | ✅ Done (2026-05-28) |

**Headline result — real 38-min mix (`sample_data/2026-05-27.mp3`):**
end-to-end **40.76s → 5.73s (7.1×)**; the analysis stage alone **36.65s → ~3s
(~12×)**. Same 461 output windows, and `local_tempo` went from `nan` (silently
broken — see below) to a real 161.5 BPM. Settled config in `src/features.py`:
`ANALYSIS_HOP_LENGTH=1024`, `ANALYSIS_N_FFT=2048`, shared STFT, `chroma_stft`
with `tuning=0.0`; `src/audio_io.py`: `res_type="soxr_lq"` at sr=22050.

---

## Quality vs baseline (real file, measured 2026-05-28)

Validated on `sample_data/2026-05-27.mp3` (38.3 min). All variants produce the
same 461 windows. Structural agreement is strong: **32/35 of the old events have
a new event within 5s**, `spectral_centroid` mean is identical, `bass_pressure`
within ~5%.

The visible change is reshuffling between the two **highest** pressure bands —
e.g. `peak`/`full` ≈ 43/116 (old) → 28/142 (new). Combined high-pressure
coverage is stable (~159 → ~170 windows); the `peak`↔`full` boundary just moved.
Cause is the coarser hop changing onset-density resolution, **not** the bass or
chroma changes. Judged acceptable for an explicitly estimate-based POC. If a
future need requires tighter fidelity, lowering `ANALYSIS_HOP_LENGTH` back toward
512 restores it — at a steep, memory-bandwidth-bound cost (see below).

`possible vocal section` count rose modestly (3 → 8–9) from the `chroma_cqt` →
`chroma_stft` swap feeding the weak vocal proxy. Acceptable; revisit if vocal
detection is upgraded (Demucs, per PLAN.md future work).

## Approaches tried and rejected

- **`n_fft=4096` for the shared STFT.** Restores `bass_pressure` *exactly* to the
  old value, but costs ~3.6× the analyze time (4.8s → 17.5s on the real file) and
  the peak/full label reshuffle persists anyway (it's hop-driven, not
  bass-resolution-driven). Not worth it. Kept `n_fft=2048`.
- **Deriving the onset envelope from the shared STFT** (mel-from-`power` instead
  of letting `onset_strength` recompute its own). 10× faster for that op, but the
  resulting envelope **anti-correlated** with the original (corr ≈ −0.22), which
  would corrupt onset density and tempo. Rejected — kept `onset_strength(y=...)`.
  It's only ~0.6s at hop=1024 anyway.
- **`sr=16000`.** Lose-lose: 44100→16000 is not a clean ratio, so it's *slower*
  to resample (3.7s vs 2.4s for 22050) **and** shifts features (centroid
  2617→1956, bass 0.229→0.255). Kept sr=22050 (exact 2:1 from 44.1kHz).
- **Native sr / no resample.** Decode is fast (2.2s) but yields 2× the samples
  (101M @ 44.1kHz), making every downstream feature ~2× slower. Rejected.

## Gotchas worth remembering

- **Synthetic audio badly underestimates real cost** (~4×). The initial
  microbenchmarks used sine+noise; on real music, dense spectra make
  `chroma_stft`, `spectral_bandwidth`, and `onset_strength` far more expensive.
  Always validate timing on a real mix.
- **`chroma_stft` estimates tuning per call** unless you pass `tuning=`. That
  hidden `estimate_tuning` cost was ~1.3s of a ~4s analyze. `tuning=0.0` cuts it
  to ~0.02s; downstream `chroma_stability` mean moved by 0.0003 (negligible).
- **Memory-bandwidth bound at fine hops.** The `|STFT|`/`power` arrays are large
  (≈406MB at hop=512, ≈203MB at hop=1024). Halving the hop *more than* halves
  runtime (hop=512 → 27s vs hop=1024 → 4.3s, a ~6× gap for 2× the frames)
  because the larger arrays thrash cache. This is why the hop is the single
  biggest lever.
- **The windowing loop was *not* the bottleneck.** Vectorizing it with
  `searchsorted` (task #8) is correct and kept (clean, avoids O(windows×frames)
  boolean masks, helps at fine hops), but profiling showed feature extraction —
  not the loop — dominates. Diagnose by profiling, not by hypothesis.
- **`local_tempo` was silently `nan` in the old code.** `librosa.feature.rhythm`
  is lazy-loaded; the old `librosa.feature.rhythm.tempo(...)` access raised
  `AttributeError`, was swallowed by the `except`, and returned `nan` every time.
  Adding `import librosa.feature.rhythm` at the top of `src/features.py` fixed it
  as a side effect — tempo now reports a real BPM.

## Open questions / not yet measured

- **Tier 3 parallelism (task #7).** Not yet implemented. With single-file
  analysis now ~6s end-to-end, per-file `ProcessPoolExecutor` in `app.py` only
  matters for multi-file batches. Measure batch wall-clock before investing.
- **Streamlit overhead.** End-to-end numbers here are the analysis pipeline only;
  the `@st.cache_data` path and UI rendering weren't separately profiled (they
  were never suspected as hot).
- **Decode lower bound.** `soxr_lq` @ 22050 = ~2.4s. An `ffmpeg` subprocess
  decode could go lower but adds a non-Python dependency; only pursue if decode
  becomes the dominant cost again after Tier 3.

---

## Investigation Log

### 2026-05-28 — Initial profiling
- Read `app.py`, `src/features.py`, `src/audio_io.py`, `PLAN.md`.
- Benchmarked each feature op on a 10-min synthetic signal (librosa 0.11.0,
  NumPy 2.4.6, single thread) — see Findings table.
- Identified redundant transforms and oversampling as the two root causes.
- Measured the optimized shared-STFT / `chroma_stft` / coarse-hop path —
  see Recommendations Tier 1 table.
- Created tasks #2–#7. No code changes landed yet (analysis only).

### 2026-05-28 — Tier 1 implemented (tasks #2–#5)
- Rewrote `analyze_audio_windows` (`src/features.py`) to compute one `|STFT|`
  (`n_fft=2048`, `hop=1024`) and derive `rms`, `spectral_centroid`,
  `spectral_bandwidth`, `spectral_flatness`, `chroma_stft`, and the bass band
  from it. Removed the dedicated `n_fft=4096` bass STFT and the per-feature
  internal transforms.
- Swapped `chroma_cqt` → `chroma_stft(S=magnitude**2)`. Column names unchanged
  (`chroma_00..11`, `chroma_stability`), so downstream labels/events/visuals are
  untouched.
- `_estimate_global_tempo` now takes the precomputed `onset_env`
  (`onset_envelope=`) instead of recomputing it.
- Introduced module constants `ANALYSIS_HOP_LENGTH=1024`, `ANALYSIS_N_FFT=2048`.
- **Verified:** end-to-end `analyze_audio_windows` 4.09s → 0.82s on 10-min
  synthetic audio (git-stash before/after), 120 windows each, no NaNs.
- Still pending: real-MP3 before/after wall-clock, and a quality diff vs the
  `chroma_cqt`/`hop=512` baseline on a real mix (see Open Questions).

### 2026-05-28 — Real-file validation + Tier 2 + tasks #8/#9
Validated everything on `sample_data/2026-05-27.mp3` (38.3 min, 92 MB) with
git-stash before/after.
- **End-to-end 40.76s → 5.73s (7.1×)**; analyze stage 36.65s → ~3s (~12×); same
  461 windows. Found and fixed the silent `local_tempo == nan` bug (lazy import).
- Profiled per stage on the real file (hop=512): chroma_stft 6.79s, bandwidth
  3.85s, onset 3.78s, centroid 1.96s, stft 1.36s — i.e. synthetic benchmarks had
  underestimated real cost ~4×. Feature extraction, not the windowing loop, is
  the cost.
- **Task #8 — searchsorted windowing** (`src/features.py`): replaced the
  per-window boolean mask with `np.searchsorted` index slices; verified
  byte-identical frame selection. Kept despite not being the bottleneck.
- **Task #9 — `chroma_stft(tuning=0.0)`**: removed the hidden per-call tuning
  estimation (~1.3s → ~0.02s); downstream impact negligible (chroma_stability
  mean Δ0.0003, +1 vocal event).
- **Task #6 — decode** (`src/audio_io.py`): `res_type="soxr_lq"` at sr=22050
  (3.66s → 2.37s). Rejected `sr=16000` (slower, non-clean ratio) and native sr
  (2× downstream samples). Also fixed a latent bug: `rms(S=...)` needs an explicit
  `frame_length=n_fft` or it breaks for any `n_fft != 2048`.
- Quality vs baseline characterized (see "Quality vs baseline" section): 32/35
  events align within 5s; only the peak↔full band boundary shifts. Rejected
  `n_fft=4096` and onset-reuse (see "Approaches tried and rejected").
- Remaining: task #7 (parallelism) — deferred; single-file is now fast enough
  that it only helps multi-file batches.
