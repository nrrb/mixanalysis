# PLAN.md: DJ Mix Flow Analyzer Proof of Concept

## Goal

Build a proof-of-concept Python + Streamlit app that accepts one or more DJ mix MP3 files and produces an intuitive, non-numeric analysis of each mix.

The app should help a novice DJ:

- Compare multiple DJ mixes visually.
- Find key moments such as drops, relief sections, vocal sections, buildups, and likely transitions.
- Understand the flow of a mix in plain English.
- See the mix as a timeline of pressure, relief, vocals, and structural changes.
- Avoid requiring the user to interpret raw numerical audio features.

The internal analysis can use numeric features, but the user-facing output should use labels, summaries, annotations, colored timelines, and readable explanations.

## Non-goals for the proof of concept

Do not try to build a perfect music-analysis engine. This is a proof of concept.

Avoid these for the first version:

- Exact track identification.
- Spotify, Beatport, Rekordbox, or Serato integration.
- Perfect key detection.
- Perfect downbeat detection.
- Full source separation by default.
- Real-time analysis.
- A production-ready database.
- User accounts.
- Cloud deployment.
- Complex machine learning training.

The first version should be local, simple, inspectable, and useful enough to study a mix.

## Recommended stack

Use Python for the analysis engine and Streamlit for the interface.

Suggested libraries:

```txt
streamlit
librosa
numpy
pandas
scipy
plotly
soundfile
pydub
python-dotenv
```

Optional later libraries:

```txt
demucs
essentia
madmom
scikit-learn
```

Notes:

- Use `librosa` for MP3 loading, beat tracking, onset strength, RMS, spectral features, chroma, and time conversion.
- Use `pandas` for windowed feature tables.
- Use `plotly` for interactive timelines and visual flow maps.
- Use `streamlit` for file upload, controls, charts, summaries, and comparison pages.
- Use `demucs` later if vocal detection from the full mix is too weak.

## Project structure

Create this structure:

```txt
dj-mix-flow-analyzer/
  README.md
  PLAN.md
  requirements.txt
  app.py
  src/
    __init__.py
    audio_io.py
    features.py
    events.py
    summaries.py
    visuals.py
    compare.py
    utils.py
  outputs/
    .gitkeep
  sample_data/
    .gitkeep
```

## User workflow

The user should be able to run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:

1. Upload one or more MP3 files.
2. For each uploaded file, label it as either a **goal-level mix** (a reference mix the user is aspiring toward) or an **aspiring mix** (one of the user's own mixes being studied). Exactly one goal-level mix is expected per session.
3. Click an analyze button.
4. View a plain-English summary for each mix.
5. View an annotated visual timeline for each mix.
6. View a list of key moments.
7. If a goal-level mix and at least one aspiring mix are present, view a comparison page that contrasts each aspiring mix against the goal-level mix and offers plain-English suggestions for closing the gap.

## Proof-of-concept UI requirements

### Streamlit sidebar

Include controls for:

- Window size in seconds: default 10.
- Hop size in seconds: default 5.
- Sensitivity: `conservative`, `balanced`, `sensitive`.
- Minimum event spacing: default 30 seconds.
- Analysis mode: `fast` only for POC.

### Main app pages or sections

Use Streamlit tabs:

1. `Upload`
2. `Mix Report`
3. `Key Moments`
4. `Visual Flow`
5. `Compare Mixes`
6. `Debug Data`, collapsed or optional

The `Debug Data` tab may show numbers for development, but the main user experience should avoid numeric interpretation.

### Mix role labeling (Upload tab)

After files are uploaded but before the analyze button is pressed, the `Upload` tab must let the user assign a **role** to each file:

- `goal-level mix` — a reference mix the user is trying to sound more like.
- `aspiring mix` — one of the user's own mixes being studied against the goal.

Requirements:

- Render one role control per uploaded file, for example a `st.radio` or `st.selectbox` keyed by the file name, defaulting to `aspiring mix`.
- Encourage exactly one `goal-level mix`. If zero or more than one are selected, show a non-blocking warning (`st.warning`) but still allow analysis so the per-mix reports remain useful.
- Persist the chosen role alongside each result so downstream tabs (especially `Compare Mixes`) can use it.
- Add a `help=` tooltip explaining the difference between the two roles, consistent with the Phase 8 contextual-help convention.

## Audio loading

Create `src/audio_io.py`.

Functions:

```python
def save_uploaded_file(uploaded_file, cache_dir: Path) -> Path:
    """Save Streamlit UploadedFile to a temporary local path and return the path."""


def load_audio(path: Path, sr: int = 22050, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio using librosa and return waveform and sample rate."""
```

Implementation notes:

- Streamlit uploaded files are file-like objects, so save them to a temporary file before passing to librosa if needed.
- Use `librosa.load` for decoding.
- Use mono audio for the POC.
- Downsample to 22050 Hz for speed. This is an exact 2:1 ratio from 44.1 kHz source audio, so decode with `res_type="soxr_lq"` for a fast, good-enough resample.
- Long DJ mixes used to be slow; after the Phase 9 optimizations a 38-minute mix analyzes in a few seconds. Still add a clear spinner/status message while analyzing.

## Feature extraction

Create `src/features.py`.

The app should compute numeric features internally, then convert them to categories.

### Required internal features

For each time window:

- Start time.
- End time.
- RMS energy.
- Low-frequency energy.
- Bass pressure ratio.
- Onset strength.
- Onset density.
- Spectral centroid.
- Spectral bandwidth.
- Spectral flatness.
- Chroma vector.
- Local tempo estimate, if feasible.

### Feature extraction function

```python
def analyze_audio_windows(
    y: np.ndarray,
    sr: int,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
) -> pd.DataFrame:
    """Return one row per analysis window with internal numeric features."""
```

### Implementation approach

Use librosa to compute frame-level features first, then aggregate into larger windows.

> **Note (Phase 9):** the original approach below let each feature recompute its
> own spectral transform, which was the dominant cost. The shipped implementation
> computes a single `|STFT|` once and derives rms/centroid/bandwidth/flatness/
> chroma/bass from it, uses `chroma_stft(tuning=0.0)` instead of `chroma_cqt`,
> coarsens the internal `hop_length` to 1024, and reuses the precomputed onset
> envelope for the tempo estimate. See [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

Suggested frame-level features (original, pre-optimization sketch):

```python
rms = librosa.feature.rms(y=y)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
flatness = librosa.feature.spectral_flatness(y=y)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
```

For bass pressure:

- Compute an STFT magnitude spectrogram.
- Sum energy in a low-frequency band, for example 20-160 Hz.
- Divide by total spectral energy.

Pseudo-code:

```python
S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
low_mask = (freqs >= 20) & (freqs <= 160)
low_energy = S[low_mask, :].sum(axis=0)
total_energy = S.sum(axis=0) + 1e-9
bass_pressure = low_energy / total_energy
```

For local tempo:

- Start with a global tempo estimate.
- For POC, local BPM can be optional or approximate.
- If local tempo is unstable, display `Tempo appears mostly stable` rather than exact BPM.

## Convert numbers into categories

Create `src/summaries.py` and `src/events.py`.

The user-facing app should use plain-English labels.

### Pressure labels

Create a function:

```python
def assign_pressure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a pressure_label column based on normalized energy, bass, brightness, and onset density."""
```

Use percentile-based thresholds within each mix so the tool adapts to different mastering levels.

Suggested labels:

- `low pressure`
- `groove / cruising`
- `building pressure`
- `full pressure`
- `peak pressure`

Do not show the calculated score as the primary output.

### Relief labels

Create a `relief_type` column based on feature drops relative to the rolling average.

Suggested labels:

- `breathing room`
- `kick/bass pullback`
- `density drop`
- `quiet reset`
- `no clear relief`

### Vocal proxy for POC

For the proof of concept, do not require source separation.

Use a weak proxy:

- Increased midrange energy.
- Chroma stability.
- Reduced bass pressure.
- Spectral features consistent with voice-like content.

Label this as `possible vocal section`, not `vocal section`.

Later, add Demucs or a vocal activity detector.

### Key and harmonic movement

For POC:

- Compute chroma features.
- Estimate a rough key label only if confidence seems adequate.
- Otherwise show `tonal center unclear`.

Avoid presenting key labels as authoritative.

## Event detection

Create `src/events.py`.

Detect these event types:

1. Drop candidate.
2. Relief section.
3. Buildup candidate.
4. Possible vocal section.
5. Likely transition.
6. Sustained pressure run.

Each event should have:

```python
@dataclass
class MixEvent:
    start: float
    end: float | None
    event_type: str
    title: str
    description: str
    confidence: str  # low, medium, high
```

### Drop detection

A drop candidate is a window where pressure rises sharply after a lower-pressure region.

Rule-of-thumb:

- Current pressure is high or peak.
- Previous 2-6 windows had lower pressure.
- Bass pressure increases.
- Onset density increases.
- RMS energy increases.

User-facing label examples:

- `Major drop candidate`
- `Kick and bass return`
- `Pressure jump`

Example description:

> The mix appears to move from a reduced-pressure section into a heavier, denser section here.

### Relief detection

A relief section is a run of windows where pressure, bass, or density drops below the local average.

User-facing label examples:

- `Breathing room`
- `Bass pullback`
- `Reduced-density section`
- `Reset before next push`

Example description:

> The low end and rhythmic density pull back here, creating a short reset before the next pressure section.

### Buildup detection

A buildup candidate is a run of windows with rising pressure leading into a drop or peak.

User-facing label examples:

- `Building pressure`
- `Rising tension`
- `Pre-drop lift`

Example description:

> Several features rise together across this section, suggesting a buildup into the next major moment.

### Transition detection

A likely transition is a region where timbre, bass profile, and chroma change while rhythm continues.

User-facing label examples:

- `Likely transition`
- `Long blend candidate`
- `Abrupt switch candidate`

For POC:

- Detect large changes in spectral centroid, flatness, bass pressure, and chroma distance.
- If change happens over several windows, label `long blend candidate`.
- If change happens in one window, label `abrupt switch candidate`.

## Plain-English summary generation

Create `src/summaries.py`.

Function:

```python
def summarize_mix(df: pd.DataFrame, events: list[MixEvent], duration: float) -> dict:
    """Return a plain-English summary, tags, and learning notes."""
```

Return structure:

```python
{
  "headline": "Wave-shaped hard-driving mix with recurring relief pockets",
  "summary": "This mix alternates between full-pressure sections and shorter resets...",
  "tags": ["wave-shaped", "vocal-sparse", "frequent relief", "stable tempo"],
  "learning_notes": [
    "Notice how the mix pulls back before major pressure jumps.",
    "The longest high-pressure stretch happens in the final third."
  ]
}
```

### Summary rules

Use deterministic rules, not an LLM, for the first version.

Examples:

- If high-pressure sections are frequent with little relief: `relentless / sustained pressure`.
- If pressure rises gradually across the mix: `slow-building`.
- If pressure repeatedly rises and falls: `wave-shaped`.
- If possible vocal sections are rare: `vocal-sparse`.
- If possible vocal sections cluster around relief or transitions: `vocals used as reset/transition texture`.
- If many transition candidates are long: `gradual blend style`.
- If many transition candidates are abrupt: `cut-heavy style`.

## Visuals

Create `src/visuals.py`.

Use Plotly for interactive visuals inside Streamlit.

### Visual 1: Flow map

Create a horizontal timeline where each window is a colored block labeled by pressure category.

Required:

- X-axis: time.
- Blocks: pressure labels.
- Markers: drops, relief, possible vocals, likely transitions.
- Hover text: plain-English description.

Do not make users read raw feature values.

Function:

```python
def make_flow_map(df: pd.DataFrame, events: list[MixEvent]) -> plotly.graph_objects.Figure:
    """Create an intuitive annotated timeline of the mix."""
```

### Visual 2: Mix silhouette

Create a smooth area chart showing perceived pressure over time.

Use labels and event markers rather than numeric y-axis interpretation.

Function:

```python
def make_pressure_silhouette(df: pd.DataFrame, events: list[MixEvent]) -> plotly.graph_objects.Figure:
    """Create an intuitive shape of the mix's pressure and relief."""
```

Hide or soften numerical axis labels if possible.

### Visual 3: Layered presence map

Create a stacked timeline with lanes:

- Pressure.
- Bass-heavy sections.
- Possible vocals.
- Relief.
- Transitions.

Function:

```python
def make_layered_presence_map(df: pd.DataFrame, events: list[MixEvent]) -> plotly.graph_objects.Figure:
    """Create a lane-based timeline of mix characteristics."""
```

### Visual 4: Multiple mix comparison

If multiple files are uploaded, create one compact timeline strip per mix.

Function:

```python
def make_comparison_strips(results: list[MixAnalysisResult]) -> plotly.graph_objects.Figure:
    """Compare multiple mixes as stacked flow timelines."""
```

Comparison should make these visible:

- Which mix reaches pressure earlier.
- Which mix has more relief.
- Which mix is more vocal-heavy.
- Which mix has longer high-pressure runs.
- Which mix looks more wave-shaped vs relentless.

## Data model

Create a simple result object in `src/compare.py` or `src/utils.py`.

```python
@dataclass
class MixAnalysisResult:
    name: str
    duration: float
    feature_df: pd.DataFrame
    events: list[MixEvent]
    summary: dict
    role: str = "aspiring"  # "goal" or "aspiring"
```

## Streamlit app layout

Create `app.py`.

Pseudo-code:

```python
import streamlit as st
from pathlib import Path

from src.audio_io import save_uploaded_file, load_audio
from src.features import analyze_audio_windows
from src.events import detect_events
from src.summaries import assign_pressure_labels, summarize_mix
from src.visuals import make_flow_map, make_pressure_silhouette, make_layered_presence_map, make_comparison_strips

st.set_page_config(page_title="DJ Mix Flow Analyzer", layout="wide")

st.title("DJ Mix Flow Analyzer")
st.caption("Upload DJ mixes and study their pressure, relief, drops, vocals, and pacing as visual flow maps.")

uploaded_files = st.file_uploader(
    "Upload MP3 mixes",
    type=["mp3", "wav", "m4a"],
    accept_multiple_files=True,
)

# Assign a role to each uploaded file before analysis.
roles = {}
if uploaded_files:
    st.subheader("Label each mix")
    for uploaded_file in uploaded_files:
        roles[uploaded_file.name] = st.radio(
            uploaded_file.name,
            ["aspiring mix", "goal-level mix"],
            index=0,
            horizontal=True,
            key=f"role_{uploaded_file.name}",
            help="Mark the reference mix you want to sound like as the goal-level mix; mark your own mixes as aspiring.",
        )
    goal_count = sum(1 for r in roles.values() if r == "goal-level mix")
    if goal_count != 1:
        st.warning("Pick exactly one goal-level mix to enable gap suggestions. You can still analyze without one.")

with st.sidebar:
    window_seconds = st.slider("Analysis window", 5, 30, 10)
    hop_seconds = st.slider("Timeline detail", 2, 15, 5)
    sensitivity = st.selectbox("Event sensitivity", ["conservative", "balanced", "sensitive"], index=1)
    analyze_clicked = st.button("Analyze mixes")

if uploaded_files and analyze_clicked:
    results = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            path = save_uploaded_file(uploaded_file, Path("outputs/cache"))
            y, sr = load_audio(path)
            df = analyze_audio_windows(y, sr, window_seconds, hop_seconds)
            df = assign_pressure_labels(df)
            events = detect_events(df, sensitivity=sensitivity)
            summary = summarize_mix(df, events, duration=len(y) / sr)
            role = "goal" if roles.get(uploaded_file.name) == "goal-level mix" else "aspiring"
            results.append(MixAnalysisResult(uploaded_file.name, len(y) / sr, df, events, summary, role=role))

    # Store in session state
    st.session_state["results"] = results

if "results" in st.session_state:
    results = st.session_state["results"]
    tabs = st.tabs(["Mix Report", "Key Moments", "Visual Flow", "Compare Mixes", "Debug Data"])
    # Render tabs
```

## Key Moments tab

For each mix, show event cards.

Each card should include:

- Timestamp.
- Event title.
- Plain-English explanation.
- Confidence label.

Example:

```txt
07:12 — Major drop candidate
The mix moves from reduced low-end pressure into a heavier, denser section. This is probably a key payoff moment.
Confidence: medium
```

## Mix Report tab

For each mix, show:

- Headline.
- 1-paragraph summary.
- Tags.
- Learning notes.
- Top 5 key moments.

Example:

```txt
Headline:
Wave-shaped hard-driving mix with recurring relief pockets

Summary:
This mix repeatedly alternates between high-pressure sections and shorter resets. The strongest pressure run appears in the final third, while likely vocal moments cluster around lower-pressure sections.

Tags:
wave-shaped, frequent relief, vocal-sparse, gradual blend style

Learning notes:
- Notice how the mix often pulls back before pressure jumps.
- The most intense sections are separated by short reset pockets rather than one continuous wall of sound.
```

## Compare Mixes tab

Only show if at least two mixes were uploaded.

Include:

- Stacked comparison strips.
- Plain-English comparison summary.
- Mix cards with character tags.

Example comparison text:

```txt
Mix A reaches full pressure earlier and stays more relentless.
Mix B has more frequent relief pockets and a clearer wave shape.
Mix C appears more transition-heavy, with more possible long-blend regions.
```

### Goal vs. aspiring suggestions

When exactly one mix is labeled `goal-level mix`, add a dedicated section that compares each `aspiring` mix against the goal and offers plain-English, actionable suggestions for closing the gap.

Function:

```python
def suggest_toward_goal(goal: MixAnalysisResult, aspiring: MixAnalysisResult) -> dict:
    """Compare one aspiring mix to the goal mix and return plain-English suggestions.

    Returns:
        {
          "headline": "Closer to the goal than most, but the energy peaks too early",
          "suggestions": [
            "The goal mix waits until about two-thirds in before its biggest drop; "
            "yours peaks in the first third. Try holding back your hardest section.",
            "Your goal has roughly twice as many relief pockets. Add a breather "
            "after long high-pressure runs to let the energy breathe.",
          ],
          "matched": [
            "Your tempo stability and transition style already match the goal well.",
          ],
        }
    """
```

Comparison rules (deterministic, no LLM for the first version). Compare summary-level, mix-wide characteristics rather than aligning timelines window-by-window:

- **Pressure shape:** if the goal is wave-shaped but the aspiring mix is relentless, suggest adding relief; if the reverse, suggest sustaining energy longer.
- **Relief frequency:** compare counts/density of relief sections and suggest adding or removing breathing room.
- **Peak placement:** compare where the highest-pressure run falls (early/middle/late third) and suggest moving the climax.
- **Vocal usage:** compare possible-vocal density and whether vocals cluster around relief/transitions.
- **Transition style:** compare long-blend vs. abrupt-cut tendencies and suggest blending more or cutting more decisively.
- **Buildup usage:** compare how often pressure rises gradually into peaks and suggest more deliberate buildups if the goal uses them.

Output guidelines:

- Phrase every suggestion as a concrete, encouraging action ("Try holding your biggest drop until later"), not a numeric diff.
- Surface what already matches the goal under `matched`, so the user gets positive reinforcement, not only corrections.
- Reuse the cautious confidence language from the rest of the app ("appears to", "tends to") since these are estimates.

### Compare Mixes layout with a goal mix

```txt
Goal mix: reference_set_2024.mp3

Aspiring: my_practice_set.mp3
Headline: Strong energy, but it peaks too early to match the goal's slow build.
Suggestions:
- The goal mix saves its hardest drop for the final third; yours lands in the
  first few minutes. Try arranging tracks so the biggest moment comes later.
- The goal breathes more — it has noticeably more relief pockets. Add a reset
  after your longest high-pressure run.
Already matching:
- Your tempo feels just as stable, and your transitions blend similarly.
```

If no goal-level mix is labeled, fall back to the neutral side-by-side comparison above and show a hint that labeling a goal mix unlocks tailored suggestions.

## Debug Data tab

Show the underlying DataFrame only for inspection.

Use:

```python
st.dataframe(result.feature_df)
```

This tab is allowed to show numbers because it is for debugging, not the main user experience.

## Acceptance criteria

The POC is complete when:

1. The app runs locally with `streamlit run app.py`.
2. The user can upload at least one MP3.
3. The app analyzes the file without crashing.
4. The app shows a plain-English summary.
5. The app shows an annotated visual timeline.
6. The app shows key moments with timestamps and explanations.
7. The app can compare at least two uploaded mixes with stacked visual strips.
8. The main UI avoids requiring users to interpret raw numbers.
9. Debug data is available separately for development.

## Suggested implementation phases for Codex

### Phase 1: Skeleton app

- Create project structure.
- Create `requirements.txt`.
- Create Streamlit upload UI.
- Save uploaded files locally.
- Load audio with librosa.
- Display audio duration and basic success message.

### Phase 2: Feature extraction

- Implement `analyze_audio_windows`.
- Compute RMS, bass pressure, onset strength, spectral centroid, bandwidth, flatness, chroma summary.
- Aggregate features into windows.
- Return a DataFrame.
- Show DataFrame in Debug tab.

### Phase 3: Labels and summaries

- Normalize features within each mix.
- Add pressure labels.
- Add relief labels.
- Create deterministic summary rules.
- Show headline, tags, and learning notes.

### Phase 4: Event detection

- Detect drop candidates.
- Detect relief sections.
- Detect buildup candidates.
- Detect possible vocal sections using a weak proxy.
- Detect likely transitions using feature-change rules.
- Create event cards with timestamps.

### Phase 5: Visuals

- Build Plotly flow map.
- Build pressure silhouette.
- Build layered presence map.
- Add event markers and hover text.
- Render in Streamlit.

### Phase 6: Comparison

- Support multiple uploaded files.
- Store each result in a `MixAnalysisResult` dataclass.
- Create stacked comparison strips.
- Generate a plain-English comparison summary.

### Phase 7: Polish

- Add caching with `st.cache_data` where useful.
- Improve error handling for unsupported files.
- Add a README with install/run instructions.
- Add clear warnings that event labels are estimates.
- Add sample output screenshots later.

### Phase 8: Contextual Help and Universal Time Formatting

#### Contextual help for sidebar controls

Add a `help=` tooltip to every Streamlit sidebar widget so users understand what each setting does without external documentation.

Use Streamlit's built-in `help` parameter, which adds a hoverable `?` icon next to the control.

Suggested tooltip copy for each control:

**Analysis window** (`window_seconds` slider):
> Controls how many seconds of audio are grouped together for each analysis snapshot. Larger values (20–30 s) smooth out noise and reveal broad patterns; smaller values (5–8 s) capture faster-moving moments like quick transitions or brief relief pockets. Default 10 s works well for most mixes.

**Timeline detail** (`hop_seconds` slider):
> How often a new analysis snapshot is taken. Smaller values produce more data points and smoother transitions in the charts, at the cost of longer analysis time. If the hop is equal to the window, there is no overlap; lower values add overlap and increase granularity. Default 5 s gives good resolution without being too slow.

**Event sensitivity** (`sensitivity` selectbox):
> How readily the analyzer flags key moments.
> - Conservative: only reports events with strong, consistent evidence. Fewer results, higher confidence.
> - Balanced: moderate threshold, good starting point for most mixes.
> - Sensitive: catches subtle pressure shifts and smaller transitions, but may include more false positives. Useful for low-energy or ambient mixes.

**Minimum event spacing** (`min_event_spacing` slider/input, if exposed):
> Prevents two events of the same type from being flagged too close together. Increase this to declutter busy sections; decrease it to see more granular detail in a complex part of the mix.

**Analysis mode** (`analysis_mode` selectbox):
> Fast mode uses a lighter feature set for quicker results. Future modes (Detailed, Deep) may add source separation and more precise tempo tracking.

#### Universal MM:SS time formatting

Apply `format_time` (already defined in the implementation details section) to every time display in the app — not just event card timestamps.

**Plotly chart axes:**

Since Plotly x-axes hold raw seconds as floats, configure custom tick labels using `tickvals` and `ticktext`:

```python
def make_time_ticks(duration_seconds: float, max_ticks: int = 12) -> tuple[list[float], list[str]]:
    """Return tick positions and MM:SS labels for a Plotly time axis."""
    interval = max(60, round(duration_seconds / max_ticks / 60) * 60)
    vals = list(range(0, int(duration_seconds) + 1, interval))
    labels = [format_time(v) for v in vals]
    return vals, labels
```

Apply to every `Figure` that has a time x-axis:

```python
tickvals, ticktext = make_time_ticks(duration)
fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
```

**All other time displays:**

Audit every place in the app that shows a time value and replace raw seconds with `format_time(seconds)`:

- Event card timestamps in the Key Moments tab.
- Top key moments listed in the Mix Report tab.
- Hover text on all Plotly charts (e.g., `customdata` or `hovertemplate` using pre-formatted strings).
- Comparison strip axis labels.
- Any debug or status text that shows timestamps.

The `format_time` helper already outputs zero-padded MM:SS (e.g., `06:31`). Do not display bare seconds like `391` anywhere in the user-facing UI after this phase.

### Phase 9: Performance optimization (complete)

Feature extraction dominated analysis time, so the pipeline was profiled and tuned. All changes are algorithmic and add no new dependencies; output windows and labels are unchanged.

`src/features.py`:

- Compute one `|STFT|` and derive rms/centroid/bandwidth/flatness/chroma/bass from it instead of letting each feature recompute its own transform.
- Replace `chroma_cqt` with `chroma_stft(tuning=0.0)`, skipping the slow CQT and the hidden per-call tuning estimation.
- Coarsen the internal `hop_length` from 512 to 1024 (output windows are 5–30 s, so ~23 ms frames were wildly oversampled).
- Reuse the precomputed onset envelope for the tempo estimate.
- Vectorize per-window aggregation with `searchsorted` instead of building a boolean mask over all frames per window.
- Fixed latent bugs: `rms(S=...)` needs an explicit `frame_length`, and a missing `librosa.feature.rhythm` import was leaving `local_tempo` silently `nan`.

`src/audio_io.py`:

- Decode with `res_type="soxr_lq"` at `sr=22050` (exact 2:1 from 44.1 kHz).

Results: on a real 38-minute mix, end-to-end time dropped from 40.76 s to 5.73 s (7.1x), with the analysis stage ~12x faster, identical 461 windows, and 32/35 events aligning within 5 s of the baseline. App-level parallelism across files was investigated and rejected as slower (the pipeline is memory-bandwidth bound and NumPy/SciPy already use multiple cores via macOS Accelerate). Full benchmarks, the quality comparison, and rejected ideas live in [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

### Phase 10: Goal vs. aspiring mix suggestions

- Add a per-file role selector (`goal-level mix` / `aspiring mix`) to the `Upload` tab, between upload and the analyze button.
- Warn (non-blocking) when not exactly one goal mix is selected.
- Add a `role` field to `MixAnalysisResult` and populate it during analysis.
- Implement `suggest_toward_goal(goal, aspiring)` in `src/compare.py` using the deterministic comparison rules above.
- In the `Compare Mixes` tab, when a goal mix exists, render one suggestion card per aspiring mix (headline, suggestions, already-matching), and fall back to the neutral comparison otherwise.
- Reuse the cautious confidence language and `format_time` conventions.

## Important implementation details

### Time formatting

Create helper:

```python
def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
```

### Normalization

Use robust percentile normalization instead of min-max when possible.

```python
def robust_normalize(series):
    low = series.quantile(0.05)
    high = series.quantile(0.95)
    return ((series - low) / (high - low + 1e-9)).clip(0, 1)
```

### Event spacing

Avoid too many events.

- Merge adjacent events of the same type.
- Enforce minimum spacing between drop candidates.
- Prefer fewer, clearer key moments over many noisy detections.

### Confidence language

Use cautious labels:

- `candidate`
- `likely`
- `possible`
- `appears to`

Avoid overclaiming.

Bad:

```txt
This is definitely the first drop.
```

Good:

```txt
This is a strong drop candidate because pressure, bass, and rhythmic density rise together after a lower-pressure section.
```

### No raw-number-first UI

Avoid charts that require numeric interpretation. The main visuals should use:

- Section labels.
- Color-coded blocks.
- Event markers.
- Hover descriptions.
- Plain-English summaries.

Numbers can exist in `Debug Data` only.

## Future improvements after POC

After the proof of concept works, consider adding:

- Demucs-based vocal stem analysis.
- Better local tempo and downbeat detection.
- Key detection with confidence and Camelot labels.
- Exportable HTML reports.
- Exportable PNG timelines.
- Manual correction/editing of detected events.
- Tracklist support.
- Rekordbox cue export.
- Playlist-level comparison.
- Named reference mixes for studying DJ style.
- LLM-generated summaries from structured features, only after deterministic summaries work.

## First Codex task

Start by implementing Phase 1 and Phase 2 only.

Deliver:

- Project structure.
- `requirements.txt`.
- `app.py` with Streamlit upload UI.
- `src/audio_io.py`.
- `src/features.py`.
- A Debug tab showing the feature DataFrame for an uploaded MP3.

Do not implement advanced event detection until the basic feature table works.
