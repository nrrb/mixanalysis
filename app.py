from pathlib import Path

import streamlit as st

from src.audio_io import load_audio, save_uploaded_bytes
from src.compare import build_comparison_profiles, suggest_toward_goal, summarize_comparison
from src.events import MixEvent, detect_events
from src.features import analyze_audio_windows
from src.summaries import assign_pressure_labels, summarize_mix
from src.utils import MixAnalysisResult, format_duration
from src.visuals import (
    make_comparison_strips,
    make_flow_map,
    make_layered_presence_map,
    make_pressure_silhouette,
)


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


@st.cache_data(show_spinner=False)
def _analyze_uploaded_bytes(
    file_name: str,
    file_bytes: bytes,
    window_seconds: float,
    hop_seconds: float,
    sensitivity: str,
    minimum_spacing_seconds: float,
) -> MixAnalysisResult:
    cache_dir = Path("outputs/cache")
    path = save_uploaded_bytes(file_name, file_bytes, cache_dir)
    y, sr = load_audio(path)
    if y.size == 0:
        raise ValueError("The decoded audio was empty.")

    feature_df = analyze_audio_windows(
        y,
        sr,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    if feature_df.empty:
        raise ValueError("The audio did not produce any analysis windows.")

    feature_df = assign_pressure_labels(feature_df)
    events = detect_events(
        feature_df,
        sensitivity=sensitivity,
        minimum_spacing_seconds=minimum_spacing_seconds,
    )
    duration = len(y) / sr if sr else 0.0
    summary = summarize_mix(feature_df, events, duration=duration)
    return MixAnalysisResult(
        name=file_name,
        duration=duration,
        sample_rate=sr,
        feature_df=feature_df,
        events=events,
        summary=summary,
    )


def _select_result(results: list[MixAnalysisResult], label: str) -> MixAnalysisResult:
    names = [result.name for result in results]
    selected_name = st.selectbox(label, names)
    return next(result for result in results if result.name == selected_name)


def _render_event_card(event: MixEvent) -> None:
    end = f" - {format_duration(event.end)}" if event.end is not None else ""
    with st.container(border=True):
        st.markdown(f"**{format_duration(event.start)}{end} - {event.title}**")
        st.write(event.description)
        st.caption(f"{event.event_type} | confidence: {event.confidence}")


def _comparison_summary(results: list[MixAnalysisResult]) -> list[str]:
    return summarize_comparison(results)


def _is_supported_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def _render_estimate_warning() -> None:
    st.warning(
        "These labels are estimates from audio features. Treat drops, vocals, transitions, and keys as study cues, not facts.",
    )


def _render_comparison_card(profile) -> None:
    first_full = (
        format_duration(profile.first_full_pressure)
        if profile.first_full_pressure is not None
        else "not detected"
    )
    with st.container(border=True):
        st.markdown(f"### {profile.name}")
        st.markdown(f"**{profile.character}**")
        st.caption(profile.shape_tag)
        cols = st.columns(3)
        cols[0].metric("First full pressure", first_full)
        cols[1].metric("High pressure", _percent(profile.high_pressure_share))
        cols[2].metric("Relief", _percent(profile.relief_share))
        cols = st.columns(3)
        cols[0].metric("Possible vocals", _percent(profile.possible_vocal_share))
        cols[1].metric("Transitions", str(profile.transition_count))
        cols[2].metric("Longest pressure run", format_duration(profile.longest_high_pressure_run))


def _percent(value: float) -> str:
    return f"{round(value * 100):.0f}%"


st.set_page_config(page_title="DJ Mix Flow Analyzer", layout="wide")

st.title("DJ Mix Flow Analyzer")
st.caption(
    "Upload DJ mixes and study their pressure, relief, drops, vocals, and pacing as visual flow maps."
)
_render_estimate_warning()

with st.sidebar:
    st.header("Analysis Controls")
    window_seconds = st.slider(
        "Analysis window (s)",
        5,
        30,
        10,
        help=(
            "How many seconds of audio are grouped into each analysis snapshot. "
            "Larger values (20–30 s) smooth out noise and reveal broad patterns. "
            "Smaller values (5–8 s) capture faster-moving moments like quick transitions. "
            "Default 10 s works well for most mixes."
        ),
    )
    hop_seconds = st.slider(
        "Timeline detail (s)",
        2,
        15,
        5,
        help=(
            "How often a new snapshot is taken. Smaller values produce more data points "
            "and smoother charts, at the cost of longer analysis time. "
            "Setting this equal to the window gives no overlap; lower values add granularity. "
            "Default 5 s is a good balance."
        ),
    )
    sensitivity = st.selectbox(
        "Event sensitivity",
        ["conservative", "balanced", "sensitive"],
        index=1,
        help=(
            "How readily the analyzer flags key moments.\n\n"
            "Conservative — fewer results, higher confidence.\n\n"
            "Balanced — good starting point for most mixes.\n\n"
            "Sensitive — catches subtle shifts but may include more false positives; "
            "useful for low-energy or ambient mixes."
        ),
    )
    minimum_event_spacing = st.slider(
        "Minimum event spacing (s)",
        10,
        90,
        30,
        help=(
            "Minimum gap in seconds between two events of the same type. "
            "Increase this to declutter busy sections; decrease it to see more "
            "granular detail. Default 30 s prevents event clustering in dense passages."
        ),
    )
    analysis_mode = st.selectbox(
        "Analysis mode",
        ["fast"],
        index=0,
        help=(
            "Fast mode uses a lighter feature set for quick results. "
            "Future modes (Detailed, Deep) will add source separation "
            "and more precise tempo tracking."
        ),
    )
    analyze_clicked = st.button("Analyze mixes", type="primary")

tabs = st.tabs(
    ["Upload", "Mix Report", "Key Moments", "Visual Flow", "Compare Mixes", "Debug Data"]
)

with tabs[0]:
    uploaded_files = st.file_uploader(
        "Upload MP3, WAV, or M4A mixes",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=True,
    )

    roles: dict[str, str] = {}
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) ready for analysis.")
        st.markdown("**Label each mix**")
        for uploaded_file in uploaded_files:
            if _is_supported_file(uploaded_file.name):
                roles[uploaded_file.name] = st.radio(
                    uploaded_file.name,
                    ["aspiring mix", "goal-level mix"],
                    index=0,
                    horizontal=True,
                    key=f"role_{uploaded_file.name}",
                    help=(
                        "Mark the reference mix you want to sound more like as the "
                        "goal-level mix, and your own mixes as aspiring. With one goal "
                        "mix labeled, the Compare Mixes tab suggests how to close the gap."
                    ),
                )
            else:
                st.error(f"{uploaded_file.name} is not a supported audio type.")

        goal_count = sum(1 for role in roles.values() if role == "goal-level mix")
        if goal_count != 1:
            st.warning(
                "Pick exactly one goal-level mix to unlock tailored suggestions. "
                "You can still analyze without one."
            )
    else:
        st.info("Upload at least one mix to begin.")

    if uploaded_files and analyze_clicked:
        results: list[MixAnalysisResult] = []

        for uploaded_file in uploaded_files:
            if not _is_supported_file(uploaded_file.name):
                continue
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    result = _analyze_uploaded_bytes(
                        uploaded_file.name,
                        bytes(uploaded_file.getbuffer()),
                        float(window_seconds),
                        float(hop_seconds),
                        sensitivity,
                        float(minimum_event_spacing),
                    )
                    result.role = (
                        "goal" if roles.get(uploaded_file.name) == "goal-level mix" else "aspiring"
                    )
                    results.append(result)
                except Exception as exc:
                    st.error(f"Could not analyze {uploaded_file.name}: {exc}")

        if results:
            st.session_state["results"] = results
            st.success(f"Analyzed {len(results)} mix(es). Open Mix Report or Visual Flow.")

    if "results" in st.session_state:
        st.subheader("Analyzed Mixes")
        for result in st.session_state["results"]:
            role_label = "goal-level mix" if result.role == "goal" else "aspiring mix"
            st.write(
                f"**{result.name}** ({role_label}) - {format_duration(result.duration)} "
                f"at {result.sample_rate:,} Hz - {len(result.feature_df)} analysis windows"
            )

with tabs[1]:
    st.subheader("Mix Report")
    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see its plain-English report.")
    else:
        _render_estimate_warning()
        for result in results:
            with st.container(border=True):
                st.markdown(f"### {result.name}")
                st.markdown(f"**{result.summary['headline']}**")
                st.write(result.summary["summary"])
                st.caption(", ".join(result.summary["tags"]))

                notes = result.summary.get("learning_notes", [])
                if notes:
                    st.markdown("**Learning notes**")
                    for note in notes:
                        st.write(f"- {note}")

                top_events = result.events[:5]
                if top_events:
                    st.markdown("**Top key moments**")
                    for event in top_events:
                        st.write(f"- {format_duration(event.start)} - {event.title}")
                else:
                    st.write("No strong key moments detected yet.")

with tabs[2]:
    st.subheader("Key Moments")
    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see detected key moments.")
    else:
        _render_estimate_warning()
        selected = _select_result(results, "Key moments mix")
        if not selected.events:
            st.info("No strong event candidates were detected for this mix.")
        for event in selected.events:
            _render_event_card(event)

with tabs[3]:
    st.subheader("Visual Flow")
    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see visual flow maps.")
    else:
        _render_estimate_warning()
        selected = _select_result(results, "Visual flow mix")
        st.plotly_chart(
            make_flow_map(selected.feature_df, selected.events),
            width="stretch",
        )
        st.plotly_chart(
            make_pressure_silhouette(selected.feature_df, selected.events),
            width="stretch",
        )
        st.plotly_chart(
            make_layered_presence_map(selected.feature_df, selected.events),
            width="stretch",
        )

with tabs[4]:
    st.subheader("Compare Mixes")
    results = st.session_state.get("results", [])
    if len(results) < 2:
        st.info("Analyze at least two mixes to compare their flow.")
    else:
        _render_estimate_warning()
        st.plotly_chart(make_comparison_strips(results), width="stretch")

        goal_mixes = [result for result in results if result.role == "goal"]
        aspiring_mixes = [result for result in results if result.role != "goal"]
        if len(goal_mixes) == 1 and aspiring_mixes:
            goal = goal_mixes[0]
            st.subheader("Suggestions Toward the Goal")
            st.markdown(f"Goal mix: **{goal.name}**")
            for aspiring in aspiring_mixes:
                advice = suggest_toward_goal(goal, aspiring)
                with st.container(border=True):
                    st.markdown(f"### {aspiring.name}")
                    st.markdown(f"**{advice['headline']}**")
                    if advice["suggestions"]:
                        st.markdown("**Suggestions**")
                        for line in advice["suggestions"]:
                            st.write(f"- {line}")
                    if advice["matched"]:
                        st.markdown("**Already matching**")
                        for line in advice["matched"]:
                            st.write(f"- {line}")
        else:
            st.info(
                "Label exactly one mix as the goal-level mix on the Upload tab to unlock "
                "tailored suggestions for your aspiring mixes."
            )

        st.subheader("Side-by-Side Comparison")
        for line in _comparison_summary(results):
            st.write(line)
        st.subheader("Mix Character Cards")
        for profile in build_comparison_profiles(results):
            _render_comparison_card(profile)

with tabs[5]:
    st.subheader("Debug Data")
    st.caption("Raw feature values are shown here for development only.")

    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see its feature table.")
    else:
        selected = _select_result(results, "Debug mix")

        st.write(
            f"Duration: {format_duration(selected.duration)} | "
            f"Windows: {len(selected.feature_df)}"
        )
        st.dataframe(selected.feature_df, width="stretch")
