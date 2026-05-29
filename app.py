from pathlib import Path

import streamlit as st

from src.audio_io import load_audio, save_uploaded_file
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
    rows = []
    for result in results:
        df = result.feature_df
        high_share = float((df["pressure_score"] >= 0.62).mean()) if not df.empty else 0.0
        relief_share = float((df["relief_type"] != "no clear relief").mean()) if not df.empty else 0.0
        vocal_share = float(df["possible_vocal"].mean()) if not df.empty else 0.0
        first_full = df.loc[df["pressure_score"] >= 0.62, "start_time"]
        rows.append(
            {
                "name": result.name,
                "high_share": high_share,
                "relief_share": relief_share,
                "vocal_share": vocal_share,
                "first_full": float(first_full.iloc[0]) if not first_full.empty else None,
                "tags": ", ".join(result.summary.get("tags", [])),
            }
        )

    earliest = min(
        (row for row in rows if row["first_full"] is not None),
        key=lambda row: row["first_full"],
        default=None,
    )
    most_relief = max(rows, key=lambda row: row["relief_share"])
    most_pressure = max(rows, key=lambda row: row["high_share"])
    most_vocal = max(rows, key=lambda row: row["vocal_share"])

    lines = []
    if earliest:
        lines.append(
            f"{earliest['name']} reaches full pressure earliest, around {format_duration(earliest['first_full'])}."
        )
    lines.append(
        f"{most_pressure['name']} spends the largest share of time in full or peak pressure."
    )
    lines.append(f"{most_relief['name']} has the most frequent relief pockets.")
    if most_vocal["vocal_share"] > 0:
        lines.append(f"{most_vocal['name']} has the strongest possible-vocal signal.")
    return lines


st.set_page_config(page_title="DJ Mix Flow Analyzer", layout="wide")

st.title("DJ Mix Flow Analyzer")
st.caption(
    "Upload DJ mixes and study their pressure, relief, drops, vocals, and pacing as visual flow maps."
)

with st.sidebar:
    st.header("Analysis Controls")
    window_seconds = st.slider("Analysis window", 5, 30, 10)
    hop_seconds = st.slider("Timeline detail", 2, 15, 5)
    sensitivity = st.selectbox(
        "Event sensitivity",
        ["conservative", "balanced", "sensitive"],
        index=1,
    )
    minimum_event_spacing = st.slider("Minimum event spacing", 10, 90, 30)
    analysis_mode = st.selectbox("Analysis mode", ["fast"], index=0)
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

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) ready for analysis.")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")
    else:
        st.info("Upload at least one mix to begin.")

    if uploaded_files and analyze_clicked:
        results: list[MixAnalysisResult] = []
        cache_dir = Path("outputs/cache")

        for uploaded_file in uploaded_files:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    path = save_uploaded_file(uploaded_file, cache_dir)
                    y, sr = load_audio(path)
                    feature_df = analyze_audio_windows(
                        y,
                        sr,
                        window_seconds=float(window_seconds),
                        hop_seconds=float(hop_seconds),
                    )
                    feature_df = assign_pressure_labels(feature_df)
                    events = detect_events(
                        feature_df,
                        sensitivity=sensitivity,
                        minimum_spacing_seconds=float(minimum_event_spacing),
                    )
                    duration = len(y) / sr if sr else 0.0
                    summary = summarize_mix(feature_df, events, duration=duration)
                    results.append(
                        MixAnalysisResult(
                            name=uploaded_file.name,
                            duration=duration,
                            sample_rate=sr,
                            feature_df=feature_df,
                            events=events,
                            summary=summary,
                        )
                    )
                except Exception as exc:
                    st.error(f"Could not analyze {uploaded_file.name}: {exc}")

        if results:
            st.session_state["results"] = results
            st.success(f"Analyzed {len(results)} mix(es). Open Mix Report or Visual Flow.")

    if "results" in st.session_state:
        st.subheader("Analyzed Mixes")
        for result in st.session_state["results"]:
            st.write(
                f"**{result.name}** - {format_duration(result.duration)} "
                f"at {result.sample_rate:,} Hz - {len(result.feature_df)} analysis windows"
            )

with tabs[1]:
    st.subheader("Mix Report")
    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see its plain-English report.")
    else:
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
        st.plotly_chart(make_comparison_strips(results), width="stretch")
        for line in _comparison_summary(results):
            st.write(line)

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
