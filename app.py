from pathlib import Path

import streamlit as st

from src.audio_io import load_audio, save_uploaded_file
from src.features import analyze_audio_windows
from src.utils import MixAnalysisResult, format_duration


st.set_page_config(page_title="DJ Mix Flow Analyzer", layout="wide")

st.title("DJ Mix Flow Analyzer")
st.caption(
    "Upload DJ mixes and inspect their pressure, rhythm, bass, brightness, and tonal features."
)

with st.sidebar:
    st.header("Analysis Controls")
    window_seconds = st.slider("Analysis window", 5, 30, 10)
    hop_seconds = st.slider("Timeline detail", 2, 15, 5)
    sensitivity = st.selectbox(
        "Event sensitivity",
        ["conservative", "balanced", "sensitive"],
        index=1,
        disabled=True,
        help="Event detection starts in a later phase.",
    )
    analysis_mode = st.selectbox("Analysis mode", ["fast"], index=0)
    analyze_clicked = st.button("Analyze mixes", type="primary")

tabs = st.tabs(["Upload", "Debug Data"])

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
                    duration = len(y) / sr if sr else 0.0
                    results.append(
                        MixAnalysisResult(
                            name=uploaded_file.name,
                            duration=duration,
                            sample_rate=sr,
                            feature_df=feature_df,
                        )
                    )
                except Exception as exc:
                    st.error(f"Could not analyze {uploaded_file.name}: {exc}")

        if results:
            st.session_state["results"] = results
            st.success(f"Analyzed {len(results)} mix(es). Open Debug Data to inspect features.")

    if "results" in st.session_state:
        st.subheader("Analyzed Mixes")
        for result in st.session_state["results"]:
            st.write(
                f"**{result.name}** - {format_duration(result.duration)} "
                f"at {result.sample_rate:,} Hz - {len(result.feature_df)} analysis windows"
            )

with tabs[1]:
    st.subheader("Debug Data")
    st.caption("Raw feature values are shown here for development only.")

    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze an uploaded mix to see its feature table.")
    else:
        names = [result.name for result in results]
        selected_name = st.selectbox("Mix", names)
        selected = next(result for result in results if result.name == selected_name)

        st.write(
            f"Duration: {format_duration(selected.duration)} | "
            f"Windows: {len(selected.feature_df)}"
        )
        st.dataframe(selected.feature_df, use_container_width=True)
