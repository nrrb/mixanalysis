import pandas as pd
import plotly.graph_objects as go

from src.events import MixEvent
from src.utils import format_duration, make_time_ticks


# ---------------------------------------------------------------------------
# Visual design language and color scheme
# ---------------------------------------------------------------------------
# A single, consistent look across every on-screen figure. Change a value here
# and it updates everywhere.

# Dark, high-contrast base so colored blocks and glowing accents pop on a phone.
BG_COLOR = "#0E0B1A"
PANEL_COLOR = "#12101F"
TEXT_COLOR = "#F5F3FF"
MUTED_TEXT = "#9C97B5"
GRID_COLOR = "rgba(255, 255, 255, 0.07)"
ACCENT = "#FF2D78"  # brand / call-to-action

FONT_FAMILY = "Trebuchet MS, Inter, Segoe UI, sans-serif"

# Pressure ramp (low -> peak): indigo -> teal -> amber -> coral -> hot magenta.
# Sequential in perceived intensity, photogenic, and reasonably colorblind-safe.
PRESSURE_COLORS = {
    "low pressure": "#3B2E7E",
    "groove / cruising": "#1FB6A6",
    "building pressure": "#F4B740",
    "full pressure": "#FF6B5B",
    "peak pressure": "#FF2D78",
}

# Event accents, kept distinct from the pressure ramp.
EVENT_COLORS = {
    "drop candidate": "#FF2D78",
    "relief section": "#46E0D0",
    "buildup candidate": "#F4B740",
    "possible vocal section": "#C9A7FF",
    "likely transition": "#FFD66B",
    "sustained pressure run": "#FF6B5B",
}

# Lane colors for the layered presence map (drawn from the shared palette).
LANE_PRESSURE = "#FF6B5B"
LANE_BASS = "#7A5CFF"
LANE_VOCAL = "#C9A7FF"
LANE_RELIEF = "#46E0D0"
LANE_TRANSITION = "#FFD66B"


def _apply_base_theme(fig: go.Figure) -> None:
    """Apply the shared dark, vibrant theme to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        legend=dict(font=dict(color=TEXT_COLOR)),
        # Style the title font without materializing an empty title object: a
        # bare title.font with no text makes Plotly.js render the literal string
        # "undefined". Only set the font when a chart actually has title text.
        title=dict(font=dict(family=FONT_FAMILY, color=TEXT_COLOR)) if fig.layout.title.text else None,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=GRID_COLOR,
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=MUTED_TEXT),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=GRID_COLOR,
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=MUTED_TEXT),
    )


def make_flow_map(df: pd.DataFrame, events: list[MixEvent]) -> go.Figure:
    """Create an intuitive annotated timeline of the mix."""
    fig = go.Figure()
    if df.empty:
        return _empty_figure("No flow data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    for label, group in df.groupby("pressure_label", sort=False):
        start_fmt = group["start_time"].apply(format_duration)
        end_fmt = group["end_time"].apply(format_duration)
        customdata = list(zip(start_fmt, end_fmt, group["relief_type"]))
        fig.add_trace(
            go.Bar(
                x=group["end_time"] - group["start_time"],
                y=["Pressure"] * len(group),
                base=group["start_time"],
                orientation="h",
                name=label,
                marker_color=PRESSURE_COLORS.get(label, "#adb5bd"),
                marker_line_width=0,
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]}–%{customdata[1]}<br>"
                    f"{label}<br>%{{customdata[2]}}<extra></extra>"
                ),
            )
        )

    _add_event_markers(fig, events, y=0.0)
    fig.update_layout(
        barmode="stack",
        height=250,
        xaxis_title="Time",
        yaxis_title="",
        legend_title_text="Pressure",
        margin=dict(l=20, r=20, t=30, b=30),
    )
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    fig.update_yaxes(showticklabels=False)
    _apply_base_theme(fig)
    return fig


def make_pressure_silhouette(
    df: pd.DataFrame,
    events: list[MixEvent],
    downbeats: "np.ndarray | None" = None,
) -> go.Figure:
    """Create an intuitive shape of the mix's pressure and relief.

    When ``downbeats`` (seconds) are supplied, faint bar lines are drawn behind
    the silhouette as an optional beat-grid overlay (Phase 13).
    """
    if df.empty:
        return _empty_figure("No pressure data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    x = (df["start_time"] + df["end_time"]) / 2
    x_fmt = x.apply(format_duration)
    fig = go.Figure()
    # Optional beat grid: faint bar lines behind everything, drawn as one trace
    # with None separators so hundreds of downbeats stay cheap.
    if downbeats is not None and len(downbeats) > 0:
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for beat_time in downbeats:
            if 0.0 <= float(beat_time) <= duration:
                line_x.extend([float(beat_time), float(beat_time), None])
                line_y.extend([0.0, 1.05, None])
        if line_x:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color="rgba(255, 255, 255, 0.08)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Beat grid",
                )
            )
    # Soft glow underlay for a polished, neon-adjacent edge.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["pressure_score"],
            mode="lines",
            line=dict(color="rgba(255, 45, 120, 0.25)", width=12, shape="spline"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Vibrant gradient fill instead of a flat color.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["pressure_score"],
            mode="lines",
            line=dict(color=ACCENT, width=3, shape="spline"),
            fill="tozeroy",
            fillgradient=dict(
                type="vertical",
                colorscale=[
                    [0.0, "rgba(59, 46, 126, 0.05)"],
                    [0.5, "rgba(244, 183, 64, 0.20)"],
                    [1.0, "rgba(255, 45, 120, 0.55)"],
                ],
            ),
            name="Pressure shape",
            text=df["pressure_label"],
            customdata=x_fmt,
            hovertemplate="%{customdata}<br>%{text}<extra></extra>",
        )
    )
    _add_event_markers(fig, events)
    fig.update_layout(
        height=320,
        xaxis_title="Time",
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=[0.15, 0.40, 0.65, 0.90],
            ticktext=["low", "groove", "full", "peak"],
            range=[0, 1.05],
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=False,
    )
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    _apply_base_theme(fig)
    return fig


def make_layered_presence_map(df: pd.DataFrame, events: list[MixEvent]) -> go.Figure:
    """Create a lane-based timeline of mix characteristics."""
    if df.empty:
        return _empty_figure("No presence data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    lanes = [
        ("Pressure", df["pressure_score"] >= 0.62, LANE_PRESSURE),
        ("Bass-heavy", df["bass_norm"] >= 0.62, LANE_BASS),
        ("Possible vocals", df["possible_vocal"], LANE_VOCAL),
        ("Relief", df["relief_type"] != "no clear relief", LANE_RELIEF),
    ]
    transition_mask = _event_lane_mask(df, events, "likely transition")
    lanes.append(("Transitions", transition_mask, LANE_TRANSITION))

    fig = go.Figure()
    for lane, mask, color in lanes:
        active = df[mask]
        if active.empty:
            fig.add_trace(
                go.Bar(
                    x=[0],
                    y=[lane],
                    base=[0],
                    orientation="h",
                    marker_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            continue
        start_fmt = active["start_time"].apply(format_duration)
        end_fmt = active["end_time"].apply(format_duration)
        customdata = list(zip(start_fmt, end_fmt))
        fig.add_trace(
            go.Bar(
                x=active["end_time"] - active["start_time"],
                y=[lane] * len(active),
                base=active["start_time"],
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                name=lane,
                customdata=customdata,
                hovertemplate=f"{lane}<br>%{{customdata[0]}}–%{{customdata[1]}}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        barmode="stack",
        height=320,
        xaxis_title="Time",
        yaxis_title="",
        margin=dict(l=20, r=20, t=30, b=30),
    )
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    _apply_base_theme(fig)
    return fig


def make_comparison_strips(results: list) -> go.Figure:
    """Compare multiple mixes as stacked flow timelines."""
    fig = go.Figure()
    if not results:
        return _empty_figure("No mixes available")

    non_empty = [r for r in results if not r.feature_df.empty]
    duration = float(max(r.feature_df["end_time"].max() for r in non_empty)) if non_empty else 0.0
    tickvals, ticktext = make_time_ticks(duration)

    for result in results:
        df = result.feature_df
        for label, group in df.groupby("pressure_label", sort=False):
            start_fmt = list(group["start_time"].apply(format_duration))
            fig.add_trace(
                go.Bar(
                    x=group["end_time"] - group["start_time"],
                    y=[result.name] * len(group),
                    base=group["start_time"],
                    orientation="h",
                    marker_color=PRESSURE_COLORS.get(label, "#adb5bd"),
                    marker_line_width=0,
                    name=label,
                    legendgroup=label,
                    showlegend=not any(trace.name == label for trace in fig.data),
                    customdata=start_fmt,
                    hovertemplate=f"{result.name}<br>{label}<br>%{{customdata}}<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="stack",
        height=max(260, 90 * len(results)),
        xaxis_title="Time",
        yaxis_title="",
        legend_title_text="Pressure",
        margin=dict(l=20, r=20, t=30, b=30),
    )
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    _apply_base_theme(fig)
    return fig


def _add_event_markers(fig: go.Figure, events: list[MixEvent], y: float | None = None) -> None:
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#FFFFFF")
        # Soft glow underlay + crisp seam line for a neon-adjacent marker.
        fig.add_vline(x=event.start, line_color=_rgba(color, 0.22), line_width=8)
        fig.add_vline(x=event.start, line_color=color, line_width=2, line_dash="dot")
        fig.add_annotation(
            x=event.start,
            y=y if y is not None else 1.02,
            yref="y" if y is not None else "paper",
            text=event.title,
            showarrow=False,
            textangle=-35,
            font=dict(size=10, color=color),
        )


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _event_lane_mask(df: pd.DataFrame, events: list[MixEvent], event_type: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for event in events:
        if event.event_type != event_type:
            continue
        end = event.end if event.end is not None else event.start
        mask |= (df["start_time"] <= end) & (df["end_time"] >= event.start)
    return mask


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, font=dict(color=MUTED_TEXT))
    fig.update_layout(height=240, xaxis_visible=False, yaxis_visible=False)
    _apply_base_theme(fig)
    return fig
