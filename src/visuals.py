import pandas as pd
import plotly.graph_objects as go

from src.events import MixEvent
from src.utils import format_duration, make_time_ticks


PRESSURE_COLORS = {
    "low pressure": "#8ecae6",
    "groove / cruising": "#57cc99",
    "building pressure": "#ffb703",
    "full pressure": "#fb8500",
    "peak pressure": "#d62828",
}

EVENT_COLORS = {
    "drop candidate": "#d62828",
    "relief section": "#2a9d8f",
    "buildup candidate": "#f77f00",
    "possible vocal section": "#7b2cbf",
    "likely transition": "#264653",
    "sustained pressure run": "#bc6c25",
}


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
    return fig


def make_pressure_silhouette(df: pd.DataFrame, events: list[MixEvent]) -> go.Figure:
    """Create an intuitive shape of the mix's pressure and relief."""
    if df.empty:
        return _empty_figure("No pressure data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    x = (df["start_time"] + df["end_time"]) / 2
    x_fmt = x.apply(format_duration)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["pressure_score"],
            mode="lines",
            line=dict(color="#264653", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(42, 157, 143, 0.24)",
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
    return fig


def make_layered_presence_map(df: pd.DataFrame, events: list[MixEvent]) -> go.Figure:
    """Create a lane-based timeline of mix characteristics."""
    if df.empty:
        return _empty_figure("No presence data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    lanes = [
        ("Pressure", df["pressure_score"] >= 0.62, "#fb8500"),
        ("Bass-heavy", df["bass_norm"] >= 0.62, "#bc6c25"),
        ("Possible vocals", df["possible_vocal"], "#7b2cbf"),
        ("Relief", df["relief_type"] != "no clear relief", "#2a9d8f"),
    ]
    transition_mask = _event_lane_mask(df, events, "likely transition")
    lanes.append(("Transitions", transition_mask, "#264653"))

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
    return fig


def _add_event_markers(fig: go.Figure, events: list[MixEvent], y: float | None = None) -> None:
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#343a40")
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
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    fig.update_layout(height=240, xaxis_visible=False, yaxis_visible=False)
    return fig
