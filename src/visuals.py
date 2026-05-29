import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.events import MixEvent
from src.utils import format_duration, make_time_ticks


# ---------------------------------------------------------------------------
# Visual design language and color scheme
# ---------------------------------------------------------------------------
# A single, consistent look across every on-screen figure and every exported
# social-media image. Change a value here and it updates everywhere.

APP_NAME = "DJ Mix Flow Analyzer"

# Dark, high-contrast base so colored blocks and glowing accents pop on a phone.
BG_COLOR = "#0E0B1A"
PANEL_COLOR = "#12101F"
TEXT_COLOR = "#F5F3FF"
MUTED_TEXT = "#9C97B5"
GRID_COLOR = "rgba(255, 255, 255, 0.07)"
ACCENT = "#FF2D78"  # brand / watermark / call-to-action

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

# Social-media export dimensions (logical px; rendered at scale=2 for crispness).
SOCIAL_SIZES = {
    "square": (1080, 1080),
    "9x16": (1080, 1920),
}
SOCIAL_ASPECT_LABELS = {"square": "Square 1:1", "9x16": "Vertical 9:16"}


def _apply_base_theme(fig: go.Figure) -> None:
    """Apply the shared dark, vibrant theme to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        legend=dict(font=dict(color=TEXT_COLOR)),
        title_font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
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


def make_pressure_silhouette(df: pd.DataFrame, events: list[MixEvent]) -> go.Figure:
    """Create an intuitive shape of the mix's pressure and relief."""
    if df.empty:
        return _empty_figure("No pressure data available")

    duration = float(df["end_time"].max())
    tickvals, ticktext = make_time_ticks(duration)

    x = (df["start_time"] + df["end_time"]) / 2
    x_fmt = x.apply(format_duration)
    fig = go.Figure()
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


def make_suggestion_card_figure(goal_name: str, aspiring_name: str, advice: dict) -> go.Figure:
    """Compose a goal-vs-aspiring suggestion card as a shareable text figure."""
    lines = [f"<span style='color:{MUTED_TEXT}'>Goal mix: {goal_name}</span>", ""]
    lines.append(f"<b>{_wrap(advice.get('headline', ''), 42)}</b>")

    suggestions = advice.get("suggestions") or []
    if suggestions:
        lines.append("")
        lines.append(f"<b><span style='color:{ACCENT}'>Try this</span></b>")
        for item in suggestions[:3]:
            lines.append(f"• {_wrap(item, 48)}")

    matched = advice.get("matched") or []
    if matched:
        lines.append("")
        lines.append(f"<b><span style='color:{LANE_RELIEF}'>Already matching</span></b>")
        for item in matched[:2]:
            lines.append(f"• {_wrap(item, 48)}")

    fig = go.Figure()
    fig.add_annotation(
        x=0.04,
        y=0.94,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text="<br>".join(lines),
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=18),
    )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=30))
    _apply_base_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Shareable social-media images
# ---------------------------------------------------------------------------


def render_social_image(
    fig: go.Figure,
    aspect: str,
    title: str,
    subtitle: str = "",
    *,
    lanes: int = 1,
    fit_plot: bool = True,
    keep_annotations: bool = False,
) -> bytes:
    """Re-lay-out a flow/comparison figure for the target aspect ratio and
    return PNG bytes suitable for social sharing (high-res, titled, watermarked).

    ``aspect`` is ``"square"`` (1:1) or ``"9x16"`` (vertical). Images are rendered
    at the logical sizes in ``SOCIAL_SIZES`` with ``scale=2`` so text and markers
    stay crisp after platform re-compression.
    """
    if aspect not in SOCIAL_SIZES:
        raise ValueError(f"Unknown aspect '{aspect}'. Use one of {list(SOCIAL_SIZES)}.")

    width, height = SOCIAL_SIZES[aspect]
    is_tall = aspect == "9x16"

    social = go.Figure(fig)

    # A clean shareable look: drop the small on-screen event labels but keep the
    # glowing event seam lines (drawn as shapes), unless the figure *is* the text
    # (e.g. a suggestion card).
    if not keep_annotations:
        social.layout.annotations = ()

    title_size = 50 if is_tall else 42
    sub_size = 26 if is_tall else 23
    top = 250 if is_tall else 190
    bottom = 210 if is_tall else 185

    social.update_layout(
        width=width,
        height=height,
        margin=dict(l=70, r=70, t=top, b=bottom),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            yref="container",
            y=0.97,
            yanchor="top",
            font=dict(size=title_size, color=TEXT_COLOR, family=FONT_FAMILY),
        ),
        font=dict(size=20),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.02,
            yanchor="top",
            font=dict(size=16, color=TEXT_COLOR),
        ),
    )
    _apply_base_theme(social)

    # Re-lay-out for the frame: keep the plot in a centered vertical band sized to
    # the number of lanes rather than stretching one fat timeline to fill the
    # frame. Skip for figures without real axes (suggestion cards).
    if fit_plot:
        social.update_yaxes(domain=_social_y_domain(aspect, lanes))

    if subtitle:
        social.add_annotation(
            text=_wrap(subtitle, 64),
            x=0.5,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            yshift=24,
            showarrow=False,
            align="center",
            font=dict(size=sub_size, color=MUTED_TEXT, family=FONT_FAMILY),
        )

    # Consistent attributable watermark/footer.
    social.add_annotation(
        text=f"<b>▲ {APP_NAME}</b>",
        x=0.99,
        y=0.0,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        yshift=-130,
        showarrow=False,
        font=dict(size=20, color=ACCENT, family=FONT_FAMILY),
    )

    return social.to_image(format="png", width=width, height=height, scale=2)


def _social_y_domain(aspect: str, lanes: int) -> list[float]:
    """Centered vertical band for the plot area, sized to the lane count."""
    lanes = max(1, lanes)
    if aspect == "9x16":
        frac = min(0.55, 0.12 * lanes + 0.14)
    else:
        frac = min(0.80, 0.16 * lanes + 0.20)
    half = frac / 2
    return [0.5 - half, 0.5 + half]


def export_shareable_images(result, out_dir: Path) -> list[Path]:
    """Write square and 9:16 PNGs for a single mix's flow visuals.

    Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(result.name)
    subtitle = result.summary.get("headline", "") if getattr(result, "summary", None) else ""

    visuals = [
        ("flow", "Flow Map", make_flow_map(result.feature_df, result.events), 1),
        ("silhouette", "Pressure Shape", make_pressure_silhouette(result.feature_df, result.events), 1),
        ("presence", "Presence Map", make_layered_presence_map(result.feature_df, result.events), 5),
    ]

    paths: list[Path] = []
    for kind, nice, fig, lanes in visuals:
        for aspect in SOCIAL_SIZES:
            png = render_social_image(
                fig,
                aspect,
                title=f"{_display_name(result.name)} · {nice}",
                subtitle=subtitle,
                lanes=lanes,
            )
            path = out_dir / f"{slug}_{kind}_{aspect}.png"
            path.write_bytes(png)
            paths.append(path)
    return paths


def export_comparison_images(results: list, out_dir: Path) -> list[Path]:
    """Write square and 9:16 PNGs for the comparison strips and any
    goal-vs-aspiring suggestion cards.

    Returns the written paths.
    """
    from src.compare import suggest_toward_goal

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    strips = make_comparison_strips(results)
    for aspect in SOCIAL_SIZES:
        png = render_social_image(
            strips,
            aspect,
            title="Mix Comparison",
            subtitle="Pressure flow, side by side",
            lanes=len(results),
        )
        path = out_dir / f"comparison_strips_{aspect}.png"
        path.write_bytes(png)
        paths.append(path)

    goal_mixes = [r for r in results if getattr(r, "role", "aspiring") == "goal"]
    aspiring_mixes = [r for r in results if getattr(r, "role", "aspiring") != "goal"]
    if len(goal_mixes) == 1 and aspiring_mixes:
        goal = goal_mixes[0]
        for aspiring in aspiring_mixes:
            advice = suggest_toward_goal(goal, aspiring)
            card = make_suggestion_card_figure(goal.name, aspiring.name, advice)
            for aspect in SOCIAL_SIZES:
                png = render_social_image(
                    card,
                    aspect,
                    title=f"{_display_name(aspiring.name)} → Goal",
                    fit_plot=False,
                    keep_annotations=True,
                )
                path = out_dir / f"suggestions_{_slug(aspiring.name)}_{aspect}.png"
                path.write_bytes(png)
                paths.append(path)
    return paths


def _display_name(name: str) -> str:
    return Path(name).stem


def _slug(name: str) -> str:
    stem = Path(name).stem.lower()
    return "".join(ch if ch.isalnum() else "-" for ch in stem).strip("-") or "mix"


def _wrap(text: str, width: int) -> str:
    if not text:
        return ""
    return "<br>".join(textwrap.wrap(text, width=width)) or text


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
