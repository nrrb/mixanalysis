"""Capability probing and graceful degradation for the optional "Deep" mode.

Deep mode (Phase 12) is the foundation for the heavier, optional analyses added
in Phases 13-15: beat_this beat/downbeat tracking and Demucs vocal-stem
separation. The libraries that power them are intentionally NOT in the core
``requirements.txt`` - they live in ``requirements-deep.txt`` and are imported
lazily, so the base app always installs and runs without them.

Nothing in this module imports ``torch``/``demucs``/``beat_this`` at load time;
the probes use :func:`importlib.util.find_spec`, so a missing optional dependency
can never raise on import.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_demucs() -> bool:
    """True when Demucs (and torch) are importable for vocal-stem separation."""
    return _module_available("demucs") and _module_available("torch")


def has_beat_this() -> bool:
    """True when beat_this (and torch) are importable for beat/downbeat tracking."""
    return _module_available("beat_this") and _module_available("torch")


@dataclass
class DeepCapabilities:
    """Which optional Deep-mode engines are installed."""

    demucs: bool
    beat_this: bool

    @property
    def any_available(self) -> bool:
        return self.demucs or self.beat_this

    @property
    def all_available(self) -> bool:
        return self.demucs and self.beat_this


def probe_capabilities() -> DeepCapabilities:
    """Check which optional Deep dependencies are currently importable."""
    return DeepCapabilities(demucs=has_demucs(), beat_this=has_beat_this())


@dataclass
class ResolvedMode:
    """The analysis mode actually used, plus any user-facing fallback notes."""

    requested: str
    effective: str
    capabilities: DeepCapabilities
    messages: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return self.requested != self.effective


def resolve_analysis_mode(requested: str) -> ResolvedMode:
    """Decide which mode to actually run and collect any fallback messages.

    Deep mode needs at least one optional engine installed. When none are
    present we fall back to Fast rather than failing, and surface a message the
    UI can show. Partial availability (e.g. beat_this but not Demucs) stays in
    Deep mode; the individual features degrade to their Fast-mode equivalents on
    their own (see Phases 13-15).
    """
    requested = (requested or "fast").lower()
    capabilities = probe_capabilities()

    if requested != "deep":
        return ResolvedMode(
            requested="fast", effective="fast", capabilities=capabilities
        )

    messages: list[str] = []
    if not capabilities.any_available:
        messages.append(
            "Deep mode needs the optional `deep` extras "
            "(`pip install -r requirements-deep.txt`). None were found, so this "
            "run uses Fast mode."
        )
        return ResolvedMode(
            requested="deep",
            effective="fast",
            capabilities=capabilities,
            messages=messages,
        )

    if not capabilities.demucs:
        messages.append(
            "Demucs/torch not found - vocal detection falls back to the heuristic "
            "proxy. Install the `deep` extras for measured vocal stems."
        )
    if not capabilities.beat_this:
        messages.append(
            "beat_this not found - beat/downbeat tracking falls back to librosa. "
            "Install the `deep` extras for the neural tracker."
        )
    return ResolvedMode(
        requested="deep",
        effective="deep",
        capabilities=capabilities,
        messages=messages,
    )
