"""Small timing profile objects shared by loader and app consumers."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True, frozen=True)
class ProfileEvent:
    name: str
    duration_s: float
    detail: str = ""


@dataclass(slots=True)
class ProfileTimer:
    name: str
    detail: str = ""
    duration_s: float = 0.0


@dataclass(slots=True)
class PerformanceProfile:
    events: list[ProfileEvent] = field(default_factory=list)

    @contextmanager
    def timed(self, name: str, detail: str = "") -> Iterator[ProfileTimer]:
        timer = ProfileTimer(name=name, detail=detail)
        started_at = perf_counter()
        try:
            yield timer
        finally:
            timer.duration_s = perf_counter() - started_at
            self.events.append(ProfileEvent(name, timer.duration_s, timer.detail))

    def add(self, name: str, duration_s: float, detail: str = "") -> None:
        self.events.append(ProfileEvent(name, duration_s, detail))

    @property
    def total_s(self) -> float:
        return sum(event.duration_s for event in self.events)


def log_profile_events(
    profile: PerformanceProfile,
    logger: logging.Logger,
    *,
    title: str,
    context: str,
) -> None:
    if not profile.events:
        return
    logger.info("%s: total %.3fs for %s", title, profile.total_s, context)
    grouped: dict[tuple[str, str], float] = defaultdict(float)
    for event in profile.events:
        grouped[(event.name, event.detail)] += event.duration_s
    for (name, detail), duration_s in sorted(
        grouped.items(), key=lambda item: item[1], reverse=True
    ):
        suffix = f" {detail}" if detail else ""
        logger.info("%s: %.3fs %s%s", title, duration_s, name, suffix)


@dataclass(slots=True, frozen=True)
class ViewportProfilingConfig:
    enabled: bool
    slow_frame_ms: float
    log_every_n_interactions: int


@dataclass(slots=True)
class ViewportInteractionProfiler:
    """Aggregates viewport interaction and render timings for logs only."""

    profile: PerformanceProfile
    config: ViewportProfilingConfig
    logger: logging.Logger
    _interaction_started_at: float | None = field(default=None, init=False)
    _interaction_events: int = field(default=0, init=False)
    _interaction_count: int = field(default=0, init=False)
    _render_started_at: float | None = field(default=None, init=False)

    def start_interaction(self, detail: str = "") -> None:
        if not self.config.enabled:
            return
        self._interaction_started_at = perf_counter()
        self._interaction_events = 0
        if detail:
            self.logger.debug("viewport interaction started: %s", detail)

    def note_interaction_event(self) -> None:
        if not self.config.enabled or self._interaction_started_at is None:
            return
        self._interaction_events += 1

    def end_interaction(self, detail: str = "") -> None:
        if not self.config.enabled or self._interaction_started_at is None:
            return
        elapsed = perf_counter() - self._interaction_started_at
        self._interaction_started_at = None
        self._interaction_count += 1
        event_detail = f"events={self._interaction_events}"
        if detail:
            event_detail = f"{event_detail}; {detail}"
        self.profile.add("viewport_interaction", elapsed, event_detail)
        log_every = max(self.config.log_every_n_interactions, 1)
        if self._interaction_count % log_every == 0:
            self.logger.info(
                "viewport interaction: %.1fms, %d event(s)%s",
                elapsed * 1000.0,
                self._interaction_events,
                f", {detail}" if detail else "",
            )

    def start_render(self, detail: str = "") -> None:
        if not self.config.enabled:
            return
        self._render_started_at = perf_counter()
        if detail:
            self.logger.debug("viewport render started: %s", detail)

    def end_render(self, detail: str = "") -> None:
        if not self.config.enabled or self._render_started_at is None:
            return
        elapsed = perf_counter() - self._render_started_at
        self._render_started_at = None
        self.profile.add("viewport_render_frame", elapsed, detail)
        elapsed_ms = elapsed * 1000.0
        if elapsed_ms >= self.config.slow_frame_ms:
            suffix = f" ({detail})" if detail else ""
            self.logger.info("slow viewport frame: %.1fms%s", elapsed_ms, suffix)
