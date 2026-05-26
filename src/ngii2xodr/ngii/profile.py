"""Small timing profile objects shared by loader and app consumers."""

from __future__ import annotations

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
class PerformanceProfile:
    events: list[ProfileEvent] = field(default_factory=list)

    @contextmanager
    def timed(self, name: str, detail: str = "") -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.events.append(ProfileEvent(name, perf_counter() - started_at, detail))

    def add(self, name: str, duration_s: float, detail: str = "") -> None:
        self.events.append(ProfileEvent(name, duration_s, detail))

    @property
    def total_s(self) -> float:
        return sum(event.duration_s for event in self.events)
