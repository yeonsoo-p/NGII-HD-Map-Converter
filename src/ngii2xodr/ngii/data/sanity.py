"""Structured warning/action logs produced while loading NGII data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class SanityWarning:
    code: str
    message: str
    layer_name: str | None = None
    feature_id: str | None = None
    source_path: Path | None = None


@dataclass(slots=True, frozen=True)
class SanityAction:
    code: str
    message: str
    before: dict[str, Any]
    after: dict[str, Any]
    layer_name: str | None = None
    feature_id: str | None = None
    source_path: Path | None = None


@dataclass(slots=True)
class SanityReport:
    warnings: list[SanityWarning] = field(default_factory=list)
    actions: list[SanityAction] = field(default_factory=list)

    def warn(
        self,
        code: str,
        message: str,
        *,
        layer_name: str | None = None,
        feature_id: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.warnings.append(
            SanityWarning(
                code=code,
                message=message,
                layer_name=layer_name,
                feature_id=feature_id,
                source_path=source_path,
            )
        )

    def action(
        self,
        code: str,
        message: str,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        layer_name: str | None = None,
        feature_id: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.actions.append(
            SanityAction(
                code=code,
                message=message,
                before=before,
                after=after,
                layer_name=layer_name,
                feature_id=feature_id,
                source_path=source_path,
            )
        )
