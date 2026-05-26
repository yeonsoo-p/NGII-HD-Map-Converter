"""App-level data objects built around the canonical NGII dataset."""

from __future__ import annotations

from dataclasses import dataclass

from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.profile import PerformanceProfile


@dataclass(slots=True, frozen=True)
class FeatureRef:
    """Canonical pointer to one feature in an :class:`NGIIDataset`."""

    layer_attr: str
    feature_id: str


@dataclass(slots=True)
class LoadedMap:
    """Everything downstream systems need for one loaded NGII map."""

    dataset: NGIIDataset
    segmentation: object
    render_registry: object
    load_profile: PerformanceProfile
    segmentation_profile: PerformanceProfile
