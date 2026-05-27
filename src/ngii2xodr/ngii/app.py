"""App-level data objects built around the canonical NGII dataset."""

from __future__ import annotations

from dataclasses import dataclass

from ngii2xodr.ngii.data.dataset import NGIIDataset
from ngii2xodr.ngii.data.features import FeatureRef
from ngii2xodr.ngii.data.sanity import SanityReport
from ngii2xodr.profile import PerformanceProfile

__all__ = ["FeatureRef", "LoadedMap"]


@dataclass(slots=True)
class LoadedMap:
    """Everything downstream systems need for one loaded NGII map."""

    dataset: NGIIDataset
    sanity: SanityReport
    segmentation: object
    render_registry: object
    load_profile: PerformanceProfile
    segmentation_profile: PerformanceProfile
    viewport_profile: PerformanceProfile
