"""Configuration objects for topographic stack preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class TopographyStackConfig:
    """Parameters controlling LiDAR-derived topographic processing."""

    aoi: BaseGeometry
    target_grid_path: Path
    output_dir: Path
    buffer_meters: float = 200.0
    tpi_small_radius: float = 30.0
    """Radius of the small-scale TPI window in meters (converted to pixels internally)."""
    tpi_large_radius: float = 150.0
    """Radius of the large-scale TPI window in meters (converted to pixels internally)."""
    cache_dir: Optional[Path] = None
    max_products: int = 500
    description: str = "USGS 3DEP derived topography"


__all__ = ["TopographyStackConfig"]


