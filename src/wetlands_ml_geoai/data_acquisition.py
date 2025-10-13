"""Utilities for on-demand NAIP and wetlands data acquisition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:
    from geoai import download as geo_download
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "The geoai package is required for automatic data acquisition. "
        "Install geoai or disable the auto-download options."
    ) from exc


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NaipDownloadRequest:
    """Configuration for NAIP downloads."""

    aoi: BaseGeometry
    output_dir: Path
    year: Optional[int]
    max_items: Optional[int] = None
    overwrite: bool = False
    preview: bool = False
    target_resolution: Optional[float] = None


@dataclass(frozen=True)
class WetlandsDownloadRequest:
    """Configuration for wetlands delineation downloads."""

    aoi: BaseGeometry
    output_path: Path
    overwrite: bool = False


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_naip_tiles(request: NaipDownloadRequest) -> List[Path]:
    """Download NAIP tiles intersecting ``request.aoi`` and return file paths."""

    bbox = request.aoi.bounds
    output_dir = _ensure_directory(request.output_dir)
    LOGGER.info(
        "Downloading NAIP tiles (year=%s, max_items=%s) to %s",
        request.year,
        request.max_items,
        output_dir,
    )

    if not hasattr(geo_download, "download_naip"):
        raise AttributeError("geoai.download.download_naip is not available in this environment")

    naip_paths = geo_download.download_naip(  # type: ignore[attr-defined]
        bbox,
        output_dir=str(output_dir),
        year=request.year,
        max_items=request.max_items,
        overwrite=request.overwrite,
        preview=request.preview,
    )

    resolved = [Path(path).expanduser().resolve() for path in naip_paths]
    LOGGER.info("Fetched %s NAIP tile(s)", len(resolved))
    return resolved


def _resample_naip_tile(
    source_path: Path,
    target_resolution: float,
    destination_dir: Path,
) -> Path:
    """Resample a NAIP tile to the specified resolution in meters."""

    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError(f"NAIP raster {source_path} lacks a CRS; unable to resample.")
        if src.crs.is_geographic:
            raise ValueError(
                f"NAIP raster {source_path} uses a geographic CRS ({src.crs}); "
                "provide projected tiles before resampling."
            )
        transform, width, height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_resolution, target_resolution),
        )
        if width == src.width and height == src.height:
            # Already at desired resolution
            return source_path

        destination_dir = _ensure_directory(destination_dir)
        resampled_path = destination_dir / f"{source_path.stem}_resampled{source_path.suffix}"
        profile = src.profile.copy()
        profile.update(
            transform=transform,
            width=width,
            height=height,
        )
        with rasterio.open(resampled_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                )
    LOGGER.info("Resampled %s to %s m pixels -> %s", source_path, target_resolution, resampled_path)
    return resampled_path


def _transform_to_epsg4326(geometry: BaseGeometry, source_crs) -> BaseGeometry:
    if source_crs is None:
        return geometry
    try:
        transformed = transform_geom(source_crs, "EPSG:4326", geometry.__geo_interface__)
    except Exception:  # pragma: no cover - fallback if transform fails
        return geometry
    return shape(transformed)


def _compute_naip_union_extent(raster_paths: Iterable[Path]) -> Optional[BaseGeometry]:
    """Return the union of raster bounds as a shapely geometry."""

    geoms: List[BaseGeometry] = []
    for path in raster_paths:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            tile_geom = _transform_to_epsg4326(tile_geom, dataset.crs)
            if not tile_geom.is_valid:
                tile_geom = tile_geom.buffer(0)
            geoms.append(tile_geom)

    if not geoms:
        return None

    union_geom = unary_union(geoms)
    if hasattr(union_geom, "bounds"):
        LOGGER.info(
            "Computed NAIP union extent -> minx=%.6f, miny=%.6f, maxx=%.6f, maxy=%.6f",
            union_geom.bounds[0],
            union_geom.bounds[1],
            union_geom.bounds[2],
            union_geom.bounds[3],
        )
    return union_geom


def _download_wetlands_delineations(request: WetlandsDownloadRequest) -> Path:
    """Download wetlands delineations for the provided ``request.aoi`` bounds."""

    output_path = request.output_path
    _ensure_directory(output_path.parent)

    # Check if file exists and handle overwrite
    if output_path.exists() and not request.overwrite:
        LOGGER.info("Wetlands file already exists: %s (skipping download)", output_path)
        return output_path

    LOGGER.info("Downloading wetlands delineations to %s", output_path)

    # Get bounds from the AOI geometry
    bounds = request.aoi.bounds  # (minx, miny, maxx, maxy)
    
    # Format geometry as comma-separated bbox for NWI API
    geometry_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

    # NWI API endpoint
    nwi_url = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
    
    # Query parameters
    params = {
        "where": "1=1",
        "geometry": geometry_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }

    LOGGER.info("Querying NWI API for bounds: minx=%.6f, miny=%.6f, maxx=%.6f, maxy=%.6f",
                bounds[0], bounds[1], bounds[2], bounds[3])

    # Make the API request
    response = requests.get(nwi_url, params=params, timeout=120)
    response.raise_for_status()

    # Parse the GeoJSON response
    geojson_data = response.json()
    
    if "features" not in geojson_data or not geojson_data["features"]:
        LOGGER.warning("No wetlands features found for the specified area")
        # Create an empty GeoDataFrame with expected schema
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    else:
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")
        
        # Clip wetlands to the exact AOI extent (not just the bounding box)
        LOGGER.info("Clipping %d wetlands features to AOI extent", len(gdf))
        aoi_gdf = gpd.GeoDataFrame([{"geometry": request.aoi}], crs="EPSG:4326")
        gdf = gpd.clip(gdf, aoi_gdf)
        
        LOGGER.info("After clipping: %d wetlands features remain", len(gdf))

    # Save to file
    gdf.to_file(output_path, driver="GPKG")
    
    LOGGER.info("Saved %d wetlands features to %s", len(gdf), output_path)
    return output_path


__all__ = [
    "NaipDownloadRequest",
    "WetlandsDownloadRequest",
    "_compute_naip_union_extent",
    "_download_naip_tiles",
    "_download_wetlands_delineations",
    "_resample_naip_tile",
]


