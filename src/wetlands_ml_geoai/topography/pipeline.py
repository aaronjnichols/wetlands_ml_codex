"""Orchestrates DEM download and derivative computation."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from .config import TopographyStackConfig
from .download import download_dem_products, fetch_dem_inventory
from .processing import write_topography_raster


LOGGER = logging.getLogger(__name__)


def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]


def prepare_topography_stack(config: TopographyStackConfig) -> Path:
    """Download 3DEP tiles and compute derivative raster for ``config`` AOI."""

    buffered_geom = _buffer_geometry(config.aoi, config.buffer_meters)
    geojson = mapping(buffered_geom)
    bbox = buffered_geom.bounds

    LOGGER.info("Fetching DEM inventory for buffered AOI (buffer=%sm)", config.buffer_meters)
    products = fetch_dem_inventory(
        geojson,
        bbox=bbox,
        max_results=config.max_products,
    )
    if not products:
        raise RuntimeError("No 3DEP DEM products found for buffered AOI; adjust buffer or verify coverage.")

    LOGGER.info("DEM inventory returned %s product(s)", len(products))

    cache_dir = config.cache_dir or config.output_dir / "raw"
    LOGGER.info("Downloading DEM products to %s", cache_dir)
    dem_paths = download_dem_products(products, cache_dir)
    LOGGER.info("DEM download complete; %s file(s) ready for mosaicking", len(dem_paths))

    topography_path = config.output_dir / "topography_stack.tif"
    LOGGER.info("Computing topographic derivatives -> %s", topography_path)
    return write_topography_raster(config, dem_paths, topography_path)


__all__ = ["prepare_topography_stack"]


