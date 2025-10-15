"""DEM mosaicking and derivative computation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from scipy import ndimage

from ..stacking import FLOAT_NODATA
from .config import TopographyStackConfig


LOGGER = logging.getLogger(__name__)


def _read_transform(reference_path: Path) -> tuple[Affine, int, int, float, str]:
    with rasterio.open(reference_path) as ref:
        res_x, res_y = ref.res
        pixel_size = float((abs(res_x) + abs(res_y)) / 2)
        return ref.transform, ref.width, ref.height, pixel_size, ref.crs.to_string() if ref.crs else None


def _mosaic_dem(paths: Iterable[Path], target_transform: Affine, target_width: int, target_height: int, target_crs: str) -> np.ndarray:
    datasets = [rasterio.open(path) for path in paths]
    if not datasets:
        raise ValueError("No DEM tiles provided")
    src_crs = datasets[0].crs
    try:
        mosaicked, transform = merge(datasets, nodata=np.nan)
    finally:
        for dataset in datasets:
            dataset.close()
    result = np.full((target_height, target_width), np.nan, dtype="float32")
    warp.reproject(
        source=mosaicked[0],
        destination=result,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return result


def _compute_slope(dem: np.ndarray, pixel_size: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(dem, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    return slope.astype("float32")


def _box_mean(dem: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return dem.astype("float32", copy=False)

    window_size = radius_pixels * 2 + 1
    valid_mask = np.isfinite(dem).astype("float32")
    values = np.nan_to_num(dem, nan=0.0).astype("float32") * valid_mask
    window_area = float(window_size * window_size)

    sum_avg = ndimage.uniform_filter(
        values,
        size=window_size,
        mode="nearest",
    ) * window_area
    count_avg = ndimage.uniform_filter(
        valid_mask,
        size=window_size,
        mode="nearest",
    ) * window_area

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_avg / count_avg
    mean[count_avg == 0] = np.nan
    return mean.astype("float32")


def _compute_tpi(dem: np.ndarray, radius: float, pixel_size: float) -> np.ndarray:
    radius_pixels = int(max(radius / pixel_size, 1))
    local_mean = _box_mean(dem, radius_pixels)
    tpi = dem - local_mean
    return tpi.astype("float32")


def _compute_depression_depth(dem: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 2)
    filled = ndimage.grey_closing(dem, footprint=structure, mode="nearest")
    depth = filled - dem
    depth[depth < 0] = 0
    return depth.astype("float32")


def write_topography_raster(config: TopographyStackConfig, dem_paths: Iterable[Path], output_path: Path) -> Path:
    dem_paths_list = list(dem_paths)
    transform, width, height, pixel_size, crs = _read_transform(config.target_grid_path)
    LOGGER.info(
        "Mosaicking %s DEM tile(s) into target grid %s",
        len(dem_paths_list),
        config.target_grid_path,
    )
    dem = _mosaic_dem(dem_paths_list, transform, width, height, crs)
    dem_mask = np.isnan(dem)

    LOGGER.info("Computing slope raster")
    slope = _compute_slope(dem, pixel_size)
    slope[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing TPI (radius=%sm)", config.tpi_small_radius)
    tpi_small = _compute_tpi(dem, config.tpi_small_radius, pixel_size)
    tpi_small[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing TPI (radius=%sm)", config.tpi_large_radius)
    tpi_large = _compute_tpi(dem, config.tpi_large_radius, pixel_size)
    tpi_large[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing depression depth")
    depression = _compute_depression_depth(dem)
    depression[dem_mask] = FLOAT_NODATA

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": FLOAT_NODATA,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    bands = np.stack([slope, tpi_small, tpi_large, depression])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing derivatives -> %s", output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(bands)
        dst.set_band_description(1, "Slope")
        dst.set_band_description(2, "TPI_small")
        dst.set_band_description(3, "TPI_large")
        dst.set_band_description(4, "DepressionDepth")

    LOGGER.info("Wrote topography raster -> %s", output_path)
    return output_path


__all__ = ["write_topography_raster"]


