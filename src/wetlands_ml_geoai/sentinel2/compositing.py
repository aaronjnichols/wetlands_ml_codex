"""Sentinel-2 seasonal compositing orchestration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401 - registers the rio accessor
import stackstac
import xarray as xr
from dask import compute as dask_compute
from pystac import Item
from pystac_client import Client
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from shapely import wkt
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from skimage.morphology import binary_dilation

from ..data_acquisition import (
    NaipDownloadRequest,
    WetlandsDownloadRequest,
    _compute_naip_union_extent,
    _download_naip_tiles,
    _download_wetlands_delineations,
    _resample_naip_tile,
)
from ..stacking import FLOAT_NODATA
from ..topography import TopographyStackConfig, prepare_topography_stack
from .manifests import NAIP_BAND_LABELS, write_stack_manifest
from .progress import LoggingProgressBar, RasterProgress, format_duration

SENTINEL_COLLECTION = "sentinel-2-l2a"
SENTINEL_BANDS = ["B03", "B04", "B05", "B06", "B08", "B11", "B12"]
SCL_MASK_VALUES = {3, 8, 9, 10, 11}
DEFAULT_SEASONS: Tuple[str, ...] = ("SPR", "SUM", "FAL")
SEASON_WINDOWS: Dict[str, Tuple[int, int, int, int]] = {
    "SPR": (3, 1, 5, 31),
    "SUM": (6, 1, 8, 31),
    "FAL": (9, 1, 11, 30),
}
SENTINEL_SCALE_FACTOR = 1 / 10000
SCL_ASSET_ID = "scl"

LOGGER = logging.getLogger(__name__)


def collect_naip_sources(candidates: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            for pattern in ("*.tif", "*.tiff"):
                for child in sorted(path.glob(pattern)):
                    if child.is_file():
                        paths.append(child)
        else:
            paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def prepare_naip_reference(
    sources: Sequence[Path],
    working_dir: Path,
    target_resolution: Optional[float] = None,
) -> Tuple[Path, Dict[str, Any], Sequence[str]]:
    if not sources:
        raise ValueError("No NAIP rasters were provided.")
    crs = None
    resolution = None
    band_count = None
    for src_path in sources:
        with rasterio.open(src_path) as dataset:
            if crs is None:
                crs = dataset.crs
                resolution = dataset.res
                band_count = dataset.count
            else:
                if dataset.crs != crs:
                    raise ValueError("All NAIP rasters must share the same CRS.")
                if not np.allclose(dataset.res, resolution, atol=1e-6):
                    raise ValueError("All NAIP rasters must share the same pixel size.")
                if dataset.count != band_count:
                    raise ValueError("All NAIP rasters must share the same band count.")

    if len(sources) == 1:
        source_path = Path(sources[0])
        reference_path = source_path
        if target_resolution is not None:
            reference_path = _resample_naip_tile(
                source_path,
                target_resolution,
                working_dir / "naip_resampled",
            )
        with rasterio.open(reference_path) as reference:
            profile = {
                "crs": reference.crs.to_string() if reference.crs else None,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
            }
            labels = NAIP_BAND_LABELS[: reference.count]
        return reference_path, profile, labels

    mosaic_path = working_dir / "naip_mosaic.tif"
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(path) for path in sources]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
        )
        profile.setdefault("tiled", True)
        profile.setdefault("compress", "deflate")
        profile.setdefault("BIGTIFF", "IF_SAFER")
        with rasterio.open(mosaic_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()

    reference_path = mosaic_path
    if target_resolution is not None:
        reference_path = _resample_naip_tile(
            mosaic_path,
            target_resolution,
            working_dir / "naip_resampled",
        )

    with rasterio.open(reference_path) as reference:
        profile = {
            "crs": reference.crs.to_string() if reference.crs else None,
            "transform": reference.transform,
            "width": reference.width,
            "height": reference.height,
        }
        labels = NAIP_BAND_LABELS[: reference.count]
    LOGGER.info("Generated NAIP reference at %s", reference_path)
    return reference_path, profile, labels


@dataclass(frozen=True)
class SeasonConfig:
    name: str
    start: date
    end: date


def parse_aoi(aoi: str) -> BaseGeometry:
    candidate = aoi.strip()
    path = Path(candidate)
    geom: Optional[BaseGeometry] = None

    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".gpkg", ".shp"}:
            gdf = gpd.read_file(path)
            if gdf.empty:
                raise ValueError(f"AOI file '{path}' contains no features.")
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            else:
                LOGGER.warning("AOI file %s has no CRS; assuming EPSG:4326 coordinates.", path)
            geom_series = gdf.geometry.dropna()
            if geom_series.empty:
                raise ValueError(f"AOI file '{path}' contains no valid geometries.")
            geom = geom_series.union_all()
        else:
            candidate = path.read_text(encoding="utf-8").strip()

    if geom is None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            geom = shape(payload.get("geometry", payload))
        elif isinstance(payload, list) and len(payload) == 4:
            geom = box(*payload)
        else:
            if "," in candidate and candidate.count(",") == 3:
                parts = [float(x) for x in candidate.split(",")]
                geom = box(*parts)
            else:
                geom = wkt.loads(candidate)

    if geom.is_empty:
        raise ValueError("AOI geometry is empty.")
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def season_date_range(year: int, season: str) -> SeasonConfig:
    if season not in SEASON_WINDOWS:
        raise ValueError(f"Unsupported season '{season}'. Supported: {sorted(SEASON_WINDOWS)}")
    sm, sd, em, ed = SEASON_WINDOWS[season]
    return SeasonConfig(season, date(year, sm, sd), date(year, em, ed))


def fetch_items(
    client: Client,
    geometry: BaseGeometry,
    season: str,
    years: Sequence[int],
    cloud_cover: float,
) -> List[Item]:
    items: Dict[str, Item] = {}
    geojson = mapping(geometry)
    for year in years:
        cfg = season_date_range(year, season)
        search = client.search(
            collections=[SENTINEL_COLLECTION],
            intersects=geojson,
            datetime=f"{cfg.start.isoformat()}/{cfg.end.isoformat()}",
            query={"eo:cloud_cover": {"lt": cloud_cover}},
        )
        for item in search.get_items():
            items[item.id] = item
    return list(items.values())


def stack_bands(
    items: Sequence[Item],
    bounds: Tuple[float, float, float, float],
    chunks: Optional[int] = 2048,
) -> xr.DataArray:
    if not items:
        raise ValueError("No Sentinel-2 items available for stacking.")
    epsg = items[0].properties.get("proj:epsg")
    if epsg is None:
        raise ValueError("Sentinel-2 item missing proj:epsg metadata.")
    asset_ids = []
    for band in SENTINEL_BANDS:
        asset_id = band_mapping = {
            "B03": "green",
            "B04": "red",
            "B05": "rededge1",
            "B06": "rededge2",
            "B08": "nir",
            "B11": "swir16",
            "B12": "swir22",
        }[band]
        if asset_id not in items[0].assets:
            raise ValueError(f"Missing Sentinel-2 asset '{asset_id}' on first item.")
        asset_ids.append(asset_id)
    data = stackstac.stack(
        items,
        assets=asset_ids,
        resolution=10,
        epsg=int(epsg),
        bounds_latlon=bounds,
        chunksize={"x": chunks, "y": chunks} if chunks else None,
        dtype="float64",
        rescale=False,
        properties=False,
    )
    data = data.reset_coords(drop=True)
    data.rio.write_crs(int(epsg), inplace=True)
    data = data.assign_coords({"band": SENTINEL_BANDS})
    return data * SENTINEL_SCALE_FACTOR


def stack_scl(
    items: Sequence[Item],
    bounds: Tuple[float, float, float, float],
    chunks: Optional[int] = 2048,
) -> xr.DataArray:
    epsg = items[0].properties.get("proj:epsg")
    if epsg is None:
        raise ValueError("Sentinel-2 item missing proj:epsg metadata.")
    if SCL_ASSET_ID not in items[0].assets:
        raise ValueError(f"Sentinel-2 item missing {SCL_ASSET_ID} asset.")
    scl = stackstac.stack(
        items,
        assets=[SCL_ASSET_ID],
        resolution=10,
        epsg=int(epsg),
        bounds_latlon=bounds,
        chunksize={"x": chunks, "y": chunks} if chunks else None,
        dtype="float64",
        rescale=False,
        properties=False,
    ).squeeze("band")
    scl = scl.reset_coords(drop=True)
    scl.rio.write_crs(int(epsg), inplace=True)
    return scl


def build_mask(scl: xr.DataArray, dilation: int) -> xr.DataArray:
    mask = xr.ones_like(scl, dtype=bool)
    for value in SCL_MASK_VALUES:
        mask = mask & (scl != value)

    if dilation > 0:
        cloudy = ~mask

        def _dilate(arr: np.ndarray) -> np.ndarray:
            return binary_dilation(arr, iterations=dilation)

        dilated = xr.apply_ufunc(
            _dilate,
            cloudy,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[bool],
        )
        mask = mask & ~dilated
    return mask


def seasonal_median(
    items: Sequence[Item],
    season: str,
    min_clear_obs: int,
    bounds: Tuple[float, float, float, float],
    mask_dilation: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    logging.info("Season %s: Building band stack from %d scenes...", season, len(items))
    t0 = time.perf_counter()
    stack = stack_bands(items, bounds)
    logging.info(
        "Season %s: Band stack prepared (lazy) in %s",
        season,
        format_duration(time.perf_counter() - t0),
    )

    logging.info("Season %s: Building SCL stack...", season)
    scl = stack_scl(items, bounds)

    mask = build_mask(scl, dilation=mask_dilation)
    mask3d = mask.expand_dims(band=stack.coords["band"]).transpose("time", "band", "y", "x")
    masked = stack.where(mask3d, other=np.nan)
    valid_counts = mask.astype("int16").sum(dim="time")

    median = masked.median(dim="time", skipna=True)
    clear_enough = (
        valid_counts >= min_clear_obs
    ).expand_dims(band=median.coords["band"]).transpose("band", "y", "x")
    median = median.where(clear_enough)

    median.rio.write_crs(stack.rio.crs, inplace=True)
    median.rio.write_transform(stack.rio.transform(), inplace=True)

    median = median.astype("float32")
    valid_counts = valid_counts.astype("int16")

    compute_label = f"Sentinel-2 {season} median"
    logging.info("Season %s: Computing median composite...", season)
    with LoggingProgressBar(compute_label, step=5):
        median, valid_counts = dask_compute(median, valid_counts)

    data = median.values
    if np.isfinite(data).any():
        max_val = float(np.nanmax(data))
        min_val = float(np.nanmin(data))
        if max_val > 1.0 + 1e-3:
            raise ValueError(f"Season {season}: reflectance exceeds 1.0 (max={max_val})")
        if min_val < -1e-3:
            raise ValueError(f"Season {season}: reflectance below 0.0 (min={min_val})")
    else:
        logging.warning("Season %s produced no clear pixels after masking.", season)

    return median, valid_counts


def write_dataarray(
    array: xr.DataArray,
    path: Path,
    band_labels: Sequence[str],
    nodata: float = FLOAT_NODATA,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.sizes["band"] != len(band_labels):
        raise ValueError("Band label count does not match array bands")

    logging.info("Writing raster to %s", path.name)
    array = array.assign_coords({"band": np.arange(1, array.sizes["band"] + 1)})
    array.rio.write_nodata(nodata, inplace=True)
    array.rio.to_raster(
        path,
        dtype="float32",
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(path, "r+") as dst:
        for idx, label in enumerate(band_labels, start=1):
            dst.set_band_description(idx, label)


def concatenate_seasons(
    seasonal: Dict[str, xr.DataArray],
    order: Sequence[str],
) -> Tuple[xr.DataArray, List[str]]:
    arrays = []
    labels: List[str] = []
    for season in order:
        arrays.append(seasonal[season])
        labels.extend([f"S2_{season}_{band}" for band in SENTINEL_BANDS])
    combined = xr.concat(arrays, dim="band")
    combined.rio.write_crs(arrays[0].rio.crs, inplace=True)
    combined.rio.write_transform(arrays[0].rio.transform(), inplace=True)
    return combined.astype("float32"), labels


def run_pipeline(
    aoi: str,
    years: Sequence[int],
    output_dir: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    cloud_cover: float = 60.0,
    min_clear_obs: int = 3,
    stac_url: str = "https://earth-search.aws.element84.com/v1",
    naip_paths: Optional[Sequence[Path]] = None,
    auto_download_naip: bool = False,
    auto_download_naip_year: Optional[int] = None,
    auto_download_naip_max_items: Optional[int] = None,
    auto_download_naip_overwrite: bool = False,
    auto_download_naip_preview: bool = False,
    auto_download_wetlands: bool = False,
    wetlands_output_path: Optional[Path] = None,
    wetlands_overwrite: bool = False,
    naip_target_resolution: Optional[float] = None,
    mask_dilation: int = 0,
    auto_download_topography: bool = False,
    topography_cache_dir: Optional[Path] = None,
    topography_buffer_meters: float = 200.0,
    topography_tpi_small: float = 30.0,
    topography_tpi_large: float = 150.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    naip_candidates: List[Path] = list(naip_paths) if naip_paths else []
    naip_sources = collect_naip_sources(naip_candidates) if naip_candidates else []
    if naip_candidates and not naip_sources:
        inputs = ", ".join(str(path) for path in naip_candidates)
        raise FileNotFoundError(f"No NAIP rasters found for --naip-path inputs: {inputs}")
    geom = parse_aoi(aoi)
    client = Client.open(stac_url)

    downloaded_naip: List[Path] = []
    if auto_download_naip:
        if naip_sources:
            logging.info("NAIP sources already provided; skipping auto download.")
        else:
            naip_request = NaipDownloadRequest(
                aoi=geom,
                output_dir=output_dir / "naip_auto",
                year=auto_download_naip_year,
                max_items=auto_download_naip_max_items,
                overwrite=auto_download_naip_overwrite,
                preview=auto_download_naip_preview,
                target_resolution=naip_target_resolution,
            )
            downloaded_naip = _download_naip_tiles(naip_request)
            naip_sources.extend(downloaded_naip)

    has_naip = bool(naip_sources)

    wetlands_path: Optional[Path] = wetlands_output_path
    if auto_download_wetlands:
        if not has_naip:
            raise ValueError(
                "Auto downloading wetlands requires NAIP tiles (either existing or downloaded)."
            )
        union_extent = _compute_naip_union_extent(naip_sources)
        if union_extent is None:
            raise RuntimeError("Unable to determine union extent from NAIP tiles.")
        target_path = wetlands_output_path or (output_dir / "wetlands_auto" / "wetlands.gpkg")
        wetlands_request = WetlandsDownloadRequest(
            aoi=union_extent,
            output_path=target_path,
            overwrite=wetlands_overwrite,
        )
        wetlands_path = _download_wetlands_delineations(wetlands_request)
        logging.info("Wetlands delineations saved to %s", wetlands_path)

    seasonal_data: Dict[str, xr.DataArray] = {}
    bounds = geom.bounds
    progress = RasterProgress(total=len(seasons))
    for season in seasons:
        logging.info("=" * 60)
        logging.info("Processing season %s", season)
        logging.info("=" * 60)
        raster_label = f"Sentinel-2 {season} median (7 band)"

        logging.info("Season %s: Fetching Sentinel-2 scenes...", season)
        items = fetch_items(client, geom, season, years, cloud_cover)
        logging.info("Season %s: Found %d scenes", season, len(items))

        if not items:
            logging.warning("No Sentinel-2 scenes found for season %s", season)
            progress.skip(raster_label)
            continue

        progress.start(raster_label)
        median, counts = seasonal_median(items, season, min_clear_obs, bounds, mask_dilation)
        seasonal_data[season] = median

        min_count = int(counts.min().item())
        max_count = int(counts.max().item())
        mean_count = float(counts.mean().item())
        logging.info(
            "Season %s clear obs stats -> min: %s, mean: %.2f, max: %s",
            season,
            min_count,
            mean_count,
            max_count,
        )

        band_labels = [f"S2_{season}_{band}" for band in SENTINEL_BANDS]
        output_path = output_dir / f"s2_{season.lower()}_median_7band.tif"
        write_dataarray(median, output_path, band_labels)
        progress.finish(raster_label)
        logging.info("Season %s: Complete!", season)

    if len(seasonal_data) != len(seasons):
        logging.warning("Skipping multi-season outputs; not all seasons were generated.")
        return

    progress.extend(1 + int(has_naip))
    composite_label = "Seasonal 21-band stack"
    progress.start(composite_label)
    combined21, labels21 = concatenate_seasons(seasonal_data, seasons)
    combined21_path = output_dir / "s2_sprsumfal_median_21band.tif"
    write_dataarray(combined21, combined21_path, labels21)
    progress.finish(composite_label)

    if has_naip:
        stack_label = "Stack manifest generation"
        progress.start(stack_label)
        naip_reference_path, reference_profile, naip_band_labels = prepare_naip_reference(
            naip_sources,
            output_dir,
            target_resolution=naip_target_resolution,
        )
        extra_sources = None
        topography_path: Optional[Path] = None
        if auto_download_topography:
            topo_output_dir = output_dir / "topography"
            topo_config = TopographyStackConfig(
                aoi=geom,
                target_grid_path=naip_reference_path,
                output_dir=topo_output_dir,
                buffer_meters=topography_buffer_meters,
                tpi_small_radius=topography_tpi_small,
                tpi_large_radius=topography_tpi_large,
                cache_dir=topography_cache_dir,
            )
            topography_path = prepare_topography_stack(topo_config)
            extra_sources = [
                {
                    "type": "topography",
                    "path": str(topography_path.resolve()),
                    "band_labels": [
                        "Slope",
                        "TPI_small",
                        "TPI_large",
                        "DepressionDepth",
                    ],
                    "resample": "bilinear",
                    "nodata": FLOAT_NODATA,
                    "description": topo_config.description,
                }
            ]
        manifest_path = write_stack_manifest(
            output_dir=output_dir,
            naip_path=naip_reference_path,
            naip_labels=naip_band_labels,
            sentinel_path=combined21_path,
            sentinel_labels=labels21,
            reference_profile=reference_profile,
            extra_sources=extra_sources,
        )
        progress.finish(stack_label)
        logging.info("Stack manifest written to %s", manifest_path)
        if topography_path:
            logging.info("Topography raster stored at %s", topography_path)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aoi", required=True, help="AOI path or geometry (GeoJSON, WKT, or bbox)")
    parser.add_argument("--years", required=True, nargs="+", type=int, help="Years to include (e.g., 2022 2023)")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated GeoTIFFs",
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=60.0,
        help="Maximum eo:cloud_cover percentage for Sentinel-2 items",
    )
    parser.add_argument(
        "--min-clear-obs",
        type=int,
        default=3,
        help="Minimum number of clear observations per pixel per season",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Season codes to process (subset of SPR SUM FAL)",
    )
    parser.add_argument(
        "--stac-url",
        default="https://earth-search.aws.element84.com/v1",
        help="STAC API endpoint for Sentinel-2 L2A",
    )
    parser.add_argument(
        "--naip-path",
        type=Path,
        nargs="+",
        help=(
            "Optional NAIP GeoTIFF(s) or directories to define the target grid. "
            "Multiple rasters will be mosaicked automatically."
        ),
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=0,
        help="Number of pixels to dilate the SCL cloud mask",
    )
    parser.add_argument(
        "--auto-download-naip",
        action="store_true",
        help="Automatically download NAIP tiles for the AOI if --naip-path is not provided.",
    )
    parser.add_argument(
        "--auto-download-naip-year",
        type=int,
        help="NAIP imagery year to request when auto downloading.",
    )
    parser.add_argument(
        "--auto-download-naip-max-items",
        type=int,
        help="Maximum number of NAIP tiles to fetch when auto downloading.",
    )
    parser.add_argument(
        "--auto-download-naip-overwrite",
        action="store_true",
        help="Allow overwriting existing auto-downloaded NAIP tiles.",
    )
    parser.add_argument(
        "--auto-download-naip-preview",
        action="store_true",
        help="Use preview mode when downloading NAIP (quicklook tiles).",
    )
    parser.add_argument(
        "--auto-download-wetlands",
        action="store_true",
        help="Automatically download wetlands delineations covering the NAIP union extent.",
    )
    parser.add_argument(
        "--wetlands-output-path",
        type=Path,
        help="Destination path for downloaded wetlands delineations (GeoPackage).",
    )
    parser.add_argument(
        "--wetlands-overwrite",
        action="store_true",
        help="Allow overwriting existing wetlands download outputs.",
    )
    parser.add_argument(
        "--naip-target-resolution",
        type=float,
        help="Optional NAIP resampling resolution in meters.",
    )
    parser.add_argument(
        "--auto-download-topography",
        action="store_true",
        help="Automatically derive LiDAR-based topographic features for the stack.",
    )
    parser.add_argument(
        "--topography-buffer-meters",
        type=float,
        default=200.0,
        help="Buffer distance (meters) applied to the AOI before DEM download.",
    )
    parser.add_argument(
        "--topography-tpi-small",
        type=float,
        default=30.0,
        help="Radius in meters for the small-scale TPI.",
    )
    parser.add_argument(
        "--topography-tpi-large",
        type=float,
        default=150.0,
        help="Radius in meters for the large-scale TPI.",
    )
    parser.add_argument(
        "--topography-cache-dir",
        type=Path,
        help="Optional directory for caching raw DEM downloads.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )


def run_from_args(args: argparse.Namespace) -> None:
    run_pipeline(
        aoi=args.aoi,
        years=args.years,
        output_dir=args.output_dir,
        seasons=tuple(args.seasons),
        cloud_cover=args.cloud_cover,
        min_clear_obs=args.min_clear_obs,
        stac_url=args.stac_url,
        naip_paths=tuple(args.naip_path) if args.naip_path else None,
        auto_download_naip=args.auto_download_naip,
        auto_download_naip_year=args.auto_download_naip_year,
        auto_download_naip_max_items=args.auto_download_naip_max_items,
        auto_download_naip_overwrite=args.auto_download_naip_overwrite,
        auto_download_naip_preview=args.auto_download_naip_preview,
        auto_download_wetlands=args.auto_download_wetlands,
        wetlands_output_path=args.wetlands_output_path,
        wetlands_overwrite=args.wetlands_overwrite,
        naip_target_resolution=args.naip_target_resolution,
        mask_dilation=args.mask_dilation,
        auto_download_topography=args.auto_download_topography,
        topography_cache_dir=args.topography_cache_dir,
        topography_buffer_meters=args.topography_buffer_meters,
        topography_tpi_small=args.topography_tpi_small,
        topography_tpi_large=args.topography_tpi_large,
    )


__all__ = [
    "configure_parser",
    "run_from_args",
    "run_pipeline",
    "collect_naip_sources",
    "prepare_naip_reference",
]

