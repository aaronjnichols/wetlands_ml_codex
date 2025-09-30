"""Sentinel-2 seasonal compositing pipeline for wetlands_ml_geoai."""
import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
import rioxarray  # noqa: F401 - needed to register the rio accessor
import xarray as xr
from dask import compute as dask_compute
from dask.diagnostics import ProgressBar
from pystac import Item
from pystac_client import Client
import stackstac
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.errors import RasterioIOError
import rasterio
from shapely import wkt
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from skimage.morphology import binary_dilation

SENTINEL_COLLECTION = "sentinel-2-l2a"
SENTINEL_BANDS = ["B03", "B04", "B05", "B06", "B08", "B11", "B12"]
SCL_MASK_VALUES = {3, 8, 9, 10, 11}
DEFAULT_SEASONS = ("SPR", "SUM", "FAL")
SEASON_WINDOWS = {
    "SPR": (3, 1, 5, 31),
    "SUM": (6, 1, 8, 31),
    "FAL": (9, 1, 11, 30),
}
SENTINEL_SCALE_FACTOR = 1 / 10000
FLOAT_NODATA = -9999.0
NAIP_BAND_LABELS = ["NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR"]

SENTINEL_ASSET_MAP = {
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
SCL_ASSET_ID = "scl"

LOGGER = logging.getLogger(__name__)


class RasterProgress:
    """Simple logger-backed tracker for raster exports."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0

    def extend(self, count: int) -> None:
        if count > 0:
            self.total += count

    def start(self, label: str) -> None:
        total = max(self.total, self.completed + 1)
        logging.info("Starting raster %d/%d: %s", self.completed + 1, total, label)

    def finish(self, label: str) -> None:
        self.completed += 1
        total = max(self.total, self.completed)
        logging.info("Finished raster %d/%d: %s", self.completed, total, label)

    def skip(self, label: str) -> None:
        if self.total > self.completed:
            self.total -= 1
        total = max(self.total, self.completed)
        logging.info("Skipping raster %s; progress %d/%d.", label, self.completed, total)



def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes}m {remainder:.2f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e9:.2f} ns"






def affine_to_list(transform):
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def compute_naip_scaling(dataset: rasterio.io.DatasetReader) -> Optional[float]:
    if dataset.nodata is None and not np.issubdtype(np.dtype(dataset.dtypes[0]), np.integer):
        return None
    dtype = np.dtype(dataset.dtypes[0])
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.max <= 0:
            return None
        return float(info.max)
    return None


def write_stack_manifest(
    output_dir: Path,
    naip_path: Path,
    naip_labels: Sequence[str],
    sentinel_path: Path,
    sentinel_labels: Sequence[str],
    reference_profile: Dict[str, Any],
) -> Path:
    manifest = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "grid": {
            "crs": reference_profile.get("crs"),
            "transform": affine_to_list(reference_profile["transform"]),
            "width": reference_profile["width"],
            "height": reference_profile["height"],
            "pixel_size": [reference_profile["transform"].a, -reference_profile["transform"].e],
            "dtype": "float32",
            "nodata": FLOAT_NODATA,
        },
        "sources": [],
    }

    naip_entry = {
        "type": "naip",
        "path": str(Path(naip_path).resolve()),
        "band_labels": list(naip_labels),
        "scale_max": None,
        "nodata": None,
    }

    with rasterio.open(naip_path) as src:
        naip_entry["nodata"] = src.nodata
        naip_entry["dtype"] = src.dtypes[0]
        scale = compute_naip_scaling(src)
        if scale:
            naip_entry["scale_max"] = scale
    manifest["sources"].append(naip_entry)

    sentinel_entry = {
        "type": "sentinel",
        "path": str(Path(sentinel_path).resolve()),
        "band_labels": list(sentinel_labels),
        "resample": "bilinear",
        "nodata": FLOAT_NODATA,
    }
    manifest["sources"].append(sentinel_entry)

    manifest_path = output_dir / "stack_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
class PercentReporter:
    """Log incremental percent completion for long-running steps."""

    def __init__(self, label: str, total: int, step: int = 10) -> None:
        self.label = label
        self.total = max(total, 1)
        self.step = max(step, 1)
        self._last_percent = -self.step
        self._start_time = time.perf_counter()
        self.update(0)

    def update(self, completed: int) -> None:
        completed = max(0, min(completed, self.total))
        percent = int((completed / self.total) * 100)
        if percent >= self._last_percent + self.step or percent >= 100 or self._last_percent < 0:
            elapsed = time.perf_counter() - self._start_time
            logging.info(
                "%s progress: %d%% (%d/%d, elapsed %s)",
                self.label,
                percent,
                completed,
                self.total,
                format_duration(elapsed),
            )
            self._last_percent = percent

    def complete(self) -> None:
        self.update(self.total)


class LoggingProgressBar(ProgressBar):
    """Dask progress bar that emits percent updates via logging."""

    def __init__(self, label: str, step: int = 10) -> None:
        super().__init__(minimum=0)
        self.label = label
        self.step = max(step, 1)
        self._last_percent = -self.step
        self._file = None

    def _draw_bar(self, frac, elapsed):
        percent = int(frac * 100)
        if percent >= self._last_percent + self.step or percent >= 100 or self._last_percent < 0:
            try:
                from dask.utils import format_time
            except ImportError:
                format_time = lambda value: f"{value:.2f}s"
            logging.info("%s progress: %d%% (elapsed %s)", self.label, percent, format_time(elapsed))
            self._last_percent = percent


@dataclass(frozen=True)
class SeasonConfig:
    name: str
    start: date
    end: date


def parse_aoi(aoi: str) -> BaseGeometry:
    """Return a shapely geometry from a path or inline geometry definition."""
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
                LOGGER.warning("AOI file %s has no CRS; assuming coordinates are EPSG:4326.", path)
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
    """Query the STAC API for Sentinel-2 items matching the filters."""
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


def build_mask(scl: xr.DataArray, dilation: int) -> xr.DataArray:
    """Create a boolean clear-sky mask from the SCL band."""
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


def stack_bands(items: Sequence[Item], bounds: Tuple[float, float, float, float], chunks: Optional[int] = 2048) -> xr.DataArray:
    if not items:
        raise ValueError("No Sentinel-2 items available for stacking.")
    epsg = items[0].properties.get("proj:epsg")
    if epsg is None:
        raise ValueError("Sentinel-2 item missing proj:epsg metadata.")
    asset_ids = []
    for band in SENTINEL_BANDS:
        asset_id = SENTINEL_ASSET_MAP.get(band)
        if asset_id is None:
            raise KeyError(f"No asset mapping defined for band {band}.")
        asset_ids.append(asset_id)
    missing_assets = [asset_id for asset_id in asset_ids if asset_id not in items[0].assets]
    if missing_assets:
        raise ValueError("Missing Sentinel-2 assets on first item: " + ", ".join(missing_assets))
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


def stack_scl(items: Sequence[Item], bounds: Tuple[float, float, float, float], chunks: Optional[int] = 2048) -> xr.DataArray:
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


def seasonal_median(
    items: Sequence[Item],
    season: str,
    min_clear_obs: int,
    bounds: Tuple[float, float, float, float],
    mask_dilation: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    stack = stack_bands(items, bounds)
    scl = stack_scl(items, bounds)
    mask = build_mask(scl, dilation=mask_dilation)

    mask3d = mask.expand_dims(band=stack.coords["band"]).transpose("time", "band", "y", "x")
    masked = stack.where(mask3d, other=np.nan)
    valid_counts = mask.astype("int16").sum(dim="time")

    median = masked.median(dim="time", skipna=True)
    clear_enough = (valid_counts >= min_clear_obs).expand_dims(band=median.coords["band"]).transpose(
        "band", "y", "x"
    )
    median = median.where(clear_enough)

    median.rio.write_crs(stack.rio.crs, inplace=True)
    median.rio.write_transform(stack.rio.transform(), inplace=True)

    median = median.astype("float32")
    valid_counts = valid_counts.astype("int16")

    compute_label = f"Sentinel-2 {season} median compute"
    with LoggingProgressBar(compute_label):
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


def concatenate_seasons(seasonal: Dict[str, xr.DataArray], order: Sequence[str]) -> Tuple[xr.DataArray, List[str]]:
    arrays = []
    labels: List[str] = []
    for season in order:
        arrays.append(seasonal[season])
        labels.extend([f"S2_{season}_{band}" for band in SENTINEL_BANDS])
    combined = xr.concat(arrays, dim="band")
    expected = len(order) * len(SENTINEL_BANDS)
    if combined.sizes["band"] != expected:
        raise ValueError("Unexpected band count in seasonal concatenation")
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
    naip_path: Optional[Path] = None,
    mask_dilation: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    geom = parse_aoi(aoi)
    client = Client.open(stac_url)

    seasonal_data: Dict[str, xr.DataArray] = {}
    bounds = geom.bounds
    progress = RasterProgress(total=len(seasons))
    for season in seasons:
        logging.info("Processing season %s", season)
        raster_label = f"Sentinel-2 {season} median (7 band)"
        items = fetch_items(client, geom, season, years, cloud_cover)
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

    if len(seasonal_data) != len(seasons):
        logging.warning("Skipping multi-season outputs; not all seasons were generated.")
        return

    progress.extend(1 + int(bool(naip_path)))
    composite_label = "Seasonal 21-band stack"
    progress.start(composite_label)
    combined21, labels21 = concatenate_seasons(seasonal_data, seasons)
    combined21_path = output_dir / "s2_sprsumfal_median_21band.tif"
    write_dataarray(combined21, combined21_path, labels21)
    progress.finish(composite_label)

    if naip_path:
        stack_label = "Stack manifest generation"
        progress.start(stack_label)
        with rasterio.open(naip_path) as reference:
            reference_profile: Dict[str, Any] = {
                "crs": reference.crs.to_string() if reference.crs else None,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
            }
            naip_band_labels = NAIP_BAND_LABELS[: reference.count]
        manifest_path = write_stack_manifest(
            output_dir=output_dir,
            naip_path=naip_path,
            naip_labels=naip_band_labels,
            sentinel_path=combined21_path,
            sentinel_labels=labels21,
            reference_profile=reference_profile,
        )
        progress.finish(stack_label)
        logging.info("Stack manifest written to %s", manifest_path)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Sentinel-2 seasonal composites.")
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
        help="Optional NAIP GeoTIFF to define the target grid for Sentinel alignment",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=0,
        help="Number of pixels to dilate the SCL cloud mask",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    run_pipeline(
        aoi=args.aoi,
        years=args.years,
        output_dir=args.output_dir,
        seasons=tuple(args.seasons),
        cloud_cover=args.cloud_cover,
        min_clear_obs=args.min_clear_obs,
        stac_url=args.stac_url,
        naip_path=args.naip_path,
        mask_dilation=args.mask_dilation,
    )


if __name__ == "__main__":
    main()
