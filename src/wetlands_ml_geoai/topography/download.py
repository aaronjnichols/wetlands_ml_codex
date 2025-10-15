"""Helpers for querying and downloading USGS 3DEP DEM tiles."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import requests


LOGGER = logging.getLogger(__name__)


TNM_BASE_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_DATASETS: Sequence[str] = (
    "Digital Elevation Model (DEM) 1 meter",
    "1-meter DEM",
    "3DEP Elevation: DEM (1 meter)",
    "Seamless 1-meter DEM (Limited Availability)",
    "DEM Source (OPR)",
    "1/9 arc-second DEM",
    "1/3 arc-second DEM",
    "1 arc-second DEM",
)


@dataclass(frozen=True)
class DemProduct:
    product_id: str
    download_url: str
    size: int
    bbox: List[float]
    last_updated: Optional[str]

    def filename(self) -> str:
        safe_id = (
            self.product_id.replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )
        return f"{safe_id}.tif"


def _build_query_params(
    bbox: Tuple[float, float, float, float],
    dataset: str,
    max_results: int,
) -> dict:
    return {
        "bbox": ",".join(str(v) for v in bbox),
        "datasets": dataset,
        "prodFormats": "GeoTIFF,IMG,TIF",
        "outputFormat": "JSON",
        "max": max_results,
    }


def _extract_primary_url(urls: Any) -> Optional[str]:
    if not urls:
        return None

    if isinstance(urls, list):
        for entry in urls:
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("URL")
                if url:
                    return str(url)
            elif isinstance(entry, str) and entry.startswith("http"):
                return entry

    if isinstance(urls, dict):
        for value in urls.values():
            if isinstance(value, dict):
                url = value.get("url") or value.get("URL")
                if url:
                    return str(url)
            elif isinstance(value, str) and value.startswith("http"):
                return value

    return None


def fetch_dem_inventory(
    aoi_geojson: dict,
    bbox: Tuple[float, float, float, float],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    session: Optional[requests.Session] = None,
    max_results: int = 100,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> List[DemProduct]:
    """Return DEM products for ``bbox`` scanning preferred datasets."""

    client = session or requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv(
            "USGS_USER_AGENT",
            "wetlands-ml-geoai/0.1 (https://github.com/atwellconsulting/wetlands_ml_codex)",
        ),
    }
    api_key = os.getenv("USGS_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    for dataset in datasets:
        params = _build_query_params(bbox, dataset, max_results)
        attempt = 0
        while True:
            attempt += 1
            LOGGER.info(
                "Querying 3DEP dataset '%s' (attempt %s) for bbox=%s",
                dataset,
                attempt,
                bbox,
            )
            response = client.get(
                TNM_BASE_URL,
                params=params,
                timeout=180,
                headers=headers,
            )
            if response.status_code in {403, 429} and attempt <= retries:
                wait = backoff_seconds * attempt
                LOGGER.warning(
                    "Dataset '%s' request returned %s; retrying in %.1f s",
                    dataset,
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            break

        data = response.json()
        items = data.get("items", [])
        if not items:
            LOGGER.info("No products found for dataset '%s'", dataset)
            continue

        products: List[DemProduct] = []
        for item in items:
            primary_url = _extract_primary_url(item.get("urls"))
            if not primary_url:
                continue
            product_id = item.get("title") or item.get("entityId") or item.get("id", "dem_tile")
            products.append(
                DemProduct(
                    product_id=str(product_id),
                    download_url=primary_url,
                    size=int(item.get("sizeInBytes", item.get("unitSize", 0))),
                    bbox=item.get("boundingBox", []),
                    last_updated=item.get("lastUpdated"),
                )
            )

        if products:
            LOGGER.info("Found %s product(s) using dataset '%s'", len(products), dataset)
            return products

    LOGGER.warning(
        "No DEM products found after scanning datasets: %s",
        ", ".join(datasets),
    )
    return []


def download_dem_products(products: Iterable[DemProduct], output_dir: Path) -> List[Path]:
    """Download DEM products to ``output_dir``; return resolved paths."""

    paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for product in products:
        target_path = output_dir / product.filename()
        if target_path.exists():
            LOGGER.info("DEM tile cached -> %s", target_path)
            paths.append(target_path.resolve())
            continue

        LOGGER.info("Downloading DEM tile %s -> %s", product.product_id, target_path)
        with session.get(product.download_url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with target_path.open("wb") as dst:
                for chunk in resp.iter_content(chunk_size=1_048_576):
                    if chunk:
                        dst.write(chunk)
        LOGGER.info("Download complete -> %s", target_path)
        paths.append(target_path.resolve())

    return paths


__all__ = ["DemProduct", "fetch_dem_inventory", "download_dem_products"]


