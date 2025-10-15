# Wetlands ML GeoAI Pipeline Overview

This document summarizes the major data and control flows in the current codebase. It serves as a baseline reference before the refactor described in `ref.plan.md`.

## Training Pipelines

- Entrypoints: `train.py` (Mask R-CNN) and `train_unet.py` (UNet) located both at project root (CLI shims) and in `src/wetlands_ml_geoai/` (full implementations).
- Shared flow:
  1. Parse CLI arguments/env vars to resolve raster, labels, manifest, tiling, and optimization parameters.
  2. Expand paths, validate existence, and derive a tile export directory.
  3. Export image/label tiles via `geoai.export_geotiff_tiles`, using NAIP rasters or manifest-backed stacks.
  4. For manifest scenarios, call `rewrite_tile_images` from `stacking.py` to assemble multi-source tiles.
  5. Derive input channel count from raster metadata or manifest band counts.
  6. Launch `geoai` model training helpers (`train_MaskRCNN_model` or `train_segmentation_model`).
  7. Persist checkpoints into `<tiles>/models*` folders.

- Unique pieces:
  - Mask R-CNN training toggles pretrained backbones and handles classification vs instance segmentation specifics.
  - UNet training computes optional target resizing and label tile statistics (`_analyze_label_tiles`).

## Inference Pipelines

- Entrypoints: `test.py` (Mask R-CNN) and `test_unet.py` in both root and `src/`.
- Flow for manifest-backed streaming inference:
  1. Parse CLI options to resolve raster/manifest, checkpoints, window sizes, thresholds, and output directories.
  2. Load manifests via `stacking.load_manifest`; instantiate `RasterStack` for windowed reads.
  3. Build Torch models (`get_instance_segmentation_model` or `get_smp_model`) with channel counts from stack metadata.
  4. Slide windows using `_compute_offsets`, gather predictions per window, and aggregate into raster arrays.
  5. Write GeoTIFF predictions and vectorize via `geoai.raster_to_vector`.

- When a raw raster path is supplied, the CLI defers to `geoai.object_detection` or `geoai.semantic_segmentation`, keeping only output-path orchestration locally.

## Sentinel-2 Seasonal Processing

- Entrypoint: `sentinel2_processing.py` (root and `src/`).
- Combined responsibilities inside `src/wetlands_ml_geoai/sentinel2_processing.py`:
  - Argument parsing and CLI dispatch.
  - AOI parsing (files, inline JSON, bbox lists, or WKT strings).
  - NAIP reference preparation, including mosaicking, resampling, and manifest scaffolding.
  - STAC queries for Sentinel-2 imagery, filtering by season, cloud masking via SCL, generating composites with Stackstac & Dask.
  - Progress reporting utilities (custom progress bar classes).
  - Optional downloads for NAIP, wetlands, and LiDAR topography that feed into the training stack.
- Manifest writing (`write_stack_manifest`) supports extra sources so LiDAR derivatives expand the stack to 28 bands when requested.

## LiDAR Topography Processing

- Entrypoint: `topography/cli.py` (Python module and `scripts/windows/run_topography_processing.bat`).
- Responsibilities:
  - Query USGS 3DEP (TNM Access) for buffered AOI coverage, download and cache 1 m DEM tiles.
  - Mosaic DEM to the NAIP/Sentinel grid, respecting a configurable buffer to avoid edge artifacts.
  - Compute derived bands: slope, TPI (small/large radii), depression depth; output float32 raster with nodata propagation.
  - Register topography raster in the stack manifest so training/inference consume a 28-band stack transparently.
  - Sentinel-2 seasonal pipeline can auto-generate these rasters when the corresponding flags are provided; the standalone CLI remains available for manual runs.