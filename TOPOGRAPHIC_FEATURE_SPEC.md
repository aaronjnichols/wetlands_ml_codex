I would like to add three additional bands derived from USGS LiDAR 1m resolution data to the full training raster. I would like this to work somewhat similarly to how the sentinel 2 rasters are derived and just the final computed rasters are saved locally. The three additional features to be added (28 total bands) are described below:

1. **Slope** — the steepness of the terrain in degrees, calculated from elevation gradients (dz/dx and dz/dy).
2. **Topographic Position Index (TPI)** — the difference between each pixel’s elevation and the mean elevation of its surrounding neighborhood.

   * We’ll use two scales:
     • **TPI_small** ≈ 30 m radius (micro-relief)
     • **TPI_large** ≈ 150 m radius (valley/floodplain context)
   * The radius in meters should be converted to pixels using the DEM’s resolution.
3. **Depression Depth** — a simple measure of how much a pixel sits in a closed depression. Compute by filling sinks in the DEM and subtracting the original elevation (`fill(DEM) – DEM`).

## Integration Summary

- New package `src/wetlands_ml_geoai/topography/` handles USGS 3DEP downloads, buffered DEM mosaicking, and derivative generation (slope, TPI_small ≈ 30 m, TPI_large ≈ 150 m, depression depth).
- CLI (`python -m wetlands_ml_geoai.topography.cli`) mirrors Sentinel-2 workflow: provide AOI, reference grid (NAIP), output directory, optional cache/buffer controls.
- DEM download logic queries TNM Access (`Digital Elevation Model (DEM) 1 meter`) using the AOI plus a configurable buffer (default 200 m) to avoid edge artifacts.
- Processing pipeline:
  1. Mosaic buffered DEM tiles and resample to the NAIP/Sentinel stack grid.
  2. Compute slope in degrees from dz/dx, dz/dy gradients.
  3. Compute TPI_small and TPI_large via neighborhood means based on radii converted to pixel units.
  4. Compute depression depth using grayscale closing (fill - original), clamp negatives to zero.
  5. Write float32 GeoTIFF with nodata propagated as `-9999.0` and band descriptions.
- Manifest integration: `write_stack_manifest` accepts `extra_sources`; the topography raster is registered under `type: "topography"` with band labels `["Slope", "TPI_small", "TPI_large", "DepressionDepth"]`, enabling 28-channel stacks for training/inference without further changes.
- Documentation updated in `docs/pipeline_overview.md` and detailed plan files under `docs/topography_*`. Validation guidance covers visual QA, histogram checks, and manifest verification.