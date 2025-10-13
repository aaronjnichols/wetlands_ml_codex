We want to create a Python script that computes a few basic, tile-friendly topographic features from one or more DEM rasters. The script should work on a single DEM or a list of DEMs and output new rasters (GeoTIFFs) containing the derived features.

Each DEM represents a small tile (for example, 512×512 px). The goal is to produce the minimum useful set of terrain features that help a wetland-detection ML model, without requiring full hydrologic processing (no large-scale flow accumulation).

### Features to calculate

1. **Slope** — the steepness of the terrain in degrees, calculated from elevation gradients (dz/dx and dz/dy).
2. **Topographic Position Index (TPI)** — the difference between each pixel’s elevation and the mean elevation of its surrounding neighborhood.

   * We’ll use two scales:
     • **TPI_small** ≈ 30 m radius (micro-relief)
     • **TPI_large** ≈ 150 m radius (valley/floodplain context)
   * The radius in meters should be converted to pixels using the DEM’s resolution.
3. **Depression Depth** — a simple measure of how much a pixel sits in a closed depression. Compute by filling sinks in the DEM and subtracting the original elevation (`fill(DEM) – DEM`).

### General behavior

* Read one or more DEM files (GeoTIFFs).
* Derive pixel size (meters per pixel) from the raster metadata to translate radii for TPI into pixel units.
* Compute each feature per DEM tile, handling nodata correctly.
* Output a multi-band GeoTIFF (float32) with one band per feature, aligned to the original DEM.

### Summary

Essentially, the script is determining:

* How steep each pixel is (**slope**)
* Whether it’s higher or lower than its local neighborhood (**TPI_small**, **TPI_large**)
* Whether it lies in a closed depression (**depression depth**)

These features together provide local hydrologic and geomorphic context that improves wetland prediction when combined with spectral data.
