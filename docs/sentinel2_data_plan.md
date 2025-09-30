# Objective

Download Sentinel-2 Level-2A scenes and build **seasonal median composites** for the **7 recommended bands** (B03, B04, B05, B06, B08, B11, B12). Concatenate the three seasonal composites (Spring, Summer, Fall) into a **21-band raster**, then align and stack with NAIP 4-band imagery to produce a **25-band GeoTIFF** for wetlands modeling.

# Inputs (parameters)

* **AOI**: polygon (GeoJSON/WKT/ESRI JSON) or bbox in EPSG:4326.
* **Years**: list of years to include (e.g., the NAIP year ±1).
* **Seasons**:
  * SPR: March–May
  * SUM: June–August
  * FAL: September–November
* **Cloud cover filter**: e.g., `eo:cloud_cover < 60` at item level.
* **Min clear obs per pixel**: e.g., ≥3 per season (else write NoData).
* **Target grid for final 25-band**: NAIP mosaic (CRS, transform, 1 m pixel size)

# Data source & assets

* **Collection**: Sentinel-2 **L2A** (surface reflectance).
* **Assets** to load per scene: `B03, B04, B05, B06, B08, B11, B12, SCL`.
* **Scale**: convert reflectance bands to **float32 0–1** using scale factor **1/10000**.

# Cloud/shadow/snow masking

* Use the **SCL** (Scene Classification Layer) per scene to mask out classes:

  * 3 (cloud shadow), 8 (cloud, med), 9 (cloud, high), 10 (cirrus), 11 (snow/ice).
* Optional: **dilate** the mask by 1 pixel to catch cloud edges.
* Optional: also drop observations with extreme reflectance outliers after masking.

# Seasonal median compositing (per season, per band)

1. Select all scenes overlapping AOI within the season window (across chosen years).
2. Apply the SCL mask to each band image.
3. **Resample 20 m bands** (B05, B06, B11, B12) to **10 m** using **bilinear** so all 7 bands share a common 10 m grid.
4. Compute the **per-pixel median** across time (ignore NaNs) for each band.
5. Require **≥ min clear obs** per pixel; otherwise set NoData.

# Outputs (files)

1. **Three seasonal GeoTIFFs** at 10 m, each with **7 bands** (order below), `float32`, `NoData=-9999`, tiled, deflate-compressed, COG-friendly:

   * `s2_spr_median_7band.tif`
   * `s2_sum_median_7band.tif`
   * `s2_fal_median_7band.tif`
2. **One 21-band GeoTIFF** (concatenate the three seasonal stacks along the band dimension) at 10 m:

   * `s2_sprsumfal_median_21band.tif`
3. **(Optional) Final 25-band GeoTIFF** aligned to NAIP:

   * Reproject the 21-band S2 composite to the **NAIP grid (CRS, 1 m)** with **bilinear** resampling.
   * **Stack order (final)**: `NAIP_R, NAIP_G, NAIP_B, NAIP_NIR,` then `S2_SPR_[B03,B04,B05,B06,B08,B11,B12], S2_SUM_[...], S2_FAL_[...]`.
   * Save as `naip_s2_25band.tif` (float32, tiled, deflate, BIGTIFF=IF\_SAFER).

# Band order specification (must be exact)

1. **S2 seasonal 7-band order (per season)**: `B03, B04, B05, B06, B08, B11, B12`.
2. **21-band**: `[SPR_7, SUM_7, FAL_7]` in that order.
3. **25-band**: `NAIP_R, NAIP_G, NAIP_B, NAIP_NIR,` then `SPR_7, SUM_7, FAL_7`.
4. Write **band descriptions** in the GeoTIFF, e.g., `S2_SPR_B03`, `S2_SUM_B11`, etc.

# Validation checks the code should perform

* Confirm all output rasters are **float32** and within **\[0,1]** (ignoring NoData).
* Print per-season counts of valid (unmasked) observations per pixel (summary stats).
* Ensure final band counts are 7/21/25 as expected and CRS/resolution/extent align.
* If NAIP grid is provided, assert the **25-band** raster exactly matches its geotransform/shape.

# Performance/robustness

* Use lazy reads / chunking (e.g., Dask) if available.
* Stream downloads; avoid loading all scenes into memory at once.
* Fail gracefully if a season has too few clear scenes; still write other seasons.

That’s the full target: fetch L2A scenes for the AOI/date windows, mask with SCL, make median **SPR/SUM/FAL** composites for **B03,B04,B05,B06,B08,B11,B12**, concatenate to 21 bands, and reproject and stack with NAIP for a **25-band** training raster—clean, reproducible, and ready for the geoai pipeline.
