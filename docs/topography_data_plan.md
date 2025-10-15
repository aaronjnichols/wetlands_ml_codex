# USGS 3DEP Acquisition Notes

## API Endpoints

- Primary download service: https://apps.nationalmap.gov/tnmaccess/v1/
- Dataset parameter for 1 m DEM: `dataset=Digital Elevation Model (DEM) 1 meter`
- Common request pattern: `https://apps.nationalmap.gov/tnmaccess/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox=<west>,<south>,<east>,<north>&format=GeoTIFF&max=500`
- GeoJSON polygon requests supported via POST with `Content-Type: application/json` and body `{"datasets": [...], "poly": {...}}`
- Responses include download URLs (`urls[0].url`), product IDs, and spatial metadata.

## Coverage and Buffering

- 1 m DEM tiles are pre-processed rasters, typically 1 km x 1 km footprints.
- Request AOI buffered by at least the largest topographic kernel (≥ 200 m for TPI_large) to avoid edge effects.
- Clip to the NAIP/Sentinel footprint after processing derivatives.

## Rate Limits & Usage

- TNM Access enforces a soft limit of ~5 requests/s; batching tiles reduces calls.
- Data is public domain (USGS). Attribution recommended: "USGS 3D Elevation Program (3DEP)".
- Large AOIs should stage downloads via pagination (`offset`, `max` parameters).

## Implementation Notes

- Cache raw DEM GeoTIFFs on disk (e.g. `data/dem/raw/` keyed by product ID).
- Track tile footprints to avoid re-downloads; the API returns `boundingBox` and `lastUpdated` fields.
- Use rasterio to mosaic buffered tiles to a single array aligned to the stack grid.
- Maintain metadata (CRS, resolution, nodata) before resampling to NAIP/Sentinel grid.


