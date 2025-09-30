# Wetlands ML GeoAI

Core pipelines for generating Sentinel-2 seasonal composites, stacking with NAIP imagery, training Mask R-CNN models, and running inference.

## Project Layout

- `src/wetlands_ml_geoai/` – reusable Python modules for compositing, stacking, training, inference, and validation.
- `train.py`, `test.py`, `sentinel2_processing.py`, `validate_seasonal_pixels.py` – thin CLI shims that delegate to the package modules.
- `tools/` – utility scripts that support local workflows (for example NAIP downloads).
- `scripts/windows/` – Windows batch launchers for common tasks.
- `docs/` – detailed project guidelines and background notes.
- `data/` – local datasets for experimentation (ignored by git; provide your own sources).

## Sentinel-2 Composites

`sentinel2_processing.py` can accept one NAIP raster or an entire directory of tiles. When you supply multiple rasters (or a folder), the CLI mosaics them automatically and writes a single stack manifest aligned to that mosaic.

## Quick Start

```bash
python setup.bat
python sentinel2_processing.py --help
python train.py --help
python test.py --help
```

Run `python -m wetlands_ml_geoai.<module>` if you prefer module execution.

Consult the files in `docs/` for processing expectations, dataset assumptions, and contribution guidelines.
