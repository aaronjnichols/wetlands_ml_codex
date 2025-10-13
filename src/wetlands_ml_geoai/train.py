"""Mask R-CNN training entry point for wetlands_ml_geoai."""
import argparse
import logging
import os
from pathlib import Path

from .stacking import load_manifest
from .training.mask_rcnn import train_mask_rcnn

DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_BUFFER = 0
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_VAL_SPLIT = 0.2


def strtobool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tiles and train the wetlands Mask R-CNN model.")
    parser.add_argument("--train-raster", default=os.getenv("TRAIN_RASTER_PATH"),
                        help="Path to the training raster (expects 4- or 25-band stack).")
    parser.add_argument("--stack-manifest", default=os.getenv("TRAIN_STACK_MANIFEST"),
                        help="Path to a stack manifest JSON (generates 25-band tiles on demand).")
    parser.add_argument("--labels", default=os.getenv("TRAIN_LABELS_PATH"),
                        help="Path to the vector training labels (GeoPackage or shapefile).")
    parser.add_argument("--tiles-dir", default=os.getenv("TRAIN_TILES_DIR"),
                        help="Directory for exported image/label tiles. Defaults to <raster_parent>/tiles.")
    parser.add_argument("--models-dir", default=os.getenv("TRAIN_MODELS_DIR"),
                        help="Directory to store trained model checkpoints. Defaults to <tiles-dir>/models.")
    parser.add_argument("--tile-size", type=int, default=int(os.getenv("TILE_SIZE", DEFAULT_TILE_SIZE)),
                        help="Tile size in pixels.")
    parser.add_argument("--stride", type=int, default=int(os.getenv("TILE_STRIDE", DEFAULT_STRIDE)),
                        help="Stride in pixels between tiles.")
    parser.add_argument("--buffer", type=int, default=int(os.getenv("TILE_BUFFER", DEFAULT_BUFFER)),
                        help="Buffer radius for tile extraction.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", DEFAULT_EPOCHS)),
                        help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", DEFAULT_LEARNING_RATE)),
                        help="Optimizer learning rate.")
    parser.add_argument("--val-split", type=float, default=float(os.getenv("VAL_SPLIT", DEFAULT_VAL_SPLIT)),
                        help="Validation split fraction.")
    parser.add_argument("--num-channels", type=int, default=None,
                        help="Override the input channel count; derived from raster if omitted.")
    default_pretrained = strtobool(os.getenv("PRETRAINED", "true"))
    parser.set_defaults(pretrained=default_pretrained)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use a backbone pretrained on ImageNet (default).")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Disable pretrained weights.")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity level.")
    args = parser.parse_args()

    if not args.train_raster and not args.stack_manifest:
        parser.error("Provide --train-raster or --stack-manifest (or set TRAIN_RASTER_PATH / TRAIN_STACK_MANIFEST).")
    if not args.labels:
        parser.error("--labels or TRAIN_LABELS_PATH must be supplied.")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    labels_path = Path(args.labels).expanduser().resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Training labels not found: {labels_path}")

    manifest_path = None
    if args.stack_manifest:
        manifest_raw = args.stack_manifest.strip()
        if manifest_raw.startswith('"') and manifest_raw.endswith('"'):
            manifest_raw = manifest_raw[1:-1]
        manifest_path = Path(manifest_raw).expanduser().resolve()
    stack_manifest = load_manifest(manifest_path) if manifest_path else None

    if stack_manifest is not None:
        naip_source = stack_manifest.naip
        if naip_source is None:
            raise ValueError("Stack manifest does not include a NAIP source.")
        base_raster = Path(naip_source.path)
    else:
        if args.train_raster is None:
            raise ValueError("--train-raster must be provided when no stack manifest is supplied.")
        base_raster = Path(args.train_raster).expanduser().resolve()

    if not base_raster.exists():
        raise FileNotFoundError(f"Training raster not found: {base_raster}")

    default_parent = manifest_path.parent if manifest_path else base_raster.parent
    tiles_dir = (
        Path(args.tiles_dir).expanduser().resolve()
        if args.tiles_dir
        else default_parent / "tiles"
    )
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else tiles_dir / "models"
    )

    train_mask_rcnn(
        labels_path=labels_path,
        train_raster=base_raster,
        tiles_dir=tiles_dir,
        models_dir=models_dir,
        stack_manifest_path=manifest_path,
        tile_size=args.tile_size,
        stride=args.stride,
        buffer_radius=args.buffer,
        num_channels_override=args.num_channels,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
    )



def _analyze_label_tiles(labels_dir: Path, max_check: int = 256) -> tuple[float, float, int]:
    paths = sorted(p for p in labels_dir.glob('*.tif'))
    if not paths:
        return 0.0, 0.0, 0
    total = 0
    all_one = 0
    cover_fracs = []
    for p in paths[:max_check]:
        try:
            import rasterio
            with rasterio.open(p) as src:
                arr = src.read(1)
        except Exception:
            continue
        total += 1
        size = arr.size
        if size == 0:
            continue
        pos = int((arr > 0).sum())
        frac = pos / float(size)
        cover_fracs.append(frac)
        if frac >= 0.999:
            all_one += 1
    if total == 0:
        return 0.0, 0.0, 0
    return all_one / total, (sum(cover_fracs) / len(cover_fracs)) if cover_fracs else 0.0, total


if __name__ == "__main__":
    main()
