"""UNet-based semantic segmentation inference entry point for wetlands_ml_geoai."""
import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import geoai
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window

from geoai.train import get_smp_model
from geoai.utils import get_device

from .stacking import FLOAT_NODATA, RasterStack, StackManifest, load_manifest

DEFAULT_WINDOW_SIZE = 512
DEFAULT_OVERLAP = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_MIN_AREA = 1000.0
DEFAULT_SIMPLIFY = 1.0
DEFAULT_ARCHITECTURE = "unet"
DEFAULT_ENCODER_NAME = "resnet34"
DEFAULT_NUM_CLASSES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation inference with a trained UNet model."
    )
    parser.add_argument(
        "--test-raster",
        default=os.getenv("TEST_RASTER_PATH"),
        help="Path to the raster to evaluate (expects 4- or 25-band stack).",
    )
    parser.add_argument(
        "--stack-manifest",
        default=os.getenv("TEST_STACK_MANIFEST"),
        help="Path to a stack manifest JSON for streaming inference.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH"),
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("PREDICTIONS_DIR"),
        help="Directory where prediction rasters/vectors will be written. Defaults to <raster_parent>/predictions.",
    )
    parser.add_argument(
        "--masks",
        default=os.getenv("PREDICTION_MASK_PATH"),
        help="Optional explicit path for the predicted mask GeoTIFF.",
    )
    parser.add_argument(
        "--vectors",
        default=os.getenv("PREDICTION_VECTOR_PATH"),
        help="Optional explicit path for the vectorized predictions.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.getenv("WINDOW_SIZE", DEFAULT_WINDOW_SIZE)),
        help="Sliding window size in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(os.getenv("WINDOW_OVERLAP", DEFAULT_OVERLAP)),
        help="Overlap in pixels between windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("INFER_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Inference batch size (used for direct GeoTIFF inference).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Override the input channel count; derived from raster if omitted.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("UNET_NUM_CLASSES", DEFAULT_NUM_CLASSES)),
        help="Number of segmentation classes (including background).",
    )
    parser.add_argument(
        "--architecture",
        default=os.getenv("UNET_ARCHITECTURE", DEFAULT_ARCHITECTURE),
        help="segmentation-models-pytorch architecture used during training (e.g., unet, deeplabv3plus).",
    )
    parser.add_argument(
        "--encoder-name",
        default=os.getenv("UNET_ENCODER_NAME", DEFAULT_ENCODER_NAME),
        help="Backbone encoder used during training (e.g., resnet34, efficientnet-b3).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=float(os.getenv("MIN_VECTOR_AREA", DEFAULT_MIN_AREA)),
        help="Minimum polygon area (square meters) to keep during vectorization.",
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=float(os.getenv("SIMPLIFY_TOLERANCE", DEFAULT_SIMPLIFY)),
        help="Douglas-Peucker tolerance for geometry simplification.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )

    args = parser.parse_args()

    if not args.test_raster and not args.stack_manifest:
        parser.error(
            "Provide --test-raster or --stack-manifest (or set TEST_RASTER_PATH / TEST_STACK_MANIFEST)."
        )
    if not args.model_path:
        parser.error("--model-path or MODEL_PATH must be supplied.")

    return args


def determine_output_paths(source_path: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source_path.parent / "predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_path = (
        Path(args.masks).expanduser().resolve()
        if args.masks
        else output_dir / f"{source_path.stem}_unet_predictions.tif"
    )
    vectors_path = (
        Path(args.vectors).expanduser().resolve()
        if args.vectors
        else output_dir / f"{source_path.stem}_unet_predictions.gpkg"
    )

    masks_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, masks_path, vectors_path


def derive_num_channels(raster_path: Path, override: Optional[int]) -> int:
    if override is not None:
        return override
    with rasterio.open(raster_path) as src:
        return src.count


def _compute_offsets(size: int, window: int, overlap: int) -> List[int]:
    if size <= window:
        return [0]
    step = max(window - overlap, 1)
    offsets = list(range(0, size - window + 1, step))
    last = size - window
    if offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def run_manifest_inference(
    manifest: StackManifest,
    model_path: Path,
    output_path: Path,
    window_size: int,
    overlap: int,
    num_channels: Optional[int],
    architecture: str,
    encoder_name: str,
    num_classes: int,
) -> None:
    manifest_obj = manifest if isinstance(manifest, StackManifest) else load_manifest(manifest)
    device = get_device()

    channel_count = num_channels
    if channel_count is None:
        with RasterStack(manifest_obj) as stack:
            channel_count = stack.band_count

    model = get_smp_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=channel_count,
        classes=num_classes,
        activation=None,
    )

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    start = time.perf_counter()

    with torch.no_grad():
        with RasterStack(manifest_obj) as stack:
            height = stack.height
            width = stack.width
            transform = stack.transform
            crs = stack.crs

            window_h = min(window_size, height)
            window_w = min(window_size, width)

            row_offsets = _compute_offsets(height, window_h, overlap)
            col_offsets = _compute_offsets(width, window_w, overlap)

            total_windows = len(row_offsets) * len(col_offsets)
            logging.info(
                "Streaming UNet inference across %s windows (%sx%s, overlap=%s)",
                total_windows,
                window_h,
                window_w,
                overlap,
            )

            prob_accumulator = np.zeros((num_classes, height, width), dtype=np.float32)
            count_accumulator = np.zeros((height, width), dtype=np.float32)

            for row_off in row_offsets:
                for col_off in col_offsets:
                    win = Window(col_off, row_off, window_w, window_h)
                    data = stack.read_window(win).astype(np.float32)
                    data = np.where(data == FLOAT_NODATA, 0.0, data)
                    tensor = torch.from_numpy(data).to(device)
                    tensor = tensor.unsqueeze(0)

                    output = model(tensor)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    if isinstance(output, torch.Tensor) and output.dim() == 4:
                        output = output[0]
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("Unexpected model output type for semantic segmentation")
                    probabilities = F.softmax(output, dim=0).cpu().numpy()

                    row_end = min(row_off + window_h, height)
                    col_end = min(col_off + window_w, width)
                    h = row_end - row_off
                    w = col_end - col_off

                    prob_accumulator[:, row_off:row_end, col_off:col_end] += probabilities[:, :h, :w]
                    count_accumulator[row_off:row_end, col_off:col_end] += 1.0

    denom = np.maximum(count_accumulator, 1e-6)
    averaged = prob_accumulator / denom[None, :, :]
    predicted = np.argmax(averaged, axis=0).astype(np.uint8)
    predicted[count_accumulator == 0] = 0

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if crs is not None:
        profile["crs"] = crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(predicted, 1)

    elapsed = time.perf_counter() - start
    logging.info("Streaming semantic inference finished in %.2f seconds", elapsed)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    manifest_path = None
    if args.stack_manifest:
        manifest_raw = args.stack_manifest.strip()
        if manifest_raw.startswith('"') and manifest_raw.endswith('"'):
            manifest_raw = manifest_raw[1:-1]
        manifest_path = Path(manifest_raw).expanduser().resolve()
    stack_manifest = load_manifest(manifest_path) if manifest_path else None

    if stack_manifest is not None:
        source_path = manifest_path
        logging.info("Using stack manifest at %s for streaming inference", manifest_path)
        test_raster = None
    else:
        if args.test_raster is None:
            raise ValueError("--test-raster must be supplied when no stack manifest is provided.")
        test_raster = Path(args.test_raster).expanduser().resolve()
        if not test_raster.exists():
            raise FileNotFoundError(f"Test raster not found: {test_raster}")
        source_path = test_raster

    output_dir, masks_path, vectors_path = determine_output_paths(source_path, args)

    if stack_manifest is not None:
        run_manifest_inference(
            manifest=stack_manifest,
            model_path=model_path,
            output_path=masks_path,
            window_size=args.window_size,
            overlap=args.overlap,
            num_channels=args.num_channels,
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_classes=args.num_classes,
        )
    else:
        num_channels = derive_num_channels(test_raster, args.num_channels)
        logging.info("Running semantic inference with %s input channels", num_channels)
        geoai.semantic_segmentation(
            str(test_raster),
            str(masks_path),
            str(model_path),
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_channels=num_channels,
            num_classes=args.num_classes,
            window_size=args.window_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )

    logging.info("Raster predictions written to %s", masks_path)

    geoai.raster_to_vector(
        str(masks_path),
        str(vectors_path),
        min_area=args.min_area,
        simplify_tolerance=args.simplify_tolerance,
    )

    logging.info("Vector predictions written to %s", vectors_path)


if __name__ == "__main__":
    main()


