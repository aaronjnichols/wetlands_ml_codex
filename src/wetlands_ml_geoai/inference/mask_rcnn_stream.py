"""Mask R-CNN streaming inference for wetlands_ml_geoai."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rasterio.windows import Window

from geoai.train import get_instance_segmentation_model
from geoai.utils import get_device

from ..stacking import FLOAT_NODATA, RasterStack, StackManifest, load_manifest


def _compute_offsets(size: int, window: int, overlap: int) -> list[int]:
    if size <= window:
        return [0]
    step = max(window - overlap, 1)
    offsets = list(range(0, size - window + 1, step))
    last = size - window
    if offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def infer_manifest(
    manifest: StackManifest | Path,
    model_path: Path,
    output_path: Path,
    window_size: int,
    overlap: int,
    confidence_threshold: float,
    num_channels: Optional[int],
) -> None:
    manifest_obj = manifest if isinstance(manifest, StackManifest) else load_manifest(manifest)
    device = get_device()

    channel_count = num_channels
    if channel_count is None:
        with RasterStack(manifest_obj) as stack:
            channel_count = stack.band_count

    model = get_instance_segmentation_model(
        num_classes=2,
        num_channels=channel_count,
        pretrained=True,
    )
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
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
                "Streaming Mask R-CNN inference across %s windows (%sx%s, overlap=%s)",
                total_windows,
                window_h,
                window_w,
                overlap,
            )

            pred_accumulator = np.zeros((height, width), dtype=np.float32)
            count_accumulator = np.zeros((height, width), dtype=np.float32)

            for row_off in row_offsets:
                for col_off in col_offsets:
                    win = Window(col_off, row_off, window_w, window_h)
                    data = stack.read_window(win)
                    data = np.where(data == FLOAT_NODATA, 0.0, data)
                    tensor = torch.from_numpy(data).to(device)
                    tensor = tensor.unsqueeze(0)

                    outputs = model(tensor)
                    output = outputs[0]

                    scores = output.get("scores")
                    masks = output.get("masks")
                    if scores is None or masks is None or len(scores) == 0:
                        continue

                    keep = scores > confidence_threshold
                    if not torch.any(keep):
                        continue

                    masks = masks[keep].squeeze(1)
                    if masks.ndim == 2:
                        masks = masks.unsqueeze(0)

                    combined = torch.max(masks, dim=0)[0] > 0.5
                    combined_np = combined.cpu().numpy().astype(np.float32)

                    row_end = min(row_off + window_h, height)
                    col_end = min(col_off + window_w, width)
                    h = row_end - row_off
                    w = col_end - col_off

                    pred_accumulator[row_off:row_end, col_off:col_end] += combined_np[:h, :w]
                    count_accumulator[row_off:row_end, col_off:col_end] += 1.0

    valid = count_accumulator > 0
    mask = np.zeros((pred_accumulator.shape[0], pred_accumulator.shape[1]), dtype=np.uint8)
    mask[valid] = (pred_accumulator[valid] / count_accumulator[valid] > 0.5).astype(np.uint8)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if crs is not None:
        profile["crs"] = crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    elapsed = time.perf_counter() - start
    logging.info("Streaming inference finished in %.2f seconds", elapsed)


__all__ = ["infer_manifest"]

