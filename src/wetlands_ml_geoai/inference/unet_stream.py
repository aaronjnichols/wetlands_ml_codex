"""UNet sliding-window inference for wetlands_ml_geoai."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from geoai.train import get_smp_model
from geoai.utils import get_device

from ..stacking import FLOAT_NODATA, RasterStack, StackManifest, load_manifest


def _compute_offsets(size: int, window: int, overlap: int) -> List[int]:
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
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
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
                    data = data / 255.0
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


__all__ = ["infer_manifest"]

