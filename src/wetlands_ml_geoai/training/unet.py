"""UNet training orchestration for wetlands_ml_geoai."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geoai

from ..stacking import RasterStack, StackManifest, load_manifest, rewrite_tile_images
from ..tiling import analyze_label_tiles, derive_num_channels


def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    train_raster: Optional[Path] = None,
    stack_manifest_path: Optional[Path] = None,
    tile_size: int = 512,
    stride: int = 256,
    buffer_radius: int = 0,
    num_channels_override: Optional[int] = None,
    num_classes: int = 2,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    batch_size: int = 4,
    epochs: int = 25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    save_best_only: bool = True,
    plot_curves: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "resize",
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume_training: bool = False,
) -> None:
    stack_manifest: Optional[StackManifest] = None
    if stack_manifest_path is not None:
        stack_manifest = load_manifest(stack_manifest_path)

    if stack_manifest is not None:
        naip_source = stack_manifest.naip
        if naip_source is None:
            raise ValueError("Stack manifest does not include a NAIP source.")
        base_raster = naip_source.path
    else:
        if train_raster is None:
            raise ValueError(
                "train_raster must be provided when no stack manifest is supplied."
            )
        base_raster = train_raster

    tiles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Exporting tiles to %s", tiles_dir)
    geoai.export_geotiff_tiles(
        in_raster=str(base_raster),
        out_folder=str(tiles_dir),
        in_class_data=str(labels_path),
        tile_size=tile_size,
        stride=stride,
        buffer_radius=buffer_radius,
    )

    images_dir = tiles_dir / "images"
    labels_dir = tiles_dir / "labels"

    if stack_manifest is not None:
        rewritten = rewrite_tile_images(stack_manifest, images_dir)
        logging.info("Rewrote %s image tiles with stack manifest", rewritten)

    if not images_dir.exists():
        raise FileNotFoundError(f"Image tiles directory missing: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label tiles directory missing: {labels_dir}")

    if stack_manifest is not None and num_channels_override is None:
        with RasterStack(stack_manifest) as stack:
            num_channels = stack.band_count
    else:
        num_channels = derive_num_channels(base_raster, num_channels_override)

    if num_channels_override is not None:
        num_channels = num_channels_override

    logging.info("Training UNet model with %s input channels", num_channels)

    all_one_frac, avg_cover, checked = analyze_label_tiles(labels_dir)
    if checked:
        logging.info(
            "Analyzed %s label tiles – %.1f%% all-one, mean foreground cover %.3f",
            checked,
            all_one_frac * 100,
            avg_cover,
        )

    geoai.train_segmentation_model(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_dir=str(models_dir),
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=num_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        val_split=val_split,
        save_best_only=save_best_only,
        plot_curves=plot_curves,
        target_size=target_size,
        resize_mode=resize_mode,
        num_workers=num_workers,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        resume_training=resume_training,
    )

    logging.info("Training complete. Models saved to %s", models_dir)


__all__ = ["train_unet"]

