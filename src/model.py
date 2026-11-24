"""Controlled Faster R-CNN MobileNetV3-Large-FPN factory and parameter summary."""

from __future__ import annotations

import math

from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from src.config import ExperimentConfig

NUM_CLASSES = 2  # 0: background, 1: UAV
MAX_TRAINABLE_BACKBONE_LAYERS = 6


def build_model(cfg: ExperimentConfig, pretrained: bool = True) -> FasterRCNN:
    """Build a controlled Faster R-CNN MobileNetV3-Large-FPN detector from config.

    Preserves the FPN feature-level count and ``num_anchors_per_location()`` when
    overriding anchor sizes so pretrained RPN head convolutional weights load
    intact with zero shape mismatch across resolution and anchor variants.
    """
    if (
        isinstance(cfg.trainable_backbone_layers, bool)
        or not isinstance(cfg.trainable_backbone_layers, int)
        or not (0 <= cfg.trainable_backbone_layers <= MAX_TRAINABLE_BACKBONE_LAYERS)
    ):
        raise ValueError(
            f"trainable_backbone_layers must be an int in [0, {MAX_TRAINABLE_BACKBONE_LAYERS}], "
            f"got {cfg.trainable_backbone_layers!r}"
        )
    if (
        isinstance(cfg.min_size, bool)
        or isinstance(cfg.max_size, bool)
        or not isinstance(cfg.min_size, int)
        or not isinstance(cfg.max_size, int)
        or cfg.min_size <= 0
        or cfg.max_size <= 0
        or cfg.min_size > cfg.max_size
    ):
        raise ValueError(
            f"Config must satisfy 0 < min_size <= max_size, "
            f"got min_size={cfg.min_size!r}, max_size={cfg.max_size!r}"
        )
    if not cfg.anchor_sizes or not all(
        not isinstance(s, bool) and isinstance(s, int) and s > 0 for s in cfg.anchor_sizes
    ):
        raise ValueError(
            f"anchor_sizes must be a non-empty sequence of positive ints, got {cfg.anchor_sizes!r}"
        )
    if not cfg.aspect_ratios or not all(
        not isinstance(r, bool)
        and isinstance(r, (int, float))
        and math.isfinite(r)
        and float(r) > 0.0
        for r in cfg.aspect_ratios
    ):
        raise ValueError(
            f"aspect_ratios must be a non-empty sequence of positive finite numbers, "
            f"got {cfg.aspect_ratios!r}"
        )

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    # TorchVision warns if trainable_backbone_layers is passed when weights and
    # weights_backbone are both None (it always falls back to 6 when unpretrained).
    backbone_layers = cfg.trainable_backbone_layers if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=backbone_layers,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
    )

    num_levels = len(model.rpn.anchor_generator.sizes)
    old_num_anchors = list(model.rpn.anchor_generator.num_anchors_per_location())
    new_anchor_generator = AnchorGenerator(
        sizes=(tuple(cfg.anchor_sizes),) * num_levels,
        aspect_ratios=(tuple(cfg.aspect_ratios),) * num_levels,
    )
    new_num_anchors = list(new_anchor_generator.num_anchors_per_location())
    if new_num_anchors != old_num_anchors:
        raise ValueError(
            "Anchor override changed num_anchors_per_location from "
            f"{old_num_anchors} to {new_num_anchors}; pretrained RPN head weights "
            "require an identical anchor count per spatial location."
        )
    model.rpn.anchor_generator = new_anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=NUM_CLASSES)
    return model


def summarize_trainable_params(model: nn.Module) -> dict[str, int]:
    """Return exact total, trainable, and frozen parameter counts for ``model``."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }
