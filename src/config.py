"""Controlled experiment configuration loader and cross-run validator."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

EXPECTED_ANCHOR_LEVELS = 5

CONTROLLED_FIELDS: tuple[str, ...] = (
    "seed",
    "trainable_backbone_layers",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_steps",
    "grad_clip_norm",
    "hflip_prob",
    "score_thresh",
    "iou_thresh",
    "aspect_ratios",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Declarative configuration for a single Anti-UAV Faster R-CNN run."""

    run_name: str
    seed: int
    min_size: int
    max_size: int
    anchor_sizes: tuple[int, ...]
    aspect_ratios: tuple[float, ...]
    trainable_backbone_layers: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip_norm: float
    hflip_prob: float
    score_thresh: float
    iou_thresh: float
    num_workers: int


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if not _is_strict_int(value):
        raise ValueError(f"Config field '{key}' must be an int, got {value!r}")
    return value


def _require_float(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if not _is_strict_number(value):
        raise ValueError(f"Config field '{key}' must be a number, got {value!r}")
    return float(value)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML configuration file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file '{cfg_path}' must contain a YAML mapping")

    expected_keys = {f.name for f in fields(ExperimentConfig)}
    actual_keys = set(raw.keys())
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"Config '{cfg_path}' missing required keys: {missing}")
    if unknown:
        raise ValueError(f"Config '{cfg_path}' has unknown keys: {unknown}")

    run_name = raw["run_name"]
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(f"Config field 'run_name' must be a non-empty string, got {run_name!r}")
    if run_name != cfg_path.stem:
        raise ValueError(
            f"Config field 'run_name' ({run_name!r}) must match filename stem ({cfg_path.stem!r})"
        )

    min_size = _require_int(raw, "min_size")
    max_size = _require_int(raw, "max_size")
    if min_size <= 0 or max_size <= 0 or min_size > max_size:
        raise ValueError(
            f"Config must satisfy 0 < min_size <= max_size, got min_size={min_size}, max_size={max_size}"
        )

    raw_anchors = raw["anchor_sizes"]
    if (
        not isinstance(raw_anchors, (list, tuple))
        or len(raw_anchors) != EXPECTED_ANCHOR_LEVELS
        or not all(_is_strict_int(a) and a > 0 for a in raw_anchors)
    ):
        raise ValueError(
            f"Config field 'anchor_sizes' must contain exactly {EXPECTED_ANCHOR_LEVELS} positive ints, "
            f"got {raw_anchors!r}"
        )
    anchor_sizes = tuple(raw_anchors)

    raw_ratios = raw["aspect_ratios"]
    if (
        not isinstance(raw_ratios, (list, tuple))
        or len(raw_ratios) == 0
        or not all(_is_strict_number(r) and r > 0 for r in raw_ratios)
    ):
        raise ValueError(
            f"Config field 'aspect_ratios' must be a non-empty sequence of positive numbers, "
            f"got {raw_ratios!r}"
        )
    aspect_ratios = tuple(float(r) for r in raw_ratios)

    return ExperimentConfig(
        run_name=run_name,
        seed=_require_int(raw, "seed"),
        min_size=min_size,
        max_size=max_size,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        trainable_backbone_layers=_require_int(raw, "trainable_backbone_layers"),
        epochs=_require_int(raw, "epochs"),
        batch_size=_require_int(raw, "batch_size"),
        lr=_require_float(raw, "lr"),
        weight_decay=_require_float(raw, "weight_decay"),
        warmup_steps=_require_int(raw, "warmup_steps"),
        grad_clip_norm=_require_float(raw, "grad_clip_norm"),
        hflip_prob=_require_float(raw, "hflip_prob"),
        score_thresh=_require_float(raw, "score_thresh"),
        iou_thresh=_require_float(raw, "iou_thresh"),
        num_workers=_require_int(raw, "num_workers"),
    )


def validate_controlled_configs(configs: list[ExperimentConfig]) -> None:
    """Assert that all controlled training and evaluation variables match across runs."""
    if not configs:
        raise ValueError("configs must contain at least one ExperimentConfig")

    ref = configs[0]
    ref_anchor_len = len(ref.anchor_sizes)

    for cfg in configs[1:]:
        for field_name in CONTROLLED_FIELDS:
            ref_val = getattr(ref, field_name)
            cur_val = getattr(cfg, field_name)
            if cur_val != ref_val:
                raise ValueError(
                    f"Controlled variable '{field_name}' differs between "
                    f"'{ref.run_name}' ({ref_val!r}) and '{cfg.run_name}' ({cur_val!r})"
                )
        if len(cfg.anchor_sizes) != ref_anchor_len:
            raise ValueError(
                f"Controlled variable 'anchor_sizes' length differs between "
                f"'{ref.run_name}' ({ref_anchor_len}) and '{cfg.run_name}' ({len(cfg.anchor_sizes)})"
            )
