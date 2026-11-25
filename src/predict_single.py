"""Single-image spot-check CLI inference tool for trained Anti-UAV Faster R-CNN checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from src.config import load_config
from src.dataset import UAV_LABEL
from src.eval import load_checkpoint, resolve_device
from src.model import build_model

COLOR_PRED = "#1E88E5"


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_score_thresh(score_thresh: float) -> float:
    if (
        not _is_strict_number(score_thresh)
        or not math.isfinite(float(score_thresh))
        or not (0.0 <= float(score_thresh) <= 1.0)
    ):
        raise ValueError(f"score_thresh must be a finite number in [0.0, 1.0], got {score_thresh!r}")
    return float(score_thresh)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _save_single_overlay(
    rgb_img: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    *,
    run_name: str,
    image_name: str,
    score_thresh: float,
    inference_ms: float,
    out_path: Path,
) -> None:
    orig_w, orig_h = rgb_img.size
    rgb_array = np.asarray(rgb_img)

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.0))
        ax.imshow(rgb_array)
        ax.set_xlim(0.0, float(orig_w))
        ax.set_ylim(float(orig_h), 0.0)
        ax.set_axis_off()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                linewidth=2.2,
                edgecolor=COLOR_PRED,
                facecolor="none",
            )
            ax.add_patch(rect)
            text_y = y1 - 4.0 if y1 > 14.0 else min(float(orig_h) - 6.0, y2 + 12.0)
            near_right = x1 > float(orig_w) * 0.78
            text_x = min(float(orig_w) - 2.0, x2) if near_right else max(2.0, x1)
            ax.text(
                text_x,
                max(10.0, text_y),
                f"UAV {score:.2f}",
                color="white",
                fontsize=8.5,
                fontweight="bold",
                ha="right" if near_right else "left",
                bbox={"facecolor": COLOR_PRED, "alpha": 0.80, "edgecolor": "none", "pad": 1.4},
            )

        ax.set_title(
            f"{run_name} | {image_name} ({orig_w}x{orig_h})\n"
            f"detections={len(boxes)} (score >= {score_thresh:.2f}) | {inference_ms:.1f} ms",
            fontsize=10.5,
            fontweight="bold",
        )
        fig.subplots_adjust(top=0.86, bottom=0.03, left=0.02, right=0.98)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)


def predict_single_image(
    config_path: Path | str,
    checkpoint_path: Path | str,
    image_path: Path | str,
    score_thresh: float = 0.50,
    out_image: Path | str | None = None,
    device: str | torch.device = "auto",
    pretrained: bool = False,
) -> dict[str, Any]:
    """Run single-image inference with a trained checkpoint and optionally save an annotated PNG."""
    score_t = _validate_score_thresh(score_thresh)
    img_path = Path(image_path)
    ckpt_path = Path(checkpoint_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    cfg = load_config(config_path)
    resolved_device = device if isinstance(device, torch.device) else resolve_device(str(device))

    model = build_model(cfg, pretrained=pretrained)
    state_dict, _ = load_checkpoint(
        ckpt_path,
        device=resolved_device,
        expected_run_name=cfg.run_name,
    )
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    with Image.open(img_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
    orig_w, orig_h = rgb_img.size
    img_tensor = F.to_tensor(rgb_img).to(resolved_device)

    with torch.no_grad():
        _sync_device(resolved_device)
        start = time.perf_counter()
        outputs = model([img_tensor])
        _sync_device(resolved_device)
        inference_ms = (time.perf_counter() - start) * 1000.0

    out = outputs[0]
    pred_boxes_t = out["boxes"].detach().cpu()
    pred_scores_t = out["scores"].detach().cpu()
    if "labels" in out:
        keep_uav = out["labels"].detach().cpu() == UAV_LABEL
        pred_boxes_t = pred_boxes_t[keep_uav]
        pred_scores_t = pred_scores_t[keep_uav]

    raw_boxes = pred_boxes_t.numpy().astype(np.float64, copy=False).reshape(-1, 4)
    raw_scores = pred_scores_t.numpy().astype(np.float64, copy=False).reshape(-1)

    kept_mask = raw_scores >= score_t
    kept_boxes = raw_boxes[kept_mask]
    kept_scores = raw_scores[kept_mask]
    if len(kept_scores) > 0:
        order = np.argsort(-kept_scores, kind="mergesort")
        kept_boxes = kept_boxes[order]
        kept_scores = kept_scores[order]

    boxes_list = [[round(float(c), 4) for c in box] for box in kept_boxes]
    scores_list = [round(float(s), 6) for s in kept_scores]

    if out_image is not None:
        _save_single_overlay(
            rgb_img,
            boxes_list,
            scores_list,
            run_name=cfg.run_name,
            image_name=img_path.name,
            score_thresh=score_t,
            inference_ms=inference_ms,
            out_path=Path(out_image),
        )

    return {
        "run_name": cfg.run_name,
        "checkpoint": str(ckpt_path),
        "image": str(img_path),
        "orig_size": [int(orig_w), int(orig_h)],
        "score_thresh": score_t,
        "num_detections": len(boxes_list),
        "boxes": boxes_list,
        "scores": scores_list,
        "inference_ms": round(float(inference_ms), 3),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.predict_single``."""
    parser = argparse.ArgumentParser(
        description="Spot-check single-image Anti-UAV Faster R-CNN inference with JSON output and optional PNG overlay."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image (.jpg/.jpeg/.png).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.50,
        help="Confidence score threshold for keeping detections (default: 0.50).",
    )
    parser.add_argument(
        "--out-image",
        type=Path,
        default=None,
        help="Optional output PNG path for annotated detection overlay.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device ('auto', 'cpu', 'cuda', ...).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for single-image Anti-UAV spot-check inference."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = predict_single_image(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        score_thresh=args.score_thresh,
        out_image=args.out_image,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
