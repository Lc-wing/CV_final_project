"""Small test-pack multi-model prediction CSV exporter and GT/TP/FP/FN comparison visualizer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from torchvision.transforms import functional as F

from src.config import ExperimentConfig, load_config
from src.dataset import UAV_LABEL
from src.eval import compute_fixed_threshold_counts, load_checkpoint, resolve_device
from src.model import build_model
from src.prepare_data import IMAGE_EXTENSIONS, parse_voc_xml

PREDICTION_CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "run_name",
    "checkpoint_name",
    "orig_w",
    "orig_h",
    "gt_count",
    "gt_boxes",
    "pred_count",
    "pred_boxes",
    "pred_scores",
    "tp",
    "fp",
    "fn",
)

COLOR_GT = "#00C853"  # Green solid
COLOR_TP = "#1E88E5"  # Blue solid
COLOR_FP = "#E53935"  # Red dashed
COLOR_FN = "#FFB300"  # Amber missed GT


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_thresholds(score_thresh: float, iou_thresh: float) -> tuple[float, float]:
    if (
        not _is_strict_number(score_thresh)
        or not math.isfinite(float(score_thresh))
        or not (0.0 <= float(score_thresh) <= 1.0)
    ):
        raise ValueError(f"score_thresh must be a finite number in [0.0, 1.0], got {score_thresh!r}")
    if (
        not _is_strict_number(iou_thresh)
        or not math.isfinite(float(iou_thresh))
        or not (0.0 < float(iou_thresh) <= 1.0)
    ):
        raise ValueError(f"iou_thresh must be a finite number in (0.0, 1.0], got {iou_thresh!r}")
    return float(score_thresh), float(iou_thresh)


def _round_boxes(boxes: Sequence[Sequence[float]] | np.ndarray) -> list[list[float]]:
    return [[round(float(c), 4) for c in box] for box in boxes]


def _round_scores(scores: Sequence[float] | np.ndarray) -> list[float]:
    return [round(float(s), 6) for s in scores]


def _load_test_pack_samples(test_pack_dir: Path) -> list[dict[str, Any]]:
    """Discover and validate all image + Pascal VOC XML pairs in ``test_pack_dir``."""
    img_dir = test_pack_dir / "img"
    xml_dir = test_pack_dir / "xml"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Test-pack image directory not found: {img_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"Test-pack XML directory not found: {xml_dir}")

    img_paths = sorted(
        (p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (p.stem, p.name),
    )
    if not img_paths:
        raise ValueError(f"No images found in test-pack directory: {img_dir}")

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for img_path in img_paths:
        sample_id = img_path.stem
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in '{img_dir}'")
        seen_ids.add(sample_id)

        xml_path = xml_dir / f"{sample_id}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation XML for test-pack image '{img_path.name}': {xml_path}"
            )

        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
        orig_w, orig_h = rgb_img.size
        gt_boxes, _ = parse_voc_xml(xml_path, img_width=orig_w, img_height=orig_h)
        samples.append(
            {
                "sample_id": sample_id,
                "img_path": img_path,
                "orig_w": int(orig_w),
                "orig_h": int(orig_h),
                "gt_boxes": np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4),
            }
        )

    return samples


def _draw_box_with_label(
    ax: plt.Axes,
    box: Sequence[float],
    *,
    edgecolor: str,
    linestyle: str,
    linewidth: float,
    label_text: str | None = None,
    img_width: int,
    img_height: int,
    pad: float = 0.0,
) -> None:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    rect = patches.Rectangle(
        (x1 - pad, y1 - pad),
        max(0.0, (x2 - x1) + 2.0 * pad),
        max(0.0, (y2 - y1) + 2.0 * pad),
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if label_text:
        text_y = y1 - 4.0 if y1 > 14.0 else min(float(img_height) - 6.0, y2 + 12.0)
        near_right = x1 > float(img_width) * 0.78
        text_x = min(float(img_width) - 2.0, x2) if near_right else max(2.0, x1)
        ax.text(
            text_x,
            max(10.0, text_y),
            label_text,
            color="white",
            fontsize=7.5,
            fontweight="bold",
            ha="right" if near_right else "left",
            bbox={"facecolor": edgecolor, "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
        )


def _save_sample_comparison_png(
    sample: dict[str, Any],
    run_panels: list[dict[str, Any]],
    out_path: Path,
    *,
    score_thresh: float,
    iou_thresh: float,
) -> None:
    """Render side-by-side GT/TP/FP/FN panels for one test-pack sample across runs."""
    with Image.open(sample["img_path"]) as pil_img:
        rgb_array = np.asarray(pil_img.convert("RGB"))

    orig_w = sample["orig_w"]
    orig_h = sample["orig_h"]
    gt_boxes = sample["gt_boxes"]
    n_runs = len(run_panels)
    fig_w = max(7.6, 5.4 * n_runs)

    fig = None
    try:
        fig, axes = plt.subplots(1, n_runs, figsize=(fig_w, 5.4), squeeze=False)
        ax_row = axes[0]

        for ax, panel in zip(ax_row, run_panels):
            ax.imshow(rgb_array)
            ax.set_xlim(0.0, float(orig_w))
            ax.set_ylim(float(orig_h), 0.0)
            ax.set_axis_off()

            # 1. Draw all GT boxes in green solid
            for gt_box in gt_boxes:
                _draw_box_with_label(
                    ax,
                    gt_box,
                    edgecolor=COLOR_GT,
                    linestyle="-",
                    linewidth=2.0,
                    label_text=None,
                    img_width=orig_w,
                    img_height=orig_h,
                )

            # 2. Highlight missed GT (FN) boxes in amber with a slight outset so they stay distinct
            for fn_box in panel["fn_boxes"]:
                _draw_box_with_label(
                    ax,
                    fn_box,
                    edgecolor=COLOR_FN,
                    linestyle="-.",
                    linewidth=2.4,
                    label_text="FN",
                    img_width=orig_w,
                    img_height=orig_h,
                    pad=2.0,
                )

            # 3. Draw TP detections in blue solid
            for tp_box, tp_score in zip(panel["tp_boxes"], panel["tp_scores"]):
                _draw_box_with_label(
                    ax,
                    tp_box,
                    edgecolor=COLOR_TP,
                    linestyle="-",
                    linewidth=2.0,
                    label_text=f"TP {tp_score:.2f}",
                    img_width=orig_w,
                    img_height=orig_h,
                )

            # 4. Draw FP detections in red dashed
            for fp_box, fp_score in zip(panel["fp_boxes"], panel["fp_scores"]):
                _draw_box_with_label(
                    ax,
                    fp_box,
                    edgecolor=COLOR_FP,
                    linestyle="--",
                    linewidth=2.0,
                    label_text=f"FP {fp_score:.2f}",
                    img_width=orig_w,
                    img_height=orig_h,
                )

            ax.set_title(
                f"{panel['run_name']}\n"
                f"TP={panel['tp']}  FP={panel['fp']}  FN={panel['fn']}  (GT={panel['gt_count']})",
                fontsize=10.0,
            )

        legend_handles = [
            Line2D([0], [0], color=COLOR_GT, lw=2.2, linestyle="-", label="GT (Ground Truth)"),
            Line2D([0], [0], color=COLOR_TP, lw=2.2, linestyle="-", label="TP (True Positive)"),
            Line2D([0], [0], color=COLOR_FP, lw=2.2, linestyle="--", label="FP (False Positive)"),
            Line2D([0], [0], color=COLOR_FN, lw=2.4, linestyle="-.", label="FN (Missed GT)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            fontsize=8.8,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.suptitle(
            f"Sample {sample['sample_id']} ({orig_w}x{orig_h}) | "
            f"score >= {score_thresh:.2f}, IoU >= {iou_thresh:.2f}",
            fontsize=11.5,
            fontweight="bold",
        )
        fig.subplots_adjust(top=0.80, bottom=0.14, left=0.02, right=0.98, wspace=0.06)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)


def run_test_pack_predictions(
    run_specs: Sequence[tuple[Path | str, Path | str]],
    test_pack_dir: Path | str,
    out_csv: Path | str,
    visuals_dir: Path | str,
    score_thresh: float = 0.50,
    iou_thresh: float = 0.50,
    device: str | torch.device = "auto",
    pretrained: bool = False,
) -> list[dict[str, Any]]:
    """Run one or more checkpoints on a small test pack, export CSV rows, and draw GT/TP/FP/FN overlays."""
    if not run_specs:
        raise ValueError("run_specs must contain at least one (config_path, checkpoint_path) pair")
    score_t, iou_t = _validate_thresholds(score_thresh, iou_thresh)
    resolved_device = device if isinstance(device, torch.device) else resolve_device(str(device))

    loaded_specs: list[tuple[ExperimentConfig, Path]] = []
    seen_runs: set[str] = set()
    for cfg_path, ckpt_path in run_specs:
        cfg = load_config(cfg_path)
        if cfg.run_name in seen_runs:
            raise ValueError(f"Duplicate run_name {cfg.run_name!r} in run_specs")
        seen_runs.add(cfg.run_name)
        loaded_specs.append((cfg, Path(ckpt_path)))

    samples = _load_test_pack_samples(Path(test_pack_dir))
    panels_by_sample: dict[str, list[dict[str, Any]]] = {s["sample_id"]: [] for s in samples}
    rows_by_sample: dict[str, list[dict[str, Any]]] = {s["sample_id"]: [] for s in samples}

    for cfg, ckpt_path in loaded_specs:
        model = build_model(cfg, pretrained=pretrained)
        state_dict, _ = load_checkpoint(
            ckpt_path,
            device=resolved_device,
            expected_run_name=cfg.run_name,
        )
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()

        with torch.no_grad():
            for sample in samples:
                with Image.open(sample["img_path"]) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                img_tensor = F.to_tensor(rgb_img).to(resolved_device)
                outputs = model([img_tensor])
                out = outputs[0]

                pred_boxes_t = out["boxes"].detach().cpu()
                pred_scores_t = out["scores"].detach().cpu()
                if "labels" in out:
                    keep_uav = out["labels"].detach().cpu() == UAV_LABEL
                    pred_boxes_t = pred_boxes_t[keep_uav]
                    pred_scores_t = pred_scores_t[keep_uav]

                raw_boxes = pred_boxes_t.numpy().astype(np.float64, copy=False).reshape(-1, 4)
                raw_scores = pred_scores_t.numpy().astype(np.float64, copy=False).reshape(-1)
                gt_boxes = sample["gt_boxes"]

                counts = compute_fixed_threshold_counts(
                    [{"sample_id": sample["sample_id"], "boxes": raw_boxes, "scores": raw_scores}],
                    [{"sample_id": sample["sample_id"], "boxes": gt_boxes}],
                    score_thresh=score_t,
                    iou_thresh=iou_t,
                )
                img_res = counts["per_image"][0]

                kept_mask = raw_scores >= score_t
                kept_boxes = raw_boxes[kept_mask]
                kept_scores = raw_scores[kept_mask]
                if len(kept_scores) > 0:
                    order = np.argsort(-kept_scores, kind="mergesort")
                    kept_boxes = kept_boxes[order]
                    kept_scores = kept_scores[order]

                row: dict[str, Any] = {
                    "sample_id": sample["sample_id"],
                    "run_name": cfg.run_name,
                    "checkpoint_name": ckpt_path.name,
                    "orig_w": sample["orig_w"],
                    "orig_h": sample["orig_h"],
                    "gt_count": int(img_res["gt_count"]),
                    "gt_boxes": _round_boxes(gt_boxes),
                    "pred_count": int(img_res["pred_count"]),
                    "pred_boxes": _round_boxes(kept_boxes),
                    "pred_scores": _round_scores(kept_scores),
                    "tp": int(img_res["tp"]),
                    "fp": int(img_res["fp"]),
                    "fn": int(img_res["fn"]),
                }
                rows_by_sample[sample["sample_id"]].append(row)

                tp_indices = img_res["tp_pred_indices"]
                fp_indices = img_res["fp_pred_indices"]
                missed_gt_indices = img_res["missed_gt_indices"]
                panels_by_sample[sample["sample_id"]].append(
                    {
                        "run_name": cfg.run_name,
                        "gt_count": int(img_res["gt_count"]),
                        "tp": int(img_res["tp"]),
                        "fp": int(img_res["fp"]),
                        "fn": int(img_res["fn"]),
                        "tp_boxes": [raw_boxes[i].tolist() for i in tp_indices],
                        "tp_scores": [float(raw_scores[i]) for i in tp_indices],
                        "fp_boxes": [raw_boxes[i].tolist() for i in fp_indices],
                        "fp_scores": [float(raw_scores[i]) for i in fp_indices],
                        "fn_boxes": [gt_boxes[i].tolist() for i in missed_gt_indices],
                    }
                )

    ordered_rows: list[dict[str, Any]] = []
    out_vis_dir = Path(visuals_dir)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sid = sample["sample_id"]
        ordered_rows.extend(rows_by_sample[sid])
        _save_sample_comparison_png(
            sample,
            panels_by_sample[sid],
            out_vis_dir / f"{sid}_comparison.png",
            score_thresh=score_t,
            iou_thresh=iou_t,
        )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTION_CSV_COLUMNS))
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "run_name": row["run_name"],
                    "checkpoint_name": row["checkpoint_name"],
                    "orig_w": row["orig_w"],
                    "orig_h": row["orig_h"],
                    "gt_count": row["gt_count"],
                    "gt_boxes": json.dumps(row["gt_boxes"]),
                    "pred_count": row["pred_count"],
                    "pred_boxes": json.dumps(row["pred_boxes"]),
                    "pred_scores": json.dumps(row["pred_scores"]),
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                }
            )

    return ordered_rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.predict_pack``."""
    parser = argparse.ArgumentParser(
        description="Run Anti-UAV checkpoints on a small test pack and export CSV predictions + GT/TP/FP/FN panels."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("CONFIG", "CHECKPOINT"),
        default=None,
        help="Repeatable (config YAML, checkpoint .pt) pair to evaluate on the test pack.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=None,
        help="Alternative to --run: list of config YAML paths (paired 1-to-1 with --checkpoints).",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Alternative to --run: list of checkpoint .pt paths (paired 1-to-1 with --configs).",
    )
    parser.add_argument(
        "--test-pack-dir",
        type=Path,
        required=True,
        help="Test-pack directory containing img/ and xml/ subdirectories.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path for test-pack predictions.",
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        required=True,
        help="Output directory for <sample_id>_comparison.png overlays.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.50,
        help="Confidence score threshold for fixed-threshold TP/FP/FN matching (default: 0.50).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.50,
        help="IoU threshold for TP/FP/FN matching (default: 0.50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device ('auto', 'cpu', 'cuda', ...).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for test-pack prediction CSV export and comparison visuals."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_specs: list[tuple[Path, Path]] = []
    if args.run is not None and (args.configs is not None or args.checkpoints is not None):
        parser.error("Use either --run CONFIG CHECKPOINT or --configs/--checkpoints, not both.")
    if args.run is not None:
        run_specs = [(Path(c), Path(k)) for c, k in args.run]
    elif args.configs is not None or args.checkpoints is not None:
        if args.configs is None or args.checkpoints is None:
            parser.error("--configs and --checkpoints must be provided together.")
        if len(args.configs) != len(args.checkpoints):
            parser.error(
                f"Mismatched lengths: {len(args.configs)} --configs vs {len(args.checkpoints)} --checkpoints."
            )
        run_specs = list(zip(args.configs, args.checkpoints))
    else:
        parser.error("Provide at least one run via --run CONFIG CHECKPOINT or --configs/--checkpoints.")

    rows = run_test_pack_predictions(
        run_specs=run_specs,
        test_pack_dir=args.test_pack_dir,
        out_csv=args.out_csv,
        visuals_dir=args.visuals_dir,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        device=args.device,
    )
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    print(
        f"Test-pack prediction complete | rows={len(rows)} "
        f"(TP={total_tp} FP={total_fp} FN={total_fn}) | "
        f"csv={args.out_csv} | visuals={args.visuals_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
