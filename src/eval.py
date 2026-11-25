"""Transparent single-class AP evaluator, original-coordinate scale binning, and latency benchmark."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.dataset import DUTAntiUAVDataset, UAV_LABEL, detection_collate_fn
from src.model import build_model
from src.prepare_data import VALID_SPLITS, assign_scale_bin

SCALE_BIN_NAMES: tuple[str, ...] = ("tiny", "small", "medium_large")
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_iou_thresh(iou_thresh: float) -> float:
    if not _is_strict_number(iou_thresh) or not math.isfinite(iou_thresh):
        raise ValueError(f"iou_thresh must be a finite number in (0.0, 1.0], got {iou_thresh!r}")
    val = float(iou_thresh)
    if not (0.0 < val <= 1.0):
        raise ValueError(f"iou_thresh must be in (0.0, 1.0], got {val!r}")
    return val


def _validate_score_thresh(score_thresh: float) -> float:
    if not _is_strict_number(score_thresh) or not math.isfinite(score_thresh):
        raise ValueError(f"score_thresh must be a finite number in [0.0, 1.0], got {score_thresh!r}")
    val = float(score_thresh)
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"score_thresh must be in [0.0, 1.0], got {val!r}")
    return val


def _to_boxes_array(boxes: Any, name: str = "boxes") -> np.ndarray:
    """Convert a tensor/array/sequence of ``[x1, y1, x2, y2]`` boxes to ``float64 (N, 4)``."""
    if isinstance(boxes, torch.Tensor):
        arr = boxes.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(boxes, dtype=np.float64)

    if arr.size == 0 and arr.ndim == 1:
        arr = arr.reshape(0, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {arr.shape}")
    if arr.size > 0 and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite coordinates")
    return arr


def _to_scores_array(scores: Any, name: str = "scores") -> np.ndarray:
    """Convert a tensor/array/sequence of confidence scores to ``float64 (N,)``."""
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(scores, dtype=np.float64)

    if arr.size == 0 and arr.ndim <= 1:
        return np.zeros((0,), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D with shape (N,), got {arr.shape}")
    if arr.size > 0 and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def _validate_orig_size(orig_size: Any, sample_id: str) -> tuple[float, float]:
    if not isinstance(orig_size, (tuple, list)) or len(orig_size) != 2:
        raise ValueError(
            f"Sample {sample_id!r} requires orig_size=(orig_w, orig_h), got {orig_size!r}"
        )
    orig_w, orig_h = orig_size
    if (
        not _is_strict_number(orig_w)
        or not _is_strict_number(orig_h)
        or not math.isfinite(orig_w)
        or not math.isfinite(orig_h)
        or float(orig_w) <= 0.0
        or float(orig_h) <= 0.0
    ):
        raise ValueError(
            f"Sample {sample_id!r} orig_size must contain positive finite numbers, got {orig_size!r}"
        )
    return float(orig_w), float(orig_h)


def _index_samples(
    predictions: Sequence[dict[str, Any]],
    ground_truths: Sequence[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate and align ground-truth and prediction records by unique ``sample_id``."""
    gt_by_id: dict[str, dict[str, Any]] = {}
    sample_order: list[str] = []

    for gt in ground_truths:
        sample_id = str(gt["sample_id"])
        if sample_id in gt_by_id:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in ground_truths")
        boxes = _to_boxes_array(gt["boxes"], name=f"ground_truths[{sample_id!r}].boxes")
        entry: dict[str, Any] = {"sample_id": sample_id, "boxes": boxes}
        if "orig_size" in gt and gt["orig_size"] is not None:
            entry["orig_size"] = gt["orig_size"]
        gt_by_id[sample_id] = entry
        sample_order.append(sample_id)

    pred_by_id: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        sample_id = str(pred["sample_id"])
        if sample_id in pred_by_id:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in predictions")
        boxes = _to_boxes_array(pred["boxes"], name=f"predictions[{sample_id!r}].boxes")
        scores = _to_scores_array(pred["scores"], name=f"predictions[{sample_id!r}].scores")
        if len(boxes) != len(scores):
            raise ValueError(
                f"Sample {sample_id!r} has mismatched boxes ({len(boxes)}) and scores ({len(scores)})"
            )
        entry = {"sample_id": sample_id, "boxes": boxes, "scores": scores}
        if "orig_size" in pred and pred["orig_size"] is not None:
            entry["orig_size"] = pred["orig_size"]
        pred_by_id[sample_id] = entry
        if sample_id not in gt_by_id:
            sample_order.append(sample_id)

    return sample_order, pred_by_id, gt_by_id


def extract_scale_edges(scale_bins: dict[str, Any]) -> dict[str, float]:
    """Extract and validate ``{'tiny_max': float, 'small_max': float}`` scale-bin edges.

    Accepts both standalone ``scale_bins.json`` (top-level ``'edges'``) and full
    ``eda_train_val.json`` (nested ``'scale_bins': {'edges': ...}``).
    """
    if not isinstance(scale_bins, dict):
        raise ValueError(f"scale_bins must be a dict, got {type(scale_bins).__name__}")

    if isinstance(scale_bins.get("edges"), dict):
        raw_edges = scale_bins["edges"]
    elif isinstance(scale_bins.get("scale_bins"), dict) and isinstance(
        scale_bins["scale_bins"].get("edges"), dict
    ):
        raw_edges = scale_bins["scale_bins"]["edges"]
    else:
        raise ValueError(
            "scale_bins must contain either top-level 'edges' or nested 'scale_bins.edges'"
        )

    if "tiny_max" not in raw_edges or "small_max" not in raw_edges:
        raise ValueError(f"scale_bins edges must contain 'tiny_max' and 'small_max', got {raw_edges!r}")

    tiny_max = raw_edges["tiny_max"]
    small_max = raw_edges["small_max"]
    if (
        not _is_strict_number(tiny_max)
        or not _is_strict_number(small_max)
        or not math.isfinite(tiny_max)
        or not math.isfinite(small_max)
        or not (0.0 < float(tiny_max) < float(small_max))
    ):
        raise ValueError(
            f"scale_bins edges must satisfy 0 < tiny_max < small_max, got {raw_edges!r}"
        )
    return {"tiny_max": float(tiny_max), "small_max": float(small_max)}


def load_scale_bins(path: str | Path) -> dict[str, Any]:
    """Load and validate a standalone ``scale_bins.json`` or ``eda_train_val.json`` file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    edges = extract_scale_edges(payload)
    if isinstance(payload.get("scale_bins"), dict) and "edges" not in payload:
        normalized = dict(payload["scale_bins"])
        normalized["edges"] = edges
        return normalized
    normalized = dict(payload)
    normalized["edges"] = edges
    return normalized


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute vectorized pairwise IoU between two ``[x1, y1, x2, y2]`` box sets.

    Uses continuous pixel-coordinate areas ``(x2 - x1) * (y2 - y1)`` and safely
    returns ``0.0`` for zero-area or non-overlapping pairs without divide warnings.
    """
    a = _to_boxes_array(boxes_a, name="boxes_a")
    b = _to_boxes_array(boxes_b, name="boxes_b")
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _match_single_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform greedy 1-to-1 matching by descending confidence score on one image.

    Returns:
        ``(order, matched_gt_for_ranked, gt_matched)`` where ``order`` indexes
        ``pred_boxes`` in descending score order, ``matched_gt_for_ranked[r]`` is
        the matched GT index (or ``-1`` if unmatched) for ranked detection ``r``,
        and ``gt_matched`` is a boolean mask over ``gt_boxes``.
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    order = np.argsort(-pred_scores, kind="mergesort")
    matched_gt_for_ranked = np.full(num_preds, -1, dtype=np.int64)
    gt_matched = np.zeros(num_gts, dtype=bool)

    if num_preds == 0 or num_gts == 0:
        return order, matched_gt_for_ranked, gt_matched

    iou_mat = compute_iou_matrix(pred_boxes[order], gt_boxes)
    for rank_idx in range(num_preds):
        ious = iou_mat[rank_idx]
        available = np.where(~gt_matched, ious, -1.0)
        best_gt = int(np.argmax(available))
        if available[best_gt] >= iou_thresh:
            gt_matched[best_gt] = True
            matched_gt_for_ranked[rank_idx] = best_gt

    return order, matched_gt_for_ranked, gt_matched


def _compute_ap_from_ranked_flags(
    tp_flags: np.ndarray,
    fp_flags: np.ndarray,
    total_gt: int,
) -> float:
    """Compute all-point interpolated Average Precision (VOC 2010+ exact envelope)."""
    if total_gt <= 0 or len(tp_flags) == 0:
        return 0.0

    cum_tp = np.cumsum(tp_flags, dtype=np.float64)
    cum_fp = np.cumsum(fp_flags, dtype=np.float64)
    recalls = cum_tp / float(total_gt)
    denom = cum_tp + cum_fp
    precisions = np.divide(cum_tp, denom, out=np.zeros_like(cum_tp), where=denom > 0.0)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))


def compute_ap_at_iou(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_thresh: float = 0.50,
) -> float:
    """Compute single-class all-point interpolated AP at a fixed IoU threshold."""
    iou_t = _validate_iou_thresh(iou_thresh)
    sample_order, pred_by_id, gt_by_id = _index_samples(predictions, ground_truths)

    total_gt = sum(len(gt_by_id[sid]["boxes"]) for sid in gt_by_id)
    if total_gt == 0:
        return 0.0

    all_scores: list[float] = []
    all_tp: list[float] = []
    all_fp: list[float] = []
    empty_boxes = np.zeros((0, 4), dtype=np.float64)

    for sample_id in sample_order:
        pred = pred_by_id.get(sample_id)
        if pred is None or len(pred["boxes"]) == 0:
            continue
        gt_boxes = gt_by_id[sample_id]["boxes"] if sample_id in gt_by_id else empty_boxes
        order, matched_gt, _ = _match_single_image(pred["boxes"], pred["scores"], gt_boxes, iou_t)
        ranked_scores = pred["scores"][order]
        for score, gt_idx in zip(ranked_scores, matched_gt):
            all_scores.append(float(score))
            if gt_idx >= 0:
                all_tp.append(1.0)
                all_fp.append(0.0)
            else:
                all_tp.append(0.0)
                all_fp.append(1.0)

    if not all_scores:
        return 0.0

    scores_arr = np.asarray(all_scores, dtype=np.float64)
    tp_arr = np.asarray(all_tp, dtype=np.float64)
    fp_arr = np.asarray(all_fp, dtype=np.float64)
    global_order = np.argsort(-scores_arr, kind="mergesort")
    return _compute_ap_from_ranked_flags(tp_arr[global_order], fp_arr[global_order], total_gt)


def compute_fixed_threshold_counts(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    score_thresh: float = 0.50,
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """Compute exact TP, FP, FN, Precision, Recall, F1, and per-image outcomes at fixed thresholds."""
    score_t = _validate_score_thresh(score_thresh)
    iou_t = _validate_iou_thresh(iou_thresh)
    sample_order, pred_by_id, gt_by_id = _index_samples(predictions, ground_truths)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    per_image: list[dict[str, Any]] = []
    empty_boxes = np.zeros((0, 4), dtype=np.float64)
    empty_scores = np.zeros((0,), dtype=np.float64)

    for sample_id in sample_order:
        gt_boxes = gt_by_id[sample_id]["boxes"] if sample_id in gt_by_id else empty_boxes
        pred = pred_by_id.get(sample_id)
        pred_boxes = pred["boxes"] if pred is not None else empty_boxes
        pred_scores = pred["scores"] if pred is not None else empty_scores

        kept_mask = pred_scores >= score_t
        kept_indices = np.flatnonzero(kept_mask)
        kept_boxes = pred_boxes[kept_indices]
        kept_scores = pred_scores[kept_indices]

        order, matched_gt, gt_matched = _match_single_image(kept_boxes, kept_scores, gt_boxes, iou_t)

        tp_pred_indices: list[int] = []
        fp_pred_indices: list[int] = []
        matched_gt_indices: list[int] = []
        for rank_idx, gt_idx in enumerate(matched_gt):
            orig_pred_idx = int(kept_indices[int(order[rank_idx])])
            if gt_idx >= 0:
                tp_pred_indices.append(orig_pred_idx)
                matched_gt_indices.append(int(gt_idx))
            else:
                fp_pred_indices.append(orig_pred_idx)

        missed_gt_indices = [int(i) for i in np.flatnonzero(~gt_matched)]
        num_gt = len(gt_boxes)
        img_tp = len(tp_pred_indices)
        img_fp = len(fp_pred_indices)
        img_fn = len(missed_gt_indices)

        total_gt += num_gt
        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn

        per_image.append(
            {
                "sample_id": sample_id,
                "gt_count": int(num_gt),
                "pred_count": int(len(kept_indices)),
                "tp": int(img_tp),
                "fp": int(img_fp),
                "fn": int(img_fn),
                "tp_pred_indices": tp_pred_indices,
                "fp_pred_indices": fp_pred_indices,
                "matched_gt_indices": matched_gt_indices,
                "missed_gt_indices": missed_gt_indices,
            }
        )

    pred_count = total_tp + total_fp
    precision = float(total_tp / pred_count) if pred_count > 0 else 0.0
    recall = float(total_tp / total_gt) if total_gt > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "score_thresh": score_t,
        "iou_thresh": iou_t,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "gt_count": int(total_gt),
        "pred_count": int(pred_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image": per_image,
    }


def _box_scale_bin(box: np.ndarray, orig_w: float, orig_h: float, edges: dict[str, float]) -> str:
    """Assign a single ``[x1, y1, x2, y2]`` box to its scale bin in original pixel coordinates."""
    box_w = max(0.0, float(box[2]) - float(box[0]))
    box_h = max(0.0, float(box[3]) - float(box[1]))
    area_ratio = (box_w * box_h) / (orig_w * orig_h)
    return assign_scale_bin(area_ratio, edges)


def evaluate_by_scale_bins(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    scale_bins: dict[str, Any],
    score_thresh: float = 0.50,
    iou_thresh: float = 0.50,
) -> dict[str, dict[str, int | float]]:
    """Evaluate fixed-threshold recall and AP50 stratified by original-image relative box area.

    Every GT box is assigned to ``tiny``, ``small``, or ``medium_large`` strictly
    using ``(w * h) / (orig_w * orig_h)`` in original pixel coordinates. When
    computing per-bin AP50 (at IoU 0.50), predictions matched to GTs in other
    bins and unmatched detections whose own original-coordinate box area falls
    outside the target bin are ignored (COCO-style scale-stratified AP).
    """
    score_t = _validate_score_thresh(score_thresh)
    iou_t = _validate_iou_thresh(iou_thresh)
    edges = extract_scale_edges(scale_bins)
    sample_order, pred_by_id, gt_by_id = _index_samples(predictions, ground_truths)

    gt_counts = {name: 0 for name in SCALE_BIN_NAMES}
    tp_counts = {name: 0 for name in SCALE_BIN_NAMES}
    fn_counts = {name: 0 for name in SCALE_BIN_NAMES}

    # Per-bin ranked AP50 pools: list of (score, is_tp)
    ap_scores: dict[str, list[float]] = {name: [] for name in SCALE_BIN_NAMES}
    ap_tp: dict[str, list[float]] = {name: [] for name in SCALE_BIN_NAMES}
    ap_fp: dict[str, list[float]] = {name: [] for name in SCALE_BIN_NAMES}

    empty_boxes = np.zeros((0, 4), dtype=np.float64)
    empty_scores = np.zeros((0,), dtype=np.float64)

    for sample_id in sample_order:
        gt_entry = gt_by_id.get(sample_id)
        pred_entry = pred_by_id.get(sample_id)
        gt_boxes = gt_entry["boxes"] if gt_entry is not None else empty_boxes
        pred_boxes = pred_entry["boxes"] if pred_entry is not None else empty_boxes
        pred_scores = pred_entry["scores"] if pred_entry is not None else empty_scores

        raw_orig_size = None
        if gt_entry is not None and "orig_size" in gt_entry:
            raw_orig_size = gt_entry["orig_size"]
        elif pred_entry is not None and "orig_size" in pred_entry:
            raw_orig_size = pred_entry["orig_size"]

        orig_dims: tuple[float, float] | None = None
        if raw_orig_size is not None:
            orig_dims = _validate_orig_size(raw_orig_size, sample_id)
        elif len(gt_boxes) > 0 or len(pred_boxes) > 0:
            raise ValueError(
                f"Sample {sample_id!r} is missing 'orig_size' required for original-coordinate scale binning"
            )

        gt_bins: list[str] = []
        if len(gt_boxes) > 0:
            assert orig_dims is not None
            orig_w, orig_h = orig_dims
            gt_bins = [_box_scale_bin(box, orig_w, orig_h, edges) for box in gt_boxes]

        # 1) Fixed-threshold TP/FN per bin (exact conservation with global fixed-threshold matching)
        kept_mask = pred_scores >= score_t
        kept_boxes = pred_boxes[kept_mask]
        kept_scores = pred_scores[kept_mask]
        _, _, fixed_gt_matched = _match_single_image(kept_boxes, kept_scores, gt_boxes, iou_t)

        for gt_idx, bin_name in enumerate(gt_bins):
            gt_counts[bin_name] += 1
            if fixed_gt_matched[gt_idx]:
                tp_counts[bin_name] += 1
            else:
                fn_counts[bin_name] += 1

        # 2) Per-bin AP50 matching at IoU 0.50 (unthresholded predictions)
        if len(pred_boxes) == 0:
            continue

        order_50, matched_gt_50, _ = _match_single_image(pred_boxes, pred_scores, gt_boxes, 0.50)
        ranked_boxes = pred_boxes[order_50]
        ranked_scores = pred_scores[order_50]
        assert orig_dims is not None
        orig_w, orig_h = orig_dims

        for box, score, gt_idx in zip(ranked_boxes, ranked_scores, matched_gt_50):
            score_f = float(score)
            if gt_idx >= 0:
                target_bin = gt_bins[int(gt_idx)]
                ap_scores[target_bin].append(score_f)
                ap_tp[target_bin].append(1.0)
                ap_fp[target_bin].append(0.0)
            else:
                pred_bin = _box_scale_bin(box, orig_w, orig_h, edges)
                ap_scores[pred_bin].append(score_f)
                ap_tp[pred_bin].append(0.0)
                ap_fp[pred_bin].append(1.0)

    results: dict[str, dict[str, int | float]] = {}
    for bin_name in SCALE_BIN_NAMES:
        gt_count = gt_counts[bin_name]
        tp = tp_counts[bin_name]
        fn = fn_counts[bin_name]
        recall = float(tp / gt_count) if gt_count > 0 else 0.0
        if gt_count == 0 or not ap_scores[bin_name]:
            ap50 = 0.0
        else:
            scores_arr = np.asarray(ap_scores[bin_name], dtype=np.float64)
            tp_arr = np.asarray(ap_tp[bin_name], dtype=np.float64)
            fp_arr = np.asarray(ap_fp[bin_name], dtype=np.float64)
            global_order = np.argsort(-scores_arr, kind="mergesort")
            ap50 = _compute_ap_from_ranked_flags(
                tp_arr[global_order], fp_arr[global_order], gt_count
            )
        results[bin_name] = {
            "gt_count": int(gt_count),
            "tp": int(tp),
            "fn": int(fn),
            "recall": recall,
            "ap50": float(ap50),
        }

    return results


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    scale_bins: dict[str, Any] | None,
    score_thresh: float = 0.50,
    iou_thresh: float = 0.50,
    *,
    include_per_image: bool = False,
) -> dict[str, Any]:
    """Compute AP50, AP75, AP50:95, fixed-threshold metrics, and optional scale-bin metrics."""
    aps = [
        compute_ap_at_iou(predictions, ground_truths, iou_thresh=thresh)
        for thresh in COCO_IOU_THRESHOLDS
    ]
    ap50 = float(aps[0])  # IoU = 0.50
    ap75 = float(aps[5])  # IoU = 0.75
    ap50_95 = float(np.mean(aps))

    fixed = compute_fixed_threshold_counts(
        predictions,
        ground_truths,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
    )
    if not include_per_image:
        fixed = {k: v for k, v in fixed.items() if k != "per_image"}

    by_scale = (
        evaluate_by_scale_bins(
            predictions,
            ground_truths,
            scale_bins,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
        )
        if scale_bins is not None
        else None
    )

    return {
        "ap50": ap50,
        "ap75": ap75,
        "ap50_95": ap50_95,
        "fixed_threshold": fixed,
        "by_scale": by_scale,
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def run_inference_on_dataset(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run model inference over a detection DataLoader, preserving original pixel coordinates."""
    dev = torch.device(device)
    was_training = model.training
    model.eval()

    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for images, targets in loader:
                device_images = [img.to(dev) for img in images]
                outputs = model(device_images)
                if len(outputs) != len(targets):
                    raise RuntimeError(
                        f"Model returned {len(outputs)} outputs for batch of {len(targets)} targets"
                    )

                for img, out, tgt in zip(images, outputs, targets):
                    if "sample_id" in tgt:
                        sample_id = str(tgt["sample_id"])
                    elif "image_id" in tgt:
                        img_id = tgt["image_id"]
                        sample_id = str(int(img_id.item()) if isinstance(img_id, torch.Tensor) else img_id)
                    else:
                        sample_id = str(len(ground_truths))

                    if "orig_size" in tgt and tgt["orig_size"] is not None:
                        orig_size = (int(tgt["orig_size"][0]), int(tgt["orig_size"][1]))
                    else:
                        orig_size = (int(img.shape[-1]), int(img.shape[-2]))

                    pred_boxes_t = out["boxes"].detach().cpu()
                    pred_scores_t = out["scores"].detach().cpu()
                    if "labels" in out:
                        pred_labels_t = out["labels"].detach().cpu()
                        keep = pred_labels_t == UAV_LABEL
                        pred_boxes_t = pred_boxes_t[keep]
                        pred_scores_t = pred_scores_t[keep]

                    pred_boxes = pred_boxes_t.numpy().astype(np.float64, copy=False).reshape(-1, 4)
                    pred_scores = pred_scores_t.numpy().astype(np.float64, copy=False).reshape(-1)
                    gt_boxes = _to_boxes_array(tgt["boxes"], name=f"target[{sample_id!r}].boxes")

                    predictions.append(
                        {
                            "sample_id": sample_id,
                            "boxes": pred_boxes,
                            "scores": pred_scores,
                        }
                    )
                    ground_truths.append(
                        {
                            "sample_id": sample_id,
                            "boxes": gt_boxes,
                            "orig_size": orig_size,
                        }
                    )
    finally:
        model.train(was_training)

    return predictions, ground_truths


def benchmark_latency(
    model: nn.Module,
    dataset: Dataset[Any] | Sequence[Any],
    device: torch.device | str,
    warmup: int = 10,
    num_images: int = 100,
) -> dict[str, Any]:
    """Benchmark synchronized single-image inference latency (excluding disk I/O)."""
    if not _is_strict_int(warmup) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative int, got {warmup!r}")
    if not _is_strict_int(num_images) or num_images <= 0:
        raise ValueError(f"num_images must be a positive int, got {num_images!r}")

    ds_len = len(dataset)
    if ds_len == 0:
        raise ValueError("dataset must not be empty when benchmarking latency")

    dev = torch.device(device)
    actual_num = min(num_images, ds_len)
    was_training = model.training
    model.eval()

    times_ms: list[float] = []
    try:
        with torch.no_grad():
            for idx in range(warmup):
                img_tensor, _ = dataset[idx % ds_len]
                img_dev = img_tensor.to(dev)
                _ = model([img_dev])
            _sync_device(dev)

            for idx in range(actual_num):
                img_tensor, _ = dataset[idx]
                img_dev = img_tensor.to(dev)
                _sync_device(dev)
                start = time.perf_counter()
                _ = model([img_dev])
                _sync_device(dev)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                times_ms.append(elapsed_ms)
    finally:
        model.train(was_training)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    fps = float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "fps": fps,
        "num_images": int(actual_num),
        "device": str(dev),
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    expected_run_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a training checkpoint dict or plain ``state_dict`` onto CPU and validate ``run_name``.

    Always deserializes with ``map_location="cpu"`` so discarded optimizer state
    never occupies GPU VRAM; callers move the model with ``model.to(device)``
    (or ``load_state_dict`` copies CPU tensors to the model's device automatically).
    """
    _ = torch.device(device)
    ckpt_path = Path(checkpoint_path)

    try:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Checkpoint '{ckpt_path}' must be a dict or state_dict, got {type(loaded).__name__}"
        )

    if "model_state_dict" in loaded:
        ckpt_run_name = loaded.get("run_name")
        if (
            expected_run_name is not None
            and ckpt_run_name is not None
            and ckpt_run_name != expected_run_name
        ):
            raise ValueError(
                f"Checkpoint run_name ({ckpt_run_name!r}) does not match "
                f"config run_name ({expected_run_name!r})"
            )
        state_dict = loaded["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint '{ckpt_path}' field 'model_state_dict' must be a dict, "
                f"got {type(state_dict).__name__}"
            )
        metadata = {k: v for k, v in loaded.items() if k != "model_state_dict"}
        return state_dict, metadata

    if loaded and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded, {}

    raise ValueError(
        f"Checkpoint '{ckpt_path}' must contain 'model_state_dict' or be a plain tensor state_dict"
    )


def resolve_device(device_arg: str) -> torch.device:
    """Resolve ``'auto'`` or an explicit PyTorch device string."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.eval``."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Anti-UAV Faster R-CNN checkpoint with scale-stratified metrics and latency benchmark."
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
        help="Path to saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root directory containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=VALID_SPLITS,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--scale-bins",
        type=Path,
        default=None,
        help="Path to scale_bins.json or eda_train_val.json (optional).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the structured evaluation JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device ('auto', 'cpu', 'cuda', ...).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional DataLoader batch size override (defaults to config batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker count override (defaults to config num_workers).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of evaluated images (for smoke tests).",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=10,
        help="Number of warmup passes before single-image latency timing (default: 10).",
    )
    parser.add_argument(
        "--latency-images",
        type=int,
        default=100,
        help="Number of single-image forward passes to time (default: 100).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scale-stratified Anti-UAV evaluation and latency benchmarking."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    scale_bins = load_scale_bins(args.scale_bins) if args.scale_bins is not None else None

    model = build_model(cfg, pretrained=False)
    state_dict, _ = load_checkpoint(
        args.checkpoint,
        device=device,
        expected_run_name=cfg.run_name,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = DUTAntiUAVDataset(
        args.data_dir / args.split,
        hflip_prob=0.0,
        max_samples=args.max_samples,
    )
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    num_workers = args.num_workers if args.num_workers is not None else cfg.num_workers
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )

    predictions, ground_truths = run_inference_on_dataset(model, loader, device=device)
    metrics = evaluate_predictions(
        predictions,
        ground_truths,
        scale_bins=scale_bins,
        score_thresh=cfg.score_thresh,
        iou_thresh=cfg.iou_thresh,
    )
    latency = benchmark_latency(
        model,
        dataset,
        device=device,
        warmup=args.latency_warmup,
        num_images=args.latency_images,
    )

    payload: dict[str, Any] = {
        "run_name": cfg.run_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "ap50": metrics["ap50"],
        "ap75": metrics["ap75"],
        "ap50_95": metrics["ap50_95"],
        "fixed_threshold": metrics["fixed_threshold"],
        "by_scale": metrics["by_scale"],
        "latency": latency,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ft = payload["fixed_threshold"]
    print(
        f"Eval complete | run={cfg.run_name} split={args.split} | "
        f"AP50={payload['ap50']:.4f} AP75={payload['ap75']:.4f} AP50:95={payload['ap50_95']:.4f} | "
        f"P={ft['precision']:.4f} R={ft['recall']:.4f} F1={ft['f1']:.4f} "
        f"(TP={ft['tp']} FP={ft['fp']} FN={ft['fn']}) | "
        f"latency={latency['mean_ms']:.2f}ms ({latency['fps']:.1f} FPS on {latency['device']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
