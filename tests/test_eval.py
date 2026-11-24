"""Unit and integration tests for scale-stratified detection evaluation and latency benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import DUTAntiUAVDataset, detection_collate_fn
from src.eval import (
    benchmark_latency,
    compute_ap_at_iou,
    compute_fixed_threshold_counts,
    compute_iou_matrix,
    evaluate_by_scale_bins,
    evaluate_predictions,
    load_checkpoint,
    main,
    run_inference_on_dataset,
)
from src.model import build_model


def _write_voc_sample(
    split_dir: Path,
    stem: str,
    size: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
) -> None:
    img_dir = split_dir / "img"
    xml_dir = split_dir / "xml"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    width, height = size
    Image.new("RGB", (width, height), color=(32, 64, 96)).save(img_dir / f"{stem}.jpg")
    objs = "".join(
        f"<object><name>UAV</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for x1, y1, x2, y2 in boxes
    )
    (xml_dir / f"{stem}.xml").write_text(
        f"<annotation><size><width>{width}</width><height>{height}</height></size>{objs}</annotation>",
        encoding="utf-8",
    )


def test_iou_ap50_and_fixed_threshold_counts() -> None:
    iou = compute_iou_matrix(
        np.array([[0, 0, 10, 10]]),
        np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
    )
    assert np.allclose(iou, [[1.0, 0.0]])

    gts = [
        {"sample_id": "a", "boxes": np.array([[10, 10, 20, 20]], dtype=float), "orig_size": (1000, 1000)},
        {"sample_id": "b", "boxes": np.array([[100, 100, 300, 300]], dtype=float), "orig_size": (1000, 1000)},
    ]
    preds = [
        {
            "sample_id": "a",
            "boxes": np.array([[10, 10, 20, 20], [500, 500, 520, 520]], dtype=float),
            "scores": np.array([0.9, 0.6]),
        },
        {
            "sample_id": "b",
            "boxes": np.array([[100, 100, 300, 300]], dtype=float),
            "scores": np.array([0.4]),
        },
    ]
    ap50 = compute_ap_at_iou(preds, gts, iou_thresh=0.50)
    assert 0.7 < ap50 < 0.9  # rank1 TP, rank2 FP, rank3 TP -> P=[1, 0.5, 0.667], R=[0.5, 0.5, 1.0] => AP=0.8333
    assert ap50 == pytest.approx(5.0 / 6.0, abs=1e-9)

    counts = compute_fixed_threshold_counts(preds, gts, score_thresh=0.50, iou_thresh=0.50)
    assert counts["tp"] == 1 and counts["fp"] == 1 and counts["fn"] == 1 and counts["gt_count"] == 2
    assert counts["precision"] == 0.5 and counts["recall"] == 0.5
    assert counts["f1"] == pytest.approx(0.5)
    assert len(counts["per_image"]) == 2


def test_scale_stratified_bins_use_original_image_coordinates() -> None:
    # Box "a" is 10x10 on 1000x1000 -> ratio 0.0001 (tiny)
    # Box "b" is 200x200 on 1000x1000 -> ratio 0.04 (medium_large)
    gts = [
        {"sample_id": "a", "boxes": np.array([[10, 10, 20, 20]], dtype=float), "orig_size": (1000, 1000)},
        {"sample_id": "b", "boxes": np.array([[100, 100, 300, 300]], dtype=float), "orig_size": (1000, 1000)},
    ]
    preds = [
        {"sample_id": "a", "boxes": np.zeros((0, 4), dtype=float), "scores": np.zeros((0,), dtype=float)},  # missed tiny
        {"sample_id": "b", "boxes": np.array([[100, 100, 300, 300]], dtype=float), "scores": np.array([0.95])},  # hit large
    ]
    bins = {"edges": {"tiny_max": 0.001, "small_max": 0.005}}
    res = evaluate_by_scale_bins(preds, gts, bins, score_thresh=0.50, iou_thresh=0.50)
    assert res["tiny"]["gt_count"] == 1 and res["tiny"]["tp"] == 0 and res["tiny"]["fn"] == 1 and res["tiny"]["recall"] == 0.0
    assert res["small"]["gt_count"] == 0 and res["small"]["tp"] == 0 and res["small"]["fn"] == 0 and res["small"]["recall"] == 0.0
    assert res["medium_large"]["gt_count"] == 1 and res["medium_large"]["tp"] == 1 and res["medium_large"]["recall"] == 1.0
    assert res["tiny"]["ap50"] == 0.0
    assert res["small"]["ap50"] == 0.0
    assert res["medium_large"]["ap50"] == pytest.approx(1.0)


def test_compute_iou_matrix_edge_cases() -> None:
    # Partial overlap: [0,0,10,10] vs [5,0,15,10] -> inter=50, union=150 -> 1/3
    # Touching edge: [0,0,10,10] vs [10,0,20,10] -> 0.0
    # Degenerate zero-area box: [5,5,5,5] vs [5,5,5,5] -> 0.0 (must emit zero RuntimeWarnings)
    iou = compute_iou_matrix(
        np.array([[0, 0, 10, 10], [5, 5, 5, 5]], dtype=float),
        np.array([[5, 0, 15, 10], [10, 0, 20, 10], [5, 5, 5, 5]], dtype=float),
    )
    assert iou.shape == (2, 3)
    assert iou[0, 0] == pytest.approx(1.0 / 3.0)
    assert iou[0, 1] == 0.0
    assert iou[0, 2] == 0.0
    assert np.all(iou[1] == 0.0)

    assert compute_iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 10, 10]])).shape == (0, 1)
    assert compute_iou_matrix(np.array([[0, 0, 10, 10]]), np.zeros((0, 4))).shape == (1, 0)
    assert compute_iou_matrix(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 0)

    with pytest.raises(ValueError, match="shape"):
        compute_iou_matrix(np.zeros((2, 3)), np.zeros((1, 4)))


def test_ap_fixed_threshold_and_ap50_95_edge_cases() -> None:
    # 1) Zero GT boxes with predictions -> all FP, AP=0, precision=0, recall=0, f1=0
    empty_gts = [{"sample_id": "empty", "boxes": np.zeros((0, 4), dtype=float), "orig_size": (200, 200)}]
    fp_preds = [
        {"sample_id": "empty", "boxes": np.array([[10, 10, 30, 30]], dtype=float), "scores": np.array([0.9])}
    ]
    assert compute_ap_at_iou(fp_preds, empty_gts, iou_thresh=0.50) == 0.0
    empty_counts = compute_fixed_threshold_counts(fp_preds, empty_gts, score_thresh=0.50, iou_thresh=0.50)
    assert empty_counts["gt_count"] == 0 and empty_counts["tp"] == 0 and empty_counts["fp"] == 1
    assert empty_counts["precision"] == 0.0 and empty_counts["recall"] == 0.0 and empty_counts["f1"] == 0.0

    # 2) Zero predictions with GT boxes -> all FN, AP=0
    single_gt = [{"sample_id": "img1", "boxes": np.array([[10, 10, 30, 30]], dtype=float), "orig_size": (200, 200)}]
    assert compute_ap_at_iou([], single_gt, iou_thresh=0.50) == 0.0
    zero_pred_counts = compute_fixed_threshold_counts([], single_gt, score_thresh=0.50, iou_thresh=0.50)
    assert zero_pred_counts["tp"] == 0 and zero_pred_counts["fp"] == 0 and zero_pred_counts["fn"] == 1
    assert zero_pred_counts["precision"] == 0.0 and zero_pred_counts["recall"] == 0.0 and zero_pred_counts["f1"] == 0.0

    # 3) Multiple predictions competing for 1 GT box: greedy 1-to-1 by descending score;
    #    extra overlapping detection must become FP. Also mix in a 0-GT empty image with an FP.
    gts = [
        {"sample_id": "s1", "boxes": np.array([[0, 0, 10, 10]], dtype=float), "orig_size": (100, 100)},
        {"sample_id": "s2_empty", "boxes": np.zeros((0, 4), dtype=float), "orig_size": (100, 100)},
    ]
    preds = [
        {
            "sample_id": "s1",
            # idx 0: lower score (0.7) with IoU 1.0; idx 1: higher score (0.9) with IoU 80/120=0.6667
            "boxes": np.array([[0, 0, 10, 10], [2, 0, 12, 10]], dtype=float),
            "scores": np.array([0.7, 0.9]),
        },
        {
            "sample_id": "s2_empty",
            # Highest score (0.95) on empty image -> rank 1 FP!
            "boxes": np.array([[10, 10, 20, 20]], dtype=float),
            "scores": np.array([0.95]),
        },
    ]
    counts = compute_fixed_threshold_counts(preds, gts, score_thresh=0.50, iou_thresh=0.50)
    assert counts["tp"] == 1 and counts["fp"] == 2 and counts["fn"] == 0 and counts["gt_count"] == 1
    assert counts["precision"] == pytest.approx(1.0 / 3.0)
    assert counts["recall"] == pytest.approx(1.0)
    assert counts["f1"] == pytest.approx(0.5)
    # In s1, pred idx 1 (score 0.9) must match GT 0; pred idx 0 (score 0.7) must be FP!
    per_img_by_id = {row["sample_id"]: row for row in counts["per_image"]}
    assert per_img_by_id["s1"]["tp_pred_indices"] == [1]
    assert per_img_by_id["s1"]["fp_pred_indices"] == [0]
    assert per_img_by_id["s1"]["missed_gt_indices"] == []
    assert per_img_by_id["s2_empty"]["tp"] == 0 and per_img_by_id["s2_empty"]["fp"] == 1

    # 4) Exact AP50, AP75, AP50:95 across COCO thresholds [0.50..0.95]
    # Box [0,0,10,10] vs [0,0,8,8]: inter=64, union=100 -> IoU = 0.64
    # Passes 0.50, 0.55, 0.60 (3 of 10 thresholds) -> AP50=1.0, AP75=0.0, AP50:95=0.3
    gt_iou = [{"sample_id": "x", "boxes": np.array([[0, 0, 10, 10]], dtype=float), "orig_size": (100, 100)}]
    pred_iou = [{"sample_id": "x", "boxes": np.array([[0, 0, 8, 8]], dtype=float), "scores": np.array([0.9])}]
    ev = evaluate_predictions(pred_iou, gt_iou, scale_bins=None, score_thresh=0.50, iou_thresh=0.50)
    assert ev["ap50"] == pytest.approx(1.0)
    assert ev["ap75"] == pytest.approx(0.0)
    assert ev["ap50_95"] == pytest.approx(0.3)
    assert ev["by_scale"] is None

    # 5) Duplicate sample_id must fail loudly
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        compute_ap_at_iou(pred_iou, [gt_iou[0], gt_iou[0]])


def test_scale_bins_cross_bin_isolation_conservation_and_eda_format() -> None:
    # Same pixel box [0, 0, 20, 20] (area=400):
    # - on 1000x1000 -> ratio 0.0004 (tiny)
    # - on 200x200 -> ratio 0.01 (medium_large)
    # Plus an image with BOTH a tiny and small box, plus a large FP on another image.
    gts = [
        {"sample_id": "img_tiny", "boxes": np.array([[0, 0, 20, 20]], dtype=float), "orig_size": (1000, 1000)},
        {
            "sample_id": "img_mixed",
            "boxes": np.array([[10, 10, 25, 25], [100, 100, 150, 150]], dtype=float),  # 225/1e6=tiny, 2500/1e6=small
            "orig_size": (1000, 1000),
        },
        {"sample_id": "img_large_small_res", "boxes": np.array([[0, 0, 20, 20]], dtype=float), "orig_size": (200, 200)},
    ]
    preds = [
        # img_tiny: true tiny detection (score 0.70)
        {"sample_id": "img_tiny", "boxes": np.array([[0, 0, 20, 20]], dtype=float), "scores": np.array([0.70])},
        # img_mixed: tiny missed, small detected (score 0.85), plus a huge medium_large FP (score 0.99)
        {
            "sample_id": "img_mixed",
            "boxes": np.array([[100, 100, 150, 150], [500, 500, 700, 700]], dtype=float),
            "scores": np.array([0.85, 0.99]),
        },
        # img_large_small_res: true medium_large detection (score 0.95)
        {
            "sample_id": "img_large_small_res",
            "boxes": np.array([[0, 0, 20, 20]], dtype=float),
            "scores": np.array([0.95]),
        },
    ]

    # Test full eda_train_val.json nested schema as well as standalone scale_bins.json schema
    eda_wrapped = {"source_splits": ["train", "val"], "scale_bins": {"edges": {"tiny_max": 0.001, "small_max": 0.005}}}
    by_scale = evaluate_by_scale_bins(preds, gts, eda_wrapped, score_thresh=0.50, iou_thresh=0.50)

    # Conservation against overall fixed-threshold counts
    fixed = compute_fixed_threshold_counts(preds, gts, score_thresh=0.50, iou_thresh=0.50)
    assert sum(v["gt_count"] for v in by_scale.values()) == fixed["gt_count"] == 4
    assert sum(v["tp"] for v in by_scale.values()) == fixed["tp"] == 3
    assert sum(v["fn"] for v in by_scale.values()) == fixed["fn"] == 1

    # Tiny: 2 GTs, 1 TP (score 0.70), 1 FN -> recall 0.5.
    # Neither the 0.95 medium_large TP nor the 0.99 medium_large FP may pollute tiny AP50!
    assert by_scale["tiny"]["gt_count"] == 2
    assert by_scale["tiny"]["tp"] == 1 and by_scale["tiny"]["fn"] == 1
    assert by_scale["tiny"]["recall"] == pytest.approx(0.5)
    assert by_scale["tiny"]["ap50"] == pytest.approx(0.5)

    # Small: 1 GT, 1 TP (score 0.85) -> recall 1.0, AP50 1.0 (unpolluted by medium_large FP/TP)
    assert by_scale["small"]["gt_count"] == 1 and by_scale["small"]["tp"] == 1
    assert by_scale["small"]["recall"] == pytest.approx(1.0)
    assert by_scale["small"]["ap50"] == pytest.approx(1.0)

    # Medium/large: 1 GT, 1 TP (score 0.95), plus 1 medium_large FP (score 0.99) -> rank1 FP, rank2 TP => AP50 = 0.5
    assert by_scale["medium_large"]["gt_count"] == 1 and by_scale["medium_large"]["tp"] == 1
    assert by_scale["medium_large"]["recall"] == pytest.approx(1.0)
    assert by_scale["medium_large"]["ap50"] == pytest.approx(0.5)


def test_run_inference_preserves_original_coords_across_resolutions_and_mode(tmp_path: Path) -> None:
    split_dir = tmp_path / "val"
    # Image 1: 600x400 (area 240,000) with a 12x12 box -> area_ratio = 144/240000 = 0.0006 (tiny)
    _write_voc_sample(split_dir, "00001", size=(600, 400), boxes=[(100, 80, 112, 92)])
    # Image 2: 600x400 empty (0 GT boxes)
    _write_voc_sample(split_dir, "00002", size=(600, 400), boxes=[])

    ds = DUTAntiUAVDataset(split_dir, hflip_prob=0.0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)
    bins = {"edges": {"tiny_max": 0.001, "small_max": 0.005}}

    for cfg_path in ("configs/res_480.yaml", "configs/res_800.yaml"):
        cfg = load_config(cfg_path)
        model = build_model(cfg, pretrained=False)
        model.train()  # Verify training mode is restored after inference

        preds, gts = run_inference_on_dataset(model, loader, device="cpu")
        assert model.training is True
        assert len(preds) == 2 and len(gts) == 2
        assert [g["sample_id"] for g in gts] == ["00001", "00002"]
        assert all(g["orig_size"] == (600, 400) for g in gts)
        assert gts[0]["boxes"].shape == (1, 4)
        assert gts[1]["boxes"].shape == (0, 4)

        for p in preds:
            assert p["boxes"].ndim == 2 and p["boxes"].shape[1] == 4
            assert p["scores"].ndim == 1 and len(p["scores"]) == len(p["boxes"])
            if len(p["boxes"]) > 0:
                # TorchVision rescales predictions strictly back to original (600, 400) pixels
                assert np.all(p["boxes"][:, 0] >= -1e-3) and np.all(p["boxes"][:, 2] <= 600.0 + 1e-3)
                assert np.all(p["boxes"][:, 1] >= -1e-3) and np.all(p["boxes"][:, 3] <= 400.0 + 1e-3)

        by_scale = evaluate_by_scale_bins(preds, gts, bins)
        # Both res_480 and res_800 MUST assign the 12x12 GT on 600x400 to 'tiny' identically
        assert by_scale["tiny"]["gt_count"] == 1
        assert by_scale["small"]["gt_count"] == 0
        assert by_scale["medium_large"]["gt_count"] == 0


def test_benchmark_latency_cpu_and_validation() -> None:
    class _DummyDetector(nn.Module):
        def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            assert not self.training
            return [
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
                for _ in images
            ]

    dummy_ds = [(torch.zeros((3, 64, 80), dtype=torch.float32), {"sample_id": str(i)}) for i in range(3)]
    model = _DummyDetector()
    model.train()

    stats = benchmark_latency(model, dummy_ds, device="cpu", warmup=2, num_images=10)
    assert model.training is True
    assert stats["num_images"] == 3  # capped cleanly at len(dataset)
    assert stats["device"] == "cpu"
    assert stats["mean_ms"] > 0.0 and stats["median_ms"] > 0.0
    assert stats["fps"] == pytest.approx(1000.0 / stats["mean_ms"])

    with pytest.raises(ValueError, match="empty"):
        benchmark_latency(model, [], device="cpu", warmup=1, num_images=1)
    with pytest.raises(ValueError, match="warmup"):
        benchmark_latency(model, dummy_ds, device="cpu", warmup=-1, num_images=1)
    with pytest.raises(ValueError, match="num_images"):
        benchmark_latency(model, dummy_ds, device="cpu", warmup=0, num_images=0)


def test_cli_eval_json_schema_and_checkpoint_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_voc_sample(data_dir / "test", "00001", size=(200, 160), boxes=[(20, 20, 40, 40)])
    _write_voc_sample(data_dir / "test", "00002", size=(200, 160), boxes=[(50, 40, 120, 110)])

    cfg = load_config("configs/res_480.yaml")
    model = build_model(cfg, pretrained=False)

    # 1) Dict checkpoint with metadata + standalone scale_bins.json
    ckpt_dict = tmp_path / "res_480_best.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "run_name": "res_480", "epoch": 1},
        ckpt_dict,
    )
    bins_path = tmp_path / "scale_bins.json"
    bins_path.write_text(
        json.dumps({"metric": "relative_box_area_ratio", "edges": {"tiny_max": 0.02, "small_max": 0.10}}),
        encoding="utf-8",
    )
    out_json = tmp_path / "metrics" / "res_480_test.json"

    exit_code = main(
        [
            "--config",
            "configs/res_480.yaml",
            "--checkpoint",
            str(ckpt_dict),
            "--data-dir",
            str(data_dir),
            "--split",
            "test",
            "--scale-bins",
            str(bins_path),
            "--out",
            str(out_json),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--latency-warmup",
            "1",
            "--latency-images",
            "2",
        ]
    )
    assert exit_code == 0
    payload: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_name"] == "res_480"
    assert payload["split"] == "test"
    assert payload["checkpoint"] == str(ckpt_dict)
    for key in ("ap50", "ap75", "ap50_95"):
        assert isinstance(payload[key], float) and 0.0 <= payload[key] <= 1.0
    ft = payload["fixed_threshold"]
    assert ft["score_thresh"] == cfg.score_thresh and ft["iou_thresh"] == cfg.iou_thresh
    assert ft["gt_count"] == 2 and ft["tp"] + ft["fn"] == 2
    for key in ("tp", "fp", "fn", "gt_count"):
        assert isinstance(ft[key], int)
    for key in ("precision", "recall", "f1"):
        assert isinstance(ft[key], float)
    assert set(payload["by_scale"].keys()) == {"tiny", "small", "medium_large"}
    for bin_stats in payload["by_scale"].values():
        assert set(bin_stats.keys()) == {"gt_count", "tp", "fn", "recall", "ap50"}
    lat = payload["latency"]
    assert set(lat.keys()) == {"mean_ms", "median_ms", "fps", "num_images", "device"}
    assert lat["num_images"] == 2 and lat["device"] == "cpu" and lat["mean_ms"] > 0.0

    # 2) Plain state_dict checkpoint + full eda_train_val.json
    ckpt_plain = tmp_path / "res_480_plain.pt"
    torch.save(model.state_dict(), ckpt_plain)
    eda_path = tmp_path / "eda_train_val.json"
    eda_path.write_text(
        json.dumps({"source_splits": ["train", "val"], "scale_bins": {"edges": {"tiny_max": 0.02, "small_max": 0.10}}}),
        encoding="utf-8",
    )
    out_plain = tmp_path / "res_480_plain_test.json"
    assert (
        main(
            [
                "--config",
                "configs/res_480.yaml",
                "--checkpoint",
                str(ckpt_plain),
                "--data-dir",
                str(data_dir),
                "--split",
                "test",
                "--scale-bins",
                str(eda_path),
                "--out",
                str(out_plain),
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--latency-warmup",
                "0",
                "--latency-images",
                "1",
            ]
        )
        == 0
    )
    assert out_plain.exists()

    # 3) Mismatched run_name in checkpoint metadata must fail loudly
    ckpt_bad = tmp_path / "wrong_run.pt"
    torch.save({"model_state_dict": model.state_dict(), "run_name": "res_800"}, ckpt_bad)
    with pytest.raises(ValueError, match="run_name"):
        load_checkpoint(ckpt_bad, device="cpu", expected_run_name="res_480")

    # 4) Checkpoint containing ExperimentConfig dataclass metadata loads cleanly via fallback
    ckpt_with_cfg = tmp_path / "res_480_with_cfg.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "run_name": "res_480", "config": cfg},
        ckpt_with_cfg,
    )
    loaded_sd, meta = load_checkpoint(ckpt_with_cfg, device="cpu", expected_run_name="res_480")
    assert set(loaded_sd.keys()) == set(model.state_dict().keys())
    assert meta["config"] == cfg


def test_interleaved_threshold_indices_and_empty_image_scale_fp() -> None:
    # Interleaved scores: idx 0 below thresh (0.2), idx 1 FP (0.9), idx 2 TP (0.7)
    gts = [
        {"sample_id": "a", "boxes": np.array([[10, 10, 20, 20]], dtype=float), "orig_size": (1000, 1000)},
        {"sample_id": "empty_bg", "boxes": np.zeros((0, 4), dtype=float), "orig_size": (1000, 1000)},
    ]
    preds = [
        {
            "sample_id": "a",
            "boxes": np.array(
                [[10, 10, 20, 20], [500, 500, 515, 515], [10, 10, 20, 20]],
                dtype=float,
            ),
            "scores": np.array([0.2, 0.9, 0.7]),
        },
        {
            "sample_id": "empty_bg",
            # High-score tiny background FP on an empty 0-GT image
            "boxes": np.array([[50, 50, 60, 60]], dtype=float),
            "scores": np.array([0.95]),
        },
    ]
    counts = compute_fixed_threshold_counts(preds, gts, score_thresh=0.50, iou_thresh=0.50)
    per_img = {row["sample_id"]: row for row in counts["per_image"]}
    assert per_img["a"]["tp_pred_indices"] == [2]
    assert per_img["a"]["fp_pred_indices"] == [1]
    assert per_img["a"]["matched_gt_indices"] == [0]
    assert per_img["a"]["missed_gt_indices"] == []

    # Tiny bin has 1 GT (sample "a") and 3 tiny detections:
    #   score 0.95 (empty_bg FP), score 0.90 (sample "a" FP), score 0.70 (sample "a" TP)
    #   -> rank 3 TP => P=[0, 0, 1/3], R=[0, 0, 1.0] => AP50 = 1/3
    bins = {"edges": {"tiny_max": 0.001, "small_max": 0.005}}
    by_scale = evaluate_by_scale_bins(preds, gts, bins, score_thresh=0.50, iou_thresh=0.50)
    assert by_scale["tiny"]["gt_count"] == 1
    assert by_scale["tiny"]["tp"] == 1 and by_scale["tiny"]["fn"] == 0
    assert by_scale["tiny"]["ap50"] == pytest.approx(1.0 / 3.0)
    assert by_scale["small"]["ap50"] == 0.0
    assert by_scale["medium_large"]["ap50"] == 0.0

