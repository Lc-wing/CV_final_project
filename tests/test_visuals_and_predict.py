"""Tests for log-based curve plotting, test-pack predictions, and single-image CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.config import load_config
from src.model import build_model
from src.plot_curves import (
    main as plot_curves_main,
    plot_scale_and_latency,
    plot_training_loss,
)
from src.predict_pack import (
    PREDICTION_CSV_COLUMNS,
    main as predict_pack_main,
    run_test_pack_predictions,
)
from src.predict_single import main as predict_single_main, predict_single_image


class _StubDetector(torch.nn.Module):
    """Deterministic stub detector returning scripted boxes/scores in original coordinates."""

    def __init__(self, outputs_by_size: dict[tuple[int, int], dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self._outputs_by_size = outputs_by_size
        self.register_buffer("_dummy", torch.zeros(1))

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
        return None

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        results: list[dict[str, torch.Tensor]] = []
        for img in images:
            h, w = int(img.shape[-2]), int(img.shape[-1])
            tpl = self._outputs_by_size[(w, h)]
            results.append({k: v.to(img.device) for k, v in tpl.items()})
        return results


def test_plot_curves_from_raw_logs_and_eval_json(tmp_path: Path) -> None:
    log = tmp_path / "res_800_epoch_log.csv"
    log.write_text(
        "run_name,epoch,train_total_loss,train_loss_classifier,train_loss_box_reg,"
        "train_loss_objectness,train_loss_rpn_box_reg,val_ap50,val_precision,val_recall,epoch_seconds,lr\n"
        "res_800,1,0.85,0.2,0.3,0.2,0.15,0.40,0.5,0.4,10.0,0.0002\n"
        "res_800,2,0.42,0.1,0.15,0.1,0.07,0.65,0.7,0.6,10.0,0.0001\n",
        encoding="utf-8",
    )
    out_loss = tmp_path / "loss.png"
    summary = plot_training_loss([log], out_loss)
    assert out_loss.exists() and out_loss.stat().st_size > 1000
    assert "res_800" in summary
    assert summary["res_800"]["log_file"] == "res_800_epoch_log.csv"
    assert summary["res_800"]["num_points"] == 2
    assert summary["res_800"]["final_loss"] == pytest.approx(0.42)

    ev = tmp_path / "res_800_test.json"
    ev.write_text(
        json.dumps(
            {
                "run_name": "res_800",
                "ap50": 0.72,
                "by_scale": {
                    "tiny": {"recall": 0.55, "gt_count": 100, "tp": 55, "fn": 45},
                    "small": {"recall": 0.80, "gt_count": 100, "tp": 80, "fn": 20},
                    "medium_large": {"recall": 0.92, "gt_count": 100, "tp": 92, "fn": 8},
                },
                "latency": {"mean_ms": 28.5, "fps": 35.1},
            }
        ),
        encoding="utf-8",
    )
    out_cmp = tmp_path / "scale_latency.png"
    plot_scale_and_latency([ev], out_cmp)
    assert out_cmp.exists() and out_cmp.stat().st_size > 1000


def test_plot_training_loss_step_axis_multi_run_and_validation(tmp_path: Path) -> None:
    log_a = tmp_path / "res_800_step_log.csv"
    log_a.write_text(
        "run_name,epoch,step,global_step,lr,total_loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg\n"
        "res_800,1,1,1,0.0001,1.20,0.4,0.4,0.2,0.2\n"
        "res_800,1,2,2,0.0002,0.75,0.2,0.25,0.15,0.15\n",
        encoding="utf-8",
    )
    log_b = tmp_path / "res_480_small_anchor_step_log.csv"
    log_b.write_text(
        "run_name,epoch,step,global_step,lr,total_loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg\n"
        "res_480_small_anchor,1,1,1,0.0001,1.35,0.45,0.45,0.25,0.20\n"
        "res_480_small_anchor,1,2,2,0.0002,0.80,0.25,0.25,0.15,0.15\n",
        encoding="utf-8",
    )

    out_step = tmp_path / "step_loss.png"
    summary = plot_training_loss([log_a, log_b], out_step, x_axis="step")
    assert out_step.exists() and out_step.stat().st_size > 1000
    assert set(summary.keys()) == {"res_800", "res_480_small_anchor"}
    assert summary["res_480_small_anchor"]["x_axis"] == "step"
    assert summary["res_480_small_anchor"]["min_loss"] == pytest.approx(0.80)

    # Mismatch between filename prefix and CSV run_name must fail hard
    bad_log = tmp_path / "res_480_epoch_log.csv"
    bad_log.write_text(
        "run_name,epoch,train_total_loss\nres_800,1,0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        plot_training_loss([bad_log], tmp_path / "bad.png", x_axis="epoch")

    # Mixed run_name rows in the same CSV must fail hard
    mixed_log = tmp_path / "res_800_epoch_log.csv"
    mixed_log.write_text(
        "run_name,epoch,train_total_loss\nres_800,1,0.5\nres_480,2,0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        plot_training_loss([mixed_log], tmp_path / "mixed.png", x_axis="epoch")

    # Passing step log when x_axis='epoch' must fail hard
    with pytest.raises(ValueError):
        plot_training_loss([log_a], tmp_path / "wrong_axis.png", x_axis="epoch")

    # Invalid x_axis must fail hard
    with pytest.raises(ValueError, match="x_axis"):
        plot_training_loss([log_a], tmp_path / "bad_axis.png", x_axis="invalid")


def test_plot_curves_cli_multi_run(tmp_path: Path) -> None:
    logs: list[Path] = []
    evals: list[Path] = []
    for run_name, ap50, tiny_tp, mean_ms in [
        ("res_800", 0.78, 62, 34.0),
        ("res_480", 0.64, 38, 16.5),
        ("res_480_small_anchor", 0.73, 56, 17.2),
    ]:
        log_path = tmp_path / f"{run_name}_epoch_log.csv"
        log_path.write_text(
            "run_name,epoch,train_total_loss,train_loss_classifier,train_loss_box_reg,"
            "train_loss_objectness,train_loss_rpn_box_reg,val_ap50,val_precision,val_recall,epoch_seconds,lr\n"
            f"{run_name},1,0.90,0.25,0.30,0.20,0.15,{ap50 - 0.15:.2f},0.60,0.50,12.0,0.0002\n"
            f"{run_name},2,0.50,0.15,0.15,0.10,0.10,{ap50:.2f},0.75,0.70,12.0,0.0001\n",
            encoding="utf-8",
        )
        logs.append(log_path)

        ev_path = tmp_path / f"{run_name}_test.json"
        ev_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "ap50": ap50,
                    "by_scale": {
                        "tiny": {"recall": tiny_tp / 100.0, "gt_count": 100, "tp": tiny_tp, "fn": 100 - tiny_tp},
                        "small": {"recall": 0.80, "gt_count": 100, "tp": 80, "fn": 20},
                        "medium_large": {"recall": 0.90, "gt_count": 100, "tp": 90, "fn": 10},
                    },
                    "latency": {"mean_ms": mean_ms, "fps": 1000.0 / mean_ms},
                }
            ),
            encoding="utf-8",
        )
        evals.append(ev_path)

    loss_out = tmp_path / "figs" / "loss.png"
    cmp_out = tmp_path / "figs" / "compare.png"
    rc = plot_curves_main(
        [
            "--logs",
            *(str(p) for p in logs),
            "--loss-out",
            str(loss_out),
            "--x-axis",
            "epoch",
            "--eval-jsons",
            *(str(p) for p in evals),
            "--compare-out",
            str(cmp_out),
        ]
    )
    assert rc == 0
    assert loss_out.exists() and loss_out.stat().st_size > 1000
    assert cmp_out.exists() and cmp_out.stat().st_size > 1000


def test_predict_pack_and_single_image_cli(tmp_path: Path) -> None:
    cfg = load_config("configs/res_480.yaml")
    model = build_model(cfg, pretrained=False)
    ckpt = tmp_path / "res_480_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "run_name": "res_480"}, ckpt)

    pack = tmp_path / "test_pack"
    (pack / "img").mkdir(parents=True)
    (pack / "xml").mkdir(parents=True)
    img_path = pack / "img" / "00010.jpg"
    Image.new("RGB", (160, 120)).save(img_path)
    (pack / "xml" / "00010.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )

    out_csv = tmp_path / "preds.csv"
    vis_dir = tmp_path / "visuals"
    rows = run_test_pack_predictions([(Path("configs/res_480.yaml"), ckpt)], pack, out_csv, vis_dir)
    assert out_csv.exists() and len(rows) == 1 and rows[0]["sample_id"] == "00010" and rows[0]["run_name"] == "res_480"
    assert (vis_dir / "00010_comparison.png").exists()

    out_single = tmp_path / "single.png"
    res = predict_single_image(Path("configs/res_480.yaml"), ckpt, img_path, score_thresh=0.5, out_image=out_single)
    assert out_single.exists() and res["run_name"] == "res_480" and "boxes" in res and "scores" in res


def test_predict_pack_csv_schema_tp_fp_fn_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = tmp_path / "test_pack"
    (pack / "img").mkdir(parents=True)
    (pack / "xml").mkdir(parents=True)

    # Sample 00001: 2 GT boxes (one will be TP, one will be FN; plus 1 FP & 1 below-threshold pred)
    Image.new("RGB", (200, 150), color=(100, 120, 140)).save(pack / "img" / "00001.jpg")
    (pack / "xml" / "00001.xml").write_text(
        "<annotation>"
        "<object><name>UAV</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>60</xmax><ymax>60</ymax></bndbox></object>"
        "<object><name>UAV</name><bndbox><xmin>120</xmin><ymin>80</ymin><xmax>160</xmax><ymax>120</ymax></bndbox></object>"
        "</annotation>",
        encoding="utf-8",
    )
    # Sample 00002: empty (0-object) image
    Image.new("RGB", (160, 120), color=(40, 40, 40)).save(pack / "img" / "00002.jpg")
    (pack / "xml" / "00002.xml").write_text("<annotation></annotation>", encoding="utf-8")

    ckpt_480 = tmp_path / "res_480_best.pt"
    ckpt_sa = tmp_path / "res_480_small_anchor_best.pt"
    torch.save({"model_state_dict": {}, "run_name": "res_480"}, ckpt_480)
    torch.save({"model_state_dict": {}, "run_name": "res_480_small_anchor"}, ckpt_sa)

    stub_outputs = {
        (200, 150): {
            "boxes": torch.tensor(
                [
                    [21.0, 21.0, 59.0, 59.0],  # TP for GT #0
                    [5.0, 100.0, 35.0, 130.0],  # FP above score_thresh
                    [120.0, 80.0, 160.0, 120.0],  # Below score_thresh -> GT #1 is FN
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.92, 0.70, 0.30], dtype=torch.float32),
            "labels": torch.tensor([1, 1, 1], dtype=torch.int64),
        },
        (160, 120): {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    }

    monkeypatch.setattr(
        "src.predict_pack.build_model",
        lambda cfg, pretrained=False: _StubDetector(stub_outputs),
    )

    out_csv = tmp_path / "out" / "test_pack_predictions.csv"
    vis_dir = tmp_path / "out" / "visuals"
    rc = predict_pack_main(
        [
            "--run",
            "configs/res_480.yaml",
            str(ckpt_480),
            "--run",
            "configs/res_480_small_anchor.yaml",
            str(ckpt_sa),
            "--test-pack-dir",
            str(pack),
            "--out-csv",
            str(out_csv),
            "--visuals-dir",
            str(vis_dir),
            "--score-thresh",
            "0.50",
            "--iou-thresh",
            "0.50",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    assert out_csv.exists()
    assert (vis_dir / "00001_comparison.png").exists()
    assert (vis_dir / "00002_comparison.png").exists()

    with out_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == PREDICTION_CSV_COLUMNS
        csv_rows = list(reader)

    assert len(csv_rows) == 4  # 2 samples x 2 runs
    first = [r for r in csv_rows if r["sample_id"] == "00001" and r["run_name"] == "res_480"][0]
    assert first["checkpoint_name"] == "res_480_best.pt"
    assert int(first["orig_w"]) == 200 and int(first["orig_h"]) == 150
    assert int(first["gt_count"]) == 2
    assert int(first["pred_count"]) == 2
    assert int(first["tp"]) == 1 and int(first["fp"]) == 1 and int(first["fn"]) == 1
    assert len(json.loads(first["gt_boxes"])) == 2
    assert len(json.loads(first["pred_boxes"])) == 2
    assert json.loads(first["pred_scores"]) == pytest.approx([0.92, 0.70])

    empty_row = [r for r in csv_rows if r["sample_id"] == "00002" and r["run_name"] == "res_480"][0]
    assert int(empty_row["gt_count"]) == 0
    assert int(empty_row["pred_count"]) == 0
    assert int(empty_row["tp"]) == 0 and int(empty_row["fp"]) == 0 and int(empty_row["fn"]) == 0

    # Mismatched checkpoint run_name must raise ValueError
    with pytest.raises(ValueError, match="run_name"):
        run_test_pack_predictions(
            [(Path("configs/res_800.yaml"), ckpt_480)],
            pack,
            tmp_path / "bad.csv",
            tmp_path / "bad_vis",
            device="cpu",
        )


def test_predict_single_schema_filtering_and_cli_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (320, 240), color=(90, 110, 130)).save(img_path)

    ckpt = tmp_path / "res_800_best.pt"
    torch.save({"model_state_dict": {}, "run_name": "res_800"}, ckpt)

    stub_outputs = {
        (320, 240): {
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 40.0, 50.0],
                    [100.0, 120.0, 140.0, 160.0],
                    [200.0, 10.0, 240.0, 50.0],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.65, 0.95, 0.25], dtype=torch.float32),
            "labels": torch.tensor([1, 1, 1], dtype=torch.int64),
        }
    }
    monkeypatch.setattr(
        "src.predict_single.build_model",
        lambda cfg, pretrained=False: _StubDetector(stub_outputs),
    )

    out_png = tmp_path / "annotated.png"
    rc = predict_single_main(
        [
            "--config",
            "configs/res_800.yaml",
            "--checkpoint",
            str(ckpt),
            "--image",
            str(img_path),
            "--score-thresh",
            "0.50",
            "--out-image",
            str(out_png),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 1000

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["run_name"] == "res_800"
    assert payload["checkpoint"] == str(ckpt)
    assert payload["image"] == str(img_path)
    assert payload["orig_size"] == [320, 240]
    assert payload["score_thresh"] == pytest.approx(0.50)
    assert payload["num_detections"] == 2
    assert payload["scores"] == pytest.approx([0.95, 0.65])
    assert payload["boxes"][0] == pytest.approx([100.0, 120.0, 140.0, 160.0])
    assert payload["boxes"][1] == pytest.approx([10.0, 20.0, 40.0, 50.0])
    assert payload["inference_ms"] >= 0.0

    # Zero detections above threshold (and out_image=None) should also work cleanly
    zero_res = predict_single_image(
        Path("configs/res_800.yaml"),
        ckpt,
        img_path,
        score_thresh=0.99,
        out_image=None,
        device="cpu",
    )
    assert zero_res["num_detections"] == 0
    assert zero_res["boxes"] == []
    assert zero_res["scores"] == []

    # Checkpoint run_name mismatch must raise ValueError
    with pytest.raises(ValueError, match="run_name"):
        predict_single_image(
            Path("configs/res_480.yaml"),
            ckpt,
            img_path,
            score_thresh=0.50,
            out_image=None,
            device="cpu",
        )
