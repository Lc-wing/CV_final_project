"""Unit and integration tests for explicit pure-PyTorch Faster R-CNN trainer."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from src.config import load_config
from src.eval import load_checkpoint, main as eval_main
from src.train import (
    EPOCH_LOG_COLUMNS,
    STEP_LOG_COLUMNS,
    _atomic_save_checkpoint,
    _load_resume_checkpoint,
    build_lr_scheduler,
    main as train_main,
    train_run,
)


def _make_mini_dataset(root: Path, *, train_count: int = 4, val_count: int = 2, include_empty_train: bool = False) -> None:
    for split, n in [("train", train_count), ("val", val_count)]:
        (root / split / "img").mkdir(parents=True, exist_ok=True)
        (root / split / "xml").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (160, 120), color=(40 + i * 20, 80, 120)).save(
                root / split / "img" / f"{i:05d}.jpg"
            )
            if split == "train" and include_empty_train and i == n - 1:
                xml_body = "<annotation></annotation>"
            else:
                xml_body = (
                    "<annotation><object><name>UAV</name><bndbox>"
                    "<xmin>30</xmin><ymin>25</ymin><xmax>70</xmax><ymax>60</ymax>"
                    "</bndbox></object></annotation>"
                )
            (root / split / "xml" / f"{i:05d}.xml").write_text(xml_body, encoding="utf-8")


class _ToyDetector(nn.Module):
    """Lightweight deterministic detector stub for fast trainer unit tests."""

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        fail_value: float = float("nan"),
        eval_scores_by_epoch: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float32))
        self.frozen = nn.Parameter(torch.tensor([0.25], dtype=torch.float32), requires_grad=False)
        self.train_calls = 0
        self.eval_pass_idx = 0
        self.fail_on_call = fail_on_call
        self.fail_value = fail_value
        self.eval_scores_by_epoch = eval_scores_by_epoch or []
        self.seen_train_sample_ids: list[list[str]] = []

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> Any:
        if self.training:
            assert targets is not None
            self.train_calls += 1
            batch_ids = [
                str(int(t["image_id"].item())) if "image_id" in t else "?"
                for t in targets
            ]
            self.seen_train_sample_ids.append(batch_ids)

            if self.fail_on_call is not None and self.train_calls == self.fail_on_call:
                bad = self.weight.sum() * 0.0 + torch.tensor(self.fail_value, device=self.weight.device)
                return {
                    "loss_classifier": bad,
                    "loss_box_reg": self.weight.pow(2).sum() * 0.1,
                    "loss_objectness": self.weight.pow(2).sum() * 0.05,
                    "loss_rpn_box_reg": self.weight.pow(2).sum() * 0.05,
                }

            base = (self.weight ** 2).sum() + images[0].mean() * 0.0
            return {
                "loss_classifier": base * 0.40,
                "loss_box_reg": base * 0.30,
                "loss_objectness": base * 0.20,
                "loss_rpn_box_reg": base * 0.10,
            }

        # Eval mode: optionally emit matching or empty detections per validation pass
        outputs: list[dict[str, torch.Tensor]] = []
        dev = images[0].device
        mode = self.eval_scores_by_epoch[self.eval_pass_idx] if self.eval_pass_idx < len(self.eval_scores_by_epoch) else 0.0
        for _ in images:
            if mode > 0.0:
                outputs.append(
                    {
                        "boxes": torch.tensor([[30.0, 25.0, 70.0, 60.0]], dtype=torch.float32, device=dev),
                        "scores": torch.tensor([mode], dtype=torch.float32, device=dev),
                        "labels": torch.tensor([1], dtype=torch.int64, device=dev),
                    }
                )
            else:
                outputs.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=dev),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=dev),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=dev),
                    }
                )
        return outputs


def test_train_run_logs_csv_and_resumes_cleanly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir, train_count=4, val_count=2, include_empty_train=True)

    cfg = load_config("configs/res_480.yaml")
    cfg = replace(cfg, batch_size=2, num_workers=0, warmup_steps=1)

    # Epoch 1 run
    res1 = train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")
    epoch_csv = log_dir / "res_480_epoch_log.csv"
    step_csv = log_dir / "res_480_step_log.csv"
    last_ckpt = ckpt_dir / "res_480_last.pt"
    best_ckpt = ckpt_dir / "res_480_best.pt"
    assert epoch_csv.exists() and step_csv.exists()
    assert last_ckpt.exists() and best_ckpt.exists()
    assert res1["completed_epochs"] == 1
    assert res1["global_step"] == 2

    # Check exact CSV columns and weights_only=True checkpoint compatibility
    with step_csv.open(encoding="utf-8") as fh:
        step_reader = csv.DictReader(fh)
        assert tuple(step_reader.fieldnames or ()) == STEP_LOG_COLUMNS
        step_rows_1 = list(step_reader)
    assert len(step_rows_1) == 2
    assert [int(r["global_step"]) for r in step_rows_1] == [1, 2]

    with epoch_csv.open(encoding="utf-8") as fh:
        epoch_reader = csv.DictReader(fh)
        assert tuple(epoch_reader.fieldnames or ()) == EPOCH_LOG_COLUMNS
        epoch_rows_1 = list(epoch_reader)
    assert len(epoch_rows_1) == 1
    epoch1_row_snapshot = dict(epoch_rows_1[0])

    raw_last = torch.load(last_ckpt, map_location="cpu", weights_only=True)
    for required_key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "global_step",
        "best_val_ap50",
        "run_name",
        "config",
    ):
        assert required_key in raw_last
    assert raw_last["run_name"] == "res_480"
    assert raw_last["epoch"] == 1
    assert raw_last["global_step"] == 2
    assert isinstance(raw_last["config"], dict)
    assert isinstance(raw_last["config"]["anchor_sizes"], list)

    epoch1_cls_weight = raw_last["model_state_dict"]["roi_heads.box_predictor.cls_score.weight"].clone()

    # Confirm src/eval.py:load_checkpoint reads both checkpoints cleanly
    sd_best, meta_best = load_checkpoint(best_ckpt, device="cpu", expected_run_name="res_480")
    assert len(sd_best) > 0 and meta_best["run_name"] == "res_480"

    # Resume to Epoch 2 without overwriting Epoch 1 row
    res2 = train_run(
        cfg,
        data_dir,
        log_dir,
        ckpt_dir,
        pretrained=False,
        resume=True,
        epochs_override=2,
        device="cpu",
    )
    raw_last_2 = torch.load(last_ckpt, map_location="cpu", weights_only=True)
    epoch2_cls_weight = raw_last_2["model_state_dict"]["roi_heads.box_predictor.cls_score.weight"]
    assert not torch.equal(epoch1_cls_weight, epoch2_cls_weight)

    with epoch_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert rows[0] == epoch1_row_snapshot
    assert all(r["run_name"] == "res_480" for r in rows)
    assert res2["completed_epochs"] == 2
    assert res2["resumed_from_epoch"] == 1
    assert res2["global_step"] == 4

    with step_csv.open(encoding="utf-8") as fh:
        step_rows_2 = list(csv.DictReader(fh))
    assert [int(r["epoch"]) for r in step_rows_2] == [1, 1, 2, 2]
    assert [int(r["global_step"]) for r in step_rows_2] == [1, 2, 3, 4]


def test_non_finite_loss_raises_without_corrupting_weights_or_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir, train_count=4, val_count=2)

    cfg = replace(load_config("configs/res_480.yaml"), batch_size=2, num_workers=0, warmup_steps=1)

    # 1) Fresh run hitting NaN on step 1 must raise before backward/step and write NO checkpoints or NaN rows
    nan_model = _ToyDetector(fail_on_call=1, fail_value=float("nan"))
    initial_weight = nan_model.weight.detach().clone()
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: nan_model)

    with pytest.raises(RuntimeError, match="Non-finite"):
        train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")

    assert torch.equal(nan_model.weight.detach(), initial_weight)
    assert not (ckpt_dir / "res_480_last.pt").exists()
    assert not (ckpt_dir / "res_480_best.pt").exists()
    with (log_dir / "res_480_step_log.csv").open(encoding="utf-8") as fh:
        assert list(csv.DictReader(fh)) == []

    # 2) Valid Epoch 1 followed by +inf spike on Epoch 2 must leave Epoch 1 checkpoints untouched
    good_model = _ToyDetector()
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: good_model)
    train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")

    last_ckpt = ckpt_dir / "res_480_last.pt"
    best_ckpt = ckpt_dir / "res_480_best.pt"
    last_bytes_before = last_ckpt.read_bytes()
    best_bytes_before = best_ckpt.read_bytes()

    inf_model = _ToyDetector(fail_on_call=1, fail_value=float("inf"))
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: inf_model)
    with pytest.raises(RuntimeError, match="Non-finite"):
        train_run(
            cfg,
            data_dir,
            log_dir,
            ckpt_dir,
            pretrained=False,
            resume=True,
            epochs_override=2,
            device="cpu",
        )

    assert last_ckpt.read_bytes() == last_bytes_before
    assert best_ckpt.read_bytes() == best_bytes_before
    assert torch.isfinite(inf_model.weight).all()
    with (log_dir / "res_480_epoch_log.csv").open(encoding="utf-8") as fh:
        assert [int(r["epoch"]) for r in csv.DictReader(fh)] == [1]


def test_resume_truncates_interrupted_rows_validates_run_name_and_handles_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir, train_count=4, val_count=2)

    cfg = replace(load_config("configs/res_480.yaml"), batch_size=2, num_workers=0, warmup_steps=1)
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: _ToyDetector())

    # Resume without existing checkpoint fails loudly
    with pytest.raises(FileNotFoundError, match="res_480_last.pt"):
        train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=2, device="cpu")

    # Complete Epoch 1
    train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")
    step_csv = log_dir / "res_480_step_log.csv"
    epoch_csv = log_dir / "res_480_epoch_log.csv"

    # Simulate interrupted Epoch 2 partial step row and stray epoch row
    with step_csv.open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(["res_480", 2, 1, 3, 0.0001, 999.0, 250.0, 250.0, 250.0, 249.0])
    with epoch_csv.open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(["res_480", 2, 999.0, 250.0, 250.0, 250.0, 249.0, 0.0, 0.0, 0.0, 1.0, 0.0001])

    # Resume to Epoch 2: partial > 1 rows must be truncated before appending fresh Epoch 2
    res2 = train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=2, device="cpu")
    assert res2["completed_epochs"] == 2

    with step_csv.open(encoding="utf-8") as fh:
        step_rows = list(csv.DictReader(fh))
    assert [int(r["epoch"]) for r in step_rows] == [1, 1, 2, 2]
    assert all(float(r["total_loss"]) < 100.0 for r in step_rows)

    with epoch_csv.open(encoding="utf-8") as fh:
        epoch_rows = list(csv.DictReader(fh))
    assert [int(r["epoch"]) for r in epoch_rows] == [1, 2]
    assert all(float(r["train_total_loss"]) < 100.0 for r in epoch_rows)

    # Resuming again at epochs_override=2 is a clean no-op
    res_noop = train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=2, device="cpu")
    assert res_noop["completed_epochs"] == 2
    assert res_noop["resumed_from_epoch"] == 2
    with epoch_csv.open(encoding="utf-8") as fh:
        assert [int(r["epoch"]) for r in csv.DictReader(fh)] == [1, 2]

    # CSV log run_name drift on resume must fail loudly
    epoch_csv_backup = epoch_csv.read_text(encoding="utf-8")
    epoch_csv.write_text(epoch_csv_backup.replace("res_480", "wrong_run"), encoding="utf-8")
    with pytest.raises(ValueError, match="run_name"):
        train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=3, device="cpu")
    epoch_csv.write_text(epoch_csv_backup, encoding="utf-8")

    # Checkpoint run_name mismatch on resume must fail loudly
    last_ckpt = ckpt_dir / "res_480_last.pt"
    tampered = torch.load(last_ckpt, map_location="cpu", weights_only=True)
    tampered["run_name"] = "res_800"
    torch.save(tampered, last_ckpt)
    with pytest.raises(ValueError, match="run_name"):
        train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=3, device="cpu")


def test_best_checkpoint_selection_and_deterministic_resume_shuffle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _make_mini_dataset(data_dir, train_count=6, val_count=2)
    cfg = replace(load_config("configs/res_480.yaml"), batch_size=2, num_workers=0, warmup_steps=2)

    # 1) Best checkpoint tracks highest val_ap50 (Epoch 1: 0.0 -> saved; Epoch 2: 1.0 -> updated; Epoch 3: 0.0 -> preserved)
    eval_schedule = [0.0, 0.9, 0.0]
    toy = _ToyDetector(eval_scores_by_epoch=eval_schedule)

    # Track validation calls by epoch via run_inference_on_dataset wrapper
    import src.train as train_mod

    real_infer = train_mod.run_inference_on_dataset

    def _counting_infer(model: nn.Module, loader: Any, device: Any) -> Any:
        res = real_infer(model, loader, device=device)
        if isinstance(model, _ToyDetector):
            model.eval_pass_idx += 1
        return res

    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: toy)
    monkeypatch.setattr("src.train.run_inference_on_dataset", _counting_infer)

    res = train_run(
        cfg,
        data_dir,
        tmp_path / "logs_best",
        tmp_path / "ckpts_best",
        pretrained=False,
        epochs_override=3,
        device="cpu",
    )
    assert res["best_val_ap50"] == pytest.approx(1.0)
    assert res["best_epoch"] == 2
    best_loaded = torch.load(tmp_path / "ckpts_best" / "res_480_best.pt", map_location="cpu", weights_only=True)
    last_loaded = torch.load(tmp_path / "ckpts_best" / "res_480_last.pt", map_location="cpu", weights_only=True)
    assert best_loaded["epoch"] == 2 and best_loaded["best_val_ap50"] == pytest.approx(1.0)
    assert last_loaded["epoch"] == 3 and last_loaded["best_val_ap50"] == pytest.approx(1.0)

    # 2) Deterministic shuffling: continuous 2 epochs matches split 1-epoch + resume-epoch-2 order
    monkeypatch.setattr("src.train.run_inference_on_dataset", real_infer)
    toy_cont = _ToyDetector()
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: toy_cont)
    train_run(cfg, data_dir, tmp_path / "logs_c", tmp_path / "ckpts_c", pretrained=False, epochs_override=2, device="cpu")

    toy_split1 = _ToyDetector()
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: toy_split1)
    train_run(cfg, data_dir, tmp_path / "logs_s", tmp_path / "ckpts_s", pretrained=False, epochs_override=1, device="cpu")

    toy_split2 = _ToyDetector()
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: toy_split2)
    train_run(
        cfg,
        data_dir,
        tmp_path / "logs_s",
        tmp_path / "ckpts_s",
        pretrained=False,
        resume=True,
        epochs_override=2,
        device="cpu",
    )

    assert toy_split1.seen_train_sample_ids + toy_split2.seen_train_sample_ids == toy_cont.seen_train_sample_ids


def test_lr_schedule_warmup_and_cosine_decay() -> None:
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = build_lr_scheduler(opt, warmup_steps=4, total_steps=8, min_lr_ratio=0.01)

    lrs: list[float] = []
    for _ in range(8):
        lrs.append(float(opt.param_groups[0]["lr"]))
        opt.zero_grad(set_to_none=True)
        (p * 0.0).sum().backward()
        opt.step()
        sched.step()

    # Warmup linearly ramps 0.25 -> 0.50 -> 0.75 -> 1.00 of peak lr
    assert lrs[:4] == pytest.approx([2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3])
    # Cosine phase decays strictly down to min_lr_ratio * peak
    assert lrs[3] > lrs[4] > lrs[5] > lrs[6] > lrs[7]
    assert lrs[7] == pytest.approx(1e-5)

    # Edge case: warmup_steps=0 and total_steps=1
    opt2 = torch.optim.AdamW([p], lr=2e-4)
    _ = build_lr_scheduler(opt2, warmup_steps=0, total_steps=1, min_lr_ratio=0.01)
    assert opt2.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_cli_smoke_test_and_eval_interoperability(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    # Provide more than smoke-test caps (10 train, 6 val) to verify --smoke-test caps to 8 / 4
    _make_mini_dataset(data_dir, train_count=10, val_count=6)

    rc = train_main(
        [
            "--config",
            "configs/res_480.yaml",
            "--data-dir",
            str(data_dir),
            "--log-dir",
            str(log_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--smoke-test",
            "--no-pretrained",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )
    assert rc == 0

    step_csv = log_dir / "res_480_step_log.csv"
    epoch_csv = log_dir / "res_480_epoch_log.csv"
    best_ckpt = ckpt_dir / "res_480_best.pt"
    assert step_csv.exists() and epoch_csv.exists() and best_ckpt.exists()

    # 8 capped train samples / batch_size 4 == exactly 2 steps in 1 epoch
    with step_csv.open(encoding="utf-8") as fh:
        step_rows = list(csv.DictReader(fh))
    assert len(step_rows) == 2

    # Downstream CLI compatibility: src.eval evaluates the produced checkpoint cleanly
    eval_out = tmp_path / "eval_val.json"
    eval_rc = eval_main(
        [
            "--config",
            "configs/res_480.yaml",
            "--checkpoint",
            str(best_ckpt),
            "--data-dir",
            str(data_dir),
            "--split",
            "val",
            "--out",
            str(eval_out),
            "--device",
            "cpu",
            "--max-samples",
            "2",
            "--num-workers",
            "0",
            "--latency-warmup",
            "0",
            "--latency-images",
            "1",
        ]
    )
    assert eval_rc == 0 and eval_out.exists()


def test_load_resume_checkpoint_always_uses_cpu_map_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir, train_count=4, val_count=2)

    cfg = replace(load_config("configs/res_480.yaml"), batch_size=2, num_workers=0, warmup_steps=1)
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: _ToyDetector())

    # Produce a real checkpoint containing AdamW optimizer state (including step tensors)
    train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")
    last_ckpt = ckpt_dir / "res_480_last.pt"

    real_torch_load = torch.load
    recorded_calls: list[tuple[Any, Any]] = []

    def _spy_torch_load(*args: Any, **kwargs: Any) -> Any:
        recorded_calls.append((kwargs.get("map_location"), kwargs.get("weights_only")))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr("src.train.torch.load", _spy_torch_load)

    # 1) Direct call loads via literal map_location="cpu" with weights_only=True
    loaded = _load_resume_checkpoint(last_ckpt, expected_run_name="res_480")
    assert recorded_calls == [("cpu", True)]
    opt_state = loaded["optimizer_state_dict"]["state"]
    assert opt_state and all(
        s["step"].device.type == "cpu" for s in opt_state.values() if "step" in s
    )

    # 2) Full train_run(resume=True) call path also passes literal map_location="cpu"
    recorded_calls.clear()
    res2 = train_run(
        cfg,
        data_dir,
        log_dir,
        ckpt_dir,
        pretrained=False,
        resume=True,
        epochs_override=2,
        device="cpu",
    )
    assert res2["completed_epochs"] == 2
    assert recorded_calls == [("cpu", True)]

    # 3) Legacy checkpoint triggering pickle.UnpicklingError fallback also uses map_location="cpu"
    legacy_ckpt = tmp_path / "legacy_last.pt"
    legacy_payload = dict(loaded)
    legacy_payload["config"] = cfg  # dataclass instance triggers weights_only=True fallback
    torch.save(legacy_payload, legacy_ckpt)

    recorded_calls.clear()
    loaded_legacy = _load_resume_checkpoint(legacy_ckpt, expected_run_name="res_480")
    assert loaded_legacy["run_name"] == "res_480"
    assert recorded_calls == [("cpu", True), ("cpu", False)]


def test_epoch_log_flushed_before_last_checkpoint_commit_and_crash_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir, train_count=4, val_count=2)

    cfg = replace(load_config("configs/res_480.yaml"), batch_size=2, num_workers=0, warmup_steps=1)
    monkeypatch.setattr("src.train.build_model", lambda _cfg, pretrained=True: _ToyDetector())

    # Complete Epoch 1 cleanly
    train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1, device="cpu")
    epoch_csv = log_dir / "res_480_epoch_log.csv"
    step_csv = log_dir / "res_480_step_log.csv"
    last_ckpt = ckpt_dir / "res_480_last.pt"

    # During Epoch 2, intercept _atomic_save_checkpoint for _last.pt:
    # verify epoch 2 is ALREADY flushed on disk in epoch_log.csv, then simulate a disconnect crash.
    flushed_epochs_at_commit: list[list[int]] = []

    def _crashing_save(payload: dict[str, Any], path: Path) -> None:
        with epoch_csv.open(encoding="utf-8") as fh:
            flushed_epochs = [int(r["epoch"]) for r in csv.DictReader(fh)]
        flushed_epochs_at_commit.append(flushed_epochs)
        if path == last_ckpt:
            raise OSError("simulated disconnect during _last.pt save")
        _atomic_save_checkpoint(payload, path)

    monkeypatch.setattr("src.train._atomic_save_checkpoint", _crashing_save)

    with pytest.raises(OSError, match="simulated disconnect"):
        train_run(
            cfg,
            data_dir,
            log_dir,
            ckpt_dir,
            pretrained=False,
            resume=True,
            epochs_override=2,
            device="cpu",
        )

    # Epoch 2 row was already flushed to epoch_log.csv BEFORE _last.pt save was attempted
    assert flushed_epochs_at_commit
    assert all(epochs == [1, 2] for epochs in flushed_epochs_at_commit)
    # _last.pt remained committed at Epoch 1
    assert torch.load(last_ckpt, map_location="cpu", weights_only=True)["epoch"] == 1

    # Restore real save and resume: uncommitted Epoch 2 CSV rows must roll back cleanly
    # and Epoch 2 must re-run with zero missing-epoch gaps or duplicate rows.
    monkeypatch.setattr("src.train._atomic_save_checkpoint", _atomic_save_checkpoint)
    res_recovered = train_run(
        cfg,
        data_dir,
        log_dir,
        ckpt_dir,
        pretrained=False,
        resume=True,
        epochs_override=2,
        device="cpu",
    )
    assert res_recovered["completed_epochs"] == 2
    assert res_recovered["resumed_from_epoch"] == 1
    assert torch.load(last_ckpt, map_location="cpu", weights_only=True)["epoch"] == 2

    with epoch_csv.open(encoding="utf-8") as fh:
        assert [int(r["epoch"]) for r in csv.DictReader(fh)] == [1, 2]
    with step_csv.open(encoding="utf-8") as fh:
        step_rows = list(csv.DictReader(fh))
    assert [int(r["epoch"]) for r in step_rows] == [1, 1, 2, 2]
    assert [int(r["global_step"]) for r in step_rows] == [1, 2, 3, 4]

