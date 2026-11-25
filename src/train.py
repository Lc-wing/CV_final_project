"""Explicit pure-PyTorch Faster R-CNN training loop, checkpoint resume, and raw CSV logging."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, load_config
from src.dataset import DUTAntiUAVDataset, detection_collate_fn, seed_everything
from src.eval import (
    compute_ap_at_iou,
    compute_fixed_threshold_counts,
    resolve_device,
    run_inference_on_dataset,
)
from src.model import build_model, summarize_trainable_params

STEP_LOG_COLUMNS: tuple[str, ...] = (
    "run_name",
    "epoch",
    "step",
    "global_step",
    "lr",
    "total_loss",
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)

EPOCH_LOG_COLUMNS: tuple[str, ...] = (
    "run_name",
    "epoch",
    "train_total_loss",
    "train_loss_classifier",
    "train_loss_box_reg",
    "train_loss_objectness",
    "train_loss_rpn_box_reg",
    "val_ap50",
    "val_precision",
    "val_recall",
    "epoch_seconds",
    "lr",
)

LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)

SMOKE_TEST_EPOCHS = 1
SMOKE_TEST_MAX_TRAIN_SAMPLES = 8
SMOKE_TEST_MAX_VAL_SAMPLES = 4


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _serialize_config(cfg: ExperimentConfig) -> dict[str, Any]:
    """Convert ``ExperimentConfig`` to a plain JSON-safe dict of primitives and lists.

    Using plain dicts/lists (rather than raw dataclass instances) ensures saved
    checkpoints load cleanly with ``torch.load(..., weights_only=True)`` across
    PyTorch 2.2+ with zero security or deprecation warnings.
    """
    raw = dataclasses.asdict(cfg)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in raw.items()}


def _seed_dataloader_worker(_worker_id: int) -> None:
    """Deterministically seed each DataLoader worker from PyTorch's per-worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _move_targets_to_device(
    targets: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Extract Faster R-CNN tensor target fields and move them onto ``device``."""
    device_targets: list[dict[str, torch.Tensor]] = []
    for target in targets:
        item: dict[str, torch.Tensor] = {
            "boxes": target["boxes"].to(device),
            "labels": target["labels"].to(device),
        }
        if "image_id" in target and isinstance(target["image_id"], torch.Tensor):
            item["image_id"] = target["image_id"].to(device)
        device_targets.append(item)
    return device_targets


def _atomic_save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint via temporary sibling + atomic rename to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _load_resume_checkpoint(
    ckpt_path: Path,
    expected_run_name: str,
) -> dict[str, Any]:
    """Load and validate a ``<run_name>_last.pt`` checkpoint onto CPU for resuming training.

    Always loads with ``map_location="cpu"`` so ``AdamW`` ``step`` tensors stay
    on CPU (preventing per-step host-device syncs and device-mismatch errors on
    CUDA) and avoiding a duplicate copy of model/optimizer state in VRAM.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Resume requested but checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Resume checkpoint '{ckpt_path}' must be a dict, got {type(ckpt).__name__}"
        )

    required_keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "global_step",
        "best_val_ap50",
        "run_name",
    )
    missing = [key for key in required_keys if key not in ckpt]
    if missing:
        raise ValueError(f"Resume checkpoint '{ckpt_path}' missing required keys: {missing}")

    ckpt_run_name = ckpt["run_name"]
    if ckpt_run_name != expected_run_name:
        raise ValueError(
            f"Checkpoint run_name ({ckpt_run_name!r}) does not match "
            f"config run_name ({expected_run_name!r})"
        )

    epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    best_val_ap50 = ckpt["best_val_ap50"]
    if not _is_strict_int(epoch) or epoch < 0:
        raise ValueError(f"Checkpoint 'epoch' must be a non-negative int, got {epoch!r}")
    if not _is_strict_int(global_step) or global_step < 0:
        raise ValueError(
            f"Checkpoint 'global_step' must be a non-negative int, got {global_step!r}"
        )
    if not _is_strict_number(best_val_ap50) or not math.isfinite(float(best_val_ap50)):
        raise ValueError(
            f"Checkpoint 'best_val_ap50' must be a finite number, got {best_val_ap50!r}"
        )

    return ckpt


def _prepare_csv_log(
    csv_path: Path,
    columns: tuple[str, ...],
    expected_run_name: str,
    *,
    resume: bool,
    max_epoch: int,
) -> None:
    """Initialize a fresh raw CSV log or preserve rows ``<= max_epoch`` on resume.

    When resuming after an interrupted epoch, any partially written rows with
    ``epoch > max_epoch`` are truncated atomically before new rows are appended.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    kept_rows: list[dict[str, str]] = []

    if resume and csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is not None:
                if tuple(reader.fieldnames) != columns:
                    raise ValueError(
                        f"Existing CSV log '{csv_path}' has header {reader.fieldnames}, "
                        f"expected {list(columns)}"
                    )
                for row in reader:
                    row_run_name = row.get("run_name")
                    if row_run_name != expected_run_name:
                        raise ValueError(
                            f"Existing CSV log '{csv_path}' contains run_name "
                            f"{row_run_name!r}, expected {expected_run_name!r}"
                        )
                    raw_epoch = row.get("epoch")
                    if raw_epoch is None:
                        raise ValueError(f"Malformed CSV row missing 'epoch' in '{csv_path}': {row!r}")
                    try:
                        row_epoch = int(raw_epoch)
                    except ValueError as exc:
                        raise ValueError(
                            f"Malformed CSV row with non-integer 'epoch' in '{csv_path}': {row!r}"
                        ) from exc
                    if row_epoch <= max_epoch:
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError(
                                f"Malformed CSV row with mismatched columns in '{csv_path}': {row!r}"
                            )
                        kept_rows.append(row)

    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(columns))
            writer.writeheader()
            if kept_rows:
                writer.writerows(kept_rows)
            fh.flush()
        os.replace(tmp_path, csv_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a per-step linear-warmup followed by cosine-decay LR scheduler."""
    if not _is_strict_int(warmup_steps) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative int, got {warmup_steps!r}")
    if not _is_strict_int(total_steps) or total_steps <= 0:
        raise ValueError(f"total_steps must be a positive int, got {total_steps!r}")
    if (
        not _is_strict_number(min_lr_ratio)
        or not math.isfinite(float(min_lr_ratio))
        or not (0.0 <= float(min_lr_ratio) <= 1.0)
    ):
        raise ValueError(f"min_lr_ratio must be in [0.0, 1.0], got {min_lr_ratio!r}")

    min_ratio = float(min_lr_ratio)

    def lr_lambda(step_idx: int) -> float:
        idx = max(0, int(step_idx))
        if warmup_steps > 0 and idx < warmup_steps:
            return float(idx + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        if warmup_steps == 0:
            if total_steps <= 1:
                return 1.0
            progress = min(1.0, float(idx) / float(total_steps - 1))
        else:
            decay_steps = total_steps - warmup_steps
            progress = min(1.0, float(idx - warmup_steps + 1) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_ratio + (1.0 - min_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    trainable_params: list[nn.Parameter],
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    global_step: int,
    step_writer: csv.DictWriter[str],
    step_fh: Any,
) -> tuple[dict[str, float], int, float]:
    """Run one explicit pure-PyTorch training epoch with AMP, grad clipping, and step CSV logging.

    Every weight update executes explicitly in linear order:
    ``zero_grad -> autocast forward -> finite-loss guard -> backward -> unscale -> clip -> step -> update``.
    """
    model.train()

    total_loss_sum = 0.0
    component_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
    num_steps = 0
    last_lr = float(optimizer.param_groups[0]["lr"])

    for step_in_epoch, (images, targets) in enumerate(train_loader, start=1):
        device_images = [img.to(device, non_blocking=(device.type == "cuda")) for img in images]
        device_targets = _move_targets_to_device(targets, device)

        # 1. Clear gradients before forward pass
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass under optional CUDA Automatic Mixed Precision (AMP)
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            loss_dict = model(device_images, device_targets)
            if not isinstance(loss_dict, dict) or not loss_dict:
                raise RuntimeError(
                    f"Model must return a non-empty loss dict in train mode "
                    f"(epoch={epoch}, step={step_in_epoch}), got {type(loss_dict).__name__}"
                )
            loss = sum(loss_dict.values())

        # 3. Finite-loss guard BEFORE backward() and optimizer.step() so NaN/Inf
        #    spikes can never corrupt model weights or saved checkpoints.
        if not torch.isfinite(loss).all() or any(
            not torch.isfinite(v).all() for v in loss_dict.values()
        ):
            components_desc = ", ".join(
                f"{k}={float(v.detach().item())}" for k, v in loss_dict.items()
            )
            raise RuntimeError(
                f"Non-finite training loss detected at epoch {epoch}, step {step_in_epoch} "
                f"(global_step={global_step + 1}): total_loss={float(loss.detach().item())} "
                f"({components_desc}). Aborting before backward/step to protect weights."
            )

        current_lr = float(optimizer.param_groups[0]["lr"])

        # 4. Backpropagate, unscale AMP gradients, clip gradient norm, and update weights
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # 5. Step LR scheduler only if CUDA AMP did not skip optimizer.step() on grad overflow
        amp_step_skipped = use_amp and scaler.get_scale() < scale_before
        if not amp_step_skipped:
            scheduler.step()

        global_step += 1
        num_steps += 1
        last_lr = current_lr

        total_loss_val = float(loss.detach().item())
        total_loss_sum += total_loss_val
        comp_vals: dict[str, float] = {}
        for key in LOSS_COMPONENT_KEYS:
            val = float(loss_dict[key].detach().item()) if key in loss_dict else 0.0
            comp_vals[key] = val
            component_sums[key] += val

        # 6. Immediately persist and flush raw step-level CSV log row
        step_writer.writerow(
            {
                "run_name": cfg.run_name,
                "epoch": epoch,
                "step": step_in_epoch,
                "global_step": global_step,
                "lr": f"{current_lr:.8g}",
                "total_loss": f"{total_loss_val:.6f}",
                "loss_classifier": f"{comp_vals['loss_classifier']:.6f}",
                "loss_box_reg": f"{comp_vals['loss_box_reg']:.6f}",
                "loss_objectness": f"{comp_vals['loss_objectness']:.6f}",
                "loss_rpn_box_reg": f"{comp_vals['loss_rpn_box_reg']:.6f}",
            }
        )
        step_fh.flush()

    if num_steps == 0:
        raise RuntimeError(f"Training loader yielded 0 batches at epoch {epoch}")

    epoch_averages = {
        "train_total_loss": total_loss_sum / num_steps,
        "train_loss_classifier": component_sums["loss_classifier"] / num_steps,
        "train_loss_box_reg": component_sums["loss_box_reg"] / num_steps,
        "train_loss_objectness": component_sums["loss_objectness"] / num_steps,
        "train_loss_rpn_box_reg": component_sums["loss_rpn_box_reg"] / num_steps,
    }
    return epoch_averages, global_step, last_lr


def _validate_train_hyperparams(cfg: ExperimentConfig, total_epochs: int) -> None:
    if not _is_strict_int(total_epochs) or total_epochs <= 0:
        raise ValueError(f"total_epochs must be a positive int, got {total_epochs!r}")
    if not _is_strict_int(cfg.batch_size) or cfg.batch_size <= 0:
        raise ValueError(f"cfg.batch_size must be a positive int, got {cfg.batch_size!r}")
    if not _is_strict_number(cfg.lr) or not math.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError(f"cfg.lr must be a positive finite number, got {cfg.lr!r}")
    if not _is_strict_number(cfg.weight_decay) or not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0.0:
        raise ValueError(
            f"cfg.weight_decay must be a non-negative finite number, got {cfg.weight_decay!r}"
        )
    if not _is_strict_int(cfg.warmup_steps) or cfg.warmup_steps < 0:
        raise ValueError(f"cfg.warmup_steps must be a non-negative int, got {cfg.warmup_steps!r}")
    if (
        not _is_strict_number(cfg.grad_clip_norm)
        or not math.isfinite(cfg.grad_clip_norm)
        or cfg.grad_clip_norm <= 0.0
    ):
        raise ValueError(
            f"cfg.grad_clip_norm must be a positive finite number, got {cfg.grad_clip_norm!r}"
        )
    if not _is_strict_int(cfg.num_workers) or cfg.num_workers < 0:
        raise ValueError(f"cfg.num_workers must be a non-negative int, got {cfg.num_workers!r}")


def _build_grad_scaler(use_amp: bool) -> Any:
    """Construct an AMP GradScaler compatible with PyTorch 2.2+ without deprecation warnings."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_run(
    cfg: ExperimentConfig,
    data_dir: Path | str,
    log_dir: Path | str,
    checkpoint_dir: Path | str,
    *,
    pretrained: bool = True,
    resume: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    epochs_override: int | None = None,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Train a Faster R-CNN MobileNetV3-Large-FPN run with raw CSV logging and resume support."""
    total_epochs = epochs_override if epochs_override is not None else cfg.epochs
    _validate_train_hyperparams(cfg, total_epochs)

    data_root = Path(data_dir)
    logs_root = Path(log_dir)
    ckpts_root = Path(checkpoint_dir)
    logs_root.mkdir(parents=True, exist_ok=True)
    ckpts_root.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    resolved_device = device if isinstance(device, torch.device) else resolve_device(str(device))
    use_amp = resolved_device.type == "cuda" and torch.cuda.is_available()

    train_dataset = DUTAntiUAVDataset(
        data_root / "train",
        hflip_prob=cfg.hflip_prob,
        max_samples=max_train_samples,
    )
    val_dataset = DUTAntiUAVDataset(
        data_root / "val",
        hflip_prob=0.0,
        max_samples=max_val_samples,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty: {data_root / 'train'}")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {data_root / 'val'}")

    train_generator = torch.Generator()
    worker_init = _seed_dataloader_worker if cfg.num_workers > 0 else None
    pin_memory = resolved_device.type == "cuda"

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate_fn,
        generator=train_generator,
        worker_init_fn=worker_init,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init,
        pin_memory=pin_memory,
    )

    model = build_model(cfg, pretrained=pretrained)
    model.to(resolved_device)
    param_summary = summarize_trainable_params(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, total_epochs * steps_per_epoch)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
    )
    scaler = _build_grad_scaler(use_amp)

    step_log_path = logs_root / f"{cfg.run_name}_step_log.csv"
    epoch_log_path = logs_root / f"{cfg.run_name}_epoch_log.csv"
    last_ckpt_path = ckpts_root / f"{cfg.run_name}_last.pt"
    best_ckpt_path = ckpts_root / f"{cfg.run_name}_best.pt"

    start_epoch = 0
    global_step = 0
    best_val_ap50 = -1.0
    best_epoch = 0

    if resume:
        ckpt = _load_resume_checkpoint(
            last_ckpt_path,
            expected_run_name=cfg.run_name,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and isinstance(ckpt.get("scaler_state_dict"), dict) and ckpt["scaler_state_dict"]:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])
        best_val_ap50 = float(ckpt["best_val_ap50"])
        best_epoch = int(ckpt.get("best_epoch", start_epoch))

    _prepare_csv_log(
        step_log_path,
        STEP_LOG_COLUMNS,
        cfg.run_name,
        resume=resume,
        max_epoch=start_epoch,
    )
    _prepare_csv_log(
        epoch_log_path,
        EPOCH_LOG_COLUMNS,
        cfg.run_name,
        resume=resume,
        max_epoch=start_epoch,
    )

    serialized_cfg = _serialize_config(cfg)
    completed_epochs = start_epoch
    last_epoch_metrics: dict[str, Any] | None = None

    with (
        step_log_path.open("a", encoding="utf-8", newline="") as step_fh,
        epoch_log_path.open("a", encoding="utf-8", newline="") as epoch_fh,
    ):
        step_writer = csv.DictWriter(step_fh, fieldnames=list(STEP_LOG_COLUMNS))
        epoch_writer = csv.DictWriter(epoch_fh, fieldnames=list(EPOCH_LOG_COLUMNS))

        for epoch in range(start_epoch + 1, total_epochs + 1):
            epoch_start = time.perf_counter()
            # Per-epoch deterministic seed guarantees identical shuffle order
            # across continuous execution and interrupted+resumed runs.
            train_generator.manual_seed(cfg.seed + epoch)

            train_avgs, global_step, epoch_lr = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                trainable_params,
                cfg=cfg,
                device=resolved_device,
                use_amp=use_amp,
                epoch=epoch,
                global_step=global_step,
                step_writer=step_writer,
                step_fh=step_fh,
            )

            # End-of-epoch validation in original pixel coordinates
            val_predictions, val_ground_truths = run_inference_on_dataset(
                model,
                val_loader,
                device=resolved_device,
            )
            val_ap50 = float(
                compute_ap_at_iou(val_predictions, val_ground_truths, iou_thresh=0.50)
            )
            val_fixed = compute_fixed_threshold_counts(
                val_predictions,
                val_ground_truths,
                score_thresh=cfg.score_thresh,
                iou_thresh=cfg.iou_thresh,
            )
            val_precision = float(val_fixed["precision"])
            val_recall = float(val_fixed["recall"])
            epoch_seconds = float(time.perf_counter() - epoch_start)

            improved = val_ap50 > best_val_ap50
            if improved:
                best_val_ap50 = val_ap50
                best_epoch = epoch

            # Flush the epoch CSV row BEFORE committing checkpoints so <run_name>_last.pt
            # acts as the single atomic commit barrier for epoch completion: any crash
            # before _last.pt is replaced will be rolled back cleanly by _prepare_csv_log.
            epoch_row = {
                "run_name": cfg.run_name,
                "epoch": epoch,
                "train_total_loss": f"{train_avgs['train_total_loss']:.6f}",
                "train_loss_classifier": f"{train_avgs['train_loss_classifier']:.6f}",
                "train_loss_box_reg": f"{train_avgs['train_loss_box_reg']:.6f}",
                "train_loss_objectness": f"{train_avgs['train_loss_objectness']:.6f}",
                "train_loss_rpn_box_reg": f"{train_avgs['train_loss_rpn_box_reg']:.6f}",
                "val_ap50": f"{val_ap50:.6f}",
                "val_precision": f"{val_precision:.6f}",
                "val_recall": f"{val_recall:.6f}",
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "lr": f"{epoch_lr:.8g}",
            }
            epoch_writer.writerow(epoch_row)
            epoch_fh.flush()

            checkpoint_payload: dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_ap50": float(best_val_ap50),
                "best_epoch": best_epoch,
                "run_name": cfg.run_name,
                "config": serialized_cfg,
            }
            if improved:
                _atomic_save_checkpoint(checkpoint_payload, best_ckpt_path)
            _atomic_save_checkpoint(checkpoint_payload, last_ckpt_path)

            completed_epochs = epoch
            last_epoch_metrics = {
                "epoch": epoch,
                **train_avgs,
                "val_ap50": val_ap50,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "epoch_seconds": epoch_seconds,
                "lr": epoch_lr,
                "is_best": improved,
            }

            best_tag = " [BEST]" if improved else ""
            print(
                f"[{cfg.run_name}] Epoch {epoch:02d}/{total_epochs:02d} | "
                f"loss={train_avgs['train_total_loss']:.4f} "
                f"(cls={train_avgs['train_loss_classifier']:.4f}, "
                f"box={train_avgs['train_loss_box_reg']:.4f}, "
                f"obj={train_avgs['train_loss_objectness']:.4f}, "
                f"rpn={train_avgs['train_loss_rpn_box_reg']:.4f}) | "
                f"val_AP50={val_ap50:.4f} P={val_precision:.4f} R={val_recall:.4f} | "
                f"lr={epoch_lr:.6g} | {epoch_seconds:.1f}s{best_tag}"
            )

    return {
        "run_name": cfg.run_name,
        "device": str(resolved_device),
        "use_amp": use_amp,
        "completed_epochs": completed_epochs,
        "resumed_from_epoch": start_epoch,
        "global_step": global_step,
        "best_val_ap50": float(best_val_ap50),
        "best_epoch": best_epoch,
        "param_summary": param_summary,
        "step_log_path": str(step_log_path),
        "epoch_log_path": str(epoch_log_path),
        "last_checkpoint_path": str(last_ckpt_path),
        "best_checkpoint_path": str(best_ckpt_path),
        "last_epoch_metrics": last_epoch_metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.train``."""
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN MobileNetV3-Large-FPN on DUT Anti-UAV with explicit PyTorch loop."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root directory containing train/ and val/ splits.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/logs"),
        help="Directory for raw step/epoch CSV logs (default: results/logs).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for <run_name>_last.pt and <run_name>_best.pt checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from <checkpoint-dir>/<run_name>_last.pt while preserving prior CSV rows.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 1-epoch verification pass capped to 8 train and 4 val samples.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize model without downloading COCO pretrained weights (useful for offline smoke tests).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device ('auto', 'cpu', 'cuda', ...).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch count override (defaults to config epochs, or 1 with --smoke-test).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training dataset samples (default: 8 with --smoke-test).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on validation dataset samples (default: 4 with --smoke-test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override (defaults to config batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker count override (defaults to config num_workers).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for explicit pure-PyTorch Anti-UAV Faster R-CNN training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    epochs_override = args.epochs
    max_train_samples = args.max_train_samples
    max_val_samples = args.max_val_samples
    if args.smoke_test:
        if epochs_override is None:
            epochs_override = SMOKE_TEST_EPOCHS
        if max_train_samples is None:
            max_train_samples = SMOKE_TEST_MAX_TRAIN_SAMPLES
        if max_val_samples is None:
            max_val_samples = SMOKE_TEST_MAX_VAL_SAMPLES

    summary = train_run(
        cfg=cfg,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        pretrained=not args.no_pretrained,
        resume=args.resume,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        epochs_override=epochs_override,
        device=args.device,
    )
    params = summary["param_summary"]
    print(
        f"Training complete | run={summary['run_name']} device={summary['device']} "
        f"amp={summary['use_amp']} | epochs={summary['completed_epochs']} "
        f"global_step={summary['global_step']} | "
        f"best_val_AP50={summary['best_val_ap50']:.4f} (epoch {summary['best_epoch']}) | "
        f"params={params['trainable_params']:,}/{params['total_params']:,} trainable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
