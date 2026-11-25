# CS460 Spot-Check Walkthrough & Live Demo Guide

This guide maps the exact implementation lines and single-command demos needed during a live CS460 code walkthrough or spot-check session.

---

## 1. Explicit Pure-PyTorch Training Loop (`src/train.py:322-362`)

All three controlled runs (`res_800`, `res_480`, `res_480_small_anchor`) execute the same explicit hand-written PyTorch training loop inside `train_one_epoch()` in `src/train.py` — no high-level trainer wrappers (`Trainer`, Lightning, Ultralytics) are used:

| Step | Exact Lines | Code / Mechanism | Purpose |
| --- | --- | --- | --- |
| **1. Zero gradients** | `src/train.py:322` | `optimizer.zero_grad(set_to_none=True)` | Clears previous step gradients cleanly before the forward pass. |
| **2. AMP forward & 4 Faster R-CNN losses** | `src/train.py:325-332` | `with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):` and `loss_dict = model(device_images, device_targets)` | Computes all 4 Faster R-CNN loss terms (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`) and sums them into scalar `loss = sum(loss_dict.values())`. |
| **3. Finite-loss guard** | `src/train.py:336-346` | `if not torch.isfinite(loss).all() or any(not torch.isfinite(v).all() for v in loss_dict.values()): raise RuntimeError(...)` | Aborts *before* `backward()` and `optimizer.step()` if any loss component produces `NaN`/`Inf`, preventing silent weight or checkpoint corruption. |
| **4. Scaled backward** | `src/train.py:351` | `scaler.scale(loss).backward()` | Backpropagates gradients through trainable backbone layers (`trainable_backbone_layers=3`), FPN, RPN, and RoI box heads (no-op scaling on CPU). |
| **5. Unscale & gradient clipping** | `src/train.py:352-353` | `scaler.unscale_(optimizer)` followed by `torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)` | Unscales AMP gradients before clipping global gradient L2 norm to `10.0`. |
| **6. Optimizer step & scaler update** | `src/train.py:355-357` | `scaler.step(optimizer)` and `scaler.update()` | Updates trainable parameters and adapts AMP loss scaling factor. |
| **7. Overflow-safe LR schedule step** | `src/train.py:360-362` | `if not amp_step_skipped: scheduler.step()` | Advances the linear-warmup + cosine-decay learning-rate schedule only when CUDA AMP did not skip the optimizer step. |
| **8. Immediate raw CSV logging** | `src/train.py:377-391` | `step_writer.writerow(...)` + `step_fh.flush()` | Persists every step's `run_name`, `epoch`, `step`, `global_step`, `lr`, `total_loss`, and all 4 component losses directly to `results/logs/<run>_step_log.csv`. |

**Checkpoint Selection & Exact Resume:**
- After each epoch, `src/train.py` evaluates on `val` in `model.eval()` + `torch.no_grad()`, logs a flushed row to `results/logs/<run>_epoch_log.csv`, saves `<run>_last.pt`, and overwrites `<run>_best.pt` strictly when `val_ap50` improves.
- Deterministic per-epoch shuffling (`train_generator.manual_seed(cfg.seed + epoch)` at `src/train.py:580`) guarantees that an interrupted run resumed with `--resume` sees the exact same batch ordering and appends cleanly without duplicate rows.

---

## 2. How `GeneralizedRCNNTransform` Keeps Boxes Aligned Across `800` vs. `480`

A common detection bug is accidentally resizing images manually without transforming ground-truth boxes — or evaluating predictions in resized coordinates instead of original image pixels. This pipeline avoids both pitfalls:

1. **Dataset always emits original-coordinate tensors (`src/dataset.py:105-148`)**:
   - `DUTAntiUAVDataset.__getitem__` loads each RGB image at its native resolution `(orig_w, orig_h)` and parses ground-truth boxes `[x1, y1, x2, y2]` in original pixels.
   - The only training augmentation is box-safe horizontal flip (`hflip_prob=0.5`), which transforms `[x1, y1, x2, y2]` to `[orig_w - x2, y1, orig_w - x1, y2]` strictly in original pixel space.
2. **Internal isotropic rescaling inside `GeneralizedRCNNTransform`**:
   - When `build_model(cfg)` constructs `fasterrcnn_mobilenet_v3_large_fpn(..., min_size=cfg.min_size, max_size=cfg.max_size)` (`src/model.py:72-78`), TorchVision attaches `model.transform` (`GeneralizedRCNNTransform`).
   - During **training**, `GeneralizedRCNNTransform` isotropically scales each image so its shorter edge hits `min_size` (`800` for `res_800`, `480` for `res_480` and `res_480_small_anchor`) capped by `max_size` (`1333` vs. `800`), and applies the **exact same scale factor** to target bounding boxes.
   - During **evaluation/inference** (`model.eval()`), `model.transform.postprocess()` automatically projects predicted boxes back into the **original** `(orig_w, orig_h)` pixel coordinate system before returning predictions.
3. **Fair cross-resolution scale evaluation (`src/eval.py:420-547`)**:
   - Because `res_800`, `res_480`, and `res_480_small_anchor` all output boxes in original pixel coordinates, `evaluate_by_scale_bins()` assigns every ground-truth UAV to `tiny`, `small`, or `medium_large` using `(w * h) / (orig_w * orig_h)` in **original image coordinates** with bin edges computed strictly from `train + val` EDA (`scale_bins.json`). Thus a Tiny UAV stays Tiny across all three models.

---

## 3. Why `res_480_small_anchor` Preserves Pretrained RPN Conv Weights (`src/model.py:80-96`)

Changing RPN anchor scales from `(32, 64, 128, 256, 512)` to `(16, 32, 64, 128, 256)` does **not** re-initialize or break the pretrained COCO RPN head convolutional weights:

- The pretrained RPN head (`model.rpn.head.cls_logits` and `model.rpn.head.bbox_pred`) is parameterized only by `num_anchors_per_location` (= number of anchor sizes per FPN level × number of aspect ratios), **not** by the numerical pixel sizes of the anchors.
- In `src/model.py:80-93`, `build_model()` reads `num_levels = len(model.rpn.anchor_generator.sizes)` and `old_num_anchors = list(model.rpn.anchor_generator.num_anchors_per_location())`, constructs the new `AnchorGenerator` with 5 sizes and the same 3 aspect ratios `(0.5, 1.0, 2.0)`, and **explicitly asserts** `new_num_anchors == old_num_anchors` before assigning `model.rpn.anchor_generator`.
- Therefore, all three runs start from the **exact same** pretrained backbone, FPN, and RPN convolutional weights; only the final 2-class box predictor `FastRCNNPredictor(in_features, num_classes=2)` (`src/model.py:95-96`) is freshly initialized in every run.

---

## 4. Exact 1-Line Live Demo Commands

### A. Regenerate the Training-Loss Curve Directly from Raw CSV Logs

Loss-only 1-liner:
```bash
python -m src.plot_curves --logs results/logs/res_800_epoch_log.csv results/logs/res_480_epoch_log.csv results/logs/res_480_small_anchor_epoch_log.csv --loss-out results/figures/training_loss_curves.png --x-axis epoch
```

Full figure regeneration (both training-loss curve and scale-recall/latency comparison chart):
```bash
python -m src.plot_curves --logs results/logs/res_800_epoch_log.csv results/logs/res_480_epoch_log.csv results/logs/res_480_small_anchor_epoch_log.csv --loss-out results/figures/training_loss_curves.png --x-axis epoch --eval-jsons results/metrics/res_800_test.json results/metrics/res_480_test.json results/metrics/res_480_small_anchor_test.json --compare-out results/figures/scale_recall_and_latency.png
```

*(Note: `src.plot_curves` verifies that every CSV's `run_name` column matches its filename prefix and prints the source log filenames in the figure legend/subtitle.)*

### B. Run Single-Image Spot-Check Inference on Any Supplied Image

```bash
python -m src.predict_single --config configs/res_800.yaml --checkpoint checkpoints/res_800_best.pt --image <path_to_image> --score-thresh 0.50 --out-image spot_check_overlay.png
```

Swap `res_800` for `res_480` or `res_480_small_anchor` to compare all three checkpoints on the exact same instructor-supplied image. The CLI outputs structured JSON (`run_name`, `orig_size`, `num_detections`, `boxes`, `scores`, `inference_ms`) and saves an annotated PNG overlay.

---

## 5. Quick Traceability Checklist for Spot-Check Questions

- **Identical run names end-to-end:** `configs/<run>.yaml` (`run_name` == filename stem) -> `checkpoints/<run>_best.pt` -> `results/logs/<run>_epoch_log.csv` -> `results/metrics/<run>_test.json` -> `submission_pack/predictions/test_pack_predictions.csv` -> report tables/captions. Enforced automatically by `src/build_submission.py`.
- **Zero test leakage:** `src/prepare_data.py` computes `scale_bins.json` strictly on `train + val` splits; `src/train.py` selects `_best.pt` strictly on `val` `AP50`; `test` is evaluated once via `src/eval.py --split test`.
- **Controlled variables verified programmatically:** `src/config.py::validate_controlled_configs` asserts that `seed`, `trainable_backbone_layers`, `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_steps`, `grad_clip_norm`, `hflip_prob`, `score_thresh`, `iou_thresh`, `aspect_ratios`, and `len(anchor_sizes)` are identical across all three YAML configs.
