# Small Targets at Speed: Input Resolution and RPN Anchor Scale Sensitivity in Vision-Based Anti-UAV Detection

CS460 Computer Vision Final Project — Pure-PyTorch Single-Class Object Detection (`UAV` vs. Background) on the **DUT Anti-UAV Detection** dataset.

---

## 1. Research Question & Controlled Experimental Design

**Research Question:** How do input resolution (`min_size=800` vs. `480`) and Region Proposal Network (RPN) anchor scales affect **Tiny-UAV recall** (measured strictly in original image pixel coordinates), false-positive rate, and single-image inference latency in a lightweight two-stage detector (`Faster R-CNN MobileNetV3-Large-FPN`)?

**Hypotheses:**
1. Reducing input resolution (`res_800` -> `res_480`) accelerates inference while disproportionately harming recall on **Tiny** UAVs as small boxes shrink below backbone stride and default 32-px RPN anchors.
2. Shifting RPN anchor sizes down from `(32, 64, 128, 256, 512)` to `(16, 32, 64, 128, 256)` at `480` resolution (`res_480_small_anchor`) recovers Tiny-UAV recall at negligible latency overhead, while preserving pretrained RPN convolutional weights intact.

| Run Name (`run_name`) | Role | Config File | Input Resize (`min_size`, `max_size`) | RPN `anchor_sizes` |
| --- | --- | --- | --- | --- |
| `res_800` | Baseline | `configs/res_800.yaml` | `(800, 1333)` | `(32, 64, 128, 256, 512)` |
| `res_480` | Variant 1 | `configs/res_480.yaml` | `(480, 800)` | `(32, 64, 128, 256, 512)` |
| `res_480_small_anchor` | Variant 2 | `configs/res_480_small_anchor.yaml` | `(480, 800)` | `(16, 32, 64, 128, 256)` |

**Controlled Constants Across All Runs** (enforced by `src/config.py`):
- Architecture: TorchVision `fasterrcnn_mobilenet_v3_large_fpn` (COCO pretrained weights; `trainable_backbone_layers=3`; 2-class `FastRCNNPredictor`).
- Optimization & Schedule: `AdamW`, `lr=2e-4`, `weight_decay=1e-4`, `epochs=12`, `batch_size=8`, `warmup_steps=200` + cosine decay, `grad_clip_norm=10.0`, CUDA AMP mixed precision, `seed=42`.
- Augmentation & Thresholds: box-safe `hflip_prob=0.5`, `aspect_ratios=(0.5, 1.0, 2.0)`, fixed-threshold evaluation at `score_thresh=0.50`, `iou_thresh=0.50`.
- Best checkpoint selected strictly by validation `AP50`; scale bins (`tiny`, `small`, `medium_large`) defined strictly in original image coordinates from `train + val` EDA only.

---

## 2. Environment & Pinned Dependency Versions

Tested locally with **Python 3.12** (`3.12.16`) on Linux (`PyTorch 2.15.0+cpu`, `TorchVision 0.29.2+cpu`) and designed for CUDA GPU execution on Google Colab / Kaggle. Minimum pinned dependency constraints in `requirements.txt`:

- `Python >= 3.12`
- `torch >= 2.2.0`
- `torchvision >= 0.17.0`
- `numpy >= 1.26.0`
- `Pillow >= 10.0.0`
- `PyYAML >= 6.0.1`
- `matplotlib >= 3.8.0`
- `huggingface-hub >= 0.23.0`
- `python-docx >= 1.1.0`
- `pytest >= 8.0.0`

### Local Virtual Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation and run the unit/integration test suite:

```bash
.venv/bin/pytest -W error -v
```

---

## 3. Dataset & Checkpoint Locations

### Dataset — DUT Anti-UAV Detection
- **Source:** Official DUT Anti-UAV Detection benchmark (Zhao et al., IEEE TITS 2023; Apache-2.0), mirrored byte-for-byte on Hugging Face at [`aegean-ai/dut-anti-uav`](https://huggingface.co/datasets/aegean-ai/dut-anti-uav).
- **Official Splits:** `5,200` train (`5,243` boxes, 3 empty images) / `2,600` val (`2,621` boxes) / `2,200` test (`2,245` boxes), Pascal VOC XML annotations (`img/` + `xml/`).
- **Local / Cloud Storage:** Stored outside git under `data/dut_anti_uav/` locally. In `notebooks/run_experiments.ipynb` on Google Colab, the dataset is extracted onto fast local SSD at `/content/data/dut_anti_uav` (avoiding Drive FUSE small-file I/O overhead across ~20,000 JPEG/XML files), while all persistent outputs (checkpoints, raw CSV logs, metrics, figures, and test-pack predictions) are written to Google Drive under `MyDrive/CV_project/` (on Kaggle, `/kaggle/working/data/dut_anti_uav` and `/kaggle/working/CV_project/`).

### Trained Model Checkpoints (`.pt`)
To keep both the GitHub repository and the CS460 submission archive strictly `<= 50 MB`, full-state FP32 checkpoints (`~210 MB` each, storing `model_state_dict` plus AdamW optimizer and scheduler state for seamless `--resume` recovery) are hosted externally (Google Drive `MyDrive/CV_project/checkpoints/` — [Project Drive Folder](https://drive.google.com/drive/folders/1iO-UKK3-DpAfsj9OokLrFWhuvh5JJHdm) — and/or GitHub Releases) and indexed in `checkpoints_manifest.json` with SHA-256 checksums.

Download and verify all checkpoints with one command (after Phase 3 training populates `checkpoints_manifest.json`):

```bash
python -m scripts.download_checkpoints --manifest checkpoints_manifest.json --out-dir checkpoints
```

Download a single checkpoint or verify already-downloaded files:

```bash
python -m scripts.download_checkpoints --manifest checkpoints_manifest.json --out-dir checkpoints --runs res_800
python -m scripts.download_checkpoints --manifest checkpoints_manifest.json --out-dir checkpoints --verify-only
```

---

## 4. Exact CLI Commands

### 4.1 Download Dataset & Run `train + val` EDA

Downloads DUT Anti-UAV from `aegean-ai/dut-anti-uav`, audits Pascal VOC XML annotations, and writes `results/metrics/eda_train_val.json` plus sibling `results/metrics/scale_bins.json` (never touching `test/`):

```bash
python -m src.prepare_data --data-dir data/dut_anti_uav --eda-out results/metrics/eda_train_val.json
```

If the dataset is already present locally, skip downloading:

```bash
python -m src.prepare_data --data-dir data/dut_anti_uav --eda-out results/metrics/eda_train_val.json --skip-download
```

### 4.2 Train All 3 Controlled Models (Explicit PyTorch Loop)

Full 12-epoch training runs (writes raw step/epoch CSV logs to `results/logs/` and saves `<run>_last.pt` and best-val-AP50 `<run>_best.pt` to `checkpoints/`):

```bash
python -m src.train --config configs/res_800.yaml --data-dir data/dut_anti_uav --log-dir results/logs --checkpoint-dir checkpoints
python -m src.train --config configs/res_480.yaml --data-dir data/dut_anti_uav --log-dir results/logs --checkpoint-dir checkpoints
python -m src.train --config configs/res_480_small_anchor.yaml --data-dir data/dut_anti_uav --log-dir results/logs --checkpoint-dir checkpoints
```

Resume an interrupted run from `checkpoints/<run>_last.pt` without duplicating prior CSV rows:

```bash
python -m src.train --config configs/res_800.yaml --data-dir data/dut_anti_uav --log-dir results/logs --checkpoint-dir checkpoints --resume
```

Fast 1-epoch verification pass (8 train / 4 val samples; add `--no-pretrained` for offline CPU checks):

```bash
python -m src.train --config configs/res_800.yaml --data-dir data/dut_anti_uav --log-dir results/logs --checkpoint-dir checkpoints --smoke-test --no-pretrained
```

### 4.3 Evaluate on Held-Out `test` Split (Original-Coordinate Scale Bins + Latency)

```bash
python -m src.eval --config configs/res_800.yaml --checkpoint checkpoints/res_800_best.pt --data-dir data/dut_anti_uav --split test --scale-bins results/metrics/scale_bins.json --out results/metrics/res_800_test.json
python -m src.eval --config configs/res_480.yaml --checkpoint checkpoints/res_480_best.pt --data-dir data/dut_anti_uav --split test --scale-bins results/metrics/scale_bins.json --out results/metrics/res_480_test.json
python -m src.eval --config configs/res_480_small_anchor.yaml --checkpoint checkpoints/res_480_small_anchor_best.pt --data-dir data/dut_anti_uav --split test --scale-bins results/metrics/scale_bins.json --out results/metrics/res_480_small_anchor_test.json
```

### 4.4 Regenerate Training-Loss Curve & Scale-Recall / Latency Figures

```bash
python -m src.plot_curves --logs results/logs/res_800_epoch_log.csv results/logs/res_480_epoch_log.csv results/logs/res_480_small_anchor_epoch_log.csv --loss-out results/figures/training_loss_curves.png --x-axis epoch --eval-jsons results/metrics/res_800_test.json results/metrics/res_480_test.json results/metrics/res_480_small_anchor_test.json --compare-out results/figures/scale_recall_and_latency.png
```

### 4.5 Run Predictions & Visual Overlays on Small Test Pack

```bash
python -m src.predict_pack --run configs/res_800.yaml checkpoints/res_800_best.pt --run configs/res_480.yaml checkpoints/res_480_best.pt --run configs/res_480_small_anchor.yaml checkpoints/res_480_small_anchor_best.pt --test-pack-dir submission_pack/test_pack --out-csv submission_pack/predictions/test_pack_predictions.csv --visuals-dir submission_pack/predictions/visuals
```

### 4.6 Spot-Check Single-Image Inference

```bash
python -m src.predict_single --config configs/res_800.yaml --checkpoint checkpoints/res_800_best.pt --image <path_to_image> --score-thresh 0.50 --out-image spot_check_overlay.png
```

### 4.7 Build Official CS460 Report Draft (`.docx`)

Populates the official `CS460_Computer_Vision_Final_Report_Template.docx` with metrics tables, embedded figures, test-pack sample references, and `[YOUR OWN WORDS]` placeholders for personal interpretation before exporting to PDF:

```bash
python -m report.build_report_docx \
  --template CS460_Computer_Vision_Final_Report_Template.docx \
  --eval-jsons results/metrics/res_800_test.json results/metrics/res_480_test.json results/metrics/res_480_small_anchor_test.json \
  --loss-figure results/figures/training_loss_curves.png \
  --scale-figure results/figures/scale_recall_and_latency.png \
  --sample-visuals submission_pack/predictions/visuals/*.png \
  --pred-csv submission_pack/predictions/test_pack_predictions.csv \
  --out report/CS460_Final_Report_Draft.docx
```

### 4.8 Build Verified `<= 50 MB` Submission Zip

Validates all CS460 checklist files, enforces strict `run_name` consistency across configs/logs/predictions, excludes `.pt`/`.pth` weights and datasets, and aborts if the zip exceeds 50 MB:

```bash
python -m src.build_submission --repo-root . --report-pdf report/CS460_Final_Report.pdf --out-zip submission_build/CS460_AntiUAV_Submission.zip
```

### 4.9 Cloud GPU Execution (Google Colab / Kaggle)

Open `notebooks/run_experiments.ipynb` in Google Colab (extracts DUT Anti-UAV onto fast local SSD at `/content/data/dut_anti_uav` while auto-mounting `/content/drive/MyDrive/CV_project` for persistent checkpoints, raw CSV logs, metrics, figures, and test-pack predictions) or Kaggle Notebooks (auto-detects `/kaggle/working/data/dut_anti_uav` and `/kaggle/working/CV_project`) to execute Steps 4.1–4.6 on a free cloud GPU with live streamed logs and automatic checkpoint resume.

---

## 5. Named Sample Expected Output

> **Status:** `TBD — populate after Phase 4 real GPU evaluation on submission_pack/test_pack/`.

After Phase 4 generates `submission_pack/predictions/test_pack_predictions.csv`, replace the placeholder table below with an actual held-out sample ID from `submission_pack/test_pack/img/` and its saved predictions across all three runs:

| Field | Expected Value (Format Preview) |
| --- | --- |
| `sample_id` | `<TBD_SAMPLE_ID>` (e.g., `00001` from `submission_pack/test_pack/img/<TBD_SAMPLE_ID>.jpg`) |
| Ground Truth Box (`orig px`) | `[[x1, y1, x2, y2]]` (scale bin: `tiny` / `small` / `medium_large`) |
| `res_800` Prediction (`score >= 0.50`) | `num_detections=<N>`, `boxes=[[...]]`, `scores=[...]`, `TP/FP/FN=<...>` |
| `res_480` Prediction (`score >= 0.50`) | `num_detections=<N>`, `boxes=[[...]]`, `scores=[...]`, `TP/FP/FN=<...>` |
| `res_480_small_anchor` Prediction (`score >= 0.50`) | `num_detections=<N>`, `boxes=[[...]]`, `scores=[...]`, `TP/FP/FN=<...>` |

Example command to reproduce the named sample prediction directly:

```bash
python -m src.predict_single --config configs/res_800.yaml --checkpoint checkpoints/res_800_best.pt --image submission_pack/test_pack/img/<TBD_SAMPLE_ID>.jpg --score-thresh 0.50 --out-image results/figures/named_sample_check.png
```

---

## 6. Repository Layout

```text
configs/                     # Controlled YAML configs (res_800, res_480, res_480_small_anchor)
docs/
  spot_check_guide.md        # Live walkthrough mapping src/train.py lines & 1-line demo commands
notebooks/
  run_experiments.ipynb      # Colab (Google Drive) / Kaggle auto-detecting GPU runner
report/
  build_report_docx.py       # Official CS460 Word template (.docx) draft builder
scripts/
  download_checkpoints.py    # Streaming SHA-256 checkpoint verifier & downloader
src/
  config.py                  # YAML config loader & cross-run controlled-variable validator
  prepare_data.py            # Hugging Face downloader, VOC XML validator, train+val EDA
  dataset.py                 # DUTAntiUAVDataset, box-safe hflip, deterministic seeding
  model.py                   # Controlled Faster R-CNN MobileNetV3-Large-FPN factory
  train.py                   # Explicit pure-PyTorch training loop, resume, raw CSV logs
  eval.py                    # Single-class AP50/AP75/AP50:95, original-coordinate scale bins, latency
  plot_curves.py             # Raw-log training loss curves & scale-recall/latency figures
  predict_pack.py            # Multi-model test-pack CSV exporter & GT/TP/FP/FN overlays
  predict_single.py          # Single-image spot-check CLI
  build_submission.py        # <= 50 MB CS460 submission zip packager & consistency validator
tests/                       # Pytest unit & integration test suite
checkpoints_manifest.json    # External checkpoint URLs & SHA-256 digests
requirements.txt             # Pinned Python dependencies
```

---

## 7. Citations & Borrowed Code / Assets

1. **DUT Anti-UAV Dataset:** Jie Zhao, Jingshu Zhang, Dongdong Li, and Dong Wang, *"Vision-Based Anti-UAV Detection and Tracking,"* IEEE Transactions on Intelligent Transportation Systems (TITS), 2023. Licensed under Apache-2.0. Downloaded from the Hugging Face mirror `aegean-ai/dut-anti-uav`.
2. **Pretrained Detector Architecture & Weights:** PyTorch / TorchVision `fasterrcnn_mobilenet_v3_large_fpn` (`FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT`), `AnchorGenerator`, `FastRCNNPredictor`, and `GeneralizedRCNNTransform` (BSD-3-Clause license; Paszke et al. / TorchVision maintainers).
3. **Original Project Code:** All data preparation, VOC XML validation, dataset/augmentation, controlled model factory, explicit PyTorch training/resume loop, transparent AP50 & original-coordinate scale-bin evaluator, log-based plotting, prediction/spot-check CLIs, Word report builder, submission packager, and automated pytest test suites in `src/`, `report/`, `scripts/`, and `tests/` were written specifically for this project without third-party trainer frameworks.

---

## 8. AI Assistance Disclosure

In accordance with CS460 academic integrity guidelines, AI coding assistants were used during development for software engineering support: structuring modular Python pipelines, drafting unit/integration tests, debugging PyTorch/TorchVision API details, and automating report `.docx` assembly from verified metric files. All research hypotheses, experimental controls, training/evaluation executions, quantitative result verification, final written interpretations (`[YOUR OWN WORDS]` sections in the submitted report), and end-to-end code comprehension and live spot-check explanations remain the author's own work and responsibility.
