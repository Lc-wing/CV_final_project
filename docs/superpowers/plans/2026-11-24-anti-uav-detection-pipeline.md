# Anti-UAV Resolution & Anchor Sensitivity Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally verify the complete pure-PyTorch DUT Anti-UAV object detection codebase (data prep/EDA, Faster R-CNN MobileNetV3-Large-FPN training, scale-stratified evaluation, plotting, small test-pack & single-image spot-check CLIs, official `.docx` report builder, `<= 50 MB` submission packager, and Colab/Kaggle launcher) ready for Phase 3 cloud GPU runs.

**Architecture:** Modular CLI-driven Python package under `src/` paired with declarative YAML configs under `configs/` so every experiment (`res_800`, `res_480`, `res_480_small_anchor`) shares identical code paths, seed initialization, and evaluation rules. A thin `notebooks/run_experiments.ipynb` mounts Google Drive `MyDrive/CV_project` on Colab (or Kaggle working dir) and invokes the exact same CLI commands documented in `README.md`.

**Tech Stack:** Python 3.12, PyTorch + TorchVision (`fasterrcnn_mobilenet_v3_large_fpn`), NumPy, Pillow, PyYAML, Matplotlib, `huggingface_hub`, `python-docx`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-24-anti-uav-design.md`

## Global Constraints

- Pure PyTorch + TorchVision `fasterrcnn_mobilenet_v3_large_fpn` with an explicit hand-written training loop (`zero_grad` → forward → `backward` → `step`); no opaque training frameworks.
- Controlled 3-run matrix: `res_800` (`min_size=800, max_size=1333`, anchors `32..512`), `res_480` (`min_size=480, max_size=800`, anchors `32..512`), `res_480_small_anchor` (`min_size=480, max_size=800`, anchors `16..256`); all share `seed=42`, `trainable_backbone_layers=3`, optimizer, LR schedule, and horizontal-flip augmentation.
- Run name (`res_800`, `res_480`, `res_480_small_anchor`) must match verbatim across config filename, raw CSV loss log filename & column, checkpoint filename, prediction CSV, and plot legend/caption.
- Evaluation and scale binning (`tiny`, `small`, `medium_large`) must occur strictly in original image pixel coordinates; scale bin thresholds are computed from `train + val` EDA only (never `test`).
- Final course upload archive must remain strictly `<= 50 MB` (`50 * 1024 * 1024` bytes); full FP32 checkpoints are hosted externally with SHA-256 verification and never committed to Git.
- Local VM has CPU only; local development uses `.venv` with CPU PyTorch wheels while `requirements.txt` remains Colab/Kaggle CUDA-compatible.
- Dual storage: keep GitHub repo `Lc-wing/CV_final_project` clean (`data/`, `checkpoints/`, `.venv/` gitignored) and sync code/docs/artifacts to Google Drive `MyDrive/CV_project` (`1iO-UKK3-DpAfsj9OokLrFWhuvh5JJHdm`).

## Review Focus

1. **Empty / 0-object images & malformed VOC XML boxes**: DUT Anti-UAV contains 3 training images with zero `<object>` entries, and real XMLs can have out-of-bounds or zero-area boxes (`xmax <= xmin`); XML parsing, dataset loading, horizontal flip, and Faster R-CNN forward/loss must handle empty `(0, 4)` targets and skip degenerate boxes without crashing (pinned in Tasks 2 & 3).
2. **Pretrained RPN conv weight mismatch when overriding anchors**: Changing `anchor_sizes` in `res_480_small_anchor` must preserve exact FPN level count and `num_anchors_per_location` (15 anchors/loc) so pretrained RPN conv weights load intact with zero shape errors (pinned in Task 3).
3. **Non-finite loss or mid-training Colab disconnect**: A loss spike (`NaN`/`Inf`) must raise/guard cleanly instead of corrupting checkpoints, and `--resume` must continue from `<run_name>_last.pt` while appending new epochs to the raw CSV log without erasing prior epoch history (pinned in Task 4).
4. **Coordinate-space mismatch during scale-stratified evaluation**: Resized inputs (`480` vs `800`) must never leak into scale-bin assignment; bins and IoU matching must operate strictly in original `(orig_w, orig_h)` coordinates, returning exact `tp`/`fn`/`gt_count` alongside percentages (pinned in Task 5).
5. **Silent `> 50 MB` submission package or run-name drift**: `build_submission.py` must reject archives exceeding 50 MB or any mismatch between config `run_name`, raw CSV log `run_name`, and prediction CSV `run_name` (pinned in Task 7).

---

## File Structure Overview

- `.gitignore`, `requirements.txt`: Clean Git exclusions + pinned dependencies.
- `configs/{res_800,res_480,res_480_small_anchor}.yaml`: Declarative run configs.
- `src/config.py`: `ExperimentConfig` loader + cross-run controlled-variable validator.
- `src/prepare_data.py`: Hugging Face downloader, VOC XML parser/validator, `train+val` EDA & scale-bin calculator.
- `src/dataset.py`: `DUTAntiUAVDataset`, box-safe horizontal flip, `detection_collate_fn`, `seed_everything`.
- `src/model.py`: Controlled `fasterrcnn_mobilenet_v3_large_fpn` builder with anchor override & trainable parameter summary.
- `src/train.py`: Explicit PyTorch training loop with AMP, warmup + cosine schedule, gradient clip, resume, and raw step/epoch CSV loggers.
- `src/eval.py`: Transparent single-class `AP50`/`AP75`/`AP50:95`, fixed-threshold P/R/F1/TP/FP/FN, original-coordinate scale-stratified evaluation, and single-image latency benchmark.
- `src/plot_curves.py`: Loss-curve & comparison plot generator reading solely from raw CSV logs and eval JSONs.
- `src/predict_pack.py`: Test-pack evaluator exporting `test_pack_predictions.csv` and side-by-side GT/TP/FP/FN overlays.
- `src/predict_single.py`: Spot-check single-image CLI inference tool.
- `report/build_report_docx.py`: Template-preserving `.docx` report draft generator built on `CS460_Computer_Vision_Final_Report_Template.docx`.
- `src/build_submission.py`: CS460 checklist & `<= 50 MB` submission zip verifier/packager.
- `scripts/download_checkpoints.py`: Manifest-driven FP32 checkpoint downloader + SHA-256 checker.
- `notebooks/run_experiments.ipynb`: Thin auto-detecting Colab/Kaggle launcher invoking exact `README.md` CLI commands.
- `docs/spot_check_guide.md`, `README.md`: Spot-check walkthrough & complete CS460 README.

---

### Task 1: Environment, Repo Scaffold & Controlled Config Loader

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `configs/res_800.yaml`
- Create: `configs/res_480.yaml`
- Create: `configs/res_480_small_anchor.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML files under `configs/`.
- Produces:
  - `ExperimentConfig` dataclass (`run_name: str`, `seed: int`, `min_size: int`, `max_size: int`, `anchor_sizes: tuple[int, ...]`, `aspect_ratios: tuple[float, ...]`, `trainable_backbone_layers: int`, `epochs: int`, `batch_size: int`, `lr: float`, `weight_decay: float`, `warmup_steps: int`, `grad_clip_norm: float`, `hflip_prob: float`, `score_thresh: float`, `iou_thresh: float`, `num_workers: int`).
  - `load_config(path: str | Path) -> ExperimentConfig`
  - `validate_controlled_configs(configs: list[ExperimentConfig]) -> None`

- [ ] **Step 1: Create local CPU `.venv`, `.gitignore`, `pyproject.toml`, and `requirements.txt`**

Create `.gitignore` excluding `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/`, `checkpoints/`, `*.pt`, `*.pth`, `*.zip`, `.DS_Store`, `submission_build/`.  
Create `pyproject.toml` with `[tool.pytest.ini_options]` setting `pythonpath = ["."]` and `testpaths = ["tests"]`.  
Create `requirements.txt` listing `torch>=2.2.0`, `torchvision>=0.17.0`, `numpy>=1.26.0`, `Pillow>=10.0.0`, `PyYAML>=6.0.1`, `matplotlib>=3.8.0`, `huggingface-hub>=0.23.0`, `python-docx>=1.1.0`, `pytest>=8.0.0`.  
Initialize `.venv` with `python3 -m venv .venv` and install CPU wheels (`torch`, `torchvision` via `https://download.pytorch.org/whl/cpu`) plus the remaining packages from `requirements.txt`.

- [ ] **Step 2: Write failing tests in `tests/test_config.py`**

```python
from pathlib import Path
import pytest
from src.config import ExperimentConfig, load_config, validate_controlled_configs


def test_load_all_three_experiment_configs():
    cfgs = [
        load_config(Path("configs/res_800.yaml")),
        load_config(Path("configs/res_480.yaml")),
        load_config(Path("configs/res_480_small_anchor.yaml")),
    ]
    assert [c.run_name for c in cfgs] == ["res_800", "res_480", "res_480_small_anchor"]
    assert cfgs[0].min_size == 800 and cfgs[0].max_size == 1333
    assert cfgs[1].min_size == 480 and cfgs[1].max_size == 800
    assert cfgs[2].min_size == 480 and cfgs[2].max_size == 800
    assert cfgs[0].anchor_sizes == (32, 64, 128, 256, 512)
    assert cfgs[1].anchor_sizes == (32, 64, 128, 256, 512)
    assert cfgs[2].anchor_sizes == (16, 32, 64, 128, 256)
    validate_controlled_configs(cfgs)


def test_run_name_must_match_filename_stem(tmp_path):
    bad = tmp_path / "wrong_name.yaml"
    bad.write_text(Path("configs/res_800.yaml").read_text())
    with pytest.raises(ValueError, match="run_name"):
        load_config(bad)


def test_validate_controlled_configs_rejects_seed_drift():
    c1 = load_config("configs/res_800.yaml")
    c2 = load_config("configs/res_480.yaml")
    c2_bad = ExperimentConfig(**{**c2.__dict__, "seed": 99})
    with pytest.raises(ValueError, match="seed"):
        validate_controlled_configs([c1, c2_bad])
```

- [ ] **Step 3: Run pytest to verify tests fail**

Run: `.venv/bin/pytest tests/test_config.py -v`  
Expected: FAIL (`ModuleNotFoundError: No module named 'src.config'`).

- [ ] **Step 4: Create `src/__init__.py`, `src/config.py`, and the 3 YAML configs**

Implement `ExperimentConfig`, `load_config(path: str | Path) -> ExperimentConfig` (enforcing `cfg.run_name == Path(path).stem`, 5 positive ints in `anchor_sizes`, `min_size <= max_size`), and `validate_controlled_configs(configs: list[ExperimentConfig]) -> None` (asserting identical `seed`, `trainable_backbone_layers`, `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_steps`, `grad_clip_norm`, `hflip_prob`, `score_thresh`, `iou_thresh`, `aspect_ratios`, and `len(anchor_sizes)` across all runs).  
Write `configs/res_800.yaml`, `configs/res_480.yaml`, and `configs/res_480_small_anchor.yaml` with `seed: 42`, `trainable_backbone_layers: 3`, `epochs: 12`, `batch_size: 8`, `lr: 0.0002` (AdamW), `weight_decay: 0.0001`, `warmup_steps: 200`, `grad_clip_norm: 10.0`, `hflip_prob: 0.5`, `score_thresh: 0.5`, `iou_thresh: 0.5`, `aspect_ratios: [0.5, 1.0, 2.0]`, `num_workers: 2`.

- [ ] **Step 5: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_config.py -v`  
Expected: `3 passed`.  
Commit: `git add .gitignore requirements.txt src/__init__.py src/config.py configs tests/test_config.py docs/superpowers && git commit -m "chore: scaffold project configs and controlled config validator"`

---

### Task 2: DUT Anti-UAV Downloader, VOC XML Validator & Train+Val EDA

**Files:**
- Create: `src/prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: Pascal VOC XML files in `<split>/xml/*.xml` and JPEG images in `<split>/img/*.jpg`; `huggingface_hub.hf_hub_download` for `aegean-ai/dut-anti-uav`.
- Produces:
  - `parse_voc_xml(xml_path: Path, img_width: int, img_height: int) -> tuple[list[list[float]], int]` (returns valid clamped `[xmin, ymin, xmax, ymax]` boxes and count of dropped degenerate boxes).
  - `SplitAudit` dataclass (`split: str`, `num_images: int`, `num_empty_images: int`, `num_boxes: int`, `num_dropped_boxes: int`, `missing_xml: list[str]`, `area_ratios: list[float]`, `equiv_sides: list[float]`, `widths: list[int]`, `heights: list[int]`).
  - `scan_split(split_dir: Path, split_name: str) -> SplitAudit`
  - `compute_scale_bins(train_audit: SplitAudit, val_audit: SplitAudit) -> dict` (keys: `metric`, `edges` (`tiny_max`, `small_max`), `train_counts`, `val_counts`, `source_splits=["train", "val"]`).
  - `download_dut_anti_uav(data_dir: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> None`
  - CLI: `python -m src.prepare_data --data-dir <dir> --eda-out <json> [--skip-download]`

- [ ] **Step 1: Write failing tests in `tests/test_prepare_data.py` (including Review Focus #1)**

```python
from pathlib import Path
from PIL import Image
from src.prepare_data import parse_voc_xml, scan_split, compute_scale_bins


def _write_xml(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    objs = "".join(
        f"<object><name>UAV</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for x1, y1, x2, y2 in boxes
    )
    path.write_text(f"<annotation><size><width>100</width><height>100</height></size>{objs}</annotation>")


def test_parse_voc_xml_handles_empty_and_malformed_boxes(tmp_path):
    empty_xml = tmp_path / "empty.xml"
    _write_xml(empty_xml, [])
    boxes, dropped = parse_voc_xml(empty_xml, img_width=100, img_height=100)
    assert boxes == [] and dropped == 0

    mixed_xml = tmp_path / "mixed.xml"
    _write_xml(mixed_xml, [(-5, 10, 120, 40), (50, 50, 50, 60), (60, 40, 20, 80)])
    boxes, dropped = parse_voc_xml(mixed_xml, img_width=100, img_height=100)
    assert boxes == [[0.0, 10.0, 100.0, 40.0]]
    assert dropped == 2


def test_scan_split_and_scale_bins_from_train_val_only(tmp_path):
    for split, ratios in [("train", [0.0005, 0.002, 0.02]), ("val", [0.0004, 0.003, 0.03])]:
        img_dir = tmp_path / split / "img"
        xml_dir = tmp_path / split / "xml"
        img_dir.mkdir(parents=True)
        xml_dir.mkdir(parents=True)
        for idx, r in enumerate(ratios):
            Image.new("RGB", (1000, 1000)).save(img_dir / f"{idx:05d}.jpg")
            side = (r * 1000 * 1000) ** 0.5
            _write_xml(xml_dir / f"{idx:05d}.xml", [(10, 10, 10 + side, 10 + side)])
        # Add 1 empty image in train (mirrors DUT Anti-UAV's 3 empty train images)
        if split == "train":
            Image.new("RGB", (1000, 1000)).save(img_dir / "99999.jpg")
            _write_xml(xml_dir / "99999.xml", [])

    train_a = scan_split(tmp_path / "train", "train")
    val_a = scan_split(tmp_path / "val", "val")
    assert train_a.num_images == 4 and train_a.num_empty_images == 1 and train_a.num_boxes == 3
    bins = compute_scale_bins(train_a, val_a)
    assert bins["source_splits"] == ["train", "val"]
    assert 0.0 < bins["edges"]["tiny_max"] < bins["edges"]["small_max"]
    assert sum(bins["train_counts"].values()) == 3
    assert sum(bins["val_counts"].values()) == 3
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_prepare_data.py -v`  
Expected: FAIL (`ModuleNotFoundError: No module named 'src.prepare_data'`).

- [ ] **Step 3: Implement `src/prepare_data.py`**

Implement XML parsing (clamping to `[0, W]`/`[0, H]`, dropping `xmax <= xmin` or `ymax <= ymin`), split auditing, `compute_scale_bins` (computing `tiny` / `small` / `medium_large` thresholds from `train + val` combined relative box area ratio tertiles rounded to clean 2-sig-fig bounds, e.g. `tiny_max`, `small_max`, with fallback defaults if sample count < 3), idempotent Hugging Face download of `detection/{train,val,test}.zip` from `aegean-ai/dut-anti-uav`, and CLI saving `eda_train_val.json` (plus `scale_bins.json`).

- [ ] **Step 4: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_prepare_data.py -v`  
Expected: `2 passed`.  
Commit: `git add src/prepare_data.py tests/test_prepare_data.py && git commit -m "feat: add DUT Anti-UAV downloader, VOC XML validator, and train+val EDA"`

---

### Task 3: PyTorch Dataset, Box-Safe Augmentation & Controlled Faster R-CNN Factory

**Files:**
- Create: `src/dataset.py`
- Create: `src/model.py`
- Test: `tests/test_dataset_and_model.py`

**Interfaces:**
- Consumes: `ExperimentConfig` (`src/config.py`), `parse_voc_xml` (`src/prepare_data.py`).
- Produces:
  - `seed_everything(seed: int) -> None`
  - `DUTAntiUAVDataset(split_dir: Path, hflip_prob: float = 0.0, max_samples: int | None = None)` returning `(image_tensor: FloatTensor[3, H, W], target: dict)` where `target` contains `boxes: FloatTensor[N, 4]`, `labels: Int64Tensor[N]`, `image_id: Tensor[1]`, `sample_id: str`, `orig_size: tuple[int, int]` (`(W, H)`).
  - `detection_collate_fn(batch)` -> `(list[Tensor], list[dict])`
  - `build_model(cfg: ExperimentConfig, pretrained: bool = True) -> FasterRCNN`
  - `summarize_trainable_params(model: nn.Module) -> dict[str, int]` (`total_params`, `trainable_params`, `frozen_params`)

- [ ] **Step 1: Write failing tests in `tests/test_dataset_and_model.py` (including Review Focus #1 & #2)**

```python
from pathlib import Path
import torch
from PIL import Image
from src.config import load_config
from src.dataset import DUTAntiUAVDataset, detection_collate_fn, seed_everything
from src.model import build_model, summarize_trainable_params


def test_dataset_empty_image_and_horizontal_flip(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "00001.jpg")
    (tmp_path / "xml" / "00001.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox>"
        "<xmin>20</xmin><ymin>10</ymin><xmax>50</xmax><ymax>30</ymax>"
        "</bndbox></object></annotation>"
    )
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "00002.jpg")
    (tmp_path / "xml" / "00002.xml").write_text("<annotation></annotation>")

    ds = DUTAntiUAVDataset(tmp_path, hflip_prob=1.0)
    img1, tgt1 = ds[0]
    assert img1.shape == (3, 100, 200)
    assert torch.allclose(tgt1["boxes"], torch.tensor([[150.0, 10.0, 180.0, 30.0]]))
    assert tgt1["labels"].tolist() == [1]
    assert tgt1["orig_size"] == (200, 100)

    img2, tgt2 = ds[1]
    assert tgt2["boxes"].shape == (0, 4) and tgt2["boxes"].dtype == torch.float32
    assert tgt2["labels"].shape == (0,) and tgt2["labels"].dtype == torch.int64


def test_build_model_preserves_rpn_weights_and_anchors_across_variants():
    cfg_480 = load_config("configs/res_480.yaml")
    cfg_small = load_config("configs/res_480_small_anchor.yaml")

    seed_everything(cfg_480.seed)
    m_480 = build_model(cfg_480, pretrained=False)
    seed_everything(cfg_small.seed)
    m_small = build_model(cfg_small, pretrained=False)

    assert len(m_small.rpn.anchor_generator.sizes) == len(m_480.rpn.anchor_generator.sizes)
    assert m_small.rpn.anchor_generator.num_anchors_per_location() == m_480.rpn.anchor_generator.num_anchors_per_location()
    assert m_small.rpn.anchor_generator.sizes[0] == (16, 32, 64, 128, 256)
    assert m_480.rpn.anchor_generator.sizes[0] == (32, 64, 128, 256, 512)
    assert torch.equal(m_480.rpn.head.cls_logits.weight, m_small.rpn.head.cls_logits.weight)
    assert torch.equal(m_480.roi_heads.box_predictor.cls_score.weight, m_small.roi_heads.box_predictor.cls_score.weight)

    stats = summarize_trainable_params(m_small)
    assert 0 < stats["trainable_params"] <= stats["total_params"]

    # Forward + loss with both a normal target and an empty (0, 4) target
    m_small.train()
    imgs = [torch.rand(3, 120, 160), torch.rand(3, 120, 160)]
    tgts = [
        {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]), "labels": torch.tensor([1], dtype=torch.int64)},
        {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    losses = m_small(imgs, tgts)
    total = sum(losses.values())
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_dataset_and_model.py -v`  
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/dataset.py` and `src/model.py`**

In `src/dataset.py`: implement `seed_everything`, `DUTAntiUAVDataset` (sorted sample stems for deterministic indexing, `torchvision.transforms.functional.to_tensor`, box-safe horizontal flip), and `detection_collate_fn`.  
In `src/model.py`: implement `build_model(cfg, pretrained=True)` loading `fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None, weights_backbone=None, trainable_backbone_layers=cfg.trainable_backbone_layers, min_size=cfg.min_size, max_size=cfg.max_size)`, inspecting `num_levels = len(model.rpn.anchor_generator.sizes)` and `old_num_anchors = model.rpn.anchor_generator.num_anchors_per_location()`, replacing `model.rpn.anchor_generator = AnchorGenerator(sizes=(tuple(cfg.anchor_sizes),) * num_levels, aspect_ratios=(tuple(cfg.aspect_ratios),) * num_levels)`, asserting `model.rpn.anchor_generator.num_anchors_per_location() == old_num_anchors`, and replacing `model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)`. Implement `summarize_trainable_params`.

- [ ] **Step 4: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_dataset_and_model.py -v`  
Expected: `2 passed`.  
Commit: `git add src/dataset.py src/model.py tests/test_dataset_and_model.py && git commit -m "feat: implement DUT Anti-UAV dataset and controlled Faster R-CNN factory"`

---

### Task 4: Scale-Stratified Detection Evaluator & Latency Benchmark (`src/eval.py`)

*(Built before `src/train.py` so `train.py` directly imports `compute_ap_at_iou` and `compute_fixed_threshold_counts` for validation checkpoint selection—zero duplicate AP code!)*

**Files:**
- Create: `src/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `DUTAntiUAVDataset`, `build_model`, `scale_bins` dict (`edges.tiny_max`, `edges.small_max`).
- Produces:
  - `compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray`
  - `compute_ap_at_iou(predictions: list[dict], ground_truths: list[dict], iou_thresh: float = 0.50) -> float`
  - `compute_fixed_threshold_counts(predictions: list[dict], ground_truths: list[dict], score_thresh: float = 0.50, iou_thresh: float = 0.50) -> dict` (`tp`, `fp`, `fn`, `gt_count`, `precision`, `recall`, `f1`, `per_image`: list of per-sample outcome dicts)
  - `evaluate_by_scale_bins(predictions: list[dict], ground_truths: list[dict], scale_bins: dict, score_thresh: float = 0.50, iou_thresh: float = 0.50) -> dict` (per-bin `gt_count`, `tp`, `fn`, `recall`, `ap50`)
  - `evaluate_predictions(predictions: list[dict], ground_truths: list[dict], scale_bins: dict | None, score_thresh: float = 0.50, iou_thresh: float = 0.50) -> dict` (`ap50`, `ap75`, `ap50_95`, `fixed_threshold`, `by_scale`)
  - `run_inference_on_dataset(model, loader, device) -> tuple[list[dict], list[dict]]`
  - `benchmark_latency(model, dataset, device, warmup: int = 10, num_images: int = 100) -> dict` (`mean_ms`, `median_ms`, `fps`, `num_images`, `device`)
  - CLI: `python -m src.eval --config <yaml> --checkpoint <pt> --data-dir <dir> --split test --scale-bins <json> --out <json>`

- [ ] **Step 1: Write failing tests in `tests/test_eval.py` (including Review Focus #4)**

```python
import numpy as np
from src.eval import (
    compute_iou_matrix,
    compute_ap_at_iou,
    compute_fixed_threshold_counts,
    evaluate_by_scale_bins,
)


def test_iou_ap50_and_fixed_threshold_counts():
    iou = compute_iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [20, 20, 30, 30]]))
    assert np.allclose(iou, [[1.0, 0.0]])

    gts = [
        {"sample_id": "a", "boxes": np.array([[10, 10, 20, 20]], dtype=float), "orig_size": (1000, 1000)},
        {"sample_id": "b", "boxes": np.array([[100, 100, 300, 300]], dtype=float), "orig_size": (1000, 1000)},
    ]
    preds = [
        {"sample_id": "a", "boxes": np.array([[10, 10, 20, 20], [500, 500, 520, 520]], dtype=float), "scores": np.array([0.9, 0.6])},
        {"sample_id": "b", "boxes": np.array([[100, 100, 300, 300]], dtype=float), "scores": np.array([0.4])},
    ]
    ap50 = compute_ap_at_iou(preds, gts, iou_thresh=0.50)
    assert 0.7 < ap50 < 0.9  # rank1 TP, rank2 FP, rank3 TP -> P=[1, 0.5, 0.667], R=[0.5, 0.5, 1.0] => AP=0.8333

    counts = compute_fixed_threshold_counts(preds, gts, score_thresh=0.50, iou_thresh=0.50)
    assert counts["tp"] == 1 and counts["fp"] == 1 and counts["fn"] == 1 and counts["gt_count"] == 2
    assert counts["precision"] == 0.5 and counts["recall"] == 0.5


def test_scale_stratified_bins_use_original_image_coordinates():
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
    assert res["medium_large"]["gt_count"] == 1 and res["medium_large"]["tp"] == 1 and res["medium_large"]["recall"] == 1.0
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_eval.py -v`  
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/eval.py`**

Implement vectorized IoU matrix, all-point interpolated single-class AP (`compute_ap_at_iou`), greedy 1-to-1 fixed-threshold matching (`compute_fixed_threshold_counts`), original-coordinate scale binning (`evaluate_by_scale_bins` classifying each GT by `(w*h)/(orig_w*orig_h)` into `tiny`, `small`, `medium_large`, ignoring predictions matched to other bins when computing per-bin AP50 so cross-bin boxes aren't unfairly counted as FP against another bin), dataset inference loop, GPU/CPU synchronized latency benchmark, and CLI saving structured JSON.

- [ ] **Step 4: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_eval.py -v`  
Expected: `2 passed`.  
Commit: `git add src/eval.py tests/test_eval.py && git commit -m "feat: add transparent single-class AP50 and original-coordinate scale evaluator"`

---

### Task 5: Explicit Pure-PyTorch Training Loop, Resume & Raw CSV Logger (`src/train.py`)

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_config` (`src/config.py`), `DUTAntiUAVDataset`, `detection_collate_fn`, `seed_everything` (`src/dataset.py`), `build_model`, `summarize_trainable_params` (`src/model.py`), `run_inference_on_dataset`, `compute_ap_at_iou`, `compute_fixed_threshold_counts` (`src/eval.py`).
- Produces:
  - `train_run(cfg: ExperimentConfig, data_dir: Path, log_dir: Path, checkpoint_dir: Path, *, pretrained: bool = True, resume: bool = False, max_train_samples: int | None = None, max_val_samples: int | None = None, epochs_override: int | None = None) -> dict`
  - Raw CSVs: `<log_dir>/<run_name>_step_log.csv` and `<log_dir>/<run_name>_epoch_log.csv`
  - Checkpoints: `<checkpoint_dir>/<run_name>_last.pt` and `<checkpoint_dir>/<run_name>_best.pt` (storing `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`, `best_val_ap50`, `run_name`, `config`)
  - CLI: `python -m src.train --config <yaml> --data-dir <dir> --log-dir results/logs --checkpoint-dir checkpoints [--resume] [--smoke-test]`

- [ ] **Step 1: Write failing tests in `tests/test_train.py` (including Review Focus #3)**

```python
import csv
from pathlib import Path
from PIL import Image
from src.config import load_config
from src.train import train_run


def _make_mini_dataset(root: Path) -> None:
    for split, n in [("train", 4), ("val", 2)]:
        (root / split / "img").mkdir(parents=True)
        (root / split / "xml").mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (160, 120), color=(40 + i * 20, 80, 120)).save(root / split / "img" / f"{i:05d}.jpg")
            (root / split / "xml" / f"{i:05d}.xml").write_text(
                "<annotation><object><name>UAV</name><bndbox>"
                "<xmin>30</xmin><ymin>25</ymin><xmax>70</xmax><ymax>60</ymax>"
                "</bndbox></object></annotation>"
            )


def test_train_run_logs_csv_and_resumes_cleanly(tmp_path):
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    _make_mini_dataset(data_dir)

    cfg = load_config("configs/res_480.yaml")
    cfg = cfg.__class__(**{**cfg.__dict__, "batch_size": 2, "num_workers": 0, "warmup_steps": 1})

    # Epoch 1 run
    res1 = train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, epochs_override=1)
    epoch_csv = log_dir / "res_480_epoch_log.csv"
    step_csv = log_dir / "res_480_step_log.csv"
    assert epoch_csv.exists() and step_csv.exists()
    assert (ckpt_dir / "res_480_last.pt").exists() and (ckpt_dir / "res_480_best.pt").exists()

    # Resume to Epoch 2 without overwriting Epoch 1 row
    res2 = train_run(cfg, data_dir, log_dir, ckpt_dir, pretrained=False, resume=True, epochs_override=2)
    with epoch_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert all(r["run_name"] == "res_480" for r in rows)
    assert res2["completed_epochs"] == 2
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_train.py -v`  
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/train.py`**

Implement `train_run` with explicit PyTorch loop (`optimizer.zero_grad(set_to_none=True)`, optional CUDA AMP, `loss_dict = model(images, targets)`, `loss = sum(loss_dict.values())`, explicit `if not torch.isfinite(loss): raise RuntimeError(...)`, `scaler.scale(loss).backward()`, `clip_grad_norm_(params, cfg.grad_clip_norm)`, `scaler.step(optimizer)`), linear warmup + cosine decay LR schedule, deterministic `torch.Generator` for batch shuffling, immediate flush on every CSV row, resume preserving existing CSV rows `<= start_epoch`, and `--smoke-test` CLI flag (`epochs=1`, `max_train_samples=8`, `max_val_samples=4`, `--no-pretrained` optional).

- [ ] **Step 4: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_train.py -v`  
Expected: `1 passed`.  
Commit: `git add src/train.py tests/test_train.py && git commit -m "feat: add explicit PyTorch Faster R-CNN trainer with resume and raw CSV logs"`

---

### Task 6: Curve Regeneration, Test-Pack Prediction & Spot-Check Single-Image CLI

**Files:**
- Create: `src/plot_curves.py`
- Create: `src/predict_pack.py`
- Create: `src/predict_single.py`
- Test: `tests/test_visuals_and_predict.py`

**Interfaces:**
- Consumes: Raw `*_epoch_log.csv` / `*_step_log.csv` (`src/train.py`), eval JSONs (`src/eval.py`), `load_config` (`src/config.py`), `build_model` (`src/model.py`), `compute_fixed_threshold_counts` (`src/eval.py`).
- Produces:
  - `plot_training_loss(log_paths: list[Path], out_path: Path, x_axis: str = "epoch") -> dict` (reads raw CSVs, verifies `run_name` matches filename stem, writes PNG, returns plotted summary dict).
  - `plot_scale_and_latency(eval_json_paths: list[Path], out_path: Path) -> None`
  - `run_test_pack_predictions(run_specs: list[tuple[Path, Path]], test_pack_dir: Path, out_csv: Path, visuals_dir: Path, score_thresh: float = 0.50, iou_thresh: float = 0.50) -> list[dict]`
  - `predict_single_image(config_path: Path, checkpoint_path: Path, image_path: Path, score_thresh: float = 0.50, out_image: Path | None = None) -> dict`
  - CLIs: `python -m src.plot_curves ...`, `python -m src.predict_pack ...`, `python -m src.predict_single ...`

- [ ] **Step 1: Write failing tests in `tests/test_visuals_and_predict.py`**

```python
import csv
import json
from pathlib import Path
import torch
from PIL import Image
from src.config import load_config
from src.model import build_model
from src.plot_curves import plot_training_loss, plot_scale_and_latency
from src.predict_pack import run_test_pack_predictions
from src.predict_single import predict_single_image


def test_plot_curves_from_raw_logs_and_eval_json(tmp_path):
    log = tmp_path / "res_800_epoch_log.csv"
    log.write_text(
        "run_name,epoch,train_total_loss,train_loss_classifier,train_loss_box_reg,"
        "train_loss_objectness,train_loss_rpn_box_reg,val_ap50,val_precision,val_recall,epoch_seconds,lr\n"
        "res_800,1,0.85,0.2,0.3,0.2,0.15,0.40,0.5,0.4,10.0,0.0002\n"
        "res_800,2,0.42,0.1,0.15,0.1,0.07,0.65,0.7,0.6,10.0,0.0001\n"
    )
    out_loss = tmp_path / "loss.png"
    summary = plot_training_loss([log], out_loss)
    assert out_loss.exists() and out_loss.stat().st_size > 1000
    assert "res_800" in summary

    ev = tmp_path / "res_800_test.json"
    ev.write_text(json.dumps({
        "run_name": "res_800",
        "ap50": 0.72,
        "by_scale": {
            "tiny": {"recall": 0.55, "gt_count": 100, "tp": 55, "fn": 45},
            "small": {"recall": 0.80, "gt_count": 100, "tp": 80, "fn": 20},
            "medium_large": {"recall": 0.92, "gt_count": 100, "tp": 92, "fn": 8},
        },
        "latency": {"mean_ms": 28.5, "fps": 35.1},
    }))
    out_cmp = tmp_path / "scale_latency.png"
    plot_scale_and_latency([ev], out_cmp)
    assert out_cmp.exists() and out_cmp.stat().st_size > 1000


def test_predict_pack_and_single_image_cli(tmp_path):
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
        "<annotation><object><name>UAV</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object></annotation>"
    )

    out_csv = tmp_path / "preds.csv"
    vis_dir = tmp_path / "visuals"
    rows = run_test_pack_predictions([(Path("configs/res_480.yaml"), ckpt)], pack, out_csv, vis_dir)
    assert out_csv.exists() and len(rows) == 1 and rows[0]["sample_id"] == "00010" and rows[0]["run_name"] == "res_480"
    assert (vis_dir / "00010_comparison.png").exists()

    out_single = tmp_path / "single.png"
    res = predict_single_image(Path("configs/res_480.yaml"), ckpt, img_path, score_thresh=0.5, out_image=out_single)
    assert out_single.exists() and res["run_name"] == "res_480" and "boxes" in res and "scores" in res
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_visuals_and_predict.py -v`  
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/plot_curves.py`, `src/predict_pack.py`, and `src/predict_single.py`**

Use headless `matplotlib.use("Agg")`.  
In `src/plot_curves.py`: verify each log's `run_name` column matches its filename prefix, plot clean multi-run curves with explicit run names and log filenames in legend/subtitle.  
In `src/predict_pack.py`: run each `(config, checkpoint)` on `test_pack/img/*.jpg`, compute TP/FP/FN against `test_pack/xml/*.xml` at fixed `score_thresh` & `iou_thresh`, save `test_pack_predictions.csv`, and draw side-by-side panels (1 panel per model) overlaying GT (green), TP (blue), FP (red dashed), and missed FN (amber) with a shared legend.  
In `src/predict_single.py`: load checkpoint cleanly (`weights_only=False` or tensor-safe dict read), run 1-image inference in `model.eval()` + `torch.no_grad()`, measure latency, print formatted JSON/table output, and optionally save overlay PNG.

- [ ] **Step 4: Run pytest to verify pass and commit**

Run: `.venv/bin/pytest tests/test_visuals_and_predict.py -v`  
Expected: `2 passed`.  
Commit: `git add src/plot_curves.py src/predict_pack.py src/predict_single.py tests/test_visuals_and_predict.py && git commit -m "feat: add log-based plotting, test-pack predictor, and spot-check single-image CLI"`

---

### Task 7: Official Report `.docx` Builder, `<= 50 MB` Submission Packager, Checkpoint Downloader & Colab/Kaggle Launcher

**Files:**
- Create: `report/__init__.py`
- Create: `report/build_report_docx.py`
- Create: `src/build_submission.py`
- Create: `scripts/__init__.py`
- Create: `scripts/download_checkpoints.py`
- Create: `checkpoints_manifest.json`
- Create: `notebooks/run_experiments.ipynb`
- Create: `docs/spot_check_guide.md`
- Create: `README.md`
- Test: `tests/test_packaging_and_report.py`

**Interfaces:**
- Consumes: Official template `CS460_Computer_Vision_Final_Report_Template.docx`, `configs/*.yaml`, `results/logs/*_epoch_log.csv`, `results/metrics/*_test.json`, `results/figures/*.png`, `submission_pack/test_pack/`, `submission_pack/predictions/test_pack_predictions.csv`.
- Produces:
  - `build_report_docx(template_path: Path, eval_jsons: list[Path], loss_figure: Path, scale_figure: Path, sample_visuals: list[Path], pred_csv: Path, out_docx: Path) -> Path`
  - `build_submission_zip(repo_root: Path, out_zip: Path, report_pdf: Path, max_bytes: int = 50 * 1024 * 1024) -> dict` (validates checklist items, run-name consistency across configs/logs/predictions, no `.pt` weights accidentally bundled, and `zip_size <= max_bytes`).
  - `verify_sha256(path: Path, expected_hex: str) -> bool` + manifest downloader CLI in `scripts/download_checkpoints.py`.
  - `notebooks/run_experiments.ipynb` (Colab Drive mount `/content/drive/MyDrive/CV_project` + Kaggle auto-detect, calling exact `README.md` commands).

- [ ] **Step 1: Write failing tests in `tests/test_packaging_and_report.py` (including Review Focus #5)**

```python
import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image
from docx import Document
from report.build_report_docx import build_report_docx
from src.build_submission import build_submission_zip
from scripts.download_checkpoints import verify_sha256


def test_build_submission_enforces_size_limit_and_run_name_consistency(tmp_path):
    # Minimal compliant repo tree
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "results" / "logs").mkdir(parents=True)
    (tmp_path / "submission_pack" / "test_pack" / "img").mkdir(parents=True)
    (tmp_path / "submission_pack" / "test_pack" / "xml").mkdir(parents=True)
    (tmp_path / "submission_pack" / "predictions").mkdir(parents=True)
    (tmp_path / "README.md").write_text("# CS460 Project\n")
    (tmp_path / "requirements.txt").write_text("torch>=2.2.0\n")
    (tmp_path / "src" / "train.py").write_text("# train\n")
    (tmp_path / "configs" / "res_800.yaml").write_text("run_name: res_800\n")
    (tmp_path / "results" / "logs" / "res_800_epoch_log.csv").write_text("run_name,epoch,train_total_loss\nres_800,1,0.5\n")
    Image.new("RGB", (64, 64)).save(tmp_path / "submission_pack" / "test_pack" / "img" / "00001.jpg")
    (tmp_path / "submission_pack" / "test_pack" / "xml" / "00001.xml").write_text("<annotation/>")
    (tmp_path / "submission_pack" / "predictions" / "test_pack_predictions.csv").write_text("sample_id,run_name\n00001,res_800\n")
    report_pdf = tmp_path / "report.pdf"
    report_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    out_zip = tmp_path / "sub.zip"
    info = build_submission_zip(tmp_path, out_zip, report_pdf)
    assert out_zip.exists() and info["size_bytes"] <= 50 * 1024 * 1024

    # Drift in run_name must fail hard
    (tmp_path / "results" / "logs" / "res_800_epoch_log.csv").write_text("run_name,epoch,train_total_loss\nwrong_run,1,0.5\n")
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)

    # Exceeding max_bytes must fail hard
    (tmp_path / "results" / "logs" / "res_800_epoch_log.csv").write_text("run_name,epoch,train_total_loss\nres_800,1,0.5\n")
    with pytest.raises(ValueError, match="exceeds"):
        build_submission_zip(tmp_path, out_zip, report_pdf, max_bytes=100)


def test_build_report_docx_and_sha256_helper(tmp_path):
    tpl = tmp_path / "template.docx"
    doc = Document()
    doc.add_paragraph("2609 CS460")
    doc.add_heading("5 Results and Discussion", level=1)
    t = doc.add_table(rows=2, cols=4)
    t.rows[0].cells[0].text = "Approach"
    doc.save(tpl)

    ev = tmp_path / "res_800_test.json"
    ev.write_text(json.dumps({
        "run_name": "res_800", "ap50": 0.72, "ap75": 0.41, "ap50_95": 0.40,
        "fixed_threshold": {"precision": 0.80, "recall": 0.70, "tp": 1571, "fp": 392, "fn": 674},
        "by_scale": {"tiny": {"recall": 0.50, "tp": 300, "gt_count": 600}},
        "latency": {"mean_ms": 25.0, "fps": 40.0},
    }))
    fig = tmp_path / "loss.png"
    Image.new("RGB", (200, 100)).save(fig)
    pred_csv = tmp_path / "preds.csv"
    pred_csv.write_text("sample_id,run_name,gt_count,pred_count,tp,fp,fn\n00001,res_800,1,1,1,0,0\n")

    out_docx = tmp_path / "draft.docx"
    build_report_docx(tpl, [ev], fig, fig, [fig], pred_csv, out_docx)
    assert out_docx.exists() and out_docx.stat().st_size > 1000
    assert verify_sha256(out_docx, hashlib.sha256(out_docx.read_bytes()).hexdigest())
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `.venv/bin/pytest tests/test_packaging_and_report.py -v`  
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement report builder, submission packager, checkpoint downloader, Colab/Kaggle notebook, `docs/spot_check_guide.md`, and `README.md`**

Implement `report/build_report_docx.py`, `src/build_submission.py`, `scripts/download_checkpoints.py`, `checkpoints_manifest.json` (placeholder entries ready to populate after Phase 3 GPU runs), `notebooks/run_experiments.ipynb` (clean JSON notebook auto-detecting Colab `/content/drive/MyDrive/CV_project` vs Kaggle `/kaggle/working`), `docs/spot_check_guide.md` (referencing exact lines of `src/train.py` and exact demo commands), and `README.md` (complete CS460 checklist sections including AI-assistance disclosure).

- [ ] **Step 4: Run full test suite & end-to-end local smoke verification**

Run: `.venv/bin/pytest -v`  
Expected: All unit & integration tests pass (`12+ passed`, `0 failed`).  
Run end-to-end synthetic smoke test (`src.train --smoke-test` → `src.eval` → `src.plot_curves` → `src.predict_pack` → `src.predict_single`) verifying every CLI command exits 0 cleanly.

- [ ] **Step 5: Commit, push to GitHub, and sync code & docs snapshot to Google Drive `CV_project/`**

Commit all code/tests/docs/notebook cleanly to `Lc-wing/CV_final_project` (`main`), push to `origin main`, and upload `notebooks/run_experiments.ipynb` (with Colab MIME type `application/vnd.google.colaboratory`), `docs/spot_check_guide.md`, and the plan/code archive to Google Drive `CV_project/code/` and `CV_project/docs/`.
