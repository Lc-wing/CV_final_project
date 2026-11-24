# CS460 Computer Vision Final Project — Approved Design Spec

> **Working Title**: *Small Targets at Speed: Input Resolution and RPN Anchor Scale Sensitivity in Vision-Based Anti-UAV Detection*  
> **Repository**: `Lc-wing/CV_final_project`  
> **Google Drive Mirror**: `MyDrive/CV_project` (`https://drive.google.com/drive/folders/1iO-UKK3-DpAfsj9OokLrFWhuvh5JJHdm`)  
> **Date**: 2026-11-24

## 1. Research Question & Hypotheses

- **Task**: Single-class visible-light object detection (`UAV` vs. background) predicting bounding boxes `[x1, y1, x2, y2]` and confidence scores in original image pixel coordinates.
- **Dataset**: **DUT Anti-UAV Detection** (Zhao et al., IEEE TITS 2023; Apache-2.0; official `5,200` train / `2,600` val / `2,200` test; `5,243` / `2,621` / `2,245` boxes; 3 training images have zero objects; mirrored byte-for-byte on Hugging Face `aegean-ai/dut-anti-uav`).
- **Hypotheses**:
  1. Lowering input resolution (`res_800` -> `res_480`) speeds up inference while disproportionately reducing recall on **Tiny** UAVs (defined in original image coordinates) because tiny targets shrink below backbone stride and default 32-px RPN anchors.
  2. At `480` resolution, shifting RPN anchors down from `(32, 64, 128, 256, 512)` to `(16, 32, 64, 128, 256)` (`res_480_small_anchor`) recovers Tiny-UAV recall at negligible latency cost, potentially trading off against higher background false alarms.

## 2. Locked Decisions & Constraints

- **Compute**: Free cloud GPU (Google Colab / Kaggle Notebooks); local CPU `.venv` for unit tests and `--smoke-test`.
- **Model & Stack**: Pure PyTorch + TorchVision `fasterrcnn_mobilenet_v3_large_fpn` with explicit hand-written training loop (`zero_grad -> forward -> backward -> step`), `trainable_backbone_layers=3`, fixed `SEED=42`, AMP mixed precision on CUDA, gradient clip `max_norm=10.0`, best checkpoint selected strictly by validation `AP50`.
- **3 Controlled Runs**:
  - `res_800` *(Baseline)*: `min_size=800, max_size=1333`, anchors `(32, 64, 128, 256, 512)`
  - `res_480` *(Variant 1)*: `min_size=480, max_size=800`, anchors `(32, 64, 128, 256, 512)`
  - `res_480_small_anchor` *(Variant 2)*: `min_size=480, max_size=800`, anchors `(16, 32, 64, 128, 256)`
- **Code Layout**: Modular `src/*.py` + `configs/*.yaml` + thin Colab/Kaggle `notebooks/run_experiments.ipynb`.
- **Packaging & Storage**:
  - Submission zip strictly `<= 50 MB` (~5–10 MB without large `.pt` weights); full FP32 checkpoints hosted on GitHub Releases / Google Drive with 1-command download + SHA-256 verification.
  - Keep GitHub repository clean (no raw course zip, full dataset, or `.pt` weights committed).
  - Mirror all project-related files in Google Drive `MyDrive/CV_project/` (`course_materials`, `docs`, `code`, `data`, `checkpoints`, `results`, `report`, `submission`).
- **Report**: Draft `.docx` generated directly from official `CS460_Computer_Vision_Final_Report_Template.docx` with embedded figures/tables for user rewriting in own words and PDF export.
- **CS460 Verification & Spot-Check Rules**:
  - Identical run name across config, raw machine-generated CSV loss log, checkpoint filename, prediction CSV, and report figure caption.
  - Scale-stratified metrics defined strictly in original image coordinates from `train + val` EDA only.
  - `README.md` documents exact commands (train, regenerate curve, predict small pack), pinned versions, named-sample expected output, borrowed-code credits, and AI-assistance disclosure.
