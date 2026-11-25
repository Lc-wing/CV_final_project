"""Tests for Task 7: report .docx builder, <=50 MB submission packager, checkpoints, notebook, and docs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from docx import Document
from PIL import Image

from report.build_report_docx import build_report_docx
from scripts.download_checkpoints import download_checkpoints, verify_sha256
from src.build_submission import build_submission_zip

REPO_ROOT = Path(__file__).resolve().parents[1]


def _create_minimal_compliant_repo(root: Path) -> Path:
    """Create a minimal CS460-compliant repository layout and dummy report PDF."""
    (root / "configs").mkdir()
    (root / "src").mkdir()
    (root / "results" / "logs").mkdir(parents=True)
    (root / "submission_pack" / "test_pack" / "img").mkdir(parents=True)
    (root / "submission_pack" / "test_pack" / "xml").mkdir(parents=True)
    (root / "submission_pack" / "predictions").mkdir(parents=True)

    (root / "README.md").write_text("# CS460 Project\n", encoding="utf-8")
    (root / "requirements.txt").write_text("torch>=2.2.0\n", encoding="utf-8")
    (root / "src" / "train.py").write_text("# train\n", encoding="utf-8")
    (root / "configs" / "res_800.yaml").write_text("run_name: res_800\n", encoding="utf-8")
    (root / "results" / "logs" / "res_800_epoch_log.csv").write_text(
        "run_name,epoch,train_total_loss\nres_800,1,0.5\n",
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64)).save(root / "submission_pack" / "test_pack" / "img" / "00001.jpg")
    (root / "submission_pack" / "test_pack" / "xml" / "00001.xml").write_text(
        "<annotation/>",
        encoding="utf-8",
    )
    (root / "submission_pack" / "predictions" / "test_pack_predictions.csv").write_text(
        "sample_id,run_name\n00001,res_800\n",
        encoding="utf-8",
    )

    report_pdf = root / "report.pdf"
    report_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return report_pdf


def test_build_submission_enforces_size_limit_and_run_name_consistency(tmp_path: Path) -> None:
    report_pdf = _create_minimal_compliant_repo(tmp_path)

    # Large dummy weights in checkpoints/ and stray .pt/.pth must never be bundled.
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "res_800_best.pt").write_bytes(b"\x00" * 4096)
    (tmp_path / "results" / "stray_weights.pth").write_bytes(b"\x00" * 2048)

    out_zip = tmp_path / "sub.zip"
    info = build_submission_zip(tmp_path, out_zip, report_pdf)
    assert out_zip.exists() and info["size_bytes"] <= 50 * 1024 * 1024
    assert info["size_bytes"] == out_zip.stat().st_size

    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
    assert "report.pdf" in names
    assert "README.md" in names
    assert "configs/res_800.yaml" in names
    assert "results/logs/res_800_epoch_log.csv" in names
    assert "submission_pack/predictions/test_pack_predictions.csv" in names
    assert not any(name.endswith((".pt", ".pth")) for name in names)
    assert not any(name.startswith("checkpoints/") for name in names)

    # Drift in raw CSV log run_name must fail hard
    (tmp_path / "results" / "logs" / "res_800_epoch_log.csv").write_text(
        "run_name,epoch,train_total_loss\nwrong_run,1,0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)

    # Restore epoch log and verify drift in predictions CSV run_name fails hard
    (tmp_path / "results" / "logs" / "res_800_epoch_log.csv").write_text(
        "run_name,epoch,train_total_loss\nres_800,1,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "submission_pack" / "predictions" / "test_pack_predictions.csv").write_text(
        "sample_id,run_name\n00001,ghost_run\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)

    # Restore predictions CSV and verify config stem vs run_name drift fails hard
    (tmp_path / "submission_pack" / "predictions" / "test_pack_predictions.csv").write_text(
        "sample_id,run_name\n00001,res_800\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "res_800.yaml").write_text("run_name: mismatched\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)

    # Restore config and verify exceeding max_bytes fails hard and deletes out_zip
    (tmp_path / "configs" / "res_800.yaml").write_text("run_name: res_800\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        build_submission_zip(tmp_path, out_zip, report_pdf, max_bytes=100)
    assert not out_zip.exists()


def test_build_submission_rejects_missing_required_deliverable(tmp_path: Path) -> None:
    report_pdf = _create_minimal_compliant_repo(tmp_path)
    (tmp_path / "README.md").unlink()
    with pytest.raises(FileNotFoundError, match="README.md"):
        build_submission_zip(tmp_path, tmp_path / "sub.zip", report_pdf)


def test_build_report_docx_and_sha256_helper(tmp_path: Path) -> None:
    tpl = tmp_path / "template.docx"
    doc = Document()
    doc.add_paragraph("2609 CS460")
    doc.add_heading("5 Results and Discussion", level=1)
    t = doc.add_table(rows=2, cols=4)
    t.rows[0].cells[0].text = "Approach"
    doc.save(tpl)

    ev = tmp_path / "res_800_test.json"
    ev.write_text(
        json.dumps(
            {
                "run_name": "res_800",
                "ap50": 0.72,
                "ap75": 0.41,
                "ap50_95": 0.40,
                "fixed_threshold": {
                    "precision": 0.80,
                    "recall": 0.70,
                    "tp": 1571,
                    "fp": 392,
                    "fn": 674,
                },
                "by_scale": {"tiny": {"recall": 0.50, "tp": 300, "gt_count": 600}},
                "latency": {"mean_ms": 25.0, "fps": 40.0},
            }
        ),
        encoding="utf-8",
    )
    fig = tmp_path / "loss.png"
    Image.new("RGB", (200, 100)).save(fig)
    pred_csv = tmp_path / "preds.csv"
    pred_csv.write_text(
        "sample_id,run_name,gt_count,pred_count,tp,fp,fn\n00001,res_800,1,1,1,0,0\n",
        encoding="utf-8",
    )

    out_docx = tmp_path / "draft.docx"
    returned = build_report_docx(tpl, [ev], fig, fig, [fig], pred_csv, out_docx)
    assert returned == out_docx
    assert out_docx.exists() and out_docx.stat().st_size > 1000
    assert verify_sha256(out_docx, hashlib.sha256(out_docx.read_bytes()).hexdigest())
    assert not verify_sha256(out_docx, "0" * 64)

    built = Document(out_docx)
    all_text = "\n".join(p.text for p in built.paragraphs)
    assert "2609 CS460" in all_text
    assert "res_800" in all_text
    assert "res_800_epoch_log.csv" in all_text
    assert "00001" in all_text
    assert "[YOUR OWN WORDS]" in all_text
    # Embedded loss_figure + scale_figure + 1 sample visual
    assert len(built.inline_shapes) == 3
    # Existing 4-column table populated + scale-stratified summary table added
    assert len(built.tables) >= 2
    assert "res_800" in built.tables[0].rows[1].cells[0].text
    assert "0.720" in built.tables[0].rows[1].cells[2].text
    scale_table = built.tables[1]
    assert "Tiny Recall" in scale_table.rows[0].cells[1].text
    assert "0.500 (300/600)" in scale_table.rows[1].cells[1].text


def test_build_report_docx_populates_official_template_structure(tmp_path: Path) -> None:
    tpl = tmp_path / "official_like.docx"
    doc = Document()
    doc.add_paragraph("2609 CS460 ")
    doc.add_paragraph("[Title of your investigation]")
    doc.add_heading("4  Experimental Setup and Evaluation", level=1)
    doc.add_heading("Training record from your experiment", level=2)
    doc.add_paragraph("[Insert your training-loss curve here.]")
    doc.add_heading("Evaluation", level=2)
    doc.add_heading("5  Results and Discussion", level=1)
    tbl = doc.add_table(rows=3, cols=4)
    tbl.rows[0].cells[0].text = "Approach"
    tbl.rows[0].cells[1].text = "Design choices"
    tbl.rows[0].cells[2].text = "Evidence or result"
    tbl.rows[0].cells[3].text = "Interpretation"
    doc.add_heading("Interpretation and exploration", level=2)
    doc.add_paragraph("[Explain your own interpretation.]")
    doc.save(tpl)

    eval_paths: list[Path] = []
    for run_name, ap50, tiny_r, ms in [
        ("res_800", 0.74, 0.55, 24.5),
        ("res_480", 0.65, 0.35, 13.0),
        ("res_480_small_anchor", 0.70, 0.49, 13.4),
    ]:
        p = tmp_path / f"{run_name}_test.json"
        p.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "ap50": ap50,
                    "ap75": ap50 - 0.2,
                    "ap50_95": ap50 - 0.25,
                    "fixed_threshold": {"precision": 0.8, "recall": 0.7, "tp": 100, "fp": 25, "fn": 43},
                    "by_scale": {
                        "tiny": {"recall": tiny_r, "tp": int(tiny_r * 100), "gt_count": 100},
                        "small": {"recall": 0.75, "tp": 75, "gt_count": 100},
                        "medium_large": {"recall": 0.90, "tp": 90, "gt_count": 100},
                    },
                    "latency": {"mean_ms": ms, "fps": 1000.0 / ms},
                }
            ),
            encoding="utf-8",
        )
        eval_paths.append(p)

    loss_fig = tmp_path / "training_loss_curves.png"
    scale_fig = tmp_path / "scale_recall_and_latency.png"
    vis_fig = tmp_path / "00010_comparison.png"
    for p in (loss_fig, scale_fig, vis_fig):
        Image.new("RGB", (240, 120)).save(p)

    pred_csv = tmp_path / "test_pack_predictions.csv"
    pred_csv.write_text(
        "sample_id,run_name,gt_count,pred_count,tp,fp,fn\n"
        "00010,res_800,1,1,1,0,0\n"
        "00010,res_480,1,0,0,0,1\n"
        "00010,res_480_small_anchor,1,1,1,0,0\n",
        encoding="utf-8",
    )

    out_docx = tmp_path / "official_draft.docx"
    build_report_docx(tpl, eval_paths, loss_fig, scale_fig, [vis_fig], pred_csv, out_docx)
    built = Document(out_docx)

    assert len(built.tables[0].rows) == 4  # header + 3 runs
    assert "res_480_small_anchor" in built.tables[0].rows[3].cells[0].text
    assert len(built.tables[1].rows) == 4  # header + 3 runs
    assert len(built.inline_shapes) == 3
    all_text = "\n".join(p.text for p in built.paragraphs)
    assert "Small Targets at Speed" in all_text
    assert "res_480_small_anchor_epoch_log.csv" in all_text
    assert "00010" in all_text


def test_download_checkpoints_manifest_placeholder_and_local_file(tmp_path: Path) -> None:
    # Repository manifest must declare all 3 controlled runs with placeholders
    repo_manifest = REPO_ROOT / "checkpoints_manifest.json"
    assert repo_manifest.is_file()
    manifest_data = json.loads(repo_manifest.read_text(encoding="utf-8"))
    entries = manifest_data["checkpoints"]
    assert {e["run_name"] for e in entries} == {"res_800", "res_480", "res_480_small_anchor"}

    # Downloading placeholder manifest must fail cleanly with actionable message
    with pytest.raises(ValueError, match="TBD_AFTER_PHASE_3_TRAINING|placeholder"):
        download_checkpoints(repo_manifest, tmp_path / "ckpts")

    # Local file:// download with valid SHA-256 succeeds and verifies checksum
    src_file = tmp_path / "res_800_best.pt"
    payload = b"fake-checkpoint-bytes-for-sha256-test"
    src_file.write_bytes(payload)
    good_sha = hashlib.sha256(payload).hexdigest()

    good_manifest = tmp_path / "good_manifest.json"
    good_manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_name": "res_800",
                        "filename": "res_800_best.pt",
                        "url": src_file.as_uri(),
                        "sha256": good_sha.upper(),  # test case-insensitivity
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "downloaded"
    downloaded = download_checkpoints(good_manifest, out_dir, run_names=["res_800"])
    assert downloaded["res_800"] == out_dir / "res_800_best.pt"
    assert verify_sha256(downloaded["res_800"], good_sha)

    # Corrupted SHA-256 must fail hard and not leave bad checkpoint in output_dir
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_name": "res_480",
                        "filename": "res_480_best.pt",
                        "url": src_file.as_uri(),
                        "sha256": "f" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        download_checkpoints(bad_manifest, out_dir)
    assert not (out_dir / "res_480_best.pt").exists()


def test_notebook_and_documentation_artifacts_are_valid() -> None:
    nb_path = REPO_ROOT / "notebooks" / "run_experiments.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 6
    full_source = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
    )
    assert "/content/drive/MyDrive/CV_project" in full_source
    assert "/kaggle/working" in full_source
    for cmd in (
        "src.prepare_data",
        "src.train",
        "src.eval",
        "src.plot_curves",
        "src.predict_pack",
        "src.predict_single",
    ):
        assert cmd in full_source

    guide_path = REPO_ROOT / "docs" / "spot_check_guide.md"
    assert guide_path.is_file()
    guide_text = guide_path.read_text(encoding="utf-8")
    assert "optimizer.zero_grad" in guide_text
    assert "GeneralizedRCNNTransform" in guide_text
    assert "src.plot_curves" in guide_text
    assert "src.predict_single" in guide_text

    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file()
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "AI Assistance Disclosure" in readme_text
    assert "aegean-ai/dut-anti-uav" in readme_text
    assert "scripts.download_checkpoints" in readme_text
    assert "src.build_submission" in readme_text
    assert "report.build_report_docx" in readme_text


def test_build_submission_checks_step_logs_eval_jsons_and_cli(tmp_path: Path) -> None:
    from src.build_submission import main as submission_main

    report_pdf = _create_minimal_compliant_repo(tmp_path)
    (tmp_path / "results" / "metrics").mkdir(parents=True)
    (tmp_path / "results" / "logs" / "res_800_step_log.csv").write_text(
        "run_name,epoch,step,global_step,total_loss\nres_800,1,1,1,0.4\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "metrics" / "res_800_test.json").write_text(
        json.dumps({"run_name": "res_800", "ap50": 0.7}),
        encoding="utf-8",
    )

    out_zip = tmp_path / "cli_sub.zip"
    rc = submission_main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-pdf",
            str(report_pdf),
            "--out-zip",
            str(out_zip),
        ]
    )
    assert rc == 0 and out_zip.is_file()

    # Step log run_name drift must fail hard
    (tmp_path / "results" / "logs" / "res_800_step_log.csv").write_text(
        "run_name,epoch,step,global_step,total_loss\nwrong_step_run,1,1,1,0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)

    # Restore step log and test eval JSON run_name drift
    (tmp_path / "results" / "logs" / "res_800_step_log.csv").write_text(
        "run_name,epoch,step,global_step,total_loss\nres_800,1,1,1,0.4\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "metrics" / "res_800_test.json").write_text(
        json.dumps({"run_name": "other_run", "ap50": 0.7}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_name"):
        build_submission_zip(tmp_path, out_zip, report_pdf)


def test_download_checkpoints_cli_verify_only_and_cache_hit(tmp_path: Path) -> None:
    from scripts.download_checkpoints import main as dl_main

    out_dir = tmp_path / "ckpts"
    out_dir.mkdir()
    payload = b"cached-checkpoint-content"
    cached = out_dir / "res_800_best.pt"
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_name": "res_800",
                        "filename": "res_800_best.pt",
                        "url": "https://example.invalid/unreachable.pt",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Cache hit succeeds without making any network request
    rc = dl_main(["--manifest", str(manifest), "--out-dir", str(out_dir)])
    assert rc == 0

    # --verify-only also succeeds cleanly
    rc_verify = dl_main(
        ["--manifest", str(manifest), "--out-dir", str(out_dir), "--verify-only"]
    )
    assert rc_verify == 0

    # Path-traversal filename in manifest must be rejected
    bad_name_manifest = tmp_path / "traversal_manifest.json"
    bad_name_manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_name": "res_800",
                        "filename": "../escape.pt",
                        "url": "https://example.invalid/x.pt",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="plain filename"):
        download_checkpoints(bad_name_manifest, out_dir)


