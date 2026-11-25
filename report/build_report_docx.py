"""Build a CS460 Final Project Report draft (.docx) from the official Word template."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Inches

WORKING_TITLE = (
    "Small Targets at Speed: Input Resolution and RPN Anchor Scale Sensitivity "
    "in Vision-Based Anti-UAV Detection"
)

KNOWN_APPROACH_METADATA: dict[str, tuple[str, str]] = {
    "res_800": (
        "res_800 (Baseline)",
        "min_size=800, max_size=1333; anchors=(32, 64, 128, 256, 512); "
        "Faster R-CNN MobileNetV3-Large-FPN, trainable_backbone_layers=3, seed=42",
    ),
    "res_480": (
        "res_480 (Variant 1)",
        "min_size=480, max_size=800; anchors=(32, 64, 128, 256, 512); "
        "all other training/eval hyperparameters identical to res_800",
    ),
    "res_480_small_anchor": (
        "res_480_small_anchor (Variant 2)",
        "min_size=480, max_size=800; anchors=(16, 32, 64, 128, 256); "
        "pretrained RPN conv weights preserved (same num_anchors_per_location)",
    ),
}

SCALE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("tiny", "Tiny Recall (TP/GT)"),
    ("small", "Small Recall (TP/GT)"),
    ("medium_large", "Medium/Large Recall (TP/GT)"),
)


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any, field: str, source: Path) -> float:
    if not _is_strict_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"Expected finite numeric '{field}' in '{source.name}', got {value!r}")
    return float(value)


def _load_eval_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Evaluation JSON '{path.name}' must be an object")

    run_name = raw.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f"Evaluation JSON '{path.name}' missing non-empty string 'run_name'")
    run_name = run_name.strip()

    ap50 = _as_float(raw.get("ap50"), "ap50", path)
    ap75 = _as_float(raw.get("ap75"), "ap75", path)
    ap50_95 = _as_float(raw.get("ap50_95"), "ap50_95", path)

    ft = raw.get("fixed_threshold")
    if not isinstance(ft, dict):
        raise ValueError(f"Evaluation JSON '{path.name}' missing 'fixed_threshold' object")
    precision = _as_float(ft.get("precision"), "fixed_threshold.precision", path)
    recall = _as_float(ft.get("recall"), "fixed_threshold.recall", path)
    tp = int(ft.get("tp", 0))
    fp = int(ft.get("fp", 0))
    fn = int(ft.get("fn", 0))

    latency = raw.get("latency")
    if not isinstance(latency, dict):
        raise ValueError(f"Evaluation JSON '{path.name}' missing 'latency' object")
    mean_ms = _as_float(latency.get("mean_ms"), "latency.mean_ms", path)
    fps = _as_float(latency.get("fps"), "latency.fps", path)

    raw_scale = raw.get("by_scale")
    if not isinstance(raw_scale, dict):
        raise ValueError(f"Evaluation JSON '{path.name}' missing 'by_scale' object")

    by_scale: dict[str, dict[str, Any]] = {}
    for bin_key, _ in SCALE_COLUMNS:
        bin_info = raw_scale.get(bin_key)
        if isinstance(bin_info, dict) and "recall" in bin_info:
            bin_recall = _as_float(bin_info["recall"], f"by_scale.{bin_key}.recall", path)
            bin_tp = int(bin_info.get("tp", 0))
            bin_gt = int(bin_info.get("gt_count", 0))
            by_scale[bin_key] = {"recall": bin_recall, "tp": bin_tp, "gt_count": bin_gt}

    return {
        "run_name": run_name,
        "json_file": path.name,
        "log_file": f"{run_name}_epoch_log.csv",
        "ap50": ap50,
        "ap75": ap75,
        "ap50_95": ap50_95,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "by_scale": by_scale,
        "mean_ms": mean_ms,
        "fps": fps,
    }


def _load_predictions_summary(pred_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not pred_csv.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")
    with pred_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV '{pred_csv.name}' is empty (missing header)")
        for required_col in ("sample_id", "run_name"):
            if required_col not in reader.fieldnames:
                raise ValueError(
                    f"Prediction CSV '{pred_csv.name}' missing required column '{required_col}'"
                )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Prediction CSV '{pred_csv.name}' contains no prediction rows")

    sample_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        sid = (row.get("sample_id") or "").strip()
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            sample_ids.append(sid)
    return sample_ids, rows


def _format_scale_cell(by_scale: dict[str, dict[str, Any]], bin_key: str) -> str:
    info = by_scale.get(bin_key)
    if info is None:
        return "N/A"
    return f"{info['recall']:.3f} ({info['tp']}/{info['gt_count']})"


def _format_evidence_summary(rec: dict[str, Any]) -> str:
    tiny_str = _format_scale_cell(rec["by_scale"], "tiny")
    return (
        f"AP50={rec['ap50']:.3f}, AP75={rec['ap75']:.3f}, AP50:95={rec['ap50_95']:.3f}; "
        f"P={rec['precision']:.3f}, R={rec['recall']:.3f} "
        f"(TP={rec['tp']}, FP={rec['fp']}, FN={rec['fn']}); "
        f"Tiny R={tiny_str}; {rec['mean_ms']:.1f} ms/img ({rec['fps']:.1f} FPS)"
    )


def _find_comparison_table(doc: Document) -> Any | None:
    for table in doc.tables:
        if len(table.columns) >= 4 and len(table.rows) >= 1:
            first_header = table.rows[0].cells[0].text.strip().lower()
            if "approach" in first_header:
                return table
    return None


def _populate_comparison_table(doc: Document, records: list[dict[str, Any]]) -> Any:
    table = _find_comparison_table(doc)
    if table is None:
        doc.add_heading("Approach Comparison Summary", level=2)
        table = doc.add_table(rows=1 + len(records), cols=4)

    headers = ("Approach", "Design choices", "Evidence or result", "Interpretation")
    for col_idx, header in enumerate(headers):
        table.rows[0].cells[col_idx].text = header

    target_rows = 1 + len(records)
    while len(table.rows) < target_rows:
        table.add_row()
    while len(table.rows) > target_rows:
        row_to_remove = table.rows[-1]
        table._tbl.remove(row_to_remove._tr)

    for row_idx, rec in enumerate(records, start=1):
        run_name = rec["run_name"]
        approach_label, design_desc = KNOWN_APPROACH_METADATA.get(
            run_name,
            (run_name, f"Controlled Faster R-CNN run '{run_name}' (see configs/{run_name}.yaml)"),
        )
        row_cells = table.rows[row_idx].cells
        row_cells[0].text = approach_label
        row_cells[1].text = design_desc
        row_cells[2].text = _format_evidence_summary(rec)
        row_cells[3].text = (
            f"[YOUR OWN WORDS] Interpret {run_name} performance across Tiny/Small/Medium-Large "
            f"scales, false positives, and inference latency ({rec['mean_ms']:.1f} ms/img)."
        )
    return table


def _find_paragraph_matching(doc: Document, needle: str) -> Any | None:
    lowered = needle.lower()
    for para in doc.paragraphs:
        if lowered in para.text.lower():
            return para
    return None


def _move_after(anchor_element: Any, new_element: Any) -> Any:
    """Move an existing OOXML body child directly after ``anchor_element``."""
    anchor_element.addnext(new_element)
    return new_element


def _insert_training_record_section(
    doc: Document,
    records: list[dict[str, Any]],
    loss_figure: Path,
) -> None:
    run_names = ", ".join(rec["run_name"] for rec in records)
    log_files = ", ".join(rec["log_file"] for rec in records)

    caption_text = (
        f"Figure 1. Training loss curves regenerated directly from raw machine-generated CSV logs "
        f"({log_files}) for runs [{run_names}]. All runs train Faster R-CNN MobileNetV3-Large-FPN "
        f"with trainable_backbone_layers=3 (plus FPN, RPN, and RoI box heads) under fixed seed=42."
    )
    prompt_text = (
        f"[YOUR OWN WORDS] Examine a specific epoch or step interval on the training-loss curve "
        f"({log_files}): describe what changed during warmup vs. cosine decay, explain one training "
        f"decision (e.g., gradient clipping, LR warmup, or best-AP50 checkpoint selection), and cite "
        f"the logged values that supported that decision."
    )

    placeholder = _find_paragraph_matching(doc, "insert your training-loss curve")
    if placeholder is not None:
        img_para = placeholder.insert_paragraph_before()
        img_para.add_run().add_picture(str(loss_figure), width=Inches(6.0))
        placeholder.text = caption_text
        prompt_para = doc.add_paragraph(prompt_text)
        _move_after(placeholder._p, prompt_para._p)
        return

    heading = _find_paragraph_matching(doc, "training record")
    if heading is not None:
        img_para = doc.add_paragraph()
        img_para.add_run().add_picture(str(loss_figure), width=Inches(6.0))
        cap_para = doc.add_paragraph(caption_text)
        prompt_para = doc.add_paragraph(prompt_text)
        anchor = _move_after(heading._p, img_para._p)
        anchor = _move_after(anchor, cap_para._p)
        _move_after(anchor, prompt_para._p)
        return

    doc.add_heading("Training Loss Record", level=2)
    img_para = doc.add_paragraph()
    img_para.add_run().add_picture(str(loss_figure), width=Inches(6.0))
    doc.add_paragraph(caption_text)
    doc.add_paragraph(prompt_text)


def _insert_scale_summary_and_figure(
    doc: Document,
    comparison_table: Any,
    records: list[dict[str, Any]],
    scale_figure: Path,
) -> None:
    table_caption = doc.add_paragraph(
        "Table 2. Scale-stratified test-set recall (computed strictly in original image pixel "
        "coordinates from train+val scale bins), AP50, and single-image inference latency."
    )
    scale_table = doc.add_table(rows=1 + len(records), cols=6)
    scale_headers = (
        "Run",
        "Tiny Recall (TP/GT)",
        "Small Recall (TP/GT)",
        "Medium/Large Recall (TP/GT)",
        "AP50",
        "Latency (ms/img)",
    )
    for col_idx, header in enumerate(scale_headers):
        scale_table.rows[0].cells[col_idx].text = header

    for row_idx, rec in enumerate(records, start=1):
        cells = scale_table.rows[row_idx].cells
        cells[0].text = rec["run_name"]
        cells[1].text = _format_scale_cell(rec["by_scale"], "tiny")
        cells[2].text = _format_scale_cell(rec["by_scale"], "small")
        cells[3].text = _format_scale_cell(rec["by_scale"], "medium_large")
        cells[4].text = f"{rec['ap50']:.3f}"
        cells[5].text = f"{rec['mean_ms']:.2f}"

    fig_para = doc.add_paragraph()
    fig_para.add_run().add_picture(str(scale_figure), width=Inches(6.0))
    eval_files = ", ".join(rec["json_file"] for rec in records)
    fig_caption = doc.add_paragraph(
        f"Figure 2. Scale-stratified recall (Tiny / Small / Medium-Large in original image "
        f"coordinates) and single-image inference latency across runs ({eval_files})."
    )

    anchor = comparison_table._tbl
    for element in (table_caption._p, scale_table._tbl, fig_para._p, fig_caption._p):
        anchor = _move_after(anchor, element)


def _insert_qualitative_sample_section(
    doc: Document,
    records: list[dict[str, Any]],
    sample_visuals: list[Path],
    sample_ids: list[str],
    pred_rows: list[dict[str, str]],
    pred_csv: Path,
) -> None:
    run_names = ", ".join(rec["run_name"] for rec in records)
    featured_ids = sample_ids[:5]
    ids_label = ", ".join(featured_ids)

    # Summarize per-sample counts for the first few sample IDs from test_pack_predictions.csv
    row_lines: list[str] = []
    for row in pred_rows:
        sid = (row.get("sample_id") or "").strip()
        if sid not in featured_ids:
            continue
        rname = (row.get("run_name") or "").strip()
        gt_c = row.get("gt_count", "?")
        pred_c = row.get("pred_count", "?")
        tp = row.get("tp", "?")
        fp = row.get("fp", "?")
        fn = row.get("fn", "?")
        row_lines.append(
            f"sample_id={sid} | run={rname} | GT={gt_c}, Pred={pred_c} (TP={tp}, FP={fp}, FN={fn})"
        )

    new_elements: list[Any] = []
    anchor_para = _find_paragraph_matching(doc, "explain your own interpretation")
    if anchor_para is None:
        anchor_para = _find_paragraph_matching(doc, "interpretation and exploration")

    if anchor_para is None:
        heading = doc.add_heading("Qualitative Test-Pack Comparisons", level=2)
        new_elements.append(heading._p)

    for idx, vis_path in enumerate(sample_visuals, start=1):
        img_p = doc.add_paragraph()
        img_p.add_run().add_picture(str(vis_path), width=Inches(6.0))
        cap_p = doc.add_paragraph(
            f"Figure {2 + idx}. Qualitative comparison panel ({vis_path.name}) across [{run_names}]. "
            f"Overlays show Ground Truth (green solid), True Positive (blue solid), "
            f"False Positive (red dashed), and missed False Negative (amber)."
        )
        new_elements.extend([img_p._p, cap_p._p])

    summary_p = doc.add_paragraph(
        f"Test-pack prediction reference ({pred_csv.name}; {len(sample_ids)} sample(s) total, "
        f"highlighting sample IDs [{ids_label}]): "
        + ("; ".join(row_lines[:9]) if row_lines else f"sample IDs {ids_label}.")
    )
    prompt_p = doc.add_paragraph(
        f"[YOUR OWN WORDS] Discuss concrete test-pack sample ID(s) [{ids_label}] from "
        f"{pred_csv.name}: compare ground truth against saved predictions across [{run_names}], "
        f"identify any Tiny-UAV recovery or shared background false alarm, and state how this "
        f"evidence refines your explanation of resolution vs. anchor scale."
    )
    new_elements.extend([summary_p._p, prompt_p._p])

    if anchor_para is not None:
        anchor = anchor_para._p
        for el in new_elements:
            anchor = _move_after(anchor, el)


def build_report_docx(
    template_path: Path | str,
    eval_jsons: Sequence[Path | str],
    loss_figure: Path | str,
    scale_figure: Path | str,
    sample_visuals: Sequence[Path | str],
    pred_csv: Path | str,
    out_docx: Path | str,
) -> Path:
    """Populate the official CS460 Word template with verified metrics, tables, and figures.

    Preserves the template's header, styles, and section hierarchy while inserting
    populated comparison and scale-stratified tables, embedded figures referencing
    exact run names and raw CSV log filenames, test-pack sample references, and
    ``[YOUR OWN WORDS]`` prompts for personal interpretation.
    """
    tpl_path = Path(template_path)
    if not tpl_path.is_file():
        raise FileNotFoundError(f"Report template .docx not found: {tpl_path}")

    eval_paths = [Path(p) for p in eval_jsons]
    if not eval_paths:
        raise ValueError("eval_jsons must contain at least one evaluation JSON path")

    loss_path = Path(loss_figure)
    scale_path = Path(scale_figure)
    pred_path = Path(pred_csv)
    out_path = Path(out_docx)
    visual_paths = [Path(p) for p in sample_visuals]

    if not loss_path.is_file():
        raise FileNotFoundError(f"Training loss figure not found: {loss_path}")
    if not scale_path.is_file():
        raise FileNotFoundError(f"Scale/latency comparison figure not found: {scale_path}")
    for vis_path in visual_paths:
        if not vis_path.is_file():
            raise FileNotFoundError(f"Sample visual figure not found: {vis_path}")

    records: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for ev_path in eval_paths:
        rec = _load_eval_record(ev_path)
        if rec["run_name"] in seen_runs:
            raise ValueError(f"Duplicate run_name {rec['run_name']!r} in eval_jsons")
        seen_runs.add(rec["run_name"])
        records.append(rec)

    sample_ids, pred_rows = _load_predictions_summary(pred_path)

    doc = Document(str(tpl_path))

    title_placeholder = _find_paragraph_matching(doc, "[title of your investigation]")
    if title_placeholder is not None:
        title_placeholder.text = WORKING_TITLE

    comparison_table = _populate_comparison_table(doc, records)
    _insert_training_record_section(doc, records, loss_path)
    _insert_scale_summary_and_figure(doc, comparison_table, records, scale_path)
    _insert_qualitative_sample_section(
        doc,
        records,
        visual_paths,
        sample_ids,
        pred_rows,
        pred_path,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m report.build_report_docx``."""
    parser = argparse.ArgumentParser(
        description="Build a CS460 Final Project Report draft (.docx) from the official template."
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Path to CS460_Computer_Vision_Final_Report_Template.docx.",
    )
    parser.add_argument(
        "--eval-jsons",
        nargs="+",
        type=Path,
        required=True,
        help="Evaluation JSON files produced by src.eval.",
    )
    parser.add_argument(
        "--loss-figure",
        type=Path,
        required=True,
        help="Training loss curve PNG produced by src.plot_curves.",
    )
    parser.add_argument(
        "--scale-figure",
        type=Path,
        required=True,
        help="Scale-recall & latency comparison PNG produced by src.plot_curves.",
    )
    parser.add_argument(
        "--sample-visuals",
        nargs="*",
        type=Path,
        default=[],
        help="Qualitative comparison PNGs produced by src.predict_pack.",
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="Test-pack predictions CSV produced by src.predict_pack.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the populated draft .docx file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building the official CS460 Word report draft."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out = build_report_docx(
        template_path=args.template,
        eval_jsons=args.eval_jsons,
        loss_figure=args.loss_figure,
        scale_figure=args.scale_figure,
        sample_visuals=args.sample_visuals,
        pred_csv=args.pred_csv,
        out_docx=args.out,
    )
    print(f"Built CS460 report draft -> {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
