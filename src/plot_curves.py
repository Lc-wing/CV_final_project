"""Log-based training loss and scale/latency comparison figure generator."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.eval import SCALE_BIN_NAMES

VALID_X_AXES: tuple[str, ...] = ("epoch", "step")

RUN_COLORS: dict[str, str] = {
    "res_800": "#1F77B4",
    "res_480": "#D62728",
    "res_480_small_anchor": "#2CA02C",
}
FALLBACK_COLORS: tuple[str, ...] = (
    "#1F77B4",
    "#D62728",
    "#2CA02C",
    "#9467BD",
    "#FF7F0E",
    "#17BECF",
)
SCALE_LABELS: dict[str, str] = {
    "tiny": "Tiny",
    "small": "Small",
    "medium_large": "Medium-Large",
}


def _color_for_run(run_name: str, index: int) -> str:
    if run_name in RUN_COLORS:
        return RUN_COLORS[run_name]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_finite_float(raw: str | None, field: str, path: Path, row_num: int) -> float:
    if raw is None or not raw.strip():
        raise ValueError(f"Missing '{field}' at row {row_num} in '{path.name}'")
    try:
        val = float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric '{field}'={raw!r} at row {row_num} in '{path.name}'"
        ) from exc
    if not math.isfinite(val):
        raise ValueError(f"Non-finite '{field}'={raw!r} at row {row_num} in '{path.name}'")
    return val


def _expected_run_name_from_log_path(log_path: Path, x_axis: str) -> str:
    """Derive expected ``run_name`` prefix from a raw step or epoch CSV log filename."""
    stem = log_path.stem
    if x_axis == "epoch":
        if stem.endswith("_step_log"):
            raise ValueError(
                f"Log file '{log_path.name}' is a step log, incompatible with x_axis='epoch'"
            )
        expected = stem[: -len("_epoch_log")] if stem.endswith("_epoch_log") else stem
    elif x_axis == "step":
        if stem.endswith("_epoch_log"):
            raise ValueError(
                f"Log file '{log_path.name}' is an epoch log, incompatible with x_axis='step'"
            )
        expected = stem[: -len("_step_log")] if stem.endswith("_step_log") else stem
    else:
        raise ValueError(f"x_axis must be one of {VALID_X_AXES}, got {x_axis!r}")

    if not expected:
        raise ValueError(f"Could not derive a non-empty run_name prefix from '{log_path.name}'")
    return expected


def _read_training_log(log_path: Path, x_axis: str) -> dict[str, Any]:
    """Read and validate a raw epoch or step CSV log produced by ``src.train``."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log CSV not found: {log_path}")

    expected_run_name = _expected_run_name_from_log_path(log_path, x_axis)
    x_col = "epoch" if x_axis == "epoch" else "global_step"
    y_col = "train_total_loss" if x_axis == "epoch" else "total_loss"
    required_cols = ("run_name", x_col, y_col)

    with log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Training log CSV '{log_path.name}' is empty (missing header)")
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Training log CSV '{log_path.name}' missing required columns {missing} "
                f"for x_axis={x_axis!r}; found {reader.fieldnames}"
            )

        has_val_ap50 = x_axis == "epoch" and "val_ap50" in reader.fieldnames
        xs: list[float] = []
        ys: list[float] = []
        val_ap50s: list[float] = []

        for row_num, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Malformed CSV row {row_num} with extra columns in '{log_path.name}'"
                )
            row_run_name = (row.get("run_name") or "").strip()
            if not row_run_name or row_run_name != expected_run_name:
                raise ValueError(
                    f"CSV log '{log_path.name}' row {row_num} has run_name={row_run_name!r}, "
                    f"which does not match filename prefix {expected_run_name!r}"
                )
            xs.append(_parse_finite_float(row.get(x_col), x_col, log_path, row_num))
            ys.append(_parse_finite_float(row.get(y_col), y_col, log_path, row_num))
            if has_val_ap50:
                raw_ap = row.get("val_ap50")
                if raw_ap is None or not raw_ap.strip():
                    has_val_ap50 = False
                else:
                    val_ap50s.append(_parse_finite_float(raw_ap, "val_ap50", log_path, row_num))

    if not xs:
        raise ValueError(f"Training log CSV '{log_path.name}' contains no data rows")

    entry: dict[str, Any] = {
        "run_name": expected_run_name,
        "log_file": log_path.name,
        "log_path": str(log_path),
        "x_axis": x_axis,
        "num_points": len(xs),
        "x_values": xs,
        "loss_values": ys,
        "initial_loss": ys[0],
        "final_loss": ys[-1],
        "min_loss": min(ys),
    }
    if has_val_ap50 and len(val_ap50s) == len(xs):
        entry["val_ap50_values"] = val_ap50s
        entry["best_val_ap50"] = max(val_ap50s)
        entry["final_val_ap50"] = val_ap50s[-1]
    return entry


def plot_training_loss(
    log_paths: Sequence[Path | str],
    out_path: Path | str,
    x_axis: str = "epoch",
) -> dict[str, dict[str, Any]]:
    """Regenerate training loss curves directly from raw CSV logs and write a PNG figure.

    Verifies that every row's ``run_name`` matches its raw CSV filename prefix
    (e.g., ``res_800`` matches ``res_800_epoch_log.csv`` or ``res_800_step_log.csv``)
    and labels each curve with both ``run_name`` and source filename for direct
    traceability.
    """
    if x_axis not in VALID_X_AXES:
        raise ValueError(f"x_axis must be one of {VALID_X_AXES}, got {x_axis!r}")
    if not log_paths:
        raise ValueError("log_paths must contain at least one raw CSV log path")

    parsed: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for raw_path in log_paths:
        entry = _read_training_log(Path(raw_path), x_axis=x_axis)
        run_name = entry["run_name"]
        if run_name in seen_runs:
            raise ValueError(f"Duplicate run_name {run_name!r} across log_paths")
        seen_runs.add(run_name)
        parsed.append(entry)

    plot_val_ap = x_axis == "epoch" and all("val_ap50_values" in e for e in parsed)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = None
    try:
        if plot_val_ap:
            fig, (ax_loss, ax_ap) = plt.subplots(1, 2, figsize=(12.2, 5.2))
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(8.8, 5.2))
            ax_ap = None

        x_label = "Epoch" if x_axis == "epoch" else "Global Step"
        marker = "o" if x_axis == "epoch" else None

        for idx, entry in enumerate(parsed):
            run_name = entry["run_name"]
            color = _color_for_run(run_name, idx)
            label = f"{run_name} ({entry['log_file']})"
            ax_loss.plot(
                entry["x_values"],
                entry["loss_values"],
                label=label,
                color=color,
                linewidth=2.0,
                marker=marker,
                markersize=4.5 if marker else None,
            )
            if ax_ap is not None:
                ax_ap.plot(
                    entry["x_values"],
                    entry["val_ap50_values"],
                    label=label,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4.5,
                )

        ax_loss.set_xlabel(x_label)
        ax_loss.set_ylabel("Training Total Loss")
        ax_loss.set_title(f"Training Total Loss vs. {x_label}")
        if x_axis == "epoch":
            ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.legend(loc="best", fontsize=8.5, frameon=True)

        if ax_ap is not None:
            ax_ap.set_xlabel("Epoch")
            ax_ap.set_ylabel("Validation AP50")
            ax_ap.set_title("Validation AP50 vs. Epoch")
            ax_ap.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ap.set_ylim(0.0, 1.05)
            ax_ap.grid(True, linestyle="--", alpha=0.4)
            ax_ap.legend(loc="best", fontsize=8.5, frameon=True)
            suptitle = "Anti-UAV Training Loss & Validation AP50 (Regenerated from Raw CSV Logs)"
            fig.subplots_adjust(top=0.84, bottom=0.17, left=0.07, right=0.97, wspace=0.25)
        else:
            suptitle = f"Anti-UAV Training Loss Curve vs. {x_label} (Regenerated from Raw CSV Logs)"
            fig.subplots_adjust(top=0.84, bottom=0.17, left=0.10, right=0.96)

        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold")
        source_caption = "Source CSV logs: " + ", ".join(e["log_file"] for e in parsed)
        fig.text(0.5, 0.03, source_caption, ha="center", va="bottom", fontsize=8.5, color="#444444")
        fig.savefig(out_file, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)

    return {entry["run_name"]: entry for entry in parsed}


def _read_eval_json(eval_path: Path) -> dict[str, Any]:
    """Load and validate an evaluation JSON produced by ``src.eval``."""
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation JSON not found: {eval_path}")

    with eval_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation JSON '{eval_path.name}' must be a JSON object")

    run_name = payload.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f"Evaluation JSON '{eval_path.name}' missing non-empty 'run_name'")
    run_name = run_name.strip()

    ap50 = payload.get("ap50")
    if not _is_strict_number(ap50) or not math.isfinite(float(ap50)):
        raise ValueError(f"Evaluation JSON '{eval_path.name}' has invalid 'ap50': {ap50!r}")

    by_scale = payload.get("by_scale")
    if not isinstance(by_scale, dict):
        raise ValueError(
            f"Evaluation JSON '{eval_path.name}' missing 'by_scale' dict required for scale plot"
        )
    scale_data: dict[str, dict[str, float | int]] = {}
    for bin_name in SCALE_BIN_NAMES:
        bin_info = by_scale.get(bin_name)
        if not isinstance(bin_info, dict):
            raise ValueError(
                f"Evaluation JSON '{eval_path.name}' missing by_scale[{bin_name!r}]"
            )
        recall = bin_info.get("recall")
        gt_count = bin_info.get("gt_count")
        tp = bin_info.get("tp")
        fn = bin_info.get("fn")
        if not _is_strict_number(recall) or not math.isfinite(float(recall)):
            raise ValueError(
                f"Evaluation JSON '{eval_path.name}' has invalid recall in '{bin_name}': {recall!r}"
            )
        if (
            not _is_strict_int(gt_count)
            or not _is_strict_int(tp)
            or not _is_strict_int(fn)
            or gt_count < 0
            or tp < 0
            or fn < 0
        ):
            raise ValueError(
                f"Evaluation JSON '{eval_path.name}' has invalid counts in '{bin_name}': "
                f"tp={tp!r}, fn={fn!r}, gt_count={gt_count!r}"
            )
        scale_data[bin_name] = {
            "recall": float(recall),
            "gt_count": int(gt_count),
            "tp": int(tp),
            "fn": int(fn),
        }

    latency = payload.get("latency")
    if not isinstance(latency, dict):
        raise ValueError(f"Evaluation JSON '{eval_path.name}' missing 'latency' dict")
    mean_ms = latency.get("mean_ms")
    fps = latency.get("fps")
    if (
        not _is_strict_number(mean_ms)
        or not math.isfinite(float(mean_ms))
        or float(mean_ms) < 0.0
        or not _is_strict_number(fps)
        or not math.isfinite(float(fps))
        or float(fps) < 0.0
    ):
        raise ValueError(
            f"Evaluation JSON '{eval_path.name}' has invalid latency metrics: "
            f"mean_ms={mean_ms!r}, fps={fps!r}"
        )

    return {
        "run_name": run_name,
        "json_file": eval_path.name,
        "ap50": float(ap50),
        "by_scale": scale_data,
        "mean_ms": float(mean_ms),
        "fps": float(fps),
    }


def plot_scale_and_latency(
    eval_json_paths: Sequence[Path | str],
    out_path: Path | str,
) -> None:
    """Plot a 2-panel scale-stratified recall and inference latency comparison figure."""
    if not eval_json_paths:
        raise ValueError("eval_json_paths must contain at least one evaluation JSON path")

    records: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for raw_path in eval_json_paths:
        rec = _read_eval_json(Path(raw_path))
        if rec["run_name"] in seen_runs:
            raise ValueError(f"Duplicate run_name {rec['run_name']!r} across eval_json_paths")
        seen_runs.add(rec["run_name"])
        records.append(rec)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(records)
    x_bins = np.arange(len(SCALE_BIN_NAMES), dtype=np.float64)
    bar_width = min(0.27, 0.81 / float(n_runs))

    legend_cols = min(n_runs, 3)
    legend_rows = max(1, math.ceil(n_runs / legend_cols))
    title_pad = 28 + 14 * (legend_rows - 1)
    top_margin = max(0.70, 0.78 - 0.04 * (legend_rows - 1))

    fig = None
    try:
        fig, (ax_scale, ax_lat) = plt.subplots(
            1,
            2,
            figsize=(14.4, 5.9),
            gridspec_kw={"width_ratios": [1.35, 1.0]},
        )

        # Left panel: grouped scale-stratified recall bars with exact tp/gt_count annotations
        for idx, rec in enumerate(records):
            offset = (idx - (n_runs - 1) / 2.0) * bar_width
            recalls_pct = [rec["by_scale"][b]["recall"] * 100.0 for b in SCALE_BIN_NAMES]
            color = _color_for_run(rec["run_name"], idx)
            bars = ax_scale.bar(
                x_bins + offset,
                recalls_pct,
                width=bar_width * 0.90,
                label=rec["run_name"],
                color=color,
                edgecolor="#222222",
                linewidth=0.6,
            )
            for bar, bin_name in zip(bars, SCALE_BIN_NAMES):
                info = rec["by_scale"][bin_name]
                val_pct = info["recall"] * 100.0
                ax_scale.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val_pct + 1.8,
                    f"{val_pct:.1f}%\n{info['tp']}/{info['gt_count']}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )

        ax_scale.set_xticks(x_bins)
        ax_scale.set_xticklabels([SCALE_LABELS[b] for b in SCALE_BIN_NAMES])
        ax_scale.set_ylabel("Recall (%)")
        ax_scale.set_ylim(0.0, 120.0)
        ax_scale.set_title("Recall by Original-Coordinate Target Scale", pad=title_pad)
        ax_scale.grid(axis="y", linestyle="--", alpha=0.4)
        ax_scale.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_cols,
            fontsize=8.5,
            frameon=True,
        )

        # Right panel: single-image latency (ms) with FPS and overall AP50 annotations
        x_runs = np.arange(n_runs, dtype=np.float64)
        lat_vals = [rec["mean_ms"] for rec in records]
        colors = [_color_for_run(rec["run_name"], idx) for idx, rec in enumerate(records)]
        lat_bars = ax_lat.bar(
            x_runs,
            lat_vals,
            width=0.52,
            color=colors,
            edgecolor="#222222",
            linewidth=0.6,
        )
        max_ms = max(lat_vals) if lat_vals else 0.0
        y_headroom = max(max_ms * 1.38, 1.0)
        text_offset = y_headroom * 0.025
        for bar, rec in zip(lat_bars, records):
            ax_lat.text(
                bar.get_x() + bar.get_width() / 2.0,
                rec["mean_ms"] + text_offset,
                f"{rec['mean_ms']:.1f} ms\n{rec['fps']:.1f} FPS\nAP50={rec['ap50']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )

        ax_lat.set_xticks(x_runs)
        ax_lat.set_xticklabels(
            [rec["run_name"] for rec in records],
            rotation=10 if n_runs > 2 else 0,
        )
        ax_lat.set_ylabel("Mean Latency (ms / image)")
        ax_lat.set_ylim(0.0, y_headroom)
        ax_lat.set_title("Single-Image Inference Latency, FPS & Overall AP50", pad=title_pad)
        ax_lat.grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(
            "Anti-UAV Scale-Stratified Recall & Inference Speed Comparison",
            fontsize=12.5,
            fontweight="bold",
            y=0.96,
        )
        source_caption = "Source eval JSONs: " + ", ".join(rec["json_file"] for rec in records)
        fig.text(0.5, 0.03, source_caption, ha="center", va="bottom", fontsize=8.5, color="#444444")
        fig.subplots_adjust(top=top_margin, bottom=0.18, left=0.07, right=0.97, wspace=0.25)
        fig.savefig(out_file, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.plot_curves``."""
    parser = argparse.ArgumentParser(
        description="Regenerate Anti-UAV training loss curves and scale/latency plots from raw logs and eval JSONs."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=None,
        help="Raw *_epoch_log.csv or *_step_log.csv paths from src.train.",
    )
    parser.add_argument(
        "--loss-out",
        type=Path,
        default=None,
        help="Output PNG path for training loss curve.",
    )
    parser.add_argument(
        "--x-axis",
        choices=VALID_X_AXES,
        default="epoch",
        help="X-axis column mode for loss curve ('epoch' or 'step'; default: epoch).",
    )
    parser.add_argument(
        "--eval-jsons",
        nargs="+",
        type=Path,
        default=None,
        help="Evaluation JSON paths produced by src.eval.",
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Output PNG path for scale-recall and latency comparison chart.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for regenerating training curves and evaluation comparison figures."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    has_loss_req = args.logs is not None or args.loss_out is not None
    has_cmp_req = args.eval_jsons is not None or args.compare_out is not None

    if not has_loss_req and not has_cmp_req:
        parser.error("Provide --logs with --loss-out and/or --eval-jsons with --compare-out.")
    if (args.logs is None) != (args.loss_out is None):
        parser.error("--logs and --loss-out must be provided together.")
    if (args.eval_jsons is None) != (args.compare_out is None):
        parser.error("--eval-jsons and --compare-out must be provided together.")

    if args.logs is not None and args.loss_out is not None:
        summary = plot_training_loss(args.logs, args.loss_out, x_axis=args.x_axis)
        runs_desc = ", ".join(
            f"{name} (final_loss={info['final_loss']:.4f}, n={info['num_points']})"
            for name, info in summary.items()
        )
        print(f"Saved training loss plot ({args.x_axis}) -> {args.loss_out} | {runs_desc}")

    if args.eval_jsons is not None and args.compare_out is not None:
        plot_scale_and_latency(args.eval_jsons, args.compare_out)
        print(f"Saved scale & latency plot -> {args.compare_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
