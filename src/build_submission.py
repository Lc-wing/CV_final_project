"""CS460 <= 50 MB submission zip packager with strict checklist and run_name validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import yaml

from src.prepare_data import IMAGE_EXTENSIONS

DEFAULT_MAX_SUBMISSION_BYTES = 50 * 1024 * 1024

NON_EVAL_METRICS_FILENAMES: frozenset[str] = frozenset(
    {
        "eda_train_val.json",
        "scale_bins.json",
    }
)

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".pytest_cache",
        ".superpowers",
        "__pycache__",
        "data",
        "checkpoints",
        "submission_build",
        "node_modules",
    }
)

EXCLUDED_REL_DIRS: frozenset[str] = frozenset({"docs/superpowers"})

WEIGHT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
    }
)

EXCLUDED_EXTENSIONS: frozenset[str] = WEIGHT_EXTENSIONS | frozenset(
    {
        ".zip",
        ".pyc",
        ".pyo",
    }
)

EXCLUDED_FILENAMES: frozenset[str] = frozenset({".DS_Store"})


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_configs(configs_dir: Path) -> set[str]:
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Required configs directory not found: {configs_dir}")

    yaml_files = sorted(configs_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No config *.yaml files found in: {configs_dir}")

    configured_runs: set[str] = set()
    for cfg_path in yaml_files:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config '{cfg_path.name}' must contain a YAML mapping")
        run_name = raw.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError(f"Config '{cfg_path.name}' missing non-empty string 'run_name'")
        run_name = run_name.strip()
        if run_name != cfg_path.stem:
            raise ValueError(
                f"Config '{cfg_path.name}' run_name ({run_name!r}) does not match "
                f"config filename stem ({cfg_path.stem!r})"
            )
        configured_runs.add(run_name)

    return configured_runs


def _validate_raw_csv_logs(logs_dir: Path, configured_runs: set[str]) -> list[Path]:
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Required raw logs directory not found: {logs_dir}")

    epoch_logs = sorted(logs_dir.glob("*_epoch_log.csv"))
    if not epoch_logs:
        raise FileNotFoundError(f"No *_epoch_log.csv files found in: {logs_dir}")

    step_logs = sorted(logs_dir.glob("*_step_log.csv"))
    all_logs: list[tuple[Path, str]] = [
        *((p, "_epoch_log") for p in epoch_logs),
        *((p, "_step_log") for p in step_logs),
    ]

    epoch_log_runs: set[str] = set()
    for log_path, suffix in all_logs:
        expected_run = log_path.stem[: -len(suffix)]
        if not expected_run or expected_run not in configured_runs:
            raise ValueError(
                f"Raw log '{log_path.name}' prefix run_name ({expected_run!r}) is not in "
                f"configured runs {sorted(configured_runs)}"
            )

        with log_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "run_name" not in reader.fieldnames:
                raise ValueError(
                    f"Raw log '{log_path.name}' is missing required 'run_name' column"
                )
            row_count = 0
            for row_idx, row in enumerate(reader, start=2):
                row_count += 1
                row_run = (row.get("run_name") or "").strip()
                if row_run != expected_run or row_run not in configured_runs:
                    raise ValueError(
                        f"Drift in run_name at row {row_idx} of '{log_path.name}': "
                        f"got {row_run!r}, expected {expected_run!r} "
                        f"(configured runs: {sorted(configured_runs)})"
                    )
            if row_count == 0:
                raise ValueError(f"Raw log '{log_path.name}' contains no data rows")

        if suffix == "_epoch_log":
            epoch_log_runs.add(expected_run)

    missing_epoch_runs = sorted(configured_runs - epoch_log_runs)
    if missing_epoch_runs:
        raise ValueError(
            f"Missing *_epoch_log.csv for configured run_name(s): {missing_epoch_runs}"
        )

    return [p for p, _ in all_logs]


def _is_eda_json_payload(raw: dict[str, Any]) -> bool:
    return (
        "source_splits" in raw
        and "splits" in raw
        and "run_name" not in raw
        and "split" not in raw
    )


def _validate_metrics_jsons(metrics_dir: Path, configured_runs: set[str]) -> None:
    if not metrics_dir.is_dir():
        return
    for suffix in ("_test", "_val"):
        for json_path in sorted(metrics_dir.glob(f"*{suffix}.json")):
            if json_path.name in NON_EVAL_METRICS_FILENAMES:
                continue
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Evaluation JSON '{json_path.name}' must be a JSON object")
            if _is_eda_json_payload(raw):
                continue
            expected_run = json_path.stem[: -len(suffix)]
            run_name = (
                (raw.get("run_name") or "").strip()
                if isinstance(raw.get("run_name"), str)
                else ""
            )
            if run_name != expected_run or run_name not in configured_runs:
                raise ValueError(
                    f"Evaluation JSON '{json_path.name}' run_name ({run_name!r}) does not match "
                    f"filename prefix ({expected_run!r}) and configured runs {sorted(configured_runs)}"
                )


def _validate_test_pack_and_predictions(
    root: Path,
    configured_runs: set[str],
) -> None:
    img_dir = root / "submission_pack" / "test_pack" / "img"
    xml_dir = root / "submission_pack" / "test_pack" / "xml"
    pred_csv = root / "submission_pack" / "predictions" / "test_pack_predictions.csv"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Required test-pack image directory not found: {img_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"Required test-pack XML directory not found: {xml_dir}")
    if not pred_csv.is_file():
        raise FileNotFoundError(f"Required test-pack predictions CSV not found: {pred_csv}")

    img_files = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not img_files:
        raise FileNotFoundError(f"No images found in test-pack directory: {img_dir}")

    xml_files = sorted(p for p in xml_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml")
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found in test-pack directory: {xml_dir}")

    test_pack_sample_ids: set[str] = set()
    for img_path in img_files:
        if img_path.stem in test_pack_sample_ids:
            raise ValueError(
                f"Duplicate test-pack sample_id {img_path.stem!r} in '{img_dir}'"
            )
        test_pack_sample_ids.add(img_path.stem)
        expected_xml = xml_dir / f"{img_path.stem}.xml"
        if not expected_xml.is_file():
            raise FileNotFoundError(
                f"Missing XML annotation for test-pack image '{img_path.name}': {expected_xml}"
            )

    with pred_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Predictions CSV '{pred_csv.name}' is empty (missing header)")
        for required_col in ("sample_id", "run_name"):
            if required_col not in reader.fieldnames:
                raise ValueError(
                    f"Predictions CSV '{pred_csv.name}' missing required column '{required_col}'"
                )
        row_count = 0
        seen_pairs: set[tuple[str, str]] = set()
        for row_idx, row in enumerate(reader, start=2):
            row_count += 1
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(
                    f"Empty 'sample_id' at row {row_idx} of '{pred_csv.name}'"
                )
            if sample_id not in test_pack_sample_ids:
                raise ValueError(
                    f"Unknown sample_id {sample_id!r} at row {row_idx} of '{pred_csv.name}': "
                    f"not present in test_pack/img/ ({sorted(test_pack_sample_ids)})"
                )
            row_run = (row.get("run_name") or "").strip()
            if row_run not in configured_runs:
                raise ValueError(
                    f"Drift in run_name at row {row_idx} of '{pred_csv.name}': "
                    f"{row_run!r} is not in configured runs {sorted(configured_runs)}"
                )
            pair = (sample_id, row_run)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate (sample_id, run_name) pair {pair!r} at row {row_idx} "
                    f"of '{pred_csv.name}'"
                )
            seen_pairs.add(pair)
        if row_count == 0:
            raise ValueError(f"Predictions CSV '{pred_csv.name}' contains no data rows")

        expected_pairs = {
            (sid, run)
            for sid in test_pack_sample_ids
            for run in configured_runs
        }
        missing_pairs = sorted(expected_pairs - seen_pairs)
        if missing_pairs:
            raise ValueError(
                f"Predictions CSV '{pred_csv.name}' missing required (sample_id, run_name) "
                f"pair(s): {missing_pairs}"
            )


def _validate_required_deliverables(root: Path, report_path: Path) -> set[str]:
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required report PDF not found or empty: {report_path}")

    readme_path = root / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"Required README.md not found: {readme_path}")

    req_path = root / "requirements.txt"
    if not req_path.is_file():
        raise FileNotFoundError(f"Required requirements.txt not found: {req_path}")

    src_dir = root / "src"
    if not src_dir.is_dir() or not any(src_dir.glob("*.py")):
        raise FileNotFoundError(f"Required src/ directory with Python source files not found: {src_dir}")

    configured_runs = _validate_configs(root / "configs")
    _validate_raw_csv_logs(root / "results" / "logs", configured_runs)
    _validate_metrics_jsons(root / "results" / "metrics", configured_runs)
    _validate_test_pack_and_predictions(root, configured_runs)
    return configured_runs


def _collect_bundle_files(
    root: Path,
    out_zip_resolved: Path,
    report_pdf_resolved: Path,
    report_arcname: str,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Collect deterministic archive entries using pruned ``os.walk`` while refusing weights."""
    entries: list[tuple[Path, str]] = []
    excluded_weights: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        # Prune excluded directories in-place so large data/, .venv/, .git/, checkpoints/,
        # and internal docs/superpowers/ trees are never traversed.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in EXCLUDED_DIRS
            and (current_dir / d).relative_to(root).as_posix() not in EXCLUDED_REL_DIRS
        )
        for fname in sorted(filenames):
            if fname in EXCLUDED_FILENAMES:
                continue

            candidate = current_dir / fname
            if not candidate.is_file():
                continue

            resolved = candidate.resolve()
            if resolved == out_zip_resolved or resolved == report_pdf_resolved:
                continue

            arcname = candidate.relative_to(root).as_posix()
            if arcname == report_arcname:
                continue

            suffix = candidate.suffix.lower()
            if suffix in WEIGHT_EXTENSIONS:
                excluded_weights.append(arcname)
                continue
            if suffix in EXCLUDED_EXTENSIONS:
                continue

            entries.append((candidate, arcname))

    entries.sort(key=lambda item: item[1])
    return entries, excluded_weights


def build_submission_zip(
    repo_root: Path | str,
    out_zip: Path | str,
    report_pdf: Path | str,
    max_bytes: int = DEFAULT_MAX_SUBMISSION_BYTES,
) -> dict[str, Any]:
    """Validate CS460 submission invariants and package a strictly ``<= max_bytes`` zip archive.

    Enforces:
    1. All required CS460 checklist deliverables exist.
    2. Strict ``run_name`` consistency across ``configs/*.yaml``, raw CSV training logs,
       optional evaluation JSONs, and ``submission_pack/predictions/test_pack_predictions.csv``.
    3. Zero weight files (``.pt``/``.pth``) or excluded directories bundled in the zip.
    4. Final archive size strictly ``<= max_bytes`` (default 50 MB); deletes the zip and
       raises ``ValueError`` if exceeded.
    """
    if not _is_strict_int(max_bytes) or max_bytes <= 0:
        raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")

    root = Path(repo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root directory not found: {root}")

    report_path = Path(report_pdf)
    zip_path = Path(out_zip)

    configured_runs = _validate_required_deliverables(root, report_path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    out_zip_resolved = zip_path.resolve()
    report_pdf_resolved = report_path.resolve()

    report_arcname = report_path.name
    bundle_entries, excluded_weights = _collect_bundle_files(
        root,
        out_zip_resolved=out_zip_resolved,
        report_pdf_resolved=report_pdf_resolved,
        report_arcname=report_arcname,
    )

    # Defense-in-depth guard: refuse any weight file before creating archive.
    leaked_weights = [
        arcname
        for _, arcname in bundle_entries
        if Path(arcname).suffix.lower() in WEIGHT_EXTENSIONS
    ]
    if leaked_weights:
        raise ValueError(
            f"Refusing to bundle model weight files into submission zip: {leaked_weights}"
        )

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, arcname=report_arcname)
        for src_file, arcname in bundle_entries:
            zf.write(src_file, arcname=arcname)

    size_bytes = zip_path.stat().st_size
    if size_bytes > max_bytes:
        zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"Submission zip size ({size_bytes:,} bytes) exceeds max_bytes limit "
            f"({max_bytes:,} bytes); archive was removed."
        )

    return {
        "out_zip": str(zip_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4),
        "max_bytes": max_bytes,
        "num_files": 1 + len(bundle_entries),
        "runs": sorted(configured_runs),
        "report_pdf": report_arcname,
        "excluded_weight_files": excluded_weights,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.build_submission``."""
    parser = argparse.ArgumentParser(
        description="Validate CS460 checklist & run_name consistency and build a <=50 MB submission zip."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--report-pdf",
        type=Path,
        required=True,
        help="Path to the final exported CS460 report PDF.",
    )
    parser.add_argument(
        "--out-zip",
        type=Path,
        required=True,
        help="Output path for the verified submission zip archive.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_SUBMISSION_BYTES,
        help=f"Maximum allowed submission zip size in bytes (default: {DEFAULT_MAX_SUBMISSION_BYTES}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building the verified CS460 submission zip."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    info = build_submission_zip(
        repo_root=args.repo_root,
        out_zip=args.out_zip,
        report_pdf=args.report_pdf,
        max_bytes=args.max_bytes,
    )
    print(
        f"Submission zip built -> {info['out_zip']} | "
        f"size={info['size_bytes']:,} bytes ({info['size_mb']:.2f} MB / "
        f"{info['max_bytes'] / (1024 * 1024):.1f} MB limit) | "
        f"files={info['num_files']} | runs={info['runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
