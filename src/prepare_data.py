"""DUT Anti-UAV dataset downloader, Pascal VOC XML validator, and train+val EDA."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

DUT_ANTI_UAV_REPO_ID = "aegean-ai/dut-anti-uav"
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
DEFAULT_TINY_MAX = 0.001
DEFAULT_SMALL_MAX = 0.005


@dataclass
class SplitAudit:
    """Integrity and bounding-box geometry audit for a single dataset split."""

    split: str
    num_images: int
    num_empty_images: int
    num_boxes: int
    num_dropped_boxes: int
    missing_xml: list[str]
    area_ratios: list[float]
    equiv_sides: list[float]
    widths: list[int]
    heights: list[int]


def parse_voc_xml(
    xml_path: Path,
    img_width: int,
    img_height: int,
) -> tuple[list[list[float]], int]:
    """Parse a Pascal VOC XML file, clamp boxes to image bounds, and drop degenerate boxes.

    Args:
        xml_path: Path to the Pascal VOC annotation XML file.
        img_width: Original image width in pixels (must be positive).
        img_height: Original image height in pixels (must be positive).

    Returns:
        Tuple of ``(boxes, num_dropped)`` where ``boxes`` is a list of valid
        clamped ``[xmin, ymin, xmax, ymax]`` float coordinates and
        ``num_dropped`` counts malformed or degenerate ``<object>`` entries.
    """
    if (
        isinstance(img_width, bool)
        or isinstance(img_height, bool)
        or img_width <= 0
        or img_height <= 0
    ):
        raise ValueError(
            f"img_width and img_height must be positive integers, "
            f"got img_width={img_width!r}, img_height={img_height!r}"
        )

    tree = ET.parse(Path(xml_path))
    root = tree.getroot()

    boxes: list[list[float]] = []
    dropped = 0
    w_bound = float(img_width)
    h_bound = float(img_height)

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            dropped += 1
            continue

        coords: list[float] = []
        malformed = False
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            elem = bndbox.find(tag)
            if elem is None or elem.text is None:
                malformed = True
                break
            try:
                val = float(elem.text.strip())
            except ValueError:
                malformed = True
                break
            if not math.isfinite(val):
                malformed = True
                break
            coords.append(val)

        if malformed:
            dropped += 1
            continue

        xmin = max(0.0, min(w_bound, coords[0]))
        ymin = max(0.0, min(h_bound, coords[1]))
        xmax = max(0.0, min(w_bound, coords[2]))
        ymax = max(0.0, min(h_bound, coords[3]))

        if xmax <= xmin or ymax <= ymin:
            dropped += 1
            continue

        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, dropped


def scan_split(split_dir: Path, split_name: str) -> SplitAudit:
    """Audit all images and VOC XML annotations in a dataset split directory."""
    split_path = Path(split_dir)
    img_dir = split_path / "img"
    xml_dir = split_path / "xml"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    img_paths = sorted(
        p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    num_empty_images = 0
    num_boxes = 0
    num_dropped_boxes = 0
    missing_xml: list[str] = []
    area_ratios: list[float] = []
    equiv_sides: list[float] = []
    widths: list[int] = []
    heights: list[int] = []

    for img_path in img_paths:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        widths.append(int(img_w))
        heights.append(int(img_h))

        xml_path = xml_dir / f"{img_path.stem}.xml"
        if not xml_path.is_file():
            missing_xml.append(img_path.stem)
            continue

        boxes, dropped = parse_voc_xml(xml_path, img_width=img_w, img_height=img_h)
        num_dropped_boxes += dropped

        if not boxes:
            num_empty_images += 1
            continue

        num_boxes += len(boxes)
        img_area = float(img_w * img_h)
        for xmin, ymin, xmax, ymax in boxes:
            box_area = (xmax - xmin) * (ymax - ymin)
            area_ratios.append(box_area / img_area)
            equiv_sides.append(math.sqrt(box_area))

    return SplitAudit(
        split=split_name,
        num_images=len(img_paths),
        num_empty_images=num_empty_images,
        num_boxes=num_boxes,
        num_dropped_boxes=num_dropped_boxes,
        missing_xml=missing_xml,
        area_ratios=area_ratios,
        equiv_sides=equiv_sides,
        widths=widths,
        heights=heights,
    )


def _round_sig(value: float, sig: int = 2) -> float:
    """Round a finite float to ``sig`` significant figures."""
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    digits = sig - 1 - int(math.floor(math.log10(abs(value))))
    return round(float(value), digits)


def assign_scale_bin(area_ratio: float, edges: dict[str, float]) -> str:
    """Classify a relative box area ratio into ``tiny``, ``small``, or ``medium_large``."""
    if area_ratio < edges["tiny_max"]:
        return "tiny"
    if area_ratio < edges["small_max"]:
        return "small"
    return "medium_large"


def _count_scale_bins(area_ratios: list[float], edges: dict[str, float]) -> dict[str, int]:
    counts = {"tiny": 0, "small": 0, "medium_large": 0}
    for ratio in area_ratios:
        counts[assign_scale_bin(ratio, edges)] += 1
    return counts


def compute_scale_bins(train_audit: SplitAudit, val_audit: SplitAudit) -> dict[str, Any]:
    """Compute relative-area scale-bin thresholds strictly from train + val audits."""
    if train_audit.split != "train" or val_audit.split != "val":
        raise ValueError(
            "Scale bins must be computed strictly from 'train' and 'val' splits "
            f"(got {train_audit.split!r} and {val_audit.split!r})"
        )

    combined = list(train_audit.area_ratios) + list(val_audit.area_ratios)
    if len(combined) < 3:
        tiny_max, small_max = DEFAULT_TINY_MAX, DEFAULT_SMALL_MAX
    else:
        q1, q2 = (float(v) for v in np.quantile(combined, [1.0 / 3.0, 2.0 / 3.0]))
        tiny_max = _round_sig(q1, sig=2)
        small_max = _round_sig(q2, sig=2)
        if not (0.0 < tiny_max < small_max):
            if 0.0 < q1 < q2:
                tiny_max, small_max = q1, q2
            elif q1 > 0.0:
                tiny_max = _round_sig(q1 * 0.5, sig=2)
                small_max = _round_sig(q1 * 1.5, sig=2)
                if not (0.0 < tiny_max < small_max):
                    tiny_max, small_max = q1 * 0.5, q1 * 1.5
            else:
                tiny_max, small_max = DEFAULT_TINY_MAX, DEFAULT_SMALL_MAX

    edges = {"tiny_max": float(tiny_max), "small_max": float(small_max)}
    return {
        "metric": "relative_box_area_ratio",
        "edges": edges,
        "train_counts": _count_scale_bins(train_audit.area_ratios, edges),
        "val_counts": _count_scale_bins(val_audit.area_ratios, edges),
        "source_splits": ["train", "val"],
    }


def _is_split_extracted(split_dir: Path) -> bool:
    """Return True if ``split_dir`` contains non-empty ``img/`` and ``xml/`` directories."""
    img_dir = split_dir / "img"
    xml_dir = split_dir / "xml"
    if not (img_dir.is_dir() and xml_dir.is_dir()):
        return False
    has_img = any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in img_dir.iterdir()
    )
    has_xml = any(
        p.is_file() and p.suffix.lower() == ".xml" for p in xml_dir.iterdir()
    )
    return has_img and has_xml


def _extract_split_zip(zip_path: Path, split_dir: Path, split_name: str) -> None:
    """Extract a DUT Anti-UAV split archive defensively into ``split_dir/{img,xml}``.

    Handles both nested (``<split>/img/...`` or ``detection/<split>/img/...``)
    and flat (``img/...``) archive layouts while rejecting path-traversal entries.
    """
    split_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=split_dir.parent, prefix=f".tmp_{split_name}_"))
    tmp_resolved = tmp_path.resolve()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                norm_name = info.filename.replace("\\", "/")
                posix_path = PurePosixPath(norm_name)
                if (
                    norm_name.startswith("/")
                    or posix_path.is_absolute()
                    or ".." in posix_path.parts
                ):
                    raise ValueError(
                        f"Unsafe path traversal entry in archive '{zip_path}': {info.filename!r}"
                    )

                if info.is_dir():
                    continue

                parts = posix_path.parts
                if not parts:
                    continue
                if "__MACOSX" in parts or parts[-1].startswith("._") or parts[-1] == ".DS_Store":
                    continue

                subdir_idx: int | None = None
                for idx in range(len(parts) - 2, -1, -1):
                    if parts[idx] in ("img", "xml"):
                        subdir_idx = idx
                        break

                if subdir_idx is None:
                    continue

                rel_parts = parts[subdir_idx:]
                dest = tmp_path.joinpath(*rel_parts)
                if not dest.resolve().is_relative_to(tmp_resolved):
                    raise ValueError(
                        f"Unsafe extraction target outside split directory: {info.filename!r}"
                    )

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src_fh, dest.open("wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh)

        if not _is_split_extracted(tmp_path):
            raise ValueError(
                f"Archive '{zip_path}' for split '{split_name}' did not contain "
                "non-empty 'img/' and 'xml/' directories"
            )

        if split_dir.exists():
            shutil.rmtree(split_dir)
        tmp_path.rename(split_dir)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def download_dut_anti_uav(
    data_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Download and extract DUT Anti-UAV detection splits from Hugging Face idempotently."""
    for split in splits:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split {split!r}; expected one of {VALID_SPLITS}"
            )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_dir = data_path / split
        if _is_split_extracted(split_dir):
            continue
        zip_path = Path(
            hf_hub_download(
                repo_id=DUT_ANTI_UAV_REPO_ID,
                filename=f"detection/{split}.zip",
                repo_type="dataset",
            )
        )
        _extract_split_zip(zip_path, split_dir, split)


def _numeric_summary(
    values: list[int | float],
    *,
    include_percentiles: bool = True,
) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    summary: dict[str, float] = {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }
    if include_percentiles:
        p10, p25, p75, p90 = (float(v) for v in np.quantile(arr, [0.10, 0.25, 0.75, 0.90]))
        summary.update({"p10": p10, "p25": p25, "p75": p75, "p90": p90})
    return summary


def summarize_split_audit(audit: SplitAudit) -> dict[str, Any]:
    """Convert a ``SplitAudit`` into a compact JSON-serializable summary dictionary."""
    return {
        "split": audit.split,
        "num_images": audit.num_images,
        "num_empty_images": audit.num_empty_images,
        "num_boxes": audit.num_boxes,
        "num_dropped_boxes": audit.num_dropped_boxes,
        "num_missing_xml": len(audit.missing_xml),
        "missing_xml": list(audit.missing_xml),
        "image_width_stats": _numeric_summary(audit.widths, include_percentiles=False),
        "image_height_stats": _numeric_summary(audit.heights, include_percentiles=False),
        "area_ratio_stats": _numeric_summary(audit.area_ratios, include_percentiles=True),
        "equiv_side_px_stats": _numeric_summary(audit.equiv_sides, include_percentiles=True),
    }


def run_train_val_eda(
    data_dir: Path,
    eda_out: Path,
    scale_bins_out: Path | None = None,
) -> dict[str, Any]:
    """Run train+val EDA, compute scale bins, and save EDA and scale-bins JSONs."""
    data_path = Path(data_dir)
    eda_path = Path(eda_out)

    train_audit = scan_split(data_path / "train", "train")
    val_audit = scan_split(data_path / "val", "val")
    scale_bins = compute_scale_bins(train_audit, val_audit)

    eda_payload: dict[str, Any] = {
        "source_splits": ["train", "val"],
        "splits": {
            "train": summarize_split_audit(train_audit),
            "val": summarize_split_audit(val_audit),
        },
        "scale_bins": scale_bins,
    }

    eda_path.parent.mkdir(parents=True, exist_ok=True)
    eda_path.write_text(json.dumps(eda_payload, indent=2) + "\n", encoding="utf-8")

    bins_path = (
        Path(scale_bins_out)
        if scale_bins_out is not None
        else eda_path.parent / "scale_bins.json"
    )
    if bins_path.resolve() != eda_path.resolve():
        bins_path.parent.mkdir(parents=True, exist_ok=True)
        bins_path.write_text(json.dumps(scale_bins, indent=2) + "\n", encoding="utf-8")

    return eda_payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.prepare_data``."""
    parser = argparse.ArgumentParser(
        description="Download DUT Anti-UAV, validate Pascal VOC XMLs, and compute train+val EDA."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root dataset directory containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--eda-out",
        type=Path,
        required=True,
        help="Output path for the full train+val EDA JSON (also embeds scale_bins).",
    )
    parser.add_argument(
        "--scale-bins-out",
        type=Path,
        default=None,
        help="Optional output path for standalone scale_bins.json (defaults to sibling of --eda-out).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download and run EDA on existing local splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="Splits to download when --skip-download is not set (EDA always uses train+val only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for DUT Anti-UAV data preparation and train+val EDA."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.skip_download:
        download_dut_anti_uav(args.data_dir, splits=tuple(args.splits))

    eda = run_train_val_eda(
        data_dir=args.data_dir,
        eda_out=args.eda_out,
        scale_bins_out=args.scale_bins_out,
    )
    edges = eda["scale_bins"]["edges"]
    train_s = eda["splits"]["train"]
    val_s = eda["splits"]["val"]
    print(
        f"Train+Val EDA complete | "
        f"train: {train_s['num_images']} imgs ({train_s['num_empty_images']} empty, "
        f"missing_xml={train_s['num_missing_xml']}), "
        f"{train_s['num_boxes']} boxes ({train_s['num_dropped_boxes']} dropped) | "
        f"val: {val_s['num_images']} imgs ({val_s['num_empty_images']} empty, "
        f"missing_xml={val_s['num_missing_xml']}), "
        f"{val_s['num_boxes']} boxes ({val_s['num_dropped_boxes']} dropped) | "
        f"scale edges: tiny<{edges['tiny_max']}, small<{edges['small_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
