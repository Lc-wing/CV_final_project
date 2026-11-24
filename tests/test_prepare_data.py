"""Unit tests for DUT Anti-UAV downloader, VOC XML validator, and train+val EDA."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.prepare_data import (
    SplitAudit,
    compute_scale_bins,
    download_dut_anti_uav,
    main,
    parse_voc_xml,
    scan_split,
)


def _write_xml(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    objs = "".join(
        f"<object><name>UAV</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for x1, y1, x2, y2 in boxes
    )
    path.write_text(
        f"<annotation><size><width>100</width><height>100</height></size>{objs}</annotation>",
        encoding="utf-8",
    )


def _create_split(
    root: Path,
    split: str,
    ratios: list[float],
    *,
    img_size: tuple[int, int] = (1000, 1000),
    add_empty: bool = False,
) -> None:
    img_dir = root / split / "img"
    xml_dir = root / split / "xml"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    w, h = img_size
    for idx, r in enumerate(ratios):
        Image.new("RGB", (w, h)).save(img_dir / f"{idx:05d}.jpg")
        side = (r * w * h) ** 0.5
        _write_xml(xml_dir / f"{idx:05d}.xml", [(10, 10, 10 + side, 10 + side)])
    if add_empty:
        Image.new("RGB", (w, h)).save(img_dir / "99999.jpg")
        _write_xml(xml_dir / "99999.xml", [])


def test_parse_voc_xml_handles_empty_and_malformed_boxes(tmp_path: Path) -> None:
    empty_xml = tmp_path / "empty.xml"
    _write_xml(empty_xml, [])
    boxes, dropped = parse_voc_xml(empty_xml, img_width=100, img_height=100)
    assert boxes == [] and dropped == 0

    mixed_xml = tmp_path / "mixed.xml"
    _write_xml(mixed_xml, [(-5, 10, 120, 40), (50, 50, 50, 60), (60, 40, 20, 80)])
    boxes, dropped = parse_voc_xml(mixed_xml, img_width=100, img_height=100)
    assert boxes == [[0.0, 10.0, 100.0, 40.0]]
    assert dropped == 2


def test_scan_split_and_scale_bins_from_train_val_only(tmp_path: Path) -> None:
    for split, ratios in [("train", [0.0005, 0.002, 0.02]), ("val", [0.0004, 0.003, 0.03])]:
        _create_split(tmp_path, split, ratios, add_empty=(split == "train"))

    train_a = scan_split(tmp_path / "train", "train")
    val_a = scan_split(tmp_path / "val", "val")
    assert train_a.num_images == 4 and train_a.num_empty_images == 1 and train_a.num_boxes == 3
    bins = compute_scale_bins(train_a, val_a)
    assert bins["source_splits"] == ["train", "val"]
    assert 0.0 < bins["edges"]["tiny_max"] < bins["edges"]["small_max"]
    assert sum(bins["train_counts"].values()) == 3
    assert sum(bins["val_counts"].values()) == 3


def test_parse_voc_xml_drops_out_of_bounds_missing_or_non_finite_boxes(tmp_path: Path) -> None:
    xml_path = tmp_path / "tricky.xml"
    xml_path.write_text(
        "<annotation>"
        "<object><name>UAV</name></object>"
        "<object><name>UAV</name><bndbox><xmin>nan</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>"
        "<object><name>UAV</name><bndbox><xmin>-20</xmin><ymin>10</ymin><xmax>-5</xmax><ymax>30</ymax></bndbox></object>"
        "<object><name>UAV</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>50</ymax></bndbox></object>"
        "</annotation>",
        encoding="utf-8",
    )
    boxes, dropped = parse_voc_xml(xml_path, img_width=100, img_height=100)
    assert boxes == [[10.0, 20.0, 40.0, 50.0]]
    assert dropped == 3

    with pytest.raises(ValueError, match="img_width|img_height|positive"):
        parse_voc_xml(xml_path, img_width=0, img_height=100)


def test_scan_split_tracks_missing_xml_and_all_dropped_images(tmp_path: Path) -> None:
    img_dir = tmp_path / "train" / "img"
    xml_dir = tmp_path / "train" / "xml"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    Image.new("RGB", (200, 100)).save(img_dir / "00001.jpg")
    _write_xml(xml_dir / "00001.xml", [(10, 10, 30, 30)])  # 20x20 -> area=400, ratio=0.02, side=20

    Image.new("RGB", (200, 100)).save(img_dir / "00002.jpg")
    _write_xml(xml_dir / "00002.xml", [(50, 50, 50, 60)])  # degenerate -> dropped -> empty

    Image.new("RGB", (200, 100)).save(img_dir / "00003.jpg")
    # 00003.xml intentionally missing

    audit = scan_split(tmp_path / "train", "train")
    assert audit.split == "train"
    assert audit.num_images == 3
    assert audit.num_boxes == 1
    assert audit.num_dropped_boxes == 1
    assert audit.num_empty_images == 1
    assert audit.missing_xml == ["00003"]
    assert audit.widths == [200, 200, 200]
    assert audit.heights == [100, 100, 100]
    assert audit.area_ratios == pytest.approx([0.02])
    assert audit.equiv_sides == pytest.approx([20.0])


def test_compute_scale_bins_fallback_constant_values_and_split_guard() -> None:
    # Fewer than 3 samples -> fallback defaults
    small_train = SplitAudit("train", 1, 0, 1, 0, [], [0.0005], [10.0], [200], [200])
    small_val = SplitAudit("val", 1, 0, 1, 0, [], [0.002], [20.0], [200], [200])
    fallback_bins = compute_scale_bins(small_train, small_val)
    assert 0.0 < fallback_bins["edges"]["tiny_max"] < fallback_bins["edges"]["small_max"]
    assert sum(fallback_bins["train_counts"].values()) == 1
    assert sum(fallback_bins["val_counts"].values()) == 1

    # Identical area ratios -> still strictly ordered positive edges
    const_train = SplitAudit("train", 2, 0, 2, 0, [], [0.002, 0.002], [20.0, 20.0], [200, 200], [200, 200])
    const_val = SplitAudit("val", 2, 0, 2, 0, [], [0.002, 0.002], [20.0, 20.0], [200, 200], [200, 200])
    const_bins = compute_scale_bins(const_train, const_val)
    assert 0.0 < const_bins["edges"]["tiny_max"] < const_bins["edges"]["small_max"]
    assert sum(const_bins["train_counts"].values()) == 2
    assert sum(const_bins["val_counts"].values()) == 2

    # Never allow test split into scale-bin calculation
    test_audit = SplitAudit("test", 1, 0, 1, 0, [], [0.5], [100.0], [200], [200])
    with pytest.raises(ValueError, match="train.*val"):
        compute_scale_bins(small_train, test_audit)


def test_download_dut_anti_uav_handles_nested_flat_layouts_and_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    sample_img = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64)).save(sample_img)
    sample_xml = "<annotation><object><name>UAV</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>25</xmax><ymax>25</ymax></bndbox></object></annotation>"

    # Nested layout: train/img/... and train/xml/... plus __MACOSX noise
    train_zip = remote_dir / "train.zip"
    with zipfile.ZipFile(train_zip, "w") as zf:
        zf.write(sample_img, arcname="train/img/00001.jpg")
        zf.writestr("train/xml/00001.xml", sample_xml)
        zf.writestr("__MACOSX/train/img/._00001.jpg", b"junk")

    # Flat layout: img/... and xml/...
    val_zip = remote_dir / "val.zip"
    with zipfile.ZipFile(val_zip, "w") as zf:
        zf.write(sample_img, arcname="img/00002.jpg")
        zf.writestr("xml/00002.xml", sample_xml)

    calls: list[tuple[str, str, str]] = []

    def fake_hf_hub_download(*, repo_id: str, filename: str, repo_type: str, **kwargs: object) -> str:
        calls.append((repo_id, filename, repo_type))
        split = Path(filename).stem
        return str(remote_dir / f"{split}.zip")

    monkeypatch.setattr("src.prepare_data.hf_hub_download", fake_hf_hub_download)

    data_dir = tmp_path / "data"
    download_dut_anti_uav(data_dir, splits=("train", "val"))
    assert calls == [
        ("aegean-ai/dut-anti-uav", "detection/train.zip", "dataset"),
        ("aegean-ai/dut-anti-uav", "detection/val.zip", "dataset"),
    ]
    assert (data_dir / "train" / "img" / "00001.jpg").is_file()
    assert (data_dir / "train" / "xml" / "00001.xml").is_file()
    assert (data_dir / "val" / "img" / "00002.jpg").is_file()
    assert (data_dir / "val" / "xml" / "00002.xml").is_file()
    assert not (data_dir / "train" / "__MACOSX").exists()

    # Second call must be idempotent (no additional downloads)
    download_dut_anti_uav(data_dir, splits=("train", "val"))
    assert len(calls) == 2


def test_download_dut_anti_uav_rejects_zip_slip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    monkeypatch.setattr(
        "src.prepare_data.hf_hub_download",
        lambda **kwargs: str(bad_zip),
    )

    data_dir = tmp_path / "data"
    with pytest.raises(ValueError, match="Unsafe|path traversal"):
        download_dut_anti_uav(data_dir, splits=("train",))
    assert not (tmp_path / "escape.txt").exists()
    assert not list(data_dir.glob(".tmp_*"))


def test_download_dut_anti_uav_rejects_invalid_split_and_incomplete_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    with pytest.raises(ValueError, match="Invalid split"):
        download_dut_anti_uav(data_dir, splits=("unknown",))

    incomplete_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(incomplete_zip, "w") as zf:
        zf.writestr("README.txt", "no img or xml here")

    monkeypatch.setattr(
        "src.prepare_data.hf_hub_download",
        lambda **kwargs: str(incomplete_zip),
    )
    with pytest.raises(ValueError, match="img.*xml"):
        download_dut_anti_uav(data_dir, splits=("train",))
    assert not (data_dir / "train").exists()
    assert not list(data_dir.glob(".tmp_*"))


def test_cli_skip_download_writes_eda_and_scale_bins_without_touching_test(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _create_split(data_dir, "train", [0.0005, 0.002, 0.02], add_empty=True)
    _create_split(data_dir, "val", [0.0004, 0.003, 0.03])
    # Deliberately add extreme boxes in test split; CLI must never read test for EDA/scale bins
    _create_split(data_dir, "test", [0.5, 0.6, 0.7])

    out_dir = tmp_path / "results"
    eda_out = out_dir / "eda_train_val.json"
    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--eda-out",
            str(eda_out),
            "--skip-download",
        ]
    )
    assert exit_code == 0
    assert eda_out.is_file()
    sibling_bins = out_dir / "scale_bins.json"
    assert sibling_bins.is_file()

    eda = json.loads(eda_out.read_text(encoding="utf-8"))
    bins = json.loads(sibling_bins.read_text(encoding="utf-8"))

    assert eda["source_splits"] == ["train", "val"]
    assert set(eda["splits"].keys()) == {"train", "val"}
    assert eda["scale_bins"] == bins
    assert bins["source_splits"] == ["train", "val"]
    assert bins["edges"]["small_max"] < 0.1  # Unaffected by test split's 0.5-0.7 ratios

    # Verify custom --scale-bins-out path works as well
    custom_bins = tmp_path / "custom" / "my_bins.json"
    exit_code_2 = main(
        [
            "--data-dir",
            str(data_dir),
            "--eda-out",
            str(eda_out),
            "--scale-bins-out",
            str(custom_bins),
            "--skip-download",
        ]
    )
    assert exit_code_2 == 0
    assert json.loads(custom_bins.read_text(encoding="utf-8")) == bins
