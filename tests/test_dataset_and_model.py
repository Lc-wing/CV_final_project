"""Unit tests for DUT Anti-UAV PyTorch dataset, augmentation, and controlled Faster R-CNN factory."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn as tv_fasterrcnn_mobilenet_v3_large_fpn,
)

from src.config import ExperimentConfig, load_config
from src.dataset import DUTAntiUAVDataset, detection_collate_fn, seed_everything
from src.model import build_model, summarize_trainable_params


def test_dataset_empty_image_and_horizontal_flip(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "00001.jpg")
    (tmp_path / "xml" / "00001.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox>"
        "<xmin>20</xmin><ymin>10</ymin><xmax>50</xmax><ymax>30</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "00002.jpg")
    (tmp_path / "xml" / "00002.xml").write_text("<annotation></annotation>", encoding="utf-8")

    ds = DUTAntiUAVDataset(tmp_path, hflip_prob=1.0)
    img1, tgt1 = ds[0]
    assert img1.shape == (3, 100, 200)
    assert torch.allclose(tgt1["boxes"], torch.tensor([[150.0, 10.0, 180.0, 30.0]]))
    assert tgt1["labels"].tolist() == [1]
    assert tgt1["orig_size"] == (200, 100)

    img2, tgt2 = ds[1]
    assert tgt2["boxes"].shape == (0, 4) and tgt2["boxes"].dtype == torch.float32
    assert tgt2["labels"].shape == (0,) and tgt2["labels"].dtype == torch.int64


def test_build_model_preserves_rpn_weights_and_anchors_across_variants() -> None:
    cfg_480 = load_config("configs/res_480.yaml")
    cfg_small = load_config("configs/res_480_small_anchor.yaml")

    seed_everything(cfg_480.seed)
    m_480 = build_model(cfg_480, pretrained=False)
    seed_everything(cfg_small.seed)
    m_small = build_model(cfg_small, pretrained=False)

    assert len(m_small.rpn.anchor_generator.sizes) == len(m_480.rpn.anchor_generator.sizes)
    assert (
        m_small.rpn.anchor_generator.num_anchors_per_location()
        == m_480.rpn.anchor_generator.num_anchors_per_location()
    )
    assert m_small.rpn.anchor_generator.sizes[0] == (16, 32, 64, 128, 256)
    assert m_480.rpn.anchor_generator.sizes[0] == (32, 64, 128, 256, 512)
    assert torch.equal(m_480.rpn.head.cls_logits.weight, m_small.rpn.head.cls_logits.weight)
    assert torch.equal(
        m_480.roi_heads.box_predictor.cls_score.weight,
        m_small.roi_heads.box_predictor.cls_score.weight,
    )

    stats = summarize_trainable_params(m_small)
    assert 0 < stats["trainable_params"] <= stats["total_params"]

    # Forward + loss with both a normal target and an empty (0, 4) target
    m_small.train()
    imgs = [torch.rand(3, 120, 160), torch.rand(3, 120, 160)]
    tgts = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]
    losses = m_small(imgs, tgts)
    total = sum(losses.values())
    assert torch.isfinite(total)


def test_dataset_sorting_metadata_max_samples_and_pixel_box_flip(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()

    # Create asymmetric PNG (lossless) so left half is pure red, right half is pure blue
    arr = np.zeros((40, 80, 3), dtype=np.uint8)
    arr[:, :40, 0] = 255  # left half red
    arr[:, 40:, 2] = 255  # right half blue
    asymmetric = Image.fromarray(arr, mode="RGB")

    # Write files out of sorted order to confirm deterministic stem sorting
    for stem in ("00010", "00002", "00001"):
        asymmetric.save(tmp_path / "img" / f"{stem}.png")
        (tmp_path / "xml" / f"{stem}.xml").write_text(
            "<annotation>"
            "<object><name>UAV</name><bndbox><xmin>4</xmin><ymin>5</ymin><xmax>24</xmax><ymax>15</ymax></bndbox></object>"
            "<object><name>UAV</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>50</xmax><ymax>30</ymax></bndbox></object>"
            "<object><name>UAV</name><bndbox><xmin>60</xmin><ymin>10</ymin><xmax>95</xmax><ymax>25</ymax></bndbox></object>"
            "</annotation>",
            encoding="utf-8",
        )

    ds_no_flip = DUTAntiUAVDataset(tmp_path, hflip_prob=0.0)
    assert len(ds_no_flip) == 3

    sample_ids = [ds_no_flip[i][1]["sample_id"] for i in range(len(ds_no_flip))]
    assert sample_ids == ["00001", "00002", "00010"]

    img_orig, tgt_orig = ds_no_flip[0]
    assert img_orig.dtype == torch.float32
    assert img_orig.shape == (3, 40, 80)
    # Left pixel is red, right pixel is blue
    assert torch.allclose(img_orig[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(img_orig[:, 0, -1], torch.tensor([0.0, 0.0, 1.0]))
    # Degenerate middle box dropped; third box clamped to xmax=80
    assert torch.allclose(
        tgt_orig["boxes"],
        torch.tensor([[4.0, 5.0, 24.0, 15.0], [60.0, 10.0, 80.0, 25.0]]),
    )
    assert tgt_orig["labels"].tolist() == [1, 1]
    assert tgt_orig["image_id"].shape == (1,) and tgt_orig["image_id"].dtype == torch.int64
    assert tgt_orig["image_id"].item() == 0
    assert tgt_orig["orig_size"] == (80, 40)

    # Negative indexing resolves image_id to positive canonical index
    _, tgt_last = ds_no_flip[-1]
    assert tgt_last["sample_id"] == "00010"
    assert tgt_last["image_id"].item() == 2

    # Horizontal flip transforms both pixels and boxes together
    ds_flip = DUTAntiUAVDataset(tmp_path, hflip_prob=1.0, max_samples=2)
    assert len(ds_flip) == 2
    img_flip, tgt_flip = ds_flip[0]
    assert torch.allclose(img_flip[:, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(img_flip[:, 0, -1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(
        tgt_flip["boxes"],
        torch.tensor([[56.0, 5.0, 76.0, 15.0], [0.0, 10.0, 20.0, 25.0]]),
    )
    assert tgt_flip["labels"].tolist() == [1, 1]


def test_dataset_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="img"):
        DUTAntiUAVDataset(tmp_path)

    (tmp_path / "img").mkdir()
    with pytest.raises(FileNotFoundError, match="xml"):
        DUTAntiUAVDataset(tmp_path)

    (tmp_path / "xml").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "img" / "00001.jpg")
    with pytest.raises(FileNotFoundError, match="00001.xml"):
        DUTAntiUAVDataset(tmp_path)

    (tmp_path / "xml" / "00001.xml").write_text("<annotation></annotation>", encoding="utf-8")

    for bad_prob in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="hflip_prob"):
            DUTAntiUAVDataset(tmp_path, hflip_prob=bad_prob)  # type: ignore[arg-type]

    for bad_max in (0, -5, True):
        with pytest.raises(ValueError, match="max_samples"):
            DUTAntiUAVDataset(tmp_path, max_samples=bad_max)  # type: ignore[arg-type]


def test_detection_collate_fn_and_dataloader_integration(tmp_path: Path) -> None:
    assert detection_collate_fn([]) == ([], [])

    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()
    Image.new("RGB", (100, 80)).save(tmp_path / "img" / "00001.jpg")
    (tmp_path / "xml" / "00001.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox>"
        "<xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    Image.new("RGB", (140, 90)).save(tmp_path / "img" / "00002.jpg")
    (tmp_path / "xml" / "00002.xml").write_text("<annotation></annotation>", encoding="utf-8")

    ds = DUTAntiUAVDataset(tmp_path, hflip_prob=0.0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))

    assert isinstance(images, list) and isinstance(targets, list)
    assert len(images) == 2 and len(targets) == 2
    assert images[0].shape == (3, 80, 100)
    assert images[1].shape == (3, 90, 140)
    assert [t["sample_id"] for t in targets] == ["00001", "00002"]


def test_seed_everything_reproducibility_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    seed_everything(42)
    py_a = [random.random() for _ in range(3)]
    np_a = np.random.rand(4)
    th_a = torch.rand(4)

    seed_everything(42)
    py_b = [random.random() for _ in range(3)]
    np_b = np.random.rand(4)
    th_b = torch.rand(4)

    assert py_a == py_b
    assert np.array_equal(np_a, np_b)
    assert torch.equal(th_a, th_b)

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    for bad_seed in (True, -1, 2**32):
        with pytest.raises(ValueError, match="seed"):
            seed_everything(bad_seed)  # type: ignore[arg-type]
    # Failed validation must not mutate PYTHONHASHSEED
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_build_model_res800_transform_predictor_and_eval_mode() -> None:
    cfg_800 = load_config("configs/res_800.yaml")
    model = build_model(cfg_800, pretrained=False)

    assert tuple(model.transform.min_size) == (800,)
    assert model.transform.max_size == 1333
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 8
    assert model.roi_heads.score_thresh == pytest.approx(0.05)
    assert len(model.rpn.anchor_generator.sizes) == 3
    assert model.rpn.anchor_generator.num_anchors_per_location() == [15, 15, 15]
    assert all(level == (32, 64, 128, 256, 512) for level in model.rpn.anchor_generator.sizes)
    assert all(level == (0.5, 1.0, 2.0) for level in model.rpn.anchor_generator.aspect_ratios)

    model.eval()
    with torch.no_grad():
        preds = model([torch.rand(3, 120, 160)])
    assert len(preds) == 1
    assert set(preds[0].keys()) == {"boxes", "labels", "scores"}
    assert preds[0]["boxes"].ndim == 2 and preds[0]["boxes"].shape[1] == 4
    if preds[0]["labels"].numel() > 0:
        assert set(preds[0]["labels"].tolist()).issubset({1})


def test_build_model_rejects_mismatched_anchor_count_and_invalid_backbone_layers() -> None:
    cfg_base = load_config("configs/res_480.yaml")

    bad_sizes = ExperimentConfig(**{**cfg_base.__dict__, "anchor_sizes": (16, 32, 64, 128)})
    with pytest.raises(ValueError, match="num_anchors_per_location"):
        build_model(bad_sizes, pretrained=False)

    bad_ratios = ExperimentConfig(**{**cfg_base.__dict__, "aspect_ratios": (0.5, 1.0)})
    with pytest.raises(ValueError, match="num_anchors_per_location"):
        build_model(bad_ratios, pretrained=False)

    for bad_layers in (-1, 7, True):
        bad_cfg = ExperimentConfig(**{**cfg_base.__dict__, "trainable_backbone_layers": bad_layers})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="trainable_backbone_layers"):
            build_model(bad_cfg, pretrained=False)

    for bad_min, bad_max in ((0, 800), (900, 800), (True, 800)):
        bad_cfg = ExperimentConfig(**{**cfg_base.__dict__, "min_size": bad_min, "max_size": bad_max})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="min_size"):
            build_model(bad_cfg, pretrained=False)

    for bad_anchor_vals in ((0, 32, 64, 128, 256), (-16, 32, 64, 128, 256), (True, 32, 64, 128, 256)):
        bad_cfg = ExperimentConfig(**{**cfg_base.__dict__, "anchor_sizes": bad_anchor_vals})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="anchor_sizes"):
            build_model(bad_cfg, pretrained=False)

    for bad_ratio_vals in ((-0.5, 1.0, 2.0), (0.0, 1.0, 2.0), (float("nan"), 1.0, 2.0)):
        bad_cfg = ExperimentConfig(**{**cfg_base.__dict__, "aspect_ratios": bad_ratio_vals})
        with pytest.raises(ValueError, match="aspect_ratios"):
            build_model(bad_cfg, pretrained=False)


def test_build_model_pretrained_wiring_and_summarize_frozen_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Direct test of summarize_trainable_params with partially frozen module
    toy = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    for p in toy[0].parameters():
        p.requires_grad = False
    toy_stats = summarize_trainable_params(toy)
    assert toy_stats == {
        "total_params": (4 * 3 + 3) + (3 * 2 + 2),
        "trainable_params": 3 * 2 + 2,
        "frozen_params": 4 * 3 + 3,
    }

    # Verify build_model(..., pretrained=True) passes DEFAULT weights & trainable_backbone_layers
    # and preserves exact pretrained RPN conv weights after anchor override
    captured_calls: list[dict[str, Any]] = []
    pre_override_rpn_state: dict[str, torch.Tensor] = {}

    def fake_factory(**kwargs: Any) -> Any:
        captured_calls.append(dict(kwargs))
        model = tv_fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=None,
            min_size=kwargs["min_size"],
            max_size=kwargs["max_size"],
        )
        if kwargs["weights"] is not None:
            # Simulate frozen early backbone blocks as when pretrained weights are loaded
            for idx, p in enumerate(model.backbone.parameters()):
                if idx < 4:
                    p.requires_grad = False
            pre_override_rpn_state["cls_w"] = model.rpn.head.cls_logits.weight.detach().clone()
            pre_override_rpn_state["cls_b"] = model.rpn.head.cls_logits.bias.detach().clone()
            pre_override_rpn_state["box_w"] = model.rpn.head.bbox_pred.weight.detach().clone()
            pre_override_rpn_state["box_b"] = model.rpn.head.bbox_pred.bias.detach().clone()
        return model

    monkeypatch.setattr("src.model.fasterrcnn_mobilenet_v3_large_fpn", fake_factory)

    cfg_small = load_config("configs/res_480_small_anchor.yaml")
    model = build_model(cfg_small, pretrained=True)

    captured = captured_calls[0]
    assert captured["weights"] == FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    assert captured["weights_backbone"] is None
    assert captured["trainable_backbone_layers"] == cfg_small.trainable_backbone_layers
    assert captured["min_size"] == 480
    assert captured["max_size"] == 800
    assert len(model.rpn.anchor_generator.sizes) == 3
    assert model.rpn.anchor_generator.num_anchors_per_location() == [15, 15, 15]
    assert model.rpn.anchor_generator.sizes[0] == (16, 32, 64, 128, 256)
    assert model.rpn.head.cls_logits.out_channels == 15
    assert model.rpn.head.bbox_pred.out_channels == 60
    assert torch.equal(model.rpn.head.cls_logits.weight, pre_override_rpn_state["cls_w"])
    assert torch.equal(model.rpn.head.cls_logits.bias, pre_override_rpn_state["cls_b"])
    assert torch.equal(model.rpn.head.bbox_pred.weight, pre_override_rpn_state["box_w"])
    assert torch.equal(model.rpn.head.bbox_pred.bias, pre_override_rpn_state["box_b"])
    assert model.roi_heads.box_predictor.cls_score.out_features == 2

    stats = summarize_trainable_params(model)
    assert stats["frozen_params"] > 0
    assert stats["trainable_params"] > 0
    assert stats["trainable_params"] + stats["frozen_params"] == stats["total_params"]

    # Verify pretrained=False does not request any pretrained weights
    build_model(cfg_small, pretrained=False)
    unpretrained = captured_calls[1]
    assert unpretrained["weights"] is None
    assert unpretrained["weights_backbone"] is None
    assert unpretrained["trainable_backbone_layers"] is None


def test_dataset_drops_float32_collapsed_box_under_hflip_and_all_empty_backward(
    tmp_path: Path,
) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()
    # A sub-ULP box near x=0 passes float64 parse_voc_xml (1e-5 > 0.0),
    # but collapses to xmax == xmin == 1920.0 in float32 when flipped on a 1920px image.
    Image.new("RGB", (1920, 120)).save(tmp_path / "img" / "00001.jpg")
    (tmp_path / "xml" / "00001.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox>"
        "<xmin>0</xmin><ymin>10</ymin><xmax>0.00001</xmax><ymax>30</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )

    ds = DUTAntiUAVDataset(tmp_path, hflip_prob=1.0)
    img, tgt = ds[0]
    assert img.shape == (3, 120, 1920)
    assert tgt["boxes"].shape == (0, 4) and tgt["boxes"].dtype == torch.float32
    assert tgt["labels"].shape == (0,) and tgt["labels"].dtype == torch.int64

    # All-empty batch forward + backward must produce finite losses without error
    cfg_small = load_config("configs/res_480_small_anchor.yaml")
    model = build_model(cfg_small, pretrained=False)
    model.train()
    losses = model([torch.rand(3, 120, 160)], [tgt])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()


def test_dataset_stochastic_hflip_reproducibility_and_dataloader_to_model_forward(
    tmp_path: Path,
) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "xml").mkdir()
    Image.new("L", (160, 120)).save(tmp_path / "img" / "00001.jpg")  # grayscale -> RGB
    (tmp_path / "xml" / "00001.xml").write_text(
        "<annotation><object><name>UAV</name><bndbox>"
        "<xmin>20</xmin><ymin>20</ymin><xmax>50</xmax><ymax>50</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    Image.new("RGB", (160, 120)).save(tmp_path / "img" / "00002.jpg")
    (tmp_path / "xml" / "00002.xml").write_text("<annotation></annotation>", encoding="utf-8")

    ds = DUTAntiUAVDataset(tmp_path, hflip_prob=0.5)
    seed_everything(42)
    seq_a = [ds[0][1]["boxes"][0, 0].item() for _ in range(20)]
    seed_everything(42)
    seq_b = [ds[0][1]["boxes"][0, 0].item() for _ in range(20)]
    assert seq_a == seq_b
    # Both unflipped (xmin=20) and flipped (xmin=110) outcomes occur under p=0.5
    assert set(seq_a) == {20.0, 110.0}

    # End-to-end DataLoader -> Faster R-CNN forward with full dataset target dicts
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))
    assert images[0].shape == (3, 120, 160)  # grayscale converted to 3-channel RGB

    cfg_480 = load_config("configs/res_480.yaml")
    cfg_small = load_config("configs/res_480_small_anchor.yaml")
    m_480 = build_model(cfg_480, pretrained=False)
    m_small = build_model(cfg_small, pretrained=False)
    m_480.train()
    m_small.train()

    losses_480 = m_480(images, targets)
    losses_small = m_small(images, targets)
    assert torch.isfinite(sum(losses_480.values()))
    assert torch.isfinite(sum(losses_small.values()))

    # Verify actual generated RPN cell anchors in res_480_small_anchor are exactly 0.5x res_480
    assert len(m_small.rpn.anchor_generator.cell_anchors) == len(m_480.rpn.anchor_generator.cell_anchors)
    for ca_small, ca_480 in zip(
        m_small.rpn.anchor_generator.cell_anchors,
        m_480.rpn.anchor_generator.cell_anchors,
    ):
        assert torch.allclose(ca_small, ca_480 * 0.5, atol=1.0)
