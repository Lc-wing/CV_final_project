"""DUT Anti-UAV PyTorch dataset, box-safe horizontal flip augmentation, and seeding."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.prepare_data import IMAGE_EXTENSIONS, parse_voc_xml

UAV_LABEL = 1


MAX_SEED_EXCLUSIVE = 2**32


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < MAX_SEED_EXCLUSIVE):
        raise ValueError(f"seed must be an int in [0, {MAX_SEED_EXCLUSIVE - 1}], got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DUTAntiUAVDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Single-class (UAV) Pascal VOC object detection dataset for DUT Anti-UAV.

    Expects ``split_dir/img/`` and ``split_dir/xml/`` with matching filename stems.
    Empty (0-object) images are preserved with ``boxes`` of shape ``(0, 4)``
    (``torch.float32``) and ``labels`` of shape ``(0,)`` (``torch.int64``).
    """

    def __init__(
        self,
        split_dir: Path | str,
        hflip_prob: float = 0.0,
        max_samples: int | None = None,
    ) -> None:
        if (
            isinstance(hflip_prob, bool)
            or not isinstance(hflip_prob, (int, float))
            or not math.isfinite(hflip_prob)
            or not (0.0 <= float(hflip_prob) <= 1.0)
        ):
            raise ValueError(f"hflip_prob must be in [0.0, 1.0], got {hflip_prob!r}")

        if max_samples is not None and (
            isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0
        ):
            raise ValueError(
                f"max_samples must be a positive int or None, got {max_samples!r}"
            )

        self.split_dir = Path(split_dir)
        self.hflip_prob = float(hflip_prob)

        img_dir = self.split_dir / "img"
        xml_dir = self.split_dir / "xml"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not xml_dir.is_dir():
            raise FileNotFoundError(f"XML directory not found: {xml_dir}")

        img_paths = sorted(
            (
                p
                for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda p: (p.stem, p.name),
        )
        if max_samples is not None:
            img_paths = img_paths[:max_samples]

        samples: list[tuple[Path, Path]] = []
        for img_path in img_paths:
            xml_path = xml_dir / f"{img_path.stem}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(
                    f"Missing annotation XML for image '{img_path.name}': {xml_path}"
                )
            samples.append((img_path, xml_path))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_path, xml_path = self.samples[index]
        resolved_index = index if index >= 0 else len(self.samples) + index

        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
        orig_w, orig_h = rgb_img.size

        raw_boxes, _ = parse_voc_xml(xml_path, img_width=orig_w, img_height=orig_h)
        image_tensor = F.to_tensor(rgb_img)

        if raw_boxes:
            boxes = torch.tensor(raw_boxes, dtype=torch.float32)
            labels = torch.full((len(raw_boxes),), UAV_LABEL, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if self.hflip_prob > 0.0 and (
            self.hflip_prob >= 1.0 or float(torch.rand(1).item()) < self.hflip_prob
        ):
            image_tensor = F.hflip(image_tensor)
            if boxes.numel() > 0:
                old_xmin = boxes[:, 0].clone()
                old_xmax = boxes[:, 2].clone()
                boxes[:, 0] = float(orig_w) - old_xmax
                boxes[:, 2] = float(orig_w) - old_xmin

        if boxes.numel() > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(orig_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(orig_h))
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not bool(valid_mask.all()):
                boxes = boxes[valid_mask]
                labels = labels[valid_mask]

        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([resolved_index], dtype=torch.int64),
            "sample_id": img_path.stem,
            "orig_size": (int(orig_w), int(orig_h)),
        }
        return image_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Collate variable-resolution detection samples into image and target lists."""
    return [item[0] for item in batch], [item[1] for item in batch]
