#!/usr/bin/env python3
"""
Validate the IAM dataset loaders.

Checks:
  1. Both loaders (IAMDataset, IAMLmdbDataset) return correct image/text pairs
     across all three splits — renders a visual grid for manual inspection.
  2. Transforms compose correctly with IAMLmdbDataset — output is a float tensor.
  3. DataLoader with num_workers=2 completes without pickling / worker errors.

Usage:
    python ml/validate_loader.py
"""

import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from datasets.iam import IAMDataset
from datasets.iam_lmdb import IAMLmdbDataset
from transforms import BrightnessJitter, ElasticDistortion, GaussianNoise, RandomRotation

import numpy as np


class _ToTensor:
    """PIL (mode L) → (1, H, W) float32 tensor.

    Avoids torchvision.ToTensor which calls torch.from_numpy — broken when
    torch was compiled against NumPy 1.x but 2.x is installed. torch.tensor()
    copies data and works regardless of the NumPy C-API bridge state.
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # PIL.tobytes() + frombuffer avoids the torch/NumPy C-API bridge entirely.
        raw = bytearray(img.convert("L").tobytes())
        t = torch.frombuffer(raw, dtype=torch.uint8).float() / 255.0
        return t.reshape(1, img.height, img.width)

OUT_DIR = ROOT / "data" / "validation"


def _pad_collate(batch: list) -> tuple:
    """Pad variable-width tensors in a batch to the same width so they stack."""
    tensors, texts = zip(*batch)
    max_w = max(t.shape[2] for t in tensors)
    padded = [torch.nn.functional.pad(t, (0, max_w - t.shape[2])) for t in tensors]
    return torch.stack(padded), list(texts)
N_SAMPLES = 10
DISPLAY_MAX_W = 700   # max width for each image row in the output grid
ROW_PAD = 8           # vertical padding between rows
TEXT_H = 20           # height reserved for transcription label
BG = 245              # light-grey background


def _make_grid(samples: list[tuple[Image.Image, str]], title: str) -> Image.Image:
    """Build a single-column grid: image row + transcription label per sample."""
    canvas_w = DISPLAY_MAX_W + 20
    rows: list[Image.Image] = []

    for img, text in samples:
        # Scale image to fit display width, keep aspect ratio.
        w, h = img.size
        scale = min(1.0, DISPLAY_MAX_W / w)
        disp_w, disp_h = max(1, round(w * scale)), max(1, round(h * scale))
        img_disp = img.resize((disp_w, disp_h), Image.LANCZOS)

        row_h = ROW_PAD + disp_h + TEXT_H + ROW_PAD
        row = Image.new("L", (canvas_w, row_h), color=BG)
        row.paste(img_disp, (10, ROW_PAD))

        draw = ImageDraw.Draw(row)
        draw.text((10, ROW_PAD + disp_h + 2), text, fill=40)
        rows.append(row)

    title_h = 30
    total_h = title_h + sum(r.height for r in rows)
    grid = Image.new("L", (canvas_w, total_h), color=BG)

    draw = ImageDraw.Draw(grid)
    draw.text((10, 6), title, fill=40)

    y = title_h
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height

    return grid


def _validate(ds_class, label: str, split: str = "train") -> None:
    ds = ds_class(split)
    n = len(ds)
    indices = random.sample(range(n), min(N_SAMPLES, n))
    samples = [(ds[i][0], ds[i][1]) for i in indices]

    # Text report
    print(f"\n{'─' * 60}")
    print(f"{label}  ({n} samples in '{split}' split)")
    print(f"{'─' * 60}")
    for idx, (img, text) in zip(indices, samples):
        print(f"  [{idx:5d}]  size={str(img.size):>14s}  mode={img.mode}  {text!r}")

    # Visual grid
    grid = _make_grid(samples, f"{label} — {split} split (10 random samples)")
    out_path = OUT_DIR / f"{label.lower().replace(' ', '_')}_{split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"\n  Grid saved → {out_path.relative_to(ROOT)}")


def _validate_transforms() -> None:
    """Check transforms compose with the dataset and produce float tensors."""
    print(f"\n{'─' * 60}")
    print("Transforms + IAMLmdbDataset")
    print(f"{'─' * 60}")

    transform = T.Compose([
        RandomRotation(p=0.5),
        ElasticDistortion(p=0.5),
        GaussianNoise(p=0.5),
        BrightnessJitter(p=0.5),
        _ToTensor(),
    ])

    ds = IAMLmdbDataset("train", transform=transform)
    for i in random.sample(range(len(ds)), 5):
        tensor, text = ds[i]
        assert isinstance(tensor, torch.Tensor), f"expected Tensor, got {type(tensor)}"
        assert tensor.ndim == 3 and tensor.shape[0] == 1, f"unexpected shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"unexpected dtype {tensor.dtype}"
        print(f"  [{i:5d}]  shape={tuple(tensor.shape)}  dtype={tensor.dtype}  {text!r}")

    print("  OK — transforms produce (1, 32, W) float32 tensors")


def _validate_dataloader() -> None:
    """Check DataLoader with num_workers=2 completes without worker errors."""
    print(f"\n{'─' * 60}")
    print("DataLoader  (batch_size=8, num_workers=2)")
    print(f"{'─' * 60}")

    transform = T.Compose([
        RandomRotation(p=0.5),
        GaussianNoise(p=0.5),
        BrightnessJitter(p=0.5),
        _ToTensor(),
    ])

    ds = IAMLmdbDataset("train", transform=transform)
    loader = DataLoader(ds, batch_size=8, num_workers=2, collate_fn=_pad_collate)

    for batch_idx, (images, texts) in enumerate(loader):
        assert images.shape[0] == 8
        assert images.shape[1] == 1
        assert images.shape[2] == 32
        print(f"  batch {batch_idx}  images={tuple(images.shape)}  texts[0]={texts[0]!r}")
        if batch_idx == 2:
            break

    print("  OK — 3 batches loaded across 2 workers without error")


def main() -> None:
    random.seed(42)
    _validate(IAMDataset,     "IAMDataset",     split="train")
    _validate(IAMLmdbDataset, "IAMLmdbDataset", split="train")
    _validate_transforms()
    _validate_dataloader()
    print()


if __name__ == "__main__":
    main()
