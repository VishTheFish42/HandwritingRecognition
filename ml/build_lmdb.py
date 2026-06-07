#!/usr/bin/env python3
"""
Build an LMDB cache of height-normalised IAM line images for fast training I/O.

Reads from data/iam/, writes to data/iam_lmdb/{split}/.
Images are resized to --target-height (default 32px) with aspect ratio preserved.
Transcriptions are stored alongside each image so a single LMDB read returns both.

Usage:
    python ml/build_lmdb.py
    python ml/build_lmdb.py --target-height 32
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

import lmdb
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "iam"
DST_DIR = ROOT / "data" / "iam_lmdb"
SPLITS = ("train", "validation", "test")
DEFAULT_HEIGHT = 32

# Virtual address space reservations (not physical allocations on most OSes).
MAP_SIZES = {"train": 1 << 30, "validation": 1 << 28, "test": 1 << 28}


def _resize_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def _build_split(split: str, src: Path, dst: Path, target_h: int) -> None:
    txt_path = src / f"{split}.txt"
    img_dir = src / "lines" / split

    samples: list[tuple[str, str]] = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                fname, text = line.split("\t", 1)
                samples.append((fname, text))

    lmdb_path = dst / split
    lmdb_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_path), map_size=MAP_SIZES.get(split, 1 << 29))
    with env.begin(write=True) as txn:
        txn.put(b"num-samples", pickle.dumps(len(samples)))
        for idx, (fname, text) in enumerate(samples):
            img = Image.open(img_dir / fname).convert("L")
            img = _resize_height(img, target_h)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=False)

            txn.put(
                f"{idx:06d}".encode(),
                pickle.dumps({"image": buf.getvalue(), "text": text}),
            )

            if (idx + 1) % 500 == 0 or idx + 1 == len(samples):
                print(f"\r  {idx + 1}/{len(samples)}", end="", flush=True)

    env.close()
    print()

    size_mb = sum(f.stat().st_size for f in lmdb_path.iterdir()) / 1e6
    print(f"  → {lmdb_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LMDB cache for IAM dataset")
    ap.add_argument("--target-height", type=int, default=DEFAULT_HEIGHT,
                    help=f"Target image height in pixels (default: {DEFAULT_HEIGHT})")
    ap.add_argument("--src", default=str(SRC_DIR), help="Source data/iam directory")
    ap.add_argument("--dst", default=str(DST_DIR), help="Output LMDB directory")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)

    for split in SPLITS:
        if not (src / f"{split}.txt").exists():
            print(f"Skipping {split}: {src / f'{split}.txt'} not found", file=sys.stderr)
            continue
        print(f"\n[{split}]")
        _build_split(split, src, dst, args.target_height)

    print(f"\nLMDB cache ready at: {dst}")


if __name__ == "__main__":
    main()
