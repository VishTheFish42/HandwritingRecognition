#!/usr/bin/env python3
"""
Download the IAM Handwriting Database via Hugging Face (Teklia/IAM-line).

No registration required. Saves images and transcriptions to data/iam/.

Usage:
  pip install datasets
  python ml/download_iam.py

Output layout:
  data/iam/
    lines/
      train/        000000.png, 000001.png, ...
      validation/
      test/
    train.txt       tab-separated: filename<TAB>transcription
    validation.txt
    test.txt
"""

import sys
from pathlib import Path

# ml/datasets/ would shadow the installed 'datasets' package when Python adds
# the script's directory to sys.path. Remove it before any HuggingFace import.
_ml_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _ml_dir]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "iam"
SPLITS = ("train", "validation", "test")
HF_DATASET = "Teklia/IAM-line"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' not installed.\n  pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {HF_DATASET} from Hugging Face ...")
    ds = load_dataset(HF_DATASET)

    for split in SPLITS:
        split_data = ds[split]
        img_dir = DATA_DIR / "lines" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        txt_path = DATA_DIR / f"{split}.txt"

        print(f"\n[{split}] {len(split_data)} samples → {img_dir.relative_to(ROOT)}/")

        with open(txt_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(split_data):
                fname = f"{idx:06d}.png"
                img_path = img_dir / fname
                if not img_path.exists():
                    row["image"].convert("L").save(img_path)  # save as grayscale PNG
                transcription = row["text"].strip()
                f.write(f"{fname}\t{transcription}\n")

        print(f"  Wrote {txt_path.relative_to(ROOT)}")

    print(f"\nIAM dataset ready at: {DATA_DIR}")


if __name__ == "__main__":
    main()
