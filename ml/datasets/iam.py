from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "iam"

SPLITS = {"train", "validation", "test"}


class IAMDataset(Dataset):
    """IAM line-level handwriting dataset.

    Each item is a (image, transcription) tuple where image is a PIL Image
    (grayscale) unless a transform converts it. Transcriptions are raw strings;
    encode them with a vocab class at training time.

    Args:
        split: One of "train", "validation", or "test".
        data_dir: Root of the downloaded IAM data (default: data/iam/).
        transform: Optional callable applied to each PIL Image before returning.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path | str = DATA_DIR,
        transform: Callable | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "lines" / split
        self.transform = transform

        txt_path = self.data_dir / f"{split}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(
                f"{txt_path} not found — run `python ml/download_iam.py` first."
            )
        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"{self.img_dir} not found — run `python ml/download_iam.py` first."
            )

        self._samples: list[tuple[str, str]] = []
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, transcription = line.split("\t", 1)
                self._samples.append((fname, transcription))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, str]:
        fname, transcription = self._samples[idx]
        img = Image.open(self.img_dir / fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, transcription

    def __repr__(self) -> str:
        return f"IAMDataset(split={self.split!r}, n={len(self)})"
