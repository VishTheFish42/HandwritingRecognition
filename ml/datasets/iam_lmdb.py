from io import BytesIO
from pathlib import Path
from typing import Any, Callable
import pickle

import lmdb
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
LMDB_DIR = ROOT / "data" / "iam_lmdb"

SPLITS = {"train", "validation", "test"}


class IAMLmdbDataset(Dataset):
    """LMDB-backed IAM line dataset.

    Reads from the cache built by `ml/build_lmdb.py`. Faster than IAMDataset
    for training because images are pre-resized and reads are sequential.

    Each item is a (image, transcription) tuple. The LMDB env is opened lazily
    per DataLoader worker so multiprocessing works without extra setup.

    Args:
        split: One of "train", "validation", or "test".
        lmdb_dir: Parent directory of the per-split LMDB databases.
        transform: Optional callable applied to each PIL Image before returning.
    """

    def __init__(
        self,
        split: str,
        lmdb_dir: Path | str = LMDB_DIR,
        transform: Callable | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.split = split
        self._db_path = str(Path(lmdb_dir) / split)
        self.transform = transform
        self._env: lmdb.Environment | None = None

        if not Path(self._db_path).exists():
            raise FileNotFoundError(
                f"{self._db_path} not found — run `python ml/build_lmdb.py` first."
            )

        # Read length without keeping env open (env can't be pickled for workers).
        env = lmdb.open(self._db_path, readonly=True, lock=False)
        with env.begin() as txn:
            self._len: int = pickle.loads(txn.get(b"num-samples"))
        env.close()

    # ------------------------------------------------------------------
    # Pickling support for DataLoader multiprocessing
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None  # drop the unpicklable env; workers reopen it
        return state

    # ------------------------------------------------------------------

    def _env_handle(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, readonly=True, lock=False)
        return self._env

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[Any, str]:
        with self._env_handle().begin() as txn:
            record = pickle.loads(txn.get(f"{idx:06d}".encode()))
        img = Image.open(BytesIO(record["image"]))
        if self.transform is not None:
            img = self.transform(img)
        return img, record["text"]

    def __repr__(self) -> str:
        return f"IAMLmdbDataset(split={self.split!r}, n={len(self)})"
