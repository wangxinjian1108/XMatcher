from pathlib import Path
import numpy as np
from PIL import Image
import torch
from xmatcher.dataset.from_pair_list import (
    FromPairListDataset, FromPairListConfig, PreprocessConfig,
)


def _make_jpg(path: Path, h: int, w: int):
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _setup_imgs(tmp_path):
    (tmp_path / "imgs").mkdir()
    _make_jpg(tmp_path / "imgs" / "a.jpg", 480, 640)
    _make_jpg(tmp_path / "imgs" / "b.jpg", 480, 640)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\n")
    return pairs


def test_dataset_iterates_and_yields_imagepair(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(),
    ))
    items = list(ds)
    assert len(ds) == 1
    assert len(items) == 1
    p = items[0]
    assert p.image0.shape == (3, 480, 640)
    assert p.image1.shape == (3, 480, 640)
    assert p.meta0.original_size == (480, 640)
    assert p.meta0.processed_size == (480, 640)
    assert p.pair_id  # non-empty


def test_dataset_resize_long_side(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(resize_long_side=320),
    ))
    p = next(iter(ds))
    # original 480x640, long side 640 → scale 320/640=0.5; processed 240x320
    assert p.image0.shape == (3, 240, 320)
    assert p.meta0.original_size == (480, 640)
    assert p.meta0.processed_size == (240, 320)
    assert p.meta0.scale == (0.5, 0.5)


def test_dataset_image_in_zero_one_range(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(),
    ))
    p = next(iter(ds))
    assert p.image0.min() >= 0.0 and p.image0.max() <= 1.0
    assert p.image0.dtype == torch.float32


def test_dataset_pair_id_distinct_for_different_pairs(tmp_path):
    _make_jpg(tmp_path / "a.jpg", 100, 100)
    _make_jpg(tmp_path / "b.jpg", 100, 100)
    _make_jpg(tmp_path / "c.jpg", 100, 100)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\na.jpg c.jpg\n")
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs), image_root=str(tmp_path),
        preprocess=PreprocessConfig(),
    ))
    ids = [p.pair_id for p in ds]
    assert len(ids) == 2 and ids[0] != ids[1]
