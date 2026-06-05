import json
from pathlib import Path
import torch
from xmatcher.core.types import MatchResult
from xmatcher.core.io import save_npz, load_npz, result_to_manifest


def _make_result(tmp_path):
    return MatchResult(
        mkpts0=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        mkpts1=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        mconf=torch.tensor([0.9, 0.8]),
        method="lightglue",
        pair_id="scene01__0001__0002",
        runtime_ms=12.34,
        dense=None,
    )


def test_save_npz_roundtrip(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "out.npz"
    save_npz(r, p)
    loaded = load_npz(p)
    assert torch.allclose(loaded.mkpts0, r.mkpts0)
    assert torch.allclose(loaded.mkpts1, r.mkpts1)
    assert torch.allclose(loaded.mconf, r.mconf)
    assert loaded.method == "lightglue"
    assert loaded.pair_id == "scene01__0001__0002"
    assert loaded.runtime_ms == 12.34


def test_result_to_manifest_serializes_to_json(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "matches" / "x.npz"
    p.parent.mkdir()
    p.write_bytes(b"")  # placeholder; manifest uses path string
    entry = result_to_manifest(r, p)
    payload = json.dumps(entry)            # must be JSON-serializable
    parsed = json.loads(payload)
    assert parsed["pair_id"] == "scene01__0001__0002"
    assert parsed["method"] == "lightglue"
    assert parsed["num_matches"] == 2
    assert parsed["runtime_ms"] == 12.34
    assert parsed["npz_path"].endswith("x.npz")


def test_save_npz_creates_parent_dir(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "deep" / "nested" / "file.npz"
    save_npz(r, p)
    assert p.exists()


def test_save_npz_handles_empty_matches(tmp_path):
    r = MatchResult(
        mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2), mconf=torch.zeros(0),
        method="lightglue", pair_id="x", runtime_ms=1.0, dense=None,
    )
    p = tmp_path / "out.npz"
    save_npz(r, p)
    loaded = load_npz(p)
    assert loaded.mkpts0.shape == (0, 2)
