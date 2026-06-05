from pathlib import Path
import yaml
import json
import numpy as np
from PIL import Image
import torch
from pydantic import BaseModel
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register, _REGISTRY
from xmatcher.core.types import _RawOutput
from xmatcher.cli.run import main


class _CLIToyParams(BaseModel):
    pass


def _ensure_toy_registered():
    if "cli_toy" not in _REGISTRY:
        @register("cli_toy")
        class _CLIToy(BaseMatcher):
            Params = _CLIToyParams
            def _setup(self): pass
            def _forward(self, pair):
                return _RawOutput(
                    mkpts0=torch.tensor([[10.0, 10.0]]),
                    mkpts1=torch.tensor([[20.0, 20.0]]),
                    mconf=torch.tensor([0.9]),
                    dense=None,
                )


def test_cli_runs_end_to_end(tmp_path):
    _ensure_toy_registered()
    # Build minimal dataset
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    for n in ("a.jpg", "b.jpg"):
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(img_dir / n)
    (tmp_path / "pairs.txt").write_text("a.jpg b.jpg\n")

    method_cfg = tmp_path / "m.yaml"
    method_cfg.write_text(yaml.safe_dump({
        "method": "cli_toy", "device": "cpu", "params": {},
    }))
    dataset_cfg = tmp_path / "d.yaml"
    dataset_cfg.write_text(yaml.safe_dump({
        "type": "from_pair_list",
        "params": {
            "pairs_file": str(tmp_path / "pairs.txt"),
            "image_root": str(img_dir),
            "preprocess": {},
        },
    }))
    out = tmp_path / "out"

    rc = main([
        "--method-cfg", str(method_cfg),
        "--dataset-cfg", str(dataset_cfg),
        "--out", str(out),
    ])
    assert rc == 0
    assert (out / "manifest.jsonl").exists()
    assert (out / "config.snapshot.yaml").exists()
    npzs = list((out / "matches").glob("*.npz"))
    assert len(npzs) == 1
    line = (out / "manifest.jsonl").read_text().splitlines()[0]
    entry = json.loads(line)
    assert entry["method"] == "cli_toy"
    assert entry["num_matches"] == 1
