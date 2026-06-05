from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from xmatcher.core.types import MatchResult


def save_npz(result: MatchResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mkpts0=result.mkpts0.detach().cpu().numpy(),
        mkpts1=result.mkpts1.detach().cpu().numpy(),
        mconf=result.mconf.detach().cpu().numpy(),
        method=np.array(result.method),
        pair_id=np.array(result.pair_id),
        runtime_ms=np.array(result.runtime_ms),
    )


def load_npz(path: str | Path) -> MatchResult:
    z = np.load(Path(path), allow_pickle=False)
    return MatchResult(
        mkpts0=torch.from_numpy(z["mkpts0"]),
        mkpts1=torch.from_numpy(z["mkpts1"]),
        mconf=torch.from_numpy(z["mconf"]),
        method=str(z["method"]),
        pair_id=str(z["pair_id"]),
        runtime_ms=float(z["runtime_ms"]),
        dense=None,
    )


def result_to_manifest(result: MatchResult, npz_path: str | Path) -> dict:
    return {
        "pair_id": result.pair_id,
        "method": result.method,
        "num_matches": int(result.mkpts0.shape[0]),
        "runtime_ms": float(result.runtime_ms),
        "npz_path": str(npz_path),
    }


def snapshot_config(method_cfg, dataset_cfg, out_path: str | Path) -> None:
    """Dump method+dataset config plus git commit hash to a single YAML for replay."""
    import subprocess
    import yaml
    from datetime import datetime

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    snap = {
        "method_cfg": method_cfg.model_dump(),
        "dataset_cfg": dataset_cfg.model_dump(),
        "git_commit": commit,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(snap, sort_keys=False))
