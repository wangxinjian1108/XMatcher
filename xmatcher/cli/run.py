from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.io import save_npz, result_to_manifest, snapshot_config
from xmatcher.dataset.config import DatasetConfig, build_dataset
import xmatcher.methods  # noqa: F401  (registers all adapters)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xmatcher.cli.run")
    p.add_argument("--method-cfg", required=True, type=Path)
    p.add_argument("--dataset-cfg", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    p.add_argument("--no-postprocess", action="store_true",
                   help="Return processed-coord keypoints (skip unproject + mask filter).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    method_cfg = RunConfig.model_validate(yaml.safe_load(args.method_cfg.read_text()))
    dataset_cfg = DatasetConfig.model_validate(yaml.safe_load(args.dataset_cfg.read_text()))
    if args.device:
        method_cfg.device = args.device

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "matches").mkdir(exist_ok=True)
    snapshot_config(method_cfg, dataset_cfg, out / "config.snapshot.yaml")

    matcher = build_matcher(method_cfg)
    dataset = build_dataset(dataset_cfg)

    n_done = 0
    with (out / "manifest.jsonl").open("w") as mf:
        for i, pair in enumerate(dataset):
            if args.limit is not None and i >= args.limit:
                break
            res = matcher(pair, return_processed_coords=args.no_postprocess)
            npz_path = out / "matches" / f"{pair.pair_id}.npz"
            save_npz(res, npz_path)
            mf.write(json.dumps(result_to_manifest(res, npz_path)) + "\n")
            n_done += 1

    print(f"[xmatcher] {n_done} pair(s) processed → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
