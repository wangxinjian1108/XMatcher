# XMatcher

Unified keypoints matching interface — a single CLI and Python API for running
multiple keypoint matchers (LightGlue, EfficientLoFTR; RoMa v2 / XFeat planned)
through a YAML-configured pipeline.

## Quick start

```bash
# 1. Submodules
git submodule update --init --recursive

# 2. Install
pip install -e .[test] kornia pytorch-lightning yacs loguru h5py

# 3. Weights (EfficientLoFTR — LightGlue auto-downloads)
scripts/download_weights.sh efficient_loftr

# 4. Run
python -m xmatcher.cli.run \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg_demo/
```

## Docker

Two image flavors are published to GHCR on every push to `main`:

- `ghcr.io/wangxinjian1108/xmatcher:latest` — lean (~5 GB), expects weights
  to be mounted from the host.
- `ghcr.io/wangxinjian1108/xmatcher:latest-bundled` — same image with
  EfficientLoFTR + LightGlue weights baked in (~5.3 GB), no mount required.

Tagged versions (`v*`) and per-commit SHA tags work the same way, with the
`-bundled` suffix for the bundled flavor.

Local build (lean):
```bash
docker/build.sh   # build xmatcher:dev locally
docker run --rm --gpus all \
    -v $HOME/.cache/xmatcher:/root/.cache/xmatcher \
    -v $PWD/outputs:/app/outputs \
    xmatcher:dev \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg/
```

Bundled image (no weights mount needed):
```bash
docker run --rm --gpus all \
    -v $PWD/outputs:/app/outputs \
    ghcr.io/wangxinjian1108/xmatcher:latest-bundled \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg/
```

## Layout

- `xmatcher/core/` — shared types, base matcher, registry, config, IO.
- `xmatcher/methods/` — per-method adapters (each registers via `@register("name")`).
- `xmatcher/dataset/` — `PairDataset` Protocol + `FromPairListDataset`.
- `xmatcher/cli/run.py` — `--method-cfg / --dataset-cfg / --out` CLI.
- `thirdparty/` — upstream method repos as git submodules.
- `configs/` — sample method and dataset YAMLs.
- `tests/{unit,contract,smoke,fixtures}/` — three-layer test suite.

## Tests

```bash
pytest tests/unit tests/contract -v          # CI-safe (no GPU, no weights)
pytest tests/smoke -v                        # local-only (GPU + weights required)
```

## Design

See [`docs/superpowers/specs/2026-06-06-unified-kpts-matching-design.md`](docs/superpowers/specs/2026-06-06-unified-kpts-matching-design.md).
