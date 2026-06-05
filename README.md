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
