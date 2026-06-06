# Weights

This directory holds the lock file (`WEIGHTS.lock`). Actual weights are NOT in git
and NOT in the Docker image. They live under `$XMATCHER_WEIGHTS_DIR`
(default `~/.cache/xmatcher/`), populated by `scripts/download_weights.sh`.

| Method            | Source                                            | License        |
|-------------------|---------------------------------------------------|----------------|
| LightGlue         | GitHub release `cvg/LightGlue@v0.1_arxiv` (¹)     | Apache-2.0     |
| EfficientLoFTR    | HuggingFace `zju-community/efficientloftr`        | Apache-2.0     |

¹ At runtime the LightGlue adapter relies on upstream's `torch.hub`
auto-download into `~/.cache/torch/hub/checkpoints/`. The `lightglue` entries
in `WEIGHTS.lock` exist for **integrity audit** (sha256 of the pinned release
artifacts) and for **air-gapped bootstrap** — pre-fetch into
`$XMATCHER_WEIGHTS_DIR/lightglue/`, then symlink/copy the files into
`~/.cache/torch/hub/checkpoints/` (note the local `superpoint_lightglue.pth`
must be renamed to `superpoint_lightglue_v0-1_arxiv.pth` to match what
`torch.hub` expects).

## First-time bootstrap

The lock file ships with `sha256: "BOOTSTRAP"` placeholders. To fill them in:

```bash
scripts/download_weights.sh efficient_loftr --bootstrap
```

This downloads the file, prints its sha256, and prompts you to paste the value
back into `weights/WEIGHTS.lock`. After that, normal runs verify the hash:

```bash
scripts/download_weights.sh efficient_loftr
```

If the recorded sha256 ever mismatches what was downloaded, the script deletes
the file and exits non-zero — never silent success.

## Private / gated HuggingFace repos

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before running the script. The
downloader sends `Authorization: Bearer $HF_TOKEN` only on requests to
`huggingface.co` (and its subdomains); other hosts never see the token.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
scripts/download_weights.sh efficient_loftr
```
