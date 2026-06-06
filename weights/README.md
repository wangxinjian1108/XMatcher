# Weights

This directory holds the lock file (`WEIGHTS.lock`). Actual weights are NOT in git
and NOT in the Docker image. They live under `$XMATCHER_WEIGHTS_DIR`
(default `~/.cache/xmatcher/`), populated by `scripts/download_weights.sh`.

| Method            | Source                                       | License        |
|-------------------|----------------------------------------------|----------------|
| LightGlue         | Auto-downloaded by upstream (`torch.hub`)    | Apache-2.0     |
| EfficientLoFTR    | HuggingFace `zju-community/efficientloftr`   | Apache-2.0     |

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
