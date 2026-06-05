# Weights

This directory holds the lock file (`WEIGHTS.lock`). Actual weights are NOT in git
and NOT in the Docker image. They live under `$XMATCHER_WEIGHTS_DIR`
(default `~/.cache/xmatcher/`), populated by `scripts/download_weights.sh`.

| Method            | Source                                       | License        |
|-------------------|----------------------------------------------|----------------|
| LightGlue         | Auto-downloaded by upstream (`torch.hub`)    | Apache-2.0     |
| EfficientLoFTR    | Google Drive (see WEIGHTS.lock)              | Apache-2.0     |
