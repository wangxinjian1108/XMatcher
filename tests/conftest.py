import os
from pathlib import Path
import pytest
import torch


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    weights_dir = Path(os.environ.get(
        "XMATCHER_WEIGHTS_DIR",
        Path.home() / ".cache" / "xmatcher",
    ))

    skip_gpu = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)

        rw = item.get_closest_marker("requires_weights")
        if rw is None:
            continue
        method = rw.args[0]
        method_dir = weights_dir / method
        if not method_dir.exists() or not any(method_dir.iterdir()):
            item.add_marker(pytest.mark.skip(
                reason=f"weights for '{method}' not found at {method_dir}"
            ))
