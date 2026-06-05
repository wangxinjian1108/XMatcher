FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Layer 2: PyTorch (separate so common deps cache is preserved)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 3: common deps
COPY docker/requirements.common.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.common.txt

# Layer 4: per-method deps
COPY docker/requirements.lightglue.txt docker/requirements.eloftr.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.lightglue.txt && \
    pip install --no-cache-dir -r /tmp/requirements.eloftr.txt

# Layer 5: app code
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENV XMATCHER_WEIGHTS_DIR=/root/.cache/xmatcher

ENTRYPOINT ["python", "-m", "xmatcher.cli.run"]
CMD ["--help"]
