# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=docker.1ms.run/nvidia/cuda:13.2.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:${PATH} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY .python-version pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra cuda132

COPY . /workspace
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/taac-entrypoint

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cuda132

ENTRYPOINT ["taac-entrypoint"]
CMD ["/bin/bash"]
