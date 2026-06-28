ARG BASE_IMAGE=base-cuda:cu128-torch2.7.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install runtime system libs for Blender (bpy) headless operation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrender1 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# PyTorch & flash-attn are pre-installed in base-cuda image (system python).
# Install remaining packages directly into system python.
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt


WORKDIR /workspace/SkinTokens
COPY . .

EXPOSE 8087

CMD ["python", "serve.py"]
