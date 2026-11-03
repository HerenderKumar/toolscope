# Python image shared by the `api` and `gateway` services (CPU only, D-002).
# Data, results, logs, indexes and the Hugging Face model cache are mounted at runtime (see docker-compose.yml),
# so the image holds code and dependencies only.
FROM python:3.11-slim

# docker CLI: the gateway starts each LiveMCPBench server with `docker exec -i` (D-019) through the host's socket
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list \
 && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch is installed from a wheel downloaded on the host (docker/wheels/, SHA-256 checked against the PyTorch index),
# because download-r2.pytorch.org dropped long downloads from this network; requirements.txt then finds it satisfied.
COPY docker/wheels/ /wheels/
RUN pip install --no-cache-dir --retries 10 --timeout 120 /wheels/torch-2.5.1-*.whl
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt
COPY src ./src
COPY scripts ./scripts
ENV PYTHONPATH=/app/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
