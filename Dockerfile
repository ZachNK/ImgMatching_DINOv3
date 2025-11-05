FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/project

RUN printf '#!/usr/bin/env bash\npython /workspace/project/run.py "$@"\n' \
      > /usr/local/bin/run \
 && printf '#!/usr/bin/env bash\npython /workspace/project/visualize.py "$@"\n' \
      > /usr/local/bin/vis \
 && chmod +x /usr/local/bin/run /usr/local/bin/vis

CMD ["bash", "-lc", "sleep infinity"]
