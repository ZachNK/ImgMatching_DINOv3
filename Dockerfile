# CUDA + PyTorch 런타임 이미지 (필요시 버전 조정)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 라이브러리 (OpenCV 등 의존)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 작업 디렉토리: 이후 compose에서 /workspace/project 를 마운트하여 사용
WORKDIR /workspace/project

# --- 콘솔 스크립트(래퍼) 설치 ---
# 패키징 없이도, 마운트된 프로젝트 파일(runCLI.py / visualize.py)을 호출하는 래퍼를 만든다.
RUN printf '#!/usr/bin/env bash\npython /workspace/project/runCLI.py "$@"\n' \
      > /usr/local/bin/run-matching \
 && printf '#!/usr/bin/env bash\npython /workspace/project/visualize.py "$@"\n' \
      > /usr/local/bin/run-visualize \
 && chmod +x /usr/local/bin/run-matching /usr/local/bin/run-visualize

# 기본은 대기(Compose에서 exec로 명령 실행)
CMD ["bash", "-lc", "sleep infinity"]


# --- 이전 버전: 비루트 사용자 설정 포함 ---
# FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# # 기본 유틸 및 tini
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       git ca-certificates tini && \
#     rm -rf /var/lib/apt/lists/*

# # 비루트 사용자
# ARG USERNAME=appuser
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g ${GID} ${USERNAME} && \
#     useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# USER ${USERNAME}
# WORKDIR /workspace

# # 파이썬 의존성
# COPY requirements.txt /tmp/requirements.txt
# RUN python -m pip install --upgrade pip setuptools wheel && \
#     pip install --no-cache-dir -r /tmp/requirements.txt

# # 로그 버퍼링 X
# ENV PYTHONUNBUFFERED=1

# # 엔트리포인트
# ENTRYPOINT ["/usr/bin/tini","--"]
# CMD ["/bin/bash"]
