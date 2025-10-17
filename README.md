# DINOv3 Image Matching (Docker + WSL2)

DINOv3 기반 **이미지 매칭 & 시각화** 파이프라인입니다.  
WSL2 + Docker Desktop + NVIDIA GPU 환경에서 재현 가능하도록 구성되어 있으며,  
여러 가중치(백본)와 All-vs-All 매칭을 손쉽게 실행할 수 있습니다.

> 내부 매칭은 DINOv3 패치 토큰을 이용한 **mutual k-NN** 방식이며, 결과는 JSON으로 저장되고, 선택적 **RANSAC**을 거쳐 PNG로 시각화됩니다.

---

## 0) 요구사항

- **Windows 11 + WSL2** (Ubuntu 20.04/22.04)
- **Docker Desktop** (Settings → Resources → **WSL Integration** 에서 Ubuntu ON)
- **NVIDIA GPU & Driver** (컨테이너에서 GPU가 보여야 함)
- (권장) `git`, `make`

확인:
```bash
nvidia-smi
```

---

## 1) 설치

```bash
# 1) 레포 받기
git clone https://github.com/<YOUR_ORG>/<YOUR_REPO>.git
cd <YOUR_REPO>

# 2) 환경변수 템플릿 복사 → 편집
cp .env.example .env
# .env를 열어 아래 2가지를 반드시 본인 환경에 맞춰 수정
#   - <WSL 경로>로 수정 (Windows D:\... → WSL /mnt/d/...)
#   - Ubuntu 사용자명 반영 (/home/<user>/...)

# 3) 빌드 & 기동
docker compose build --no-cache
docker compose up -d

# 4) 기본 점검
docker compose exec pair nvidia-smi
docker compose exec pair bash -lc 'echo REPO_DIR=$REPO_DIR IMG_ROOT=$IMG_ROOT; ls -al $REPO_DIR | head'
```

### .env 주요 항목 (요약)

* `PROJECT_HOST` : 코드(`project/`)가 있는 WSL 경로
  예) `/home/<user>/dinov3-docker/project`
* `CODE_HOST` : **hubconf.py가 있는 dinov3 리포 루트**
  예) `/home/<user>/dinov3-src` (`hubconf.py` + `dinov3/` 폴더가 있어야 함)
* `WEIGHTS_HOST` : 가중치 디렉토리(WSL 경로)
  예) `D:\GoogleDrive\...\dinov3_weights` → `/mnt/d/GoogleDrive/.../dinov3_weights`
* `DATASET_HOST` : 데이터셋 루트(WSL 경로)
* `EXPORT_HOST` : 결과물 저장 루트(WSL 경로)
* `REPO_DIR` : 컨테이너 내부의 dinov3 리포 경로(기본 `/workspace/dinov3`)
* `IMG_ROOT` : 컨테이너 내부의 데이터셋 루트(기본 `/opt/datasets`)
* `DINOV3_BLOCK_NET=1` : `torch.hub`의 외부 다운로드 차단(재현성/보안 목적, 권장)

---

## 2) 스모크 테스트

```bash
# 예시: 400.0100 ↔ 200.0100 한 건만, vitl16 가중치
docker compose exec pair run-matching \
  --weights vitl16 \
  -a 400.0100 -b 200.0100

# 시각화 (Affine RANSAC, 포인트 표시)
docker compose exec pair run-visualize \
  --ransac affine --draw-points --alpha 180 --linewidth 3
```

생성 파일 위치:

* **매칭 JSON**: `/exports/pair_match/<weight>_<Aalt>_<Aframe>/<weight>_<Aalt.Aframe>_<Balt.Bframe>.json`
* **시각화 PNG**: `/exports/pair_viz/…/*.png`
  Windows에서는 `D:\Exports\...`로 확인 가능.

---

## 3) 대규모 실행 (All-vs-All / 여러 가중치)

```bash
# 전체 이미지 × 전체 이미지, 모든 가중치
docker compose exec pair run-matching --all-weights

# 특정 그룹만 (예: SAT-493M)
docker compose exec pair run-matching --group ViT_SAT493M

# 임의 조합
docker compose exec pair run-matching --weights vitl16 cxSmall vitl16sat
```

---

## 4) 하이퍼파라미터 / 옵션

**매칭(run-matching)**

* `--image-size` (기본 336)
* `--mutual-k` (기본 10)
* `--topk` (기본 400)
* `--max-patches` (기본 0 → 전체 사용, >0 이면 균등 서브샘플링)
* `--regex` (파일명 패턴, 기본: `.*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(...)$`)
* `--weights` / `--group` / `--all-weights` (가중치 선택)

**시각화(run-visualize)**

* `--ransac {off|affine|homography}` (기본 off)
* `--reproj-th`, `--confidence`, `--iters`
* `--max-lines`, `--linewidth`, `--draw-points`, `--alpha`

---

## 5) 트러블슈팅(FAQ)

* **hubconf.py를 못 찾음**
  `.env`의 `CODE_HOST`는 **hubconf.py가 있는 리포 루트**여야 함 → 컨테이너 내부 `REPO_DIR`(`/workspace/dinov3`)에서 `ls -al` 했을 때 `hubconf.py`가 보여야 함.

* **No images matched under …**
  `.env`의 `IMG_ROOT`가 실제 이미지 루트인지 확인.
  파일명이 기본 정규식(`..._###_####.ext`)과 다르면 `--regex`로 완화:

  ```bash
  --regex '.*_(?P<alt>\d+)_(?P<frame>\d+)\.(jpg|jpeg|png|bmp|tif|tiff|webp)$'
  ```

* **Docker 명령이 WSL에서 안 보임**
  Docker Desktop → Settings → Resources → **WSL Integration**에서 Ubuntu ON,
  PowerShell(관리자)에서 `wsl --shutdown` 후 재실행.

* **GPU가 안 보임**
  `docker compose exec pair nvidia-smi`가 실패하면 호스트 NVIDIA 드라이버/WSL 통합 점검.

---

## 6) 폴더 구조

```
project/
  imatch/            # 라이브러리 모듈들
  runCLI.py          # 매칭 실행기 (콘솔 진입점)
  visualize.py       # 시각화 스크립트
Dockerfile
docker-compose.yml
requirements.txt
.env.example         # ← 이걸 복사해 .env 작성
```

---

