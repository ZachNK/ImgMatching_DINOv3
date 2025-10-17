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
<<<<<<< HEAD
=======
````

---

## 1) 폴더 구조(권장)

Windows의 드라이브 경로는 WSL에서 `/mnt/<드라이브문자>/...` 로 접근합니다.

* (Windows) `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_weights` → (WSL) `/mnt/d/GoogleDrive/KNK_Lab/_Projects/dinov3_weights`
* (Windows) `D:\GoogleDrive\KNK_Lab\_Datasets\shinsung_data` → (WSL) `/mnt/d/GoogleDrive/KNK_Lab/_Datasets/shinsung_data`
* (Windows) `D:\Exports` → (WSL) `/mnt/d/Exports`

레포(이 프로젝트)는 WSL 홈에 두는 것을 권장:

```
~/dinov3-docker
  ├ project/
  ├ Dockerfile
  ├ docker-compose.yml
  ├ requirements.txt
  ├ .env.example  ← 이 파일을 .env 로 복사 후 편집
  └ README.md
>>>>>>> update README.md
```

---

<<<<<<< HEAD
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
=======
## 2) 설치 순서 (처음 세팅)

### 2-1) 레포 클론

```bash
git clone https://github.com/<YOUR_ORG>/<YOUR_REPO>.git
cd <YOUR_REPO>      # 예: cd dinov3-docker
```

### 2-2) dinov3 리포지토리 준비 (hubconf.py 필요)

`torch.hub.load(..., source="local")`을 위해 **리포 루트(hubconf.py가 있는 경로)** 가 필요합니다.

```bash
# WSL 홈에 클론 (경로는 자유)
git clone https://github.com/facebookresearch/dinov3.git ~/dinov3-src

# 확인: ~/dinov3-src 안에 hubconf.py와 dinov3/ 폴더가 있어야 함
ls -al ~/dinov3-src | head
```

### 2-3) 가중치(체크포인트) 설치

원하는 위치(예: Windows D 드라이브)에 아래 구조로 폴더를 준비하고, .pth 파일을 배치하세요.

```
/mnt/d/GoogleDrive/KNK_Lab/_Projects/dinov3_weights
├─ 01_ViT_LVD-1689M
│  ├─ dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
│  ├─ dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
│  ├─ dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
│  ├─ dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│  ├─ dinov3_vits16_pretrain_lvd1689m-08c60483.pth
│  └─ dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
├─ 02_ConvNeXT_LVD-1689M
│  ├─ dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
│  ├─ dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
│  ├─ dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
│  └─ dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
└─ 03_ViT_SAT-493M
   ├─ dinov3_vit7b16_pretrain_sat493m-a6675841.pth
   └─ dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

> 이 프로젝트의 **가중치 별 별칭**(CLI `--weights`에서 사용):
>
> * `vit7b16`, `vitb16`, `vith16+`, `vitl16`, `vits16`, `vits16+`
> * `cxBase`, `cxLarge`, `cxSmall`, `cxTiny`
> * `vit7b16sat`, `vitl16sat`

### 2-4) 데이터셋 설치

데이터셋 루트 아래에 실제 이미지들이 존재해야 하며(서브 폴더 허용), 기본 파일명 패턴은
`..._<ALT(3자리)>_<FRAME(4자리)>.ext` 입니다. 예:

```
/mnt/d/GoogleDrive/KNK_Lab/_Datasets/shinsung_data
├─ 250912150549_400/250912150549_400_0100.jpg
├─ 250912161658_200/250912161658_200_0100.jpg
└─ ... (기타 폴더/파일)
```

> 패턴이 다르면 `run-matching` 실행 시 `--regex` 로 맞춰줄 수 있습니다.

### 2-5) 환경변수 파일(.env) 작성

템플릿 복사 후 본인 환경에 맞춰 수정:

```bash
cp .env.example .env
nano .env
```

예시(WSL 사용자: `nkangzach` / Windows D 드라이브 기준):

```env
IMAGE_NAME=dinov3:cuda12.1-py310
TZ=Asia/Seoul

PROJECT_HOST=/home/nkangzach/dinov3-docker/project
CODE_HOST=/home/nkangzach/dinov3-src

WEIGHTS_HOST=/mnt/d/GoogleDrive/KNK_Lab/_Projects/dinov3_weights
DATASET_HOST=/mnt/d/GoogleDrive/KNK_Lab/_Datasets/shinsung_data
EXPORT_HOST=/mnt/d/Exports

REPO_DIR=/workspace/dinov3
IMG_ROOT=/opt/datasets
EXPORT_DIR=/exports/dinov3_embeds
PAIR_VIZ_DIR=/exports/pair_viz

DINOV3_BLOCK_NET=1   # 재현성/보안을 위해 허브 네트워크 차단(권장)
```

> ⚠️ `.env`는 개인 경로가 포함되므로 **레포에 커밋하지 마세요**.

### 2-6) 컨테이너 빌드 & 기동

```bash
docker compose build --no-cache
docker compose up -d

# 확인
>>>>>>> update README.md
docker compose exec pair nvidia-smi
docker compose exec pair bash -lc 'echo REPO_DIR=$REPO_DIR IMG_ROOT=$IMG_ROOT; ls -al $REPO_DIR | head'
```

<<<<<<< HEAD
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
=======
---

## 3) 스모크 테스트

```bash
# 예: 400.0100 ↔ 200.0100 한 건, vitl16 가중치
>>>>>>> update README.md
docker compose exec pair run-matching \
  --weights vitl16 \
  -a 400.0100 -b 200.0100

<<<<<<< HEAD
# 시각화 (Affine RANSAC, 포인트 표시)
=======
# 시각화 (Affine RANSAC + 포인트)
>>>>>>> update README.md
docker compose exec pair run-visualize \
  --ransac affine --draw-points --alpha 180 --linewidth 3
```

<<<<<<< HEAD
생성 파일 위치:

* **매칭 JSON**: `/exports/pair_match/<weight>_<Aalt>_<Aframe>/<weight>_<Aalt.Aframe>_<Balt.Bframe>.json`
* **시각화 PNG**: `/exports/pair_viz/…/*.png`
  Windows에서는 `D:\Exports\...`로 확인 가능.

---

## 3) 대규모 실행 (All-vs-All / 여러 가중치)
=======
출력 경로:

* 매칭 JSON: `/exports/pair_match/<weight>_<Aalt>_<Aframe>/<weight>_<Aalt.Aframe>_<Balt.Bframe>.json`
* 시각화 PNG: `/exports/pair_viz/.../*.png`
* (Windows) `D:\Exports\...` 에서 확인

---

## 4) 대규모 실행 (All-vs-All / 여러 가중치)
>>>>>>> update README.md

```bash
# 전체 이미지 × 전체 이미지, 모든 가중치
docker compose exec pair run-matching --all-weights

# 특정 그룹만 (예: SAT-493M)
docker compose exec pair run-matching --group ViT_SAT493M

# 임의 조합
docker compose exec pair run-matching --weights vitl16 cxSmall vitl16sat
```

---

<<<<<<< HEAD
## 4) 하이퍼파라미터 / 옵션
=======
## 5) 주요 옵션
>>>>>>> update README.md

**매칭(run-matching)**

* `--image-size` (기본 336)
* `--mutual-k` (기본 10)
* `--topk` (기본 400)
* `--max-patches` (기본 0 → 전체 사용, >0 이면 균등 서브샘플링)
<<<<<<< HEAD
* `--regex` (파일명 패턴, 기본: `.*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(...)$`)
* `--weights` / `--group` / `--all-weights` (가중치 선택)

**시각화(run-visualize)**

* `--ransac {off|affine|homography}` (기본 off)
=======
* `--regex` (파일명 패턴. 기본: `.*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(jpg|jpeg|png|bmp|tif|tiff|webp)$`)
* `--weights` / `--group` / `--all-weights`

**시각화(run-visualize)**

* `--ransac {off|affine|homography}`
>>>>>>> update README.md
* `--reproj-th`, `--confidence`, `--iters`
* `--max-lines`, `--linewidth`, `--draw-points`, `--alpha`

---

<<<<<<< HEAD
## 5) 트러블슈팅(FAQ)

* **hubconf.py를 못 찾음**
  `.env`의 `CODE_HOST`는 **hubconf.py가 있는 리포 루트**여야 함 → 컨테이너 내부 `REPO_DIR`(`/workspace/dinov3`)에서 `ls -al` 했을 때 `hubconf.py`가 보여야 함.

* **No images matched under …**
  `.env`의 `IMG_ROOT`가 실제 이미지 루트인지 확인.
  파일명이 기본 정규식(`..._###_####.ext`)과 다르면 `--regex`로 완화:
=======
## 6) 트러블슈팅

* **hubconf.py를 못 찾음**
  `.env`의 `CODE_HOST`가 **hubconf.py가 있는 리포 루트**인지 확인.
  컨테이너 내부 `REPO_DIR` 경로(`/workspace/dinov3`)에서 `hubconf.py`가 보여야 함.

* **No images matched under …**
  `.env`의 `IMG_ROOT`가 실제 이미지 루트인지 확인.
  파일명이 기본 패턴과 다르면 `--regex` 완화:
>>>>>>> update README.md

  ```bash
  --regex '.*_(?P<alt>\d+)_(?P<frame>\d+)\.(jpg|jpeg|png|bmp|tif|tiff|webp)$'
  ```

* **Docker 명령이 WSL에서 안 보임**
  Docker Desktop → Settings → Resources → **WSL Integration**에서 Ubuntu ON,
<<<<<<< HEAD
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
=======
  관리자 PowerShell에서 `wsl --shutdown` 후 재실행.

* **GPU가 안 보임**
  `docker compose exec pair nvidia-smi` 확인. 보이지 않으면 호스트 드라이버/WSL 통합 점검.

---

## 7) 폴더 구조

```
project/
  imatch/            # 라이브러리 모듈
  runCLI.py          # 매칭 실행기
  visualize.py       # 시각화
>>>>>>> update README.md
Dockerfile
docker-compose.yml
requirements.txt
.env.example         # ← 이걸 복사해 .env 작성
```

---

<<<<<<< HEAD

=======
>>>>>>> update README.md
