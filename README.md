# DINOv3 Image Matching (Docker)

Docker Desktop 위에서 DINOv3 기반 이미지 매칭과 시각화를 수행하기 위한 프로젝트.  
컨테이너 안에서는 1:1 매칭을 수행하도록 구성되어 있으며, 결과(JSON/PNG)는 호스트의 지정된 디렉터리에 저장.

<p align="center">
  <img src="docs/figs/sequence_runner.svg" width = "75%"/>
</p>
<p align="center"><em>전체 코드의 호출, 의존관계 시퀀스 다이어그램</em></p>


---

## 0) 요구 사항

- **Windows 11** + **Docker Desktop (v.4.46.0 이상)**  
  - Docker Desktop 환경에서 동작.  
  - Docker Desktop Settings → Resources → File Sharing 에서 프로젝트/데이터 폴더가 공유되어 있는지 확인.
- **NVIDIA GPU & 최신 드라이버** (CUDA 12.x 호환)
- **NVIDIA Container Toolkit** (Docker Desktop 설치 시 자동 포함)
- 권장 체크 명령
  ```powershell
  docker --version
  nvidia-smi
  ```


### 0-1) Docker Desktop 설치
- 개인 PC 운영체제에 맞는 Docker Desktop 다운로드 후 설치 
- 본 프로젝트는 `4.46.0` 버전 사용

- 개인 PC 시스템 환경 확인
```powershell
Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Architecture
```

  > _출력결과_:
  ```powershell
  # x64 (AMD64)
  9 

  # ARM64
  12
  ```

- Docker Desktop 설치: 
  > https://www.docker.com/

<p align="center">
  <img src="docs/figs/docker_desktop_main.png" width = "75%"/>
</p>
<p align="center"><em>(출력결과 9: AMD64 설치, 출력결과 12: ARM64 설치)</em></p>
<p align="center"><em>대부분 Desktop/노트북은 x64(AMD64), Intel CPU사용하더라도 AMD64를 받는것이 일반적</em></p>


### 0-2) Docker Desktop 설치 후 기본 설정
- Docker Desktop 실행 → 리소스 제한 (CPU/Memory) → WSL2 (Windows) 연동 등 환경 설정 확인
- 프로젝트에 필요한 GPU 관련 드라이버/Container Runtime(NVIDIA Container toolkit) 설치

### 0-3) 프로젝트 디렉터리 준비
- 로컬 경로를 미리 생성한다.\
  작업할 디렉토리: `<Your>\<Project>\<Directory>` 로 가정\
  활용 데이터셋 디렉토리: `<Your>\<Datasets>\<Directory>`로 분리\
  작업 결과 디렉토리: `<Your>\<Project_Exports>\<Directory>`로 분리\

  
  ```bash
  mkdir <Your>\<Project>\<Directory>\dinov3_main      # 본 프로젝트 경로
  mkdir <Your>\<Project>\<Directory>\dinov3_src       # DINOv3
  mkdir <Your>\<Project>\<Directory>\dinov3_weights   # DINOv3에서 제공한 백본 경로
  mkdir <Your>\<Datasets>\<Directory>\dinov_data      # 활용할 입력 데이터셋 경로
  mkdir <Your>\<Project_Exports>\<Directory>\dinov3_exports   # 본 프로젝트의 출력 저장 경로
  ```

- 그리고 Docker Desktop에 Docker Desktop Settings → Resources → File Sharing 에서 프로젝트/데이터 폴더가 공유되어 있는지 확인
<p align="center">
  <img src="docs/figs/docker_desktop_filesharing.png" width="75%">
</p>
<p align="center"><em>File Sharing에서 프로젝트/데이터 폴더가 공유되어 있는지 확인 (본 프로젝트는 D:에 공유됨)</em></p>


### 0-4) 환경 변수 파일 작성
- `.env.example` 파일을 이용하여 `.env` 파일을 생성해야 한다.
- `.env.example` 파일을 `.env` 파일명으로 복사:
  ```bash
  cp .env.example .env
  ```

- `.env`파일을 연다.
- 건드려야 할 곳은 `호스트 경로` 부분 5군데이다. 나머지는 건들지 않는 곳.

  `.env`에서 자신의 환경에 맞게 수정. 모든 경로는 **Windows 경로**로 작성.

  | 변수 | 설명 | 예시 (Windows) |
  | --- | --- | --- |
  | `PROJECT_HOST` | `project/` 폴더 실경로 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_main\project` |
  | `CODE_HOST` | dinov3 원본 리포지터리 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_src` |
  | `WEIGHTS_HOST` | `.pth` 가중치 루트 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_weights` |
| `DATASET_HOST` | 이미지 데이터셋 루트 | `D:\Datasets\dinov_data` |
| `EXPORT_HOST` | JSON/PNG 결과 저장 루트 | `D:\Project_Exports\dinov3_exports` |


### 0-5) Docker Compose 빌드 단계
- 본격적으로 docker로 사용하기에 앞서 프로젝트 루트에서  `docker compose build` 실행.
  ```powershell
  docker compose build
  ```
  
  * 그러면 빌드 하면서 마지막에 
  ```powershell
  [+] Building 1/1
  ✔ dinov3:cuda12.1-py310  Built
  ```

### 0-6) Docker 컨테이너 실행 및 확인
- Docker 컨테이너 실행:
  ```powershell
  docker compose up -d
  ```
  
  * 그러면 컨테이너가 실행 준비 완료 되었다는 것을 다음과 같이 나온다:
  ```powershell
  [+] Running 2/2
  ✔ Network dinov3_main_default  Created                   0.0s 
  ✔ Container dinov3-matching    Started                   0.5s 
  ```

- 컨테이너 실행 하는지 확인:
  ```powershell
  docker compose ps
  ```

  * 그려면 아래와 같이 나옴:
  ```bash
  NAME              IMAGE                   COMMAND                   SERVICE    CREATED         STATUS         PORTS
  dinov3-matching   dinov3:cuda12.1-py310   "bash -lc 'sleep inf…"   matching   8 seconds ago   Up 7 seconds
  ```

### 0-7) Docker 초기 진입/테스트 
- 컨테이너 쉘 진입하여 기본 매칭/시각화 명령을 한번씩 수행하고, 결과 파일이 HOST경로에 생성되는지 확인:
  ```powershell
  docker compose exec matching bash
  ```
  
  * 그러면 아래와 같이 나오면 컨테이너 쉘 진입 확인 완료
  ```powershell
  root@{...}:/workspace/project#
  ```
  * `exit` 명령어로 쉘 나오기
  ```powershell
  root@{...}:/workspace/project# exit
  ```

- GPU 인식 체크로 드라이버/Toolkit 연동 상태 확인
  ```powershell
  docker compose exec matching nvidia-smi
  ```
  * `NVIDIA-SMI ~ Driver Version ~` 등 뜨면 정상적으로 Toolkit 연동

---

## 1) 저장소 구조 & 필수 리소스

- 아래와 같이 `dinov3_main` 의 디렉터리는 다음과 같이 있어야 한다.

```bash
dinov3_main/
├─ project/
│  ├─ imatch/           # 라이브러리 모듈
│  ├─ run.py            # 매칭 실행 엔트리
│  └─ visualize.py      # 시각화 엔트리
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .env
├─ .env.example
└─ README.md
```

필수 리소스 (작업할 디렉토리: `<Your>\<Project>\<Directory>` 라고 가정)
- **본 실행 프로젝트**  
  > _예시 위치:_ `<Your>\<Project>\<Directory>\dinov3_main`

- **facebookresearch/dinov3** 저장소 (코드 참조용)  
  > _예시 위치:_ `<Your>\<Project>\<Directory>\dinov3_src`

- **사전 학습 가중치(.pth)**  
  > _예시 위치:_ `<Your>\<Project>\<Directory>\dinov3_weights`

- **매칭 대상 이미지 데이터셋**  
  > _예시 위치:_ `<Your>\<Datasets>\<Directory>\dinov_data`

- **결과 저장 디렉터리**  
  > _예시 위치:_ `<Your>\<Project_Exports>\<Directory>\dinov3_exports`


### 1-1) 프로젝트 저장


- 작업하고자 하는 디렉토리(_`<Your>\<Project>\<Directory>`_)에 먼저 접근하여 본 프로젝트를 `dinov3_main` 하위 경로에 clone한다. 

  ```Bash
  git clone https://github.com/ZachNK/ImgMatching_DINOv3.git .\dinov3_main
  ```

### 1-2) DINOv3 원본 저장


- 작업할 경로 (_`<Your>\<Project>\<Directory>`_)에서 `dinov3_src` 하위 경로에 DINOv3 원본을 저장한다.

  ```Bash
  git clone https://github.com/facebookresearch/dinov3.git .\dinov3_src
  ```

### 1-3) 백본 백본 준비 

- _`<Your>\<Project>\<Directory>`_ 경로에 `dinov3_weights` 디렉토리에 백본 데이터를 준비 한다.\
  https://github.com/facebookresearch/dinov3에 게시된 가중치를 `dinov3_weights`에 바로 저장한다.

- `dinov3_weights`에는 백본 종류별로 다시 디렉토리를 생성해야 한다:
  ```bash
  # dinov3_weights에 디렉토리 추가 생성
  New-Item -ItemType Directory -Path <Your>\<Project>\<Directory>\dinov3_weights\01_ViT_LVD-1689M -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Path <Your>\<Project>\<Directory>\dinov3_weights\02_ConvNeXT_LVD-1689M -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Path <Your>\<Project>\<Directory>\dinov3_weights\03_ViT_SAT-493M -ErrorAction SilentlyContinue
  ```

* 각 세부 디렉토리별로 pth 파일들을 이동한다 
  (아래는 CLI명령 예시)
  ```powershell
  # dinov3_weights 디렉토리에 저장된 .pth 파일들 데이터셋별로 정리

  # 1) dinov3_weights\01_ViT_LVD-1689M에 파일 이동 (ViT-S/16 distilled 이동할 때)
  Move-Item -Path <Your>\<Project>\<Directory>\dinov3_vits16_pretrain_lvd1689m-08c60483.pth -Destination <Your>\<Project>\<Directory>\dinov3_weights\01_ViT_LVD-1689M

  # ... 나머지 ViT-S+/16 distilled, ViT-B/16 distilled 등 .pth파일 이동

  # 2) dinov3_weights\02_ConvNeXT_LVD-1689M에 파일 이동 (ConvNeXt Tiny 이동할 때)
  Move-Item -Path <Your>\<Project>\<Directory>\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth -Destination <Your>\<Project>\<Directory>\dinov3_weights\01_ViT_LVD-1689M

  # ... 나머지 ConvNeXt Small, ConvNeXt Base 등 .pth파일 이동

  # 3) dinov3_weights\03_ViT_SAT-493M에 파일 이동 (ViT-L/16 distilled 이동할 때)
  Move-Item -Path <Your>\<Project>\<Directory>\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth -Destination <Your>\<Project>\<Directory>\dinov3_weights\01_ViT_LVD-1689M

  # ... 나머지 dinov3_vit7b16_pretrain_sat493m-a6675841.pth .pth파일 이동
  ```


### 1-4) 데이터셋 준비

- 데이터셋은 `dinov_data`의 디렉토리를 본 프로젝트를 수행하는 경로와 다른 곳에 저장한다. (저장하는 데이터셋, 임베딩 파일 및 쿼리 이미지 용량이 매우 크기 때문에, 적절히 메모리 환경이 충분한 곳에 골라 저장)

- 본 프로젝트의 저장 경로를 `<Your>\<Project>\<Directory>`라 한다면, 본 프로젝트에 활용할 데이터셋의 경로는 `<Your>\<Datasets>\<Directory>`에 저장함. 

- `dinov_data` 경로에 활용할 데이터셋은 아래와 같이 일관된 경로로 수정해야 한다.\
  `<ID>`는 세부 데이터셋 명이고, `<ALT>`는 항공 사진의 고도, `<FRAME>`은 해당 고도에서 촬영한 이미지 순번.

  ```bash
  <Your>\<Datasets>\<Directory>\dinov_data
    └─<Your>\<Datasets>\<Directory>\dinov_data\<ID>_<ALT>
        └─<Your>\<Datasets>\<Directory>\dinov_data\<ID>_<ALT>\<ID>_<ALT>_<FRAME>.jpg
  ```

- 본 프로젝트의 데이터셋 경로 예시
  ```bash
  D:\dinov_data
    └─D:\dinov_data\250912143954_450
        └─D:\dinov_data\250912143954_450\250912143954_450_0001.jpg
  ```

### 1-5) 디렉토리 최종

- 본 프로젝트 `dinov3_main`에서 실행한 후 도출한 결과들을 저장할 디렉토리 `dinov3_exports`에 생성한다.\
  최종 경로 상태는 아래와 같다:

  ```text
  <Your>\<Project>\<Directory>\
  ├─ dinov3_main\
  │  ├─ project\
  │  │  ├─ __pycache__\  
  │  │  ├─ imatch\
  │  │  │  ├─ extracting.py
  │  │  │  ├─ loading.py
  │  │  │  ├─ matching.py
  │  │  │  └─ ...
  │  │  ├─ json\
  │  │  │  ├─ data_key.json
  │  │  │  └─ manifest.json
  │  │  ├─ analyze_rotaion_similarity.py
  │  │  ├─ Generate_DenseFT.py
  │  │  ├─ Generate_Query.py
  │  │  ├─ run_Img2DenseFT.py
  │  │  ├─ run_manifest.py
  │  │  └─ ...
  │  ├─ Dockerfile
  │  ├─ docker-compose.yml
  │  ├─ requirements.txt
  │  ├─ .env
  │  ├─ .env.example
  │  └─ README.md
  ├─ dinov3_src\                 # facebookresearch/dinov3 clone
  │  ├─ .github
  │  ├─ __pycache__  
  │  ├─ dinov3  
  │  ├─ notebooks  
  │  ├─ .docstr.yaml
  │  └─ hubconf.py 등...
  ├─ dinov3_weights\
  │  ├─ 01_ViT_LVD-1689M\
  │  │  └─ *.pth
  │  ├─ 02_ConvNeXT_LVD-1689M\
  │  │  └─ *.pth
  │  └─ … (필요한 가중치별 디렉터리)

  <Your>\<Datasets>\<Directory>\
  └─ dinov_data\                # 매칭 대상 이미지/데이터셋
    └─ … (프로젝트별 입력 데이터)

  <Your>\<Project_Exports>\<Directory>\
  └─ dinov3_exports\             # 결과(JSON/PNG/npy) 저장
    ├─ dinov3_embeds\
    ├─ pair_match\
    └─ pair_vis\
  ```

* **추가 파일 (updated 25.10.30)**

  임베딩 파악을 위한 테스트 모듈:

    `project/Test_patch_embedding.py` 
    — runs a test/sanity script that loads DINOv3, extracts patch‑level embeddings for a chosen image, and saves or inspects the raw patch token tensors so you can verify patch pipeline behaviour.

    `project/Test_global_embedding.py` 
    — similar harness for computing the global CLS embedding from DINOv3; it hardcodes image/checkpoint paths and exports the resulting global feature vector.

    `project/feature_map.py` 
    — end‑to‑end script that loads DINOv3, processes the specified image, computes the full patch–patch cosine similarity matrix, and writes both the flat map and reshaped grid .npy files under /exports.

    `project/display_results.py` 
    — interactive CLI tool that asks for an original image and up to 100 feature-map images, then builds a Matplotlib figure showing the original on top and all selected maps below with configurable padding; optionally saves the composed figure.


---

## 2) Docker 이미지 빌드 & 컨테이너 실행

```powershell
docker compose build        # Dockerfile 변경 시 재빌드
docker compose up -d        # 컨테이너 백그라운드 실행
docker compose ps           # 상태 확인
```

변경 사항 적용 또는 `.env`를 수정한 뒤에는 `docker compose up -d --force-recreate`로 재생성.  
GPU가 인식되는지 확인:

```powershell
docker compose exec matching nvidia-smi
```

---

## 3) Embeddings

The focus of this release is producing reusable DINOv3 embeddings prior to any downstream matching or visualization. Inside the container all artifacts appear under `/exports/...`; on the host the same tree is mounted at `<Your>\<Project_Exports>\<Directory>\dinov3_exports`.

### 3-1) Token Types & Output Layout

| Token type | Files | Description | Host path example |
| --- | --- | --- | --- |
| `GlobalToken` | `.npy`, `_meta.json` (queries add `.csv`) | Stores the single CLS/global vector | `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_embeds\<weight>\<altitude>\GlobalToken` |
| `PatchToken` | `.npy`, `_meta.json` | Flattened patch tokens (N × C tensor) | `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_embeds\<weight>\<altitude>\PatchToken` |
| `PatchGrid` | `.npy`, `_meta.json` | Patch tokens reshaped to H × W × C grids | `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_embeds\<weight>\<altitude>\PatchGrid` |
| `DenseFT` | `.png` | PCA-based dense feature visualizations derived from PatchGrid | `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_embeds\<weight>\<altitude>\DenseFT` |
| `Query*` | `.npy`, `_meta.json`, Global `.csv` | Embeddings for rotated/cropped query images | `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_query_embeds\Q<weight_key>\<query_dir>` |

- Naming pattern: `TokenType_{embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{altitude}_{index}`. Query outputs append `{scene}_{altitude}_{index}_{tag}`.
- `_meta.json` just adds `_meta` to the base filename.
- DenseFT PNGs are generated from PatchGrid exports via `Generate_DenseFT.py` (dataset) or `Generate_DenseFT4Query.py` (query) and later referenced through the `files.dense_vis` slot.

```text
<Your>\<Project_Exports>\<Directory>\dinov3_exports
├─ dinov3_embeds
│  └─ vitl16
│     └─ 450
│        ├─ GlobalToken
│        │  ├─ GlobalToken_res1024_ImageNet_raw_dinov3_vitl16_SAT_450_0001.npy
│        │  └─ GlobalToken_res1024_ImageNet_raw_dinov3_vitl16_SAT_450_0001_meta.json
│        ├─ PatchToken
│        ├─ PatchGrid
│        └─ DenseFT
└─ dinov3_query_embeds
   └─ Qvitb16
      └─ Q250912161658_200
         ├─ QueryGlobal_*.npy /.csv / _meta.json
         ├─ QueryPatchToken_*.npy / _meta.json
         └─ QueryPatchGrid_*.npy / _meta.json
```

Dense feature PNGs produced from query PatchGrid tensors live under `<Your>\<Project_Exports>\<Directory>\dinov3_exports\dinov3_vis\Q<weight_key>\<query_dir>`.

### 3-2) Execution Scripts & Parameters

- `project/Test_Embedding.py`: runs `run_global_embedding()` for a single altitude/index/weight triple and emits Global/Patch/PatchGrid artifacts.
- `project/run_manifest.py`: batch runner that parses `project/json/manifest.json`, expands datasets/frames, and optionally calls `Generate_DenseFT.py` when `generate_denseft: true`.
- `project/Test_Embedding4Query.py`: scans `/exports/Q...` (host `<Your>\<Project_Exports>\<Directory>\dinov3_exports\Q250912161658_200`, etc.) to emit `QueryGlobal/QueryPatch*` outputs per query image.
- `project/Generate_DenseFT.py`, `project/Generate_DenseFT4Query.py`: convert PatchGrid tensors into 1024×1024 PCA-projected PNGs for datasets and queries respectively.
- Helper tools (`Generate_Query.py`, `Test_Embedding4Query.py`, etc.) depend on `imatch.loading` path constants, so ensure `.env` provides valid `DATASET_HOST`, `EXPORT_HOST`, and related variables.

Batch example:

```powershell
docker compose exec matching python project/run_manifest.py --manifest project/json/manifest.json
```

Single-run experiment:

```powershell
docker compose exec matching python -c "from Test_Embedding import run_global_embedding; run_global_embedding(altitude=400, index=1, weight='vitl16', target_res=1024, variant='mutual', variant_params={'norm_threshold':0.8})"
```

Key arguments for `run_global_embedding()`:

| Argument | Purpose | Notes |
| --- | --- | --- |
| `altitude` | Capture altitude registered in `data_key.json` | e.g. `400` |
| `index` | Frame index inside the altitude | `1` → `_0001` |
| `weight` | `weight_key` (`vitb16`, `vitl16`, `cxTiny`, …) | maps to `/opt/weights` |
| `target_res` | Resize resolution for the input image | default `1024`; affects PatchGrid |
| `variant` | Patch-token post-process (`raw`, `mutual`, `topk`, `subsample`, …) | implemented in `process_patch_tokens()` |
| `embedding_cfg` | Optional label inserted in filenames | default `res{target_res}_ImageNet` |
| `variant_params` | Dict overriding variant defaults | e.g. `{"topk": 256}` |
| `output_plan` | Controls which artifacts persist | `{global|patch|grid: {"npy": bool, "json": bool}}` |

`run_manifest.py` exposes the same knobs under `jobs[].embedding`; toggle `generate_denseft` to chain DenseFT creation. For query embeddings, update `QUERY_DIRS`, `VAR_WEIGHT_KEYS`, `VARIANT`, and `VARIANT_PARAMS` inside `Test_Embedding4Query.py`.

### 3-3) Patch Token Variants

`imatch/postprocess.py` registers the available strategies:

| Variant | Default params | Behavior | Output impact |
| --- | --- | --- | --- |
| `raw` | none | Keeps every patch token | `keep_ratio = 1.0`; PatchGrid size unchanged |
| `mutual` | `norm_threshold = 0.75` | Drops low-norm tokens (mutual-kNN proxy) | `matching_count` / `mutual_knn_tokens` reflect survivors |
| `topk` | `topk = 128` | Selects highest-norm `k` tokens | Logs `params.topk` and effective token count |
| `subsample` | `stride = 2` | Strided subsampling over the reshaped grid | `grid_shape` plus `keep_ratio` describe the reduced resolution |

Override these defaults via `variant_params` (single runs) or manifest `params` (batch). `_meta.json` captures the resulting ratios so you can audit the effect quickly.

### 3-4) Metadata Schema

Every `_meta.json` shares the same structure:

```json
{
  "run_id": "GlobalToken_res1024_ImageNet_raw_dinov3_vitl16_SAT_400_0001",
  "token_type": "GlobalToken",
  "config": { ... },
  "files": { ... },
  "metrics": { ... },
  "timing_ms": { ... },
  "resources": { ... }
}
```

- `config`: embedding parameters (`embedding_cfg`, `variant`, `variant_params`, `weight_id`, `dataset_type`, `altitude`, `index`, `prefix`, `target_res`, `rotations`, `aggregation`). Query outputs also include `query.source_file`, `query.tag`, and `query.query_dir`.
- `files`: pointers to emitted assets.

| Key | Meaning |
| --- | --- |
| `vector` | `.npy` containing the main tensor (Global/Patch/Grid/Query) |
| `csv` | Global token serialized as CSV (query pipeline) |
| `patch_tokens` | Patch token `.npy` reference |
| `patch_grid` | Query PatchGrid `.npy` reference |
| `dense_vis` | Placeholder for DenseFT PNGs |
| `index` | Reserved for future ANN/Faiss indices |

- `metrics`: Global tokens fix `token_count = 1`; Patch tokens log `token_count`, `embedding_dim`, `matching_count`, `mutual_knn_tokens`, `keep_ratio`; Patch grids add `grid_shape` and derived counts. Slots such as `recall@k`, `mAP`, `top1_precision` remain for downstream experiments.
- `timing_ms`: `global_forward`, `patch_forward`, `postprocess`, `index_build`, `query`, `pipeline_total`.
- `resources`: `gpu_peak_mem_mb`, `embedding_storage_bytes`, `index_size_bytes` (future use).

Query metas (`QueryGlobal_*_meta.json`, `QueryPatchToken_*_meta.json`) follow the same schema and always populate `config.query` plus `files.csv`. After creating DenseFT PNGs, place them beside the `.npy` files and update `files.dense_vis` when traceability is required.

> **Heads-up**: Matching/visualization sections will return once those flows stabilize; for now the README intentionally documents embedding steps only.

