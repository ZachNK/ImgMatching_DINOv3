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
  작업할 디렉토리: `<Your>\<Project>\<Directory>` 로 가정할 때,
  
  ```bash
  mkdir <Your>\<Project>\<Directory>\dinov3_main      # 본 프로젝트 경로
  mkdir <Your>\<Project>\<Directory>\dinov3_src       # DINOv3
  mkdir <Your>\<Project>\<Directory>\dinov3_weights   # DINOv3에서 제공한 백본 경로
  mkdir <Your>\<Project>\<Directory>\dinov3_data      # 활용할 입력 데이터셋 경로
  mkdir <Your>\<Project>\<Directory>\dinov3_exports   # 본 프로젝트의 출력 저장 경로
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
  | `DATASET_HOST` | 이미지 데이터셋 루트 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_data` |
  | `EXPORT_HOST` | JSON/PNG 결과 저장 루트 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_exports` |


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
  > _예시 위치:_ `<Your>\<Project>\<Directory>\dinov3_data`

- **결과 저장 디렉터리**  
  > _예시 위치:_ `<Your>\<Project>\<Directory>\dinov3_exports`


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

- 마찬가지로 데이터셋도 `dinov3_data` 디렉토리에 저장한다.

- `dinov3_data` 경로에 활용할 데이터셋은 아래와 같이 일관된 경로로 수정해야 한다.\
  `<ID>`는 세부 데이터셋 명이고, `<ALT>`는 항공 사진의 고도, `<FRAME>`은 해당 고도에서 촬영한 이미지 순번.

  ```bash
  <Your>\<Project>\<Directory>\dinov3_data
    └─<Your>\<Project>\<Directory>\dinov3_data\<ID>_<ALT>
        └─<Your>\<Project>\<Directory>\dinov3_data\<ID>_<ALT>\<ID>_<ALT>_<FRAME>.jpg
  ```

- 본 프로젝트의 데이터셋 경로 예시
  ```bash
  <Your>\<Project>\<Directory>\dinov3_data
    └─<Your>\<Project>\<Directory>\dinov3_data\250912143954_450
        └─<Your>\<Project>\<Directory>\dinov3_data\250912143954_450\250912143954_450_0001.jpg
  ```

### 1-5) 디렉토리 최종

- 본 프로젝트 `dinov3_main`에서 실행한 후 도출한 결과들을 저장할 디렉토리 `dinov3_exports`에 생성한다.\
  최종 경로 상태는 아래와 같다:

  ```text
  <Your>\<Project>\<Directory>\
  ├─ dinov3_main\
  │  ├─ project\
  │  │  ├─ imatch\
  │  │  ├─ run.py
  │  │  ├─ run2.py
  │  │  └─ visualize.py
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
  ├─ dinov3_data\                # 매칭 대상 이미지/데이터셋
  │  └─ … (프로젝트별 입력 데이터)
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

## 3) 매칭 실행 (`run`)

컨테이너 래퍼 명령은 `run` 

```powershell
docker compose exec matching run --weights vitl16 -a 400.0200 -b 200.0200
```

* 아래와 같이 실행 및 결과:
  * `-w vitl16`: 아래 예시의 경우, `ViT-L/16` 가중치로 활용
  * `-e`: 매칭 임베딩 결과 저장
  * `-a 400.0001`: A이미지는 `400`고도의 `1`번 이미지와
  * `-b 200.0001`: B이미지는 `200`고도의 `1`번 이미지를 서로 매칭 실행 
  
  
  <p align="center">
    <img src="docs/figs/matching_run option example.png" width="75%">
  </p>
  <p align="center"><em>이미지 매칭 실행 및 json, npy 저장 완료</em></p>
  
* **`ImageNet` 기반 학습 세트(1.28M 장의 이미지)의 픽셀 통계**:
  * 대부분의 ImageNet 기반 사전 학습 모델(ViT, DINO, MAE 등)은 학습 시, 입력을 $ (x-mean)/std $ 로 Normalization 했음.
  * 따라서 추론에서도 같은 통계를 사용하면 모델이 기대한 분포와 일치함.
  * 픽셀을 먼저 `ConvertImageDtype(torch.float32)/ToTensor()`으로 0~1 범위에 바꾼 뒤 해당 평균/표준편차를 적용해야 적절함.
  * (마찬가지로, 다른 데이터셋이나 사전학습 설정을 쓴 모델일 경우엔 그 모델이 학습 때 사용한 평균/표준편차 값으로 바꿔주는 것이 최선임)
  
* **픽셀 Normalization**:
  > 1. 먼저 이미지를 `unit8` (`0255 → float(01)`)로 변환
  <br>
  <div align='center'>
**Note**:
| `Variables` | `Description` |
| -----------: | :------------- |
| $x_{i,c}(u,v)$ | $i$ 번째 이미지의 픽셀 값 |
| $H_i, W_i$ | 해상도 |
| $N$ | 전체 이미지 수 |


* 한 채널 $c ∈ \{R,G,B\}$ 에 대한 평균:
  <details>
  <summary>mean for normalization</summary>
  $$\mu_c = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{1}{H_i W_i} \sum_{u=1}^{H_i} \sum_{v=1}^{W_i} \frac{x_{i,c}(u,v)}{255}\right)$$
  <br>
  </details>
  <br>

  > 2. 표준편차는 평균을 뺀 제곱을 평균 낸 뒤 루트 ($\sqrt{}$) 를 취한다 (보통 모집단의 표준편차를 사용). 

  <details>
  <summary>std for normalization</summary>
  
    $$\sigma_c = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{H_i W_i} \sum_{u=1}^{H_i} \sum_{v=1}^{W_i} \left(\frac{x_{i,c}(u,v)}{255} - \mu_c\right)^2 \right) }$$

  </details>

  <br>
    <details>
    <summary>ImageNet평균/분산을 누적 계산하면 다음과 같음:</summary>
      ImageNet을 순회하며 평균/분산을 누적 계산하는 스크립트

    ```powershell
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    dataset = datasets.ImageNet(
        root="/path/to/imagenet",
        split="train",
        transform=transforms.ToTensor(),  # 0~1 범위
    )

    ### DataLoader 파라미터 설명:
    # dataset: 로드할 데이터 세트
    # batch_size: 한번에 배치당 로드할 샘플 수(e.g. 256) 
    # num_workers: 데이터 로딩에 사용할 하위 프로세스의 수 (백그라운드에 이미지 읽고 전처리할 CPU스레드 수, CPU코어 수와 I/O상황에 맞게 조정)
    # 이 값들이 통계값 자체를 바꾸는 건 아님. 어디까지나 데이터와 로컬 환경에 맞는 실용적인 예시
    ###
    loader = DataLoader(dataset, batch_size=256, num_workers=8)

    mean = 0.0
    var = 0.0
    num = 0  # 누적 픽셀 수
    for images, _ in loader:
        # images shape: [B, 3, H, W]
        b, c, h, w = images.shape
        # B(dim=0)는 배치 차원
        # 3(dim=1)은 채널(RGB)원
        # H(dim=2)은 세로(높이) 차원
        # W(dim=3)은 가로(너비) 차원
        
        # 모든 픽셀 = 배치 사이즈 X 세로 x 가로
        pixels = b * h * w
        # 채널별 픽셀 합계
        mean += images.sum(dim=[0, 2, 3])
        # 각 픽셀을 제곱한 값을 채널별로 더해 ∑ x² 제공
        var += (images ** 2).sum(dim=[0, 2, 3])
        num += pixels

    mean /= num
    var /= num
    std = torch.sqrt(var - mean ** 2)
    print(mean, std)  # tensor([0.485..., 0.456..., 0.406...]) / ([0.229..., ...])
    ```
    </details>
    <br>

- `-a`, `-b`: ALT.FRAME 형식 (예: `400.0001`) \
  지정하지 않으면 모든 조합을 순회.

- `--weights`, 혹은 `-w`: 사용 가중치 변수\
  `--group`, `--all-weights` 옵션도 지원.

  | backbone | parameter |
  | -------- | ---- |
  | `ViT-S/16 distilled` | `vits16` |    
  | `ViT-S+/16 distilled` |`vits16+` |   
  | `ViT-B/16 distilled` | `vitb16` |
  | `ViT-L/16 distilled` | `vitl16` | 
  | `ViT-H+/16 distilled` | `vith16+` |
  | `ViT-7B/16` | `vit7b16` |
  | `ConvNeXt Tiny` | `cxTiny` |  
  | `ConvNeXt Small` | `cxSmall` |  
  | `ConvNeXt Base` | `cxBase` |   
  | `ConvNeXt Large` | `cxLarge` |  
  | `ViT-L/16 distilled` | `vitl16sat` | 
  | `ViT-7B/16` | `vit7b16sat` |


- 주요 튜닝 파라미터
  | 옵션 | 기본값 | 명칭 | 설명 |
  | --- | --- | --- | --- |
  | `--image-size` | 336 | 입력 해상도 | 입력 이미지를 몇개의 패치로 나뉠지 결정하는 지표 = 백본에 입력될 전처리 이후의 정사각형 해상도. 백본이 16×16 패치를 전제로 토큰을 만들기 때문에, 입력 해상도도 16의 배수로 맞춰 두는 편이 가장 안전. CLI에서 받은 값을 그대로 `build_transform()`의 첫번째 인자로 넘기며, 내부에서는 `target_size` = `image_size` * `patch_multiple`로 만들어 정사각형 리사이즈를 수행함. 결과적으로 ViT가 보는 해상도 및 토큰 수 ($N=(\frac{target_size}{patch_size_model})^{2}$)가 이 값에 의해 정해짐.  |
  | `--max-features` | 1000 | 패치 토큰 최대 개수 (균등 샘플링) | 패치 토큰을 최대 몇 개만 남길지 제한 하는 수. 토큰 수가 임계치를 넘으면 균등 간격으로 인덱스를 선택해 서브샘플링하고 , 이때 사용되는 로직은`subsample_tokens()`으로 정의됨. ViT 한 장에서 나오는 패치가 대용량일 때는 k-NN 매칭 비용을 제어함. |
  | `--match-th` | 0.1 | 유사도 절대 임계값 | 상호 k-NN으로 얻은 패치 쌍의 코사인 유사도를 절대값 임계로 컷 오프 함 (`compute_matches_mutual_knn()`). ViT 패치 임베딩은 정규화 후 내적이 되는 코사인 값이 매칭의 기준. |
  | `--keypoint-th` | 0.015 | 토큰 L2 임계값 | 패치 토큰의 L2거리를 0~1로 정규화한 후, 해당 임계값 이상인 토큰 만 남김 (`apply_keypoint_threshold()`). 해당 값이 낮을수록 대비가 약한 패치까지 살리고, 높일수록 에너지가 큰 (자기 표현력이 높은) 토큰만 남김. 모든 토큰이 걸러지면 가장 점수 높은 것 하나는 강제로 유지해서, 매칭이 빈 상태가 되지 않게 제어함. |
  | `--line-th` | 0.2 | 최고 유사도 대비 상대 임계값 | 최고 유사도 대비 상대 임계치를 적용. 최대 유사도 `sim.max()`에 `line_th`를 곱하여 두 번째 필터로 사용하고, Low ratio test처럼 상위 매칭과 너무 격차가 큰 쌍은 제거함. `match_th`가 전역적인 최소 기준이고, `line_th`는 한 이미지 쌍 내에서 상대 품질을 구분하는 역할. |

- 결과 JSON은 _`<Your>\<Project>\<Directory>\dinov3_exports/pair_match/<weight>_<ALT>_<FRAME>/…`_ 에 저장.

  <p align="center">
    <img src="docs/examples/vitl16_400_0001/JSON_vitl16_400.0001_200.0001.png" width="75%">
  </p>
  <p align="center"><em>이미지 매칭 실행 후 json파일 결과 예시</em></p>

### 3-1) 수정 버전 매칭 실행 (`run2`)


- `AutoImageProcessor/AutoModel`쓰는 방법으로 실행

  * 먼저 HF 토큰 확인: Hugging Face에서 DINOv3 모델은 gated 로그인이 필요.
    토큰 발급닥기 이전에 해당 Hugging Face 모델( facebook/dinov3-convnext-tiny-pretrain-lvd1689m 등)이 gated 모델이므로 접근 권한부터 부여 받아야 함.

  * Hugging Face에서 모델 페이지(https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m 등)를 열고, 로그인한 뒤, 접근 요청해 승인을 받아야 함.

  <p align="center">
    <img src="docs/figs/hf_facebook_weights_auth.png" width="75%">
  </p>
  <p align="center"><em>해당 백본 모델 접근 권한 승인을 위한 신청</em></p>

  * 그리고 나서, 요청이 승인될 때까지 기다리면, 모델 가중치 정상적으로 실행 가능.

  <p align="center">
    <img src="docs/figs/Gated Reops Status (pending).png" width="75%">
  </p>
  <p align="center"><em>백본 모델 접근 권한 승인 요청중</em></p>

- 승인이 완료된 것을 확인 후 명령 실행:

  <p align="center">
    <img src="docs/figs/Gated Reops Status (accepted).png" width="75%">
  </p>
  <p align="center"><em>백본 모델 접근 권한 승인 요청 완료</em></p>

  ```powershell
  docker compose exec matching bash -lc "HF_TOKEN=(token) python run2.py -w (weights) -a (a 이미지) -b (b 이미지)"
  ```

  <p align="center">
    <img src="docs/figs/matching_run2 option example.png" width="75%">
  </p>
  <p align="center"><em>이미지 매칭 (`run2`) 실행 및 json, npy저장 완료</em></p>

  <p align="center">
    <img src="docs/examples/cxTiny_400_0001/JSON_cxTiny_400.0001_200.0001.png" width="75%">
  </p>
  <p align="center"><em>이미지 매칭 (`run2`) 실행 후 json파일 결과 예시</em></p>


---

## 4) 시각화 (`vis`)

- 대화형으로 이미지 폴더를 직접 선택해 시각화
  ```powershell
  # 대화형 선택 
  docker compose exec matching vis
  ```
  * 아래와 같이 대화형으로 선택 가능:
  <p align="center">
    <img src="docs/figs/matching_visualize option example.png" width="50%">
  </p>
  <p align="center"><em>위의 경우 1을 입력하여 vitl16_400_0001에 있는 json파일을 일괄로 시각화 실행</em></p>



주요 옵션
| 옵션 | 기본값 (환경변수) | 설명 |
| --- | --- | --- |
| `--root` | `/exports/dinov3_match` (`MATCH_ROOT`) | JSON 루트 |
| `--out` | `/exports/dinov3_vis` (`VIS_ROOT`) | PNG 출력 루트 |
| `--ransac` | `homography` | `off/affine/homography` |
| `--reproj-th` | 8.0 | 투영 오차 임계값 |
| `--confidence` | 0.9999 | RANSAC 신뢰도 |
| `--max-lines` | 1000 | 그릴 매칭 수 (0이면 미표시) |
| `--draw-points` | OFF | 점 표시 여부 |

실행 결과는 호스트 `EXPORT_HOST\pair_viz\…`에서 확인.

---

## 5) 결과 확인 & 경로 정리

- JSON: `EXPORT_HOST\pair_match\<weight>_<ALT>_<FRAME>\*.json`
- PNG: `EXPORT_HOST\pair_viz\<weight>_<ALT>_<FRAME>\*.png`
- JSON 내용
  - `meta`: 실행 환경 정보
  - `advanced_settings`: 사용한 매칭/필터 파라미터 (`matching_mode`는 현재 `mutual_knn_k1_unique`)
  - `patch`: 선택된 패치 정보 (`idx_a`, `idx_b`, `similarities` 등)

- 필요 시 결과 폴더를 탐색기에서 바로 열어 확인.


  <p align="center">
    <img src="docs/examples/vitl16_400_0001/RESULT_vitl16_400.0001_200.0001.png" width="75%">
  </p>
  <p align="center"><em>시각화 결과 (ViT-L/16 distilled, 400_0001과 200_0001 매칭)</em></p>



-  `run2` 실행 결과:
  <p align="center">
    <img src="docs/examples/cxTiny_400_0001/RESULT_cxTiny_400.0001_200.0001.png" width="75%">
  </p>
  <p align="center"><em>시각화 결과 (ConvNeXt Tiny, 400_0001과 200_0001 매칭)</em></p>

---

## 6) 트러블슈팅

| 증상 | 확인 사항 & 해결 팁 |
| --- | --- |
| Docker 명령 실패 / 권한 오류 | Docker Desktop 재시작, 관리자 PowerShell에서 실행 |
| 컨테이너에서 GPU 미노출 | `docker compose exec pair nvidia-smi` 확인 → NVIDIA 드라이버/NVIDIA Container Toolkit 재설치 |
| 볼륨 마운트 실패 | Docker Desktop Settings → Resources → File Sharing 에서 각 드라이브 허용 여부 확인 |
| 매칭 JSON이 생성되지 않음 | `pairs_to_run=0` 인 경우 ALT.FRAME 조합이 존재하지 않는 것 → 데이터셋 이름/정규식 확인 |
| 1:1 매칭이 맞지 않는 것처럼 보임 | `run` 내부에서 자동으로 1:1을 강제함. PNG 상에서 선이 적게 보인다면 RANSAC 필터를 완화하거나 `vis --ransac off`로 검증 |
| 기타 로그 | `docker compose logs -f pair` 로 컨테이너 로그 확인 |
| 추가 디버깅 | 필요한 경우 `docker compose exec pair bash`로 컨테이너 내부에 진입하여 추가 디버깅을 진행 가능 |

---

  

