# DINOv3 Image Matching (Docker + WSL2)

DINOv3 기반 **이미지 매칭 & 시각화** 파이프라인입니다.  
WSL2 + Docker Desktop + NVIDIA GPU 환경에서 재현 가능하도록 구성되어 있으며,  
여러 가중치(백본)와 All-vs-All 매칭 배치를 지원합니다.

> 참고: 내부 매칭은 DINOv3 토큰의 상호 최근접(mutuual k-NN) 기반이며, 결과는 JSON으로 저장되고, RANSAC(옵션) 기반의 인라이어 필터링 후 PNG로 시각화합니다.

---

## 0) 요구사항

- **Windows 11 + WSL2 (Ubuntu 20.04/22.04)**
- **Docker Desktop** (Settings → Resources → **WSL Integration** 에서 Ubuntu ON)
- **NVIDIA GPU & Driver** (Docker가 GPU를 볼 수 있어야 함)
- (권장) `git`, `make` 등 기본 개발 도구

GPU 확인:
```bash
nvidia-smi
