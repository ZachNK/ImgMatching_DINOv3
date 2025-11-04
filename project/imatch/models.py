# project/imatch/models.py
import os
import torch
from pathlib import Path
from typing import Optional, Tuple

from imatch.paths import DINOV_BLOCK_NET

def _block_hub_net_if_needed():
    if DINOV_BLOCK_NET:
        import torch.hub as _th
        def _no_dl(*a, **k):
            raise RuntimeError("Blocked torch.hub.load_state_dict_from_url (offline)")
        _th.load_state_dict_from_url = _no_dl  # type: ignore

# def load_model(repo_dir: Path, device: str, hub_name: Optional[str], ckpt: Optional[Path]) -> Tuple[torch.nn.Module, str]:
#     """
#     torch.hub 로 로컬 리포에서 모델 생성 후, ckpt state_dict 로딩.
#     hub_name이 None이면 dinov3_vitl16 → vitb16 → vits16 순으로 시도.
#     """
#     _block_hub_net_if_needed()

#     if hub_name:
#         print(f"[model] hub.load entry='{hub_name}' from {repo_dir}")
#         hub_kwargs = {}
#         if ckpt is not None:
#             hub_kwargs["pretrained"] = False
#         try:
#             model = torch.hub.load(str(repo_dir), hub_name, source="local", trust_repo=True, **hub_kwargs)
#         except TypeError:
#             model = torch.hub.load(str(repo_dir), hub_name, source="local", trust_repo=True)
#     else:
#         tried = ["dinov3_vitl16", "dinov3_vitb16", "dinov3_vits16"]
#         last_err = None
#         model = None
#         for name in tried:
#             try:
#                 model = torch.hub.load(str(repo_dir), name, source="local", trust_repo=True,
#                                        pretrained=False if ckpt else True)
#                 hub_name = name
#                 break
#             except TypeError:
#                 model = torch.hub.load(str(repo_dir), name, source="local", trust_repo=True)
#                 hub_name = name
#                 break
#             except Exception as e:
#                 last_err = e
#         if model is None:
#             raise SystemExit(f"Failed to load hub model. Last error: {last_err}")

#     model.eval().to(device)

#     if ckpt is not None:
#         print(f"[ckpt] loading: {ckpt}")
#         # PyTorch 2.0+ 안전 로딩: weights_only=True
#         state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
#         if isinstance(state, dict) and "state_dict" in state:
#             state = state["state_dict"]
#         new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
#         missing, unexpected = model.load_state_dict(new_state, strict=False)
#         if missing:    print(f"[ckpt][warn] missing={len(missing)}")
#         if unexpected: print(f"[ckpt][warn] unexpected={len(unexpected)}")
#     return model, (hub_name or "hub_model")


def load_model(repo_dir: str, hub_entry: str, ckpt_path: str, device: str) -> Tuple[torch.nn.Module, str]:
    """
    torch.hub 로 로컬 리포에서 모델 생성 후, ckpt state_dict 로딩.
    """
    # 블록 허브 네트워크 호출이 필요한지 확인
    _block_hub_net_if_needed()

    # 모델 로드
    # model = None
    # # 허브 엔트리가 있는 경우
    # if hub_entry:
    #     try:
    #         # 허브에서 모델 로드
    #         print(f"[model] hub.load entry='{hub_entry}' from {repo_dir}")
    #         model = torch.hub.load(repo_dir, hub_entry, ckpt_path, source="local", trust_repo=True, pretrained=False)
    #     # TypeError 예외 처리
    #     except TypeError as type_err:
    #         raise SystemExit(f"Type error: {type_err}")
    #     # 기타 예외 처리
    #     except Exception as last_err:
    #         # 모델 로드 실패 시 처리
    #         if model is None:
    #             raise SystemExit(f"Failed to load hub model. Last error: {last_err}")
    #         # 기타 예외 재발생
    #         else:
    #             raise SystemExit(f"Last error: {last_err}")
    # # 허브 엔트리가 없는 경우
    # else:
    #     # 모델 로드 실패 처리
    #     raise SystemExit(f"hub entry: None")
    print(f"[model] hub.load entry='{hub_entry}' from {repo_dir}")
    model = torch.hub.load(str(repo_dir), hub_entry, source="local", trust_repo=True, pretrained=False)
    model.eval().to(device)

    # 체크 포인트 로드 및 모델 가중치 설정
    try:
        # CUDA 장치에 맞게 체크 포인트 로드 시도
        state = torch.load(ckpt_path, map_location="cuda:0", weights_only=True)
    except TypeError:
        # 실패 시 CPU에 맞게 로드
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 체크 포인트에서 'state_dict' 키가 있으면 해당 값으로 설정
    if isinstance(state, dict) and "state_dict" in state:
        # 'state_dict' 키가 있는 경우 해당 값으로 설정
        state = state["state_dict"]
    # 'module.' 접두사가 있는 키를 제거하여 모델에 맞게 정리
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # 모델에 가중치 로드, 엄격하지 않게 설정하여 누락된 키나 예기치 않은 키 경고 출력
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    
    # 경고 출력
    if missing:
        # 누락된 키 경고 출력
        print(f"[ckpt][warn] missing keys: {len(missing)}")
    if unexpected:
        # 예기치 않은 키 경고 출력
        print(f"[ckpt][warn] unexpected keys: {len(unexpected)}")

    
    return model, (hub_entry or "hub_model")
