# project/imatch/features.py
import torch
from typing import Optional

@torch.no_grad()
def extract_global_feature(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """
    글로벌 특징 추출: forward_features(x) 또는 model(x)의 적절한 텐서를 평균/압축
    """
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if isinstance(out, dict):
        if "x" in out and isinstance(out["x"], torch.Tensor) and out["x"].ndim == 3:
            feat = out["x"].mean(dim=1)
        else:
            for k in ("feat", "features", "pooled", "pooler_output"):
                if k in out and isinstance(out[k], torch.Tensor):
                    feat = out[k]
                    break
            else:
                feat = [v for v in out.values() if torch.is_tensor(v)][-1]
    else:
        feat = out

    if feat.ndim == 3:
        feat = feat.mean(dim=1)
    if feat.ndim == 4:
        feat = feat.mean(dim=(2, 3))
    return feat.squeeze(0)

@torch.no_grad()
def extract_patch_tokens(model: torch.nn.Module, x: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """
    패치 토큰(CLS 제외) 추출:
    - 우선순위: out["x_norm_patchtokens"]
    - 대안: patch 관련 키 또는 3D 텐서에서 첫 토큰 제외
    """
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if isinstance(out, dict):
        if "x_norm_patchtokens" in out and torch.is_tensor(out["x_norm_patchtokens"]):
            v = out["x_norm_patchtokens"]
            if v.ndim == 3:
                return v.squeeze(0).contiguous()
        for k in ["patch_tokens", "tokens_patch", "features_patch"]:
            v = out.get(k, None)
            if torch.is_tensor(v) and v.ndim == 3:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        for v in out.values():
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        return None

    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        return None

    if torch.is_tensor(out) and out.ndim == 3 and out.shape[1] > 16:
        return (out[:, 1:, :].squeeze(0) if out.shape[1] > 1 else out.squeeze(0)).contiguous()

    return None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    코사인 유사도(정규화 후 내적)
    """
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())
