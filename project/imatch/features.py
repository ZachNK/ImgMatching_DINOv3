# project/imatch/features.py
import torch
from typing import Optional, Tuple


@torch.no_grad()
def extract_global_feature(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """
    모델 출력에서 글로벌 특징을 추출한다.
    """
    # 입력 텐서를 지정된 장치로 이동
    x = x.to(device, non_blocking=True)
    # 모델의 특징 추출 메서드 호출
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
    # 출력에서 특징 추출
    if isinstance(out, dict):
        # 다양한 키를 확인하여 특징을 추출: "x", "feat", "features", "pooled", "pooler_output"
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
        # 출력이 튜플 또는 리스트인 경우 첫 번째 요소를 특징으로 사용
        feat = out

    # 특징 텐서의 차원에 따라 평균을 계산하여 글로벌 특징 벡터 생성
    if feat.ndim == 3:
        feat = feat.mean(dim=1)
    if feat.ndim == 4:
        feat = feat.mean(dim=(2, 3))
    return feat.squeeze(0)


@torch.no_grad()
def extract_patch_tokens(model: torch.nn.Module, x: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """
    패치 토큰(CLS 제외)을 추출한다.
    """
    # 입력 텐서를 지정된 장치로 이동
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    # 다양한 출력 형식에 대응하여 패치 토큰 추출 (딕셔너리, 튜플/리스트, 텐서)
    if isinstance(out, dict):
        
        # 'x_norm_patchtokens' 키가 있으면 해당 값 사용
        if "x_norm_patchtokens" in out and torch.is_tensor(out["x_norm_patchtokens"]):
            
            # 값이 3차원 텐서인 경우 반환
            v = out["x_norm_patchtokens"]
            
            # v.ndim: 3 인 경우 패치 토큰 반환
            if v.ndim == 3:
                # 패치 토큰 반환
                return v.squeeze(0).contiguous()
        
        # 다른 키들 중 패치 토큰을 찾아 반환: 'patch_tokens', 'tokens_patch', 'features_patch'
        for k in ["patch_tokens", "tokens_patch", "features_patch"]:
            # 키에 해당하는 값 가져오기
            v = out.get(k, None)

            # 값이 3차원 텐서인 경우 반환
            if torch.is_tensor(v) and v.ndim == 3:
                # 3차원 텐서인 경우 패치 토큰 반환
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        
        # 위에서 찾지 못한 경우, 값들 중 3차원 텐서를 찾아 반환
        for v in out.values():
            
            # 3차원 텐서인 경우 패치 토큰 반환
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                # 패치 토큰 반환 (CLS 제외)
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        
        # 찾지 못한 경우 None 반환
        return None

    # 출력이 튜플 또는 리스트인 경우 각 요소를 검사
    if isinstance(out, (tuple, list)):
        
        # 각 요소를 검사하여 3차원 텐서인 경우 패치 토큰 반환
        for v in out:
            # 3차원 텐서인 경우 패치 토큰 반환
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                # 패치 토큰 반환 (CLS 제외)
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        # 찾지 못한 경우 None 반환
        return None

    # 출력이 텐서인 경우 검사
    if torch.is_tensor(out) and out.ndim == 3 and out.shape[1] > 16:
        # 패치 토큰 반환 (CLS 제외)
        return (out[:, 1:, :].squeeze(0) if out.shape[1] > 1 else out.squeeze(0)).contiguous()

    # 찾지 못한 경우 None 반환
    return None

# 코사인 유사도 계산 함수
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    코사인 유사도를 계산한다.
    """
    # L2 정규화 후 내적 계산
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())


def apply_keypoint_threshold(
    # tokens: 필터링할 토큰 텐서
    # indx_map: 각 토큰에 대한 인덱스 매핑 텐서
    # threshold: 필터링 임계값
    
    tokens: torch.Tensor,
    idx_map: torch.Tensor,
    threshold: float,

    # 반환값: 필터링된 토큰 텐서와 인덱스 매핑 텐서의 튜플
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    토큰 L2-노름 기반의 임계값 필터링을 수행한다. 모든 토큰이 걸러지는 경우
    최고 점수 토큰을 하나 남겨 매칭 단계가 비어 있지 않도록 보장한다.
    """

    # 토큰이 비어있는 경우 바로 반환
    if tokens.numel() == 0:
        return tokens, idx_map

    # 토큰의 L2-노름 계산
    scores = torch.linalg.norm(tokens, dim=1)
    
    # 정규화
    min_s = scores.min() 
    max_s = scores.max()
    
    # 정규화된 점수 계산
    if (max_s - min_s).abs() < 1e-6:
        # 모든 점수가 동일한 경우 모든 토큰 선택
        normalized = torch.ones_like(scores)
    else:
        # 정규화된 점수 계산
        normalized = (scores - min_s) / (max_s - min_s + 1e-6)

    # 임계값 기반 마스크 생성
    mask = normalized >= threshold
    
    # 모든 토큰이 걸러지는 경우 최고 점수 토큰 하나 선택
    if not torch.any(mask):
        
        # 가장 높은 정규화 점수의 인덱스 찾기
        top_idx = torch.argmax(normalized)
        # 해당 인덱스의 마스크를 True로 설정
        mask[top_idx] = True

    # 마스크에 따라 토큰과 인덱스 매핑 필터링
    keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    
    # 필터링된 토큰과 인덱스 매핑 반환
    filtered_tokens = tokens.index_select(0, keep_idx)
    
    # 필터링된 인덱스 매핑 반환
    filtered_idx_map = idx_map.index_select(0, keep_idx)

    # 반환    
    return filtered_tokens, filtered_idx_map

