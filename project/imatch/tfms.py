# imatch/tfms.py
import math
import torch
from torchvision import transforms

def build_transform(
    ### 이미지 전처리 
    # patch_size: 패치 크기
    # patch_multiple: 패치 배수, 입력 이미지 크기를 패치 크기의 배수로 조정 (기본값 16)
    # interpolation: 크기변형 시 사용할 보간법 (기본값 "bicubic")
    # normalize: 정규화 적용 여부 (기본값 True)
    patch_size: int,
    patch_multiple: int = 16,
    interpolation: str = "bicubic",
    normalize: bool = True,
):
    target_size = patch_size * patch_multiple

    transforms_steps = [

        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(
            (target_size, target_size),
            interpolation = getattr(transforms.InterpolationMode, interpolation.upper()),
            antialias = True,
        ),
    ]

    if normalize:
        transforms_steps.append(
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(transforms_steps)
