# imatch/tfms.py
import torch
from torchvision import transforms

def build_transform(image_size: int=224):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
