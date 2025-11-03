import numpy as np 
import torch 
import torch.nn.functional as F 
from PIL import Image
name = "patch_feature_dinov3_vitb16_300_0001_grid"
grid = torch.from_numpy(np.load(f"/exports/{name}.npy"))  # (64, 64, 1024)
flat = grid.reshape(-1, grid.shape[-1])  # (4096, 1024)

feat = flat - flat.mean(dim=0, keepdim=True)
u, s, v = torch.pca_lowrank(feat, q=3)   # v: (1024, 3)

proj = feat @ v[:, :3]                   # (4096, 3)

rgb = proj.reshape(grid.shape[0], grid.shape[1], 3).numpy()
rgb -= rgb.min()
rgb /= (rgb.max() + 1e-6)

rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, 64)
rgb_up = F.interpolate(rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy()


img = Image.fromarray((rgb_up * 255).astype("uint8"))
img.save(f"/exports/{name}_dinov3_dense_features.png")