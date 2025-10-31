from pathlib import Path
import numpy as np
from PIL import Image
fileName = "dinov3_vitl16_400_0112"
mapPath = Path(f"/exports/patch_cosine_map_{fileName}.npy")
gridPath = Path(f"/exports/patch_cosine_grid_{fileName}.npy")
mapData = np.load(mapPath)
gridData = np.load(gridPath)
if mapData.dtype != np.uint8:
    mapData = mapData.astype(np.uint8)
if gridData.dtype != np.uint8:
    gridData = gridData.astype(np.uint8)

try:
    map = Image.fromarray(mapData)
    map.save(f"/exports/cosine_map_{fileName}.jpg", 'JPEG')
    print(f"[saved] cosine map -> /exports/cosine_map_{fileName}.jpg")
except Exception as e:
    print(f"Error to save cosine map: {e}")

try:
    grid = Image.fromarray(gridData)
    grid.save(f"/exports/cosine_grid_{fileName}.jpg", 'JPEG')
    print(f"[saved] cosine grid -> /exports/ccosine_grid{fileName}.jpg")
except Exception as e:
    print(f"Error to save cosine grid: {e}")


