import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. NPY 파일 불러오기
# 'your_file.npy'를 실제 NPY 파일의 로컬 경로로 변경하세요.
npy_file_path = '/exports/dinov3_embeds/vitb16/450/PatchToken/PatchToken_res1024_ImageNet_raw_dinov3_vitb16_LVD_450_0031.npy' 


try:
    data = np.load(npy_file_path)
    print(f"데이터를 성공적으로 불러왔습니다. Shape: {data.shape}, Dtype: {data.dtype}")
except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {npy_file_path}")
    exit()

# 데이터가 2차원 배열일 경우 (일반적인 흑백/그레이스케일 이미지 또는 히트맵)
if data.ndim == 2:
    # 2. 데이터 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray') # cmap='gray'는 그레이스케일로 표시합니다. 필요시 'viridis', 'hot' 등으로 변경 가능
    plt.title('Visualization of NPY data')
    plt.axis('off') # 축 정보를 숨깁니다.
    plt.colorbar() # 컬러바를 추가합니다 (선택 사항).
    plt.show()

    # 3. 이미지 파일로 저장 (Matplotlib 사용)
    # 'output_image_mpl.png' 저장 경로 및 파일명을 지정합니다.
    output_image_path_mpl = '/exports/output_image_mpl.png'
    plt.savefig(output_image_path_mpl, bbox_inches='tight', pad_inches=0)
    print(f"시각화된 이미지를 '{output_image_path_mpl}'로 저장했습니다.")

    # 4. 이미지 파일로 저장 (PIL 사용 - 더 간단한 이미지 저장)
    # 데이터 타입이 이미지 데이터 (예: uint8)에 적합해야 합니다.
    if data.dtype == np.uint8:
        output_image_path_pil = '/exports/output_image_pil.png'
        # NumPy 배열을 PIL Image 객체로 변환
        image_from_array = Image.fromarray(data)
        # 이미지 저장
        image_from_array.save(output_image_path_pil)
        print(f"이미지 데이터를 '{output_image_path_pil}'로 저장했습니다 (PIL).")
    else:
        print("PIL 저장을 건너뜁니다. 데이터 타입이 이미지 형식(예: uint8)이 아닙니다.")

# 데이터가 3차원 배열일 경우 (예: 컬러 이미지 (H, W, C) 또는 여러 채널/프레임)
elif data.ndim == 3:
    # 컬러 이미지 (높이, 너비, 채널)라고 가정
    if data.shape[-1] == 3 or data.shape[-1] == 4: # RGB 또는 RGBA
        plt.figure(figsize=(6, 6))
        plt.imshow(data)
        plt.title('Visualization of NPY data (Color)')
        plt.axis('off')
        plt.show()

        output_image_path_mpl = '/exports/output_image_color_mpl.png'
        plt.savefig(output_image_path_mpl, bbox_inches='tight', pad_inches=0)
        print(f"시각화된 컬러 이미지를 '{output_image_path_mpl}'로 저장했습니다.")

        if data.dtype == np.uint8:
            output_image_path_pil = '/exports/output_image_color_pil.png'
            image_from_array = Image.fromarray(data)
            image_from_array.save(output_image_path_pil)
            print(f"컬러 이미지 데이터를 '{output_image_path_pil}'로 저장했습니다 (PIL).")
    else:
        print("3차원 배열이지만 일반적인 컬러 이미지 형식이 아닙니다 (채널 수가 3 또는 4가 아님).")
        # 특정 채널만 시각화하는 방법 예시
        plt.imshow(data[:, :, 0], cmap='gray')
        plt.show()

else:
    print(f"지원하지 않는 배열 차원입니다: {data.ndim}차원.")

