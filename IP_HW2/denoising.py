#E94111253
from numpy import array, zeros_like, ones, zeros
from numpy import abs, max, min
from numpy import fft, float32, uint8
from filtering import *
from morphology import *
# --- import the above object/funcs, only. ---#
# !!! Import others package is forbidden.  !!!#

def denoising_func(img: array) -> array:

    def gaussian_filter_spatial(image, sigma=1.5, kernel_size=3):
        # 生成高斯核
        half = kernel_size // 2
        x = array([i - half for i in range(kernel_size)], dtype=float32)
        y = x[:, None]  
       
        kernel = 2.718281828459045 ** (-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / sum(kernel)
        
        # 邊界填充 (zero-padding)
        image_padded = my_padding(image, (kernel.shape[0]//2, kernel.shape[1]//2), mode=4)
        
        # 卷積運算
        filtered = zeros_like(image, dtype=float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered[i,j] = ((image_padded[i:i+kernel_size, j:j+kernel_size] * kernel)).sum()

        return filtered.astype(float32)
    signal_power = (img ** 2).mean()
    if signal_power > 0.36:                                     # std = 0.5 or 0.3
        denoised = gaussian_filter_spatial(img, 5, 7)
        denoised = (denoised - denoised.min())/(denoised.max() - denoised.min())
        print("std = 0.5 or 0.3")
    # elif 0.36 > signal_power > 0.34 :
    #     denoised = gaussian_filter_spatial(img, 2.5, 7)
    #     denoised = (denoised - denoised.min())/(denoised.max() - denoised.min())
    #     print("std = 0.2")
    elif 0.36 > signal_power > 0.34 :                           # std = 0.2
        denoised = gaussian_filter_spatial(img, 2.5, 7)
        denoised = (denoised - denoised.min())/(denoised.max() - denoised.min())
        print("std = 0.2")
    elif 0.34 > signal_power > 0.3 :
        """
        使用均值濾波去噪 (3x3 均值濾波器)
        Args:
            img: 輸入影像 (float32, 範圍 [0, 1])
        Returns:
            去噪後的影像 (float32, 範圍 [0, 1])
        """
        h, w = img.shape[0], img.shape[1]
        denoised = zeros((h, w), dtype=float32)
        
        # 手動實現 3x3 均值濾波
        for i in range(1, h-1):
            for j in range(1, w-1):
                # 取 3x3 鄰域並計算均值
                patch = img[i-1:i+2, j-1:j+2]
                denoised[i, j] = patch.mean()
        
        # 邊界處理 (直接複製原圖邊界)
        denoised[0, :] = img[0, :]    # 上邊界
        denoised[-1, :] = img[-1, :]  # 下邊界
        denoised[:, 0] = img[:, 0]    # 左邊界
        denoised[:, -1] = img[:, -1]  # 右邊界
        print("std = 0.2 below")
    else :
        print("error")
    
    signal_power = (denoised ** 2).mean()
    print(signal_power)
    return denoised

def deSaltPepper_func(img:array):
    h, w = img.shape[0], img.shape[1]
    result = zeros((h, w), dtype=float32)
    kernel = array([
        [1, 1],
        [1, 1]
    ])
    kernel1 = array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel2 = array([
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel3 = array([
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    # denoised = denoising_func(img)
    denoised =  opening(img, kernel2)
    # denoised =  closing(denoised, kernel1)
    # denoised =  opening(denoised, kernel1)
    denoised =  closing(denoised, kernel2)
    denoised =  erosion(denoised, kernel)              #0 or 2
    # denoised =  dilation(denoised, kernel1)
    # denoised =  opening(denoised, kernel)
    # denoised =  erosion(denoised, kernel2)
    # denoised =  opening(denoised, kernel)
    denoised =  closing(denoised, kernel2)            # 
    # denoised =  dilation(denoised, kernel7)
    result = denoised
    return result