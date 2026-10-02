#E94111253
from numpy import array, zeros_like, ones, zeros, fft
from numpy import abs, max, min
from numpy import float32, uint8

#### Problem 1
def my_padding(img:array, size: tuple, mode: int, value=None):
    """
    自定義邊界填充函數
        size: 填充大小 (int: 四邊相同, tuple: (垂直填充, 水平填充))
        mode: 填充模式 
              0 - 使用指定值填充 (需提供 value)
              1 - 填充 0
              2 - 填充 1
              3 - 鏡像填充
              4 - 重複邊緣像素
        value: 當 mode=0 時使用的填充值
    Returns:
        填充後的陣列 (與輸入同 dtype)
    """
    pad_h, pad_w = size[0], size[1]
    
    h, w = img.shape
    padded = zeros((h + 2*pad_h, w + 2*pad_w), dtype=img.dtype)
    
    # 中心區域放入原圖
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = img
    
    if mode == 0:  # 使用指定值填充
        if value is None:
            raise ValueError("mode=0 時需提供 value 參數")
        padded[:pad_h, :] = value       # 上邊界
        padded[-pad_h:, :] = value      # 下邊界
        padded[:, :pad_w] = value       # 左邊界
        padded[:, -pad_w:] = value      # 右邊界
        
    elif mode == 1:                     # 填充 0
        pass                            # 已初始化為0
        
    elif mode == 2:  # 填充 1
        padded[:pad_h, :] = 1           # 上邊界
        padded[-pad_h:, :] = 1          # 下邊界
        padded[:, :pad_w] = 1           # 左邊界
        padded[:, -pad_w:] = 1          # 右邊界
        
    elif mode == 3:  # 鏡像填充
        # 上邊界 (垂直鏡像)
        padded[:pad_h, pad_w:pad_w+w] = img[pad_h-1::-1, :]
        # 下邊界 (垂直鏡像)
        padded[-pad_h:, pad_w:pad_w+w] = img[-1:-pad_h-1:-1, :]
        # 左邊界 (水平鏡像)
        padded[pad_h:pad_h+h, :pad_w] = img[:, pad_w-1::-1]
        # 右邊界 (水平鏡像)
        padded[pad_h:pad_h+h, -pad_w:] = img[:, -1:-pad_w-1:-1]
        
    elif mode == 4:  # 重複邊緣像素
        # 上邊界
        padded[:pad_h, pad_w:pad_w+w] = img[0, :]
        # 下邊界
        padded[-pad_h:, pad_w:pad_w+w] = img[-1, :]
        # 左邊界
        padded[pad_h:pad_h+h, :pad_w] = img[:, 0].reshape(-1, 1)
        # 右邊界
        padded[pad_h:pad_h+h, -pad_w:] = img[:, -1].reshape(-1, 1)
        
    else:
        raise ValueError("不支持的 mode (0-4)")
    
    return padded

def my_2d_conv(image:array, kernel:array):          # 空間域的卷積運算
    h, w = image.shape
    k_h, k_w = kernel.shape

    padded = my_padding(image, (kernel.shape[0]//2, kernel.shape[1]//2), mode=3)
    output = zeros_like(image, dtype = float32)
    
    # 卷積計算
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = (region * kernel).sum()
    
    output[output < 0] = 0
    output[output > 255] = 255
    return output.astype(uint8)

#### Problem 2
def spatialdomain_filtering(image:array, kernel:array):                     # 空間域濾波
    # Input array size must be the same with output array.
    spatial_filtered = my_2d_conv(image, kernel)

    return spatial_filtered

def freqencydomain_filtering(image:array, kernel:array):
    # Input array size must be the same with output array.
    pad_h = (image.shape[0] - kernel.shape[0]) // 2
    pad_w = (image.shape[1] - kernel.shape[1]) // 2
    kernel_padded = zeros_like(image, dtype = float32)
    kernel_padded[pad_h:pad_h+kernel.shape[0], pad_w:pad_w+kernel.shape[1]] = kernel

    # 傅立葉變換
    F_img = fft.fft2(image)
    F_kernel = fft.fft2(fft.ifftshift(kernel_padded))

    # 頻域相乘 + 反變換
    F_filtered = F_img * F_kernel
    filtered = fft.ifft2(F_filtered).real

    # 歸一化處理
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255
    freq_filtered = filtered

    F_img = fft.ifftshift(F_img)
    F_kernel = fft.ifftshift(F_kernel)

    return freq_filtered, F_img, F_kernel

def spatial_hybrid_imaging(for_lowpass:array, for_highpass:array, kernel=None):
    # Input array size must be the same with output array.
    low_frequencies = zeros_like(for_lowpass, dtype = float32)
    high_frequencies = zeros_like(for_highpass, dtype = float32)
    
    low_frequencies = my_2d_conv(for_lowpass, kernel)
    
    # 高通濾波 (原圖 - 低通濾波)
    high_frequencies = for_highpass.astype(float32) - my_2d_conv(for_highpass, kernel)
    
    hybrid_img = (low_frequencies + high_frequencies)           # 混合影像 (低通 + 高通)
    
    hybrid_img[hybrid_img < 0] = 0
    hybrid_img[hybrid_img > 255] = 255
    
    return hybrid_img.astype(uint8), low_frequencies, high_frequencies

def freq_hybrid_imaging(for_lowpass:array, for_highpass:array, kernel=None):
    # Input array size must be the same with output array.
    pad_h = (for_lowpass.shape[0] - kernel.shape[0]) // 2
    pad_w = (for_lowpass.shape[1] - kernel.shape[1]) // 2
    kernel_padded = zeros_like(for_lowpass, dtype=float32)
    kernel_padded[pad_h : pad_h+kernel.shape[0], pad_w : pad_w+kernel.shape[1]] = kernel
    
    # 傅立葉變換
    F_lowpass = fft.fft2(for_lowpass)
    F_kernel = fft.fft2(fft.ifftshift(kernel_padded))
    lowpass_freq = F_lowpass * F_kernel
    
    # 對 for_highpass 做頻域高通濾波 (1 - 低通)
    F_highpass = fft.fft2(for_highpass)
    highpass_freq = F_highpass * (1 - F_kernel)
    
    # 混合頻譜
    hybrid_freq = lowpass_freq + highpass_freq
    
    # 反傅立葉變換並取實部
    hybrid_img = fft.ifft2(hybrid_freq).real
    
    # 歸一化到 0~255
    hybrid_img = (hybrid_img - hybrid_img.min()) / (hybrid_img.max() - hybrid_img.min()) * 255
    hybrid_img[hybrid_img < 0] = 0
    hybrid_img[hybrid_img > 255] = 255

    low_frequencies, _, _ = freqencydomain_filtering(for_lowpass, kernel)
    low_frequencies_img2, _, _ = freqencydomain_filtering(for_highpass, kernel)   # 先對 img2 模糊

    high_frequencies = for_highpass - low_frequencies_img2                  # 高頻 = 原圖 - 模糊後的圖
    high_frequencies = high_frequencies +128
    high_frequencies[high_frequencies < 0] = 0
    high_frequencies[high_frequencies > 255] = 255
    # 轉換為 uint8
    high_frequencies = (high_frequencies ).astype(uint8)
    

    return hybrid_img.astype(uint8), low_frequencies, high_frequencies 

