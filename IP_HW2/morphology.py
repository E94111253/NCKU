#E94111253
from numpy import array, zeros_like, ones, zeros, fft
from numpy import abs, max, min
from numpy import float32, uint8

def my_padding(img : array, size, mode, value=None):
    """
    自定義邊界填充函數
    Args:
        img: 輸入圖像 (2D numpy array)
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
        padded[:pad_h, :] = value          # 上邊界
        padded[-pad_h:, :] = value         # 下邊界
        padded[:, :pad_w] = value          # 左邊界
        padded[:, -pad_w:] = value         # 右邊界
        
    elif mode == 1:  # 填充 0
        pass  # 已初始化為0
        
    elif mode == 2:  # 填充 1
        padded[:pad_h, :] = 1              # 上邊界
        padded[-pad_h:, :] = 1             # 下邊界
        padded[:, :pad_w] = 1              # 左邊界
        padded[:, -pad_w:] = 1             # 右邊界
        
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
    return padded
#### Problem 1
def dilation(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape

    padded = my_padding(img, (1, 1), mode=0, value=0)
    dilated = zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            region = padded[i : i+k_h, j : j+k_w]
            dilated[i,j] = max(region * kernel)
    
    return dilated

def erosion(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    padded = my_padding(img, (1, 1), mode=0, value=255)
    eroded = zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k_h, j:j+k_w]
            eroded[i,j] = min(region * kernel)
    
    return eroded

#### Problem 2
def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)