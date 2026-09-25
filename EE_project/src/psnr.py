import numpy as np
import math
import torch

def MPSNR(img1, img2, data_range=1.0):
    """
    計算多通道 PSNR
    img1, img2 : numpy array 或 torch tensor, shape = (H, W, C)
    data_range : 最大值，通常 1.0 或 255.0
    """
    # 如果是 torch tensor → 轉 numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    ch = img1.shape[2] if img1.ndim == 3 else 1

    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(data_range / math.sqrt(mse))
    else:
        sum_psnr = 0
        for i in range(ch):
            mse = np.mean((img1[:, :, i] - img2[:, :, i]) ** 2)
            if mse == 0:
                return 100
            sum_psnr += 20 * math.log10(data_range / math.sqrt(mse))
        return sum_psnr / ch
    
def SAM(x, y):
    dot = (x*y).sum(dim=1)
    norm_x = torch.norm(x, dim=1)
    norm_y = torch.norm(y, dim=1)
    return torch.acos(dot / (norm_x * norm_y + 1e-8))
