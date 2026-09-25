import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.fast_convex import fast_convex_
# from Fast_convex_test import fast_convex_
from src.COS2A import COS2A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_fixed_D():
    aviris_wl_224 = np.linspace(400, 2500, 224)

    remove_bands = list(range(1, 11)) \
                 + list(range(104, 117)) \
                 + list(range(152, 171)) \
                 + list(range(215, 225))

    remove_idx = [b - 1 for b in remove_bands]
    keep_idx = [i for i in range(224) if i not in remove_idx]

    aviris_wl_172 = aviris_wl_224[keep_idx]

    assert len(aviris_wl_172) == 172, f"Expected 172 bands, got {len(aviris_wl_172)}"

    S2_center_nm = np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190], dtype=np.float32)
    S2_bw_nm     = np.array([20,  65,  35,  30,  15,  15,  20, 115, 20,  20,  90,   180], dtype=np.float32)

    D = np.zeros((12, 172), dtype=np.float32)

    for i in range(12):
        low = S2_center_nm[i] - S2_bw_nm[i] / 2
        high = S2_center_nm[i] + S2_bw_nm[i] / 2

        idx = np.where((aviris_wl_172 >= low) & (aviris_wl_172 <= high))[0]

        if len(idx) == 0:
            idx = np.array([np.argmin(np.abs(aviris_wl_172 - S2_center_nm[i]))])

        D[i, idx] = 1.0 / len(idx)

    D = torch.from_numpy(D).float()

    D = torch.clamp(D, min=0.0)
    D = D / (D.sum(dim=1, keepdim=True) + 1e-8)

    return D

class algo1(nn.Module):
    def __init__(self, cnmf_rank=10, cnmf_iters=50, verbose_cnmf=False, blend_fixed_D=0.10):
        super(algo1, self).__init__()
        D_init = build_fixed_D()

        self.register_buffer("D_fixed", D_init)
        self.cos2a = COS2A(D_init)
        self.nmf = fast_convex_(r=10, iters=50, lam=0.1, blur_r=2, print_every=20 if verbose_cnmf else None)
        self.highres_idx = [1, 2, 3, 7]
        self.blend_fixed_D = blend_fixed_D
        
    def forward(self, YS):
        #   YS : (B, 12, 64, 64)
        #   YH : (B, 172, 64, 64)  
        #   A  : (172, 10) 
        #   S  : (B, 10, 64, 64)
        # YDE  : (B, 172, 64, 64)
        #YSwave: (B, 172, 64, 64)
        #D_wave: (2, 12, 172)
        B, _, H, W= YS.shape
        YS = torch.clamp(YS, 0.0, 1.0)
        Y_DE = self.cos2a(YS)                                        # (B, 172, 64, 64), 粗 HSI（未做非負）
        Y_DE = torch.clamp(Y_DE, 0.0, 1.0)
        
        YS_10m = YS[:, self.highres_idx, :, :]

        D_wave = ridge_regression(Y_DE.detach(), YS_10m.detach())

        if self.blend_fixed_D > 0:
            D_prior = self.D_fixed[self.highres_idx, :].unsqueeze(0).to(D_wave.device, D_wave.dtype)
            D_wave = (1.0 - self.blend_fixed_D) * D_wave + self.blend_fixed_D * D_prior
            D_wave = torch.clamp(D_wave, min=0.0)
            D_wave = D_wave / (D_wave.sum(dim=2, keepdim=True) + 1e-8)

        A_star, S_star = self.nmf(Y_DE, YS_10m, D_wave)

        YH_star_flat = torch.bmm(A_star, S_star)              # (B,172,H*W)
        YH_star = YH_star_flat.reshape(B, 172, H, W)
        YH_star = torch.clamp(YH_star, 0.0, 1.0)

        Ym_hat = torch.bmm(D_wave, YH_star_flat).reshape(B, 4, H, W)
        Ym_hat = torch.clamp(Ym_hat, 0.0, 1.0)

        if not self.training:
            with torch.no_grad():
                mse_de = F.mse_loss(YH_star, Y_DE).item()
                mse_ms = F.mse_loss(Ym_hat, YS_10m).item()
                print(f"[algo] mse(YH_star, Y_DE)={mse_de:.6f} | mse(DYH_star, YS_10m)={mse_ms:.6f}")

        return YH_star, Ym_hat, Y_DE, D_wave


def ridge_regression(YDE, YSwave, eta=0.0001):
    """
    CHECK:
        YDE   : (B, 172, H, W)
        YSwave: (B, 12,  H, W)
    """
    # print("Ridge regression strat...")
    # print(f"[ridge_regression] YDE:{YDE.shape},   YSwave:{YSwave.shape}")
    B, Lh, H, W = YDE.shape
    _, Lm, _, _ = YSwave.shape  

    Y = YDE.reshape(B, Lh, -1)   # (B,172,N)
    S = YSwave.reshape(B, Lm, -1)    # (B,12,N)

    I = torch.eye(Lh, device=Y.device, dtype=Y.dtype).unsqueeze(0)
    YYt = torch.bmm(Y, Y.transpose(1, 2)) + eta * I      # (B,172,172)
    SYt = torch.bmm(S, Y.transpose(1, 2))                # (B,M,172)

    # Solve D * YYt = SYt  ->  YYt^T * D^T = SYt^T
    D = torch.linalg.solve(YYt.transpose(1, 2), SYt.transpose(1, 2)).transpose(1, 2)
    D = torch.clamp(D, min=0.0)
    D = D / (D.sum(dim=2, keepdim=True) + 1e-8)
    return D
    # print("Ridge regression end...")
    # print("D_wave range:", D_wave.min().item(), D_wave.max().item())

    return D_wave

if __name__ == '__main__':
    B = 2
    Ys = torch.randn(B, 12, 64, 64).to(device)

    model = algo1().to(device)

    print("=== Running Algo1 Test ===")

    YH, Ym_hat, Y_DE, D_wave = model(Ys)

    print("=== DONE ===")
    print("YH:", YH.shape)
    print("Ym_hat:", Ym_hat.shape)
    print("Y_DE:", Y_DE.shape)
    print("D_wave:", D_wave.shape)
