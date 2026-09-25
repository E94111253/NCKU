import torch
import matplotlib.pyplot as plt
from src.dataset import load_mat_file
from src.algo import algo1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = 's'      # real or simulation
# ===== 載入模型 =====
model = algo1().to(device)
if mode == 'r':
    state = torch.load("checkpoints/realS2/best_model.pth", map_location=device)
    # ===== 載入一張資料 =====
    YS = load_mat_file("dataset/crop_S_r/patch_00451.mat")  # 改成要測的 00133, 00451, 00609, 00685
    YH = load_mat_file("dataset/crop_A_r/patch_00451.mat")
    print("[Real Sentinel2]")
else:
    state = torch.load("checkpoints/simulationS2/best_model.pth", map_location=device)
    # ===== 載入一張資料 =====
    YS = load_mat_file("dataset/crop_S/patch_00033.mat")    # 改成要測的 00033, 00123, 00138, 00627, 00654
    YH = load_mat_file("dataset/crop_A/patch_00033.mat")
    print("[Simulation Sentinel2]")

missing, unexpected = model.cos2a.load_state_dict(state, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)
model.eval()



YS = torch.tensor(YS).unsqueeze(0).to(device).float()
YH = torch.tensor(YH).unsqueeze(0).to(device).float()
print("YS range      :", YS.min().item(), YS.max().item())
YS = torch.clamp(YS, 0.0, 1.0)
YH = torch.clamp(YH, 0.0, 1.0)

# ===== forward =====
YH_star, Ym_hat, Y_DE, D_wave = model(YS)

with torch.no_grad():
    Y_DE_raw = model.cos2a(YS)

Y_DE_raw_cpu = Y_DE_raw[0].detach().cpu()

print("\n=== Raw Y_DE before clamp check ===")
for b in [11, 27, 111]:
    print(
        f"band {b:03d} | "
        f"raw min={Y_DE_raw_cpu[b].min().item():.6f}, "
        f"raw max={Y_DE_raw_cpu[b].max().item():.6f}, "
        f"raw mean={Y_DE_raw_cpu[b].mean().item():.6f}, "
        f"raw std={Y_DE_raw_cpu[b].std().item():.6f}"
    )

# ===== 轉 numpy =====
YS = YS[0].detach().cpu()
YH = YH[0].detach().cpu()
Y_DE = Y_DE[0].detach().cpu()
YH_star = YH_star[0].detach().cpu()

# def bandwise_match_to_ref(pred, ref, eps=1e-8):
#     # pred, ref: (B, C, H, W)
#     pred_mean = pred.mean(dim=(2, 3), keepdim=True)
#     pred_std  = pred.std(dim=(2, 3), keepdim=True)

#     ref_mean = ref.mean(dim=(2, 3), keepdim=True)
#     ref_std  = ref.std(dim=(2, 3), keepdim=True)

#     out = (pred - pred_mean) * (ref_std / (pred_std + eps)) + ref_mean
    
#     return torch.clamp(out, 0.0, float(ref.max()))

def psnr(pred, target, eps=1e-12):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = torch.mean((pred - target) ** 2)
    return (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse + eps))).item()

def sam_deg(pred, target, eps=1e-8):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    p = pred.reshape(pred.shape[0], -1).T
    t = target.reshape(target.shape[0], -1).T

    dot = torch.sum(p * t, dim=1)
    p_norm = torch.norm(p, dim=1)
    t_norm = torch.norm(t, dim=1)

    cos = dot / (p_norm * t_norm + eps)
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)

    return torch.acos(cos).mean().item() * 180.0 / 3.14159265

def print_stats(name, x):
    print(
        f"{name:14s} | "
        f"min={x.min().item():.6f} | "
        f"max={x.max().item():.6f} | "
        f"mean={x.mean().item():.6f} | "
        f"std={x.std().item():.6f}")

YH_star_raw = YH_star.clone()

print("YS range      :", YS.min().item(), YS.max().item())
print("YH range      :", YH.min().item(), YH.max().item())
print("Y_DE range    :", Y_DE.min().item(), Y_DE.max().item())
print("YH_star range :", YH_star.min().item(), YH_star.max().item())


r_idx, g_idx, b_idx = 23, 13, 5

rgb_S = torch.stack([YS[3], YS[2], YS[1]], dim=0).permute(1, 2, 0)
rgb_GT = torch.stack([YH[r_idx], YH[g_idx], YH[b_idx]], dim=0).permute(1, 2, 0)
rgb_DE = torch.stack([Y_DE[r_idx], Y_DE[g_idx], Y_DE[b_idx]], dim=0).permute(1, 2, 0)
rgb_star = torch.stack([YH_star_raw[r_idx], YH_star_raw[g_idx], YH_star_raw[b_idx]], dim=0).permute(1, 2, 0)

ref_min = torch.min(torch.stack([rgb_GT.min(), rgb_DE.min(), rgb_star.min()]))
ref_max = torch.max(torch.stack([rgb_GT.max(), rgb_DE.max(), rgb_star.max()]))

def normalize_rgb(x, mode="1", ref_min=None, ref_max=None):
    """
    mode:
      "1. matlab"     : 每個 RGB channel 分別做 1%-99% percentile stretch
      "2. individual" : 整張 RGB 自己 min-max
      "3. shared"     : 用共同 ref_min/ref_max
    """

    x = x.clone()
    x[~torch.isfinite(x)] = 0

    if mode == "1":
        out = torch.zeros_like(x)

        for ch in range(x.shape[2]):
            band = x[:, :, ch]
            low = torch.quantile(band.reshape(-1), 0.01)
            high = torch.quantile(band.reshape(-1), 0.99)

            if torch.abs(high - low) < 1e-8:
                out[:, :, ch] = 0
            else:
                band = (band - low) / (high - low + 1e-8)
                out[:, :, ch] = torch.clamp(band, 0.0, 1.0)

        x = out

    elif mode == "2":
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    elif mode == "3":
        gamma = 0.8
        low=1 
        high=99
        x = x.clone()
        x[~torch.isfinite(x)] = 0

        lo = torch.quantile(x.reshape(-1), low / 100)
        hi = torch.quantile(x.reshape(-1), high / 100)

        x = (x - lo) / (hi - lo + 1e-8)
        x = torch.clamp(x, 0.0, 1.0)

        x = x ** gamma
        return x

    
    return torch.clamp(x, 0.0, 1.0)

display_mode = "3"             # 1: MATLAB, 2: individual, 3: shared, 
gamma = 1.0 
# if display_mode == "3":
#     ref_min = torch.min(torch.stack([rgb_GT.min(), rgb_DE.min(), rgb_star.min()]))
#     ref_max = torch.max(torch.stack([rgb_GT.max(), rgb_DE.max(), rgb_star.max()]))
# else:
#     ref_min, ref_max = None, None

rgb_S = normalize_rgb(rgb_S, mode=display_mode)
rgb_GT = normalize_rgb(rgb_GT, mode=display_mode, ref_min=ref_min, ref_max=ref_max)
rgb_DE = normalize_rgb(rgb_DE, mode=display_mode, ref_min=ref_min, ref_max=ref_max)
rgb_star = normalize_rgb(rgb_star, mode=display_mode, ref_min=ref_min, ref_max=ref_max)

print("\n==============================")
print("Basic statistics")
print("==============================")
print_stats("YS", YS)
print_stats("GT YH", YH)
print_stats("Y_DE", Y_DE)
print_stats("YH_raw", YH_star_raw)

print("==============================")
print("Metrics vs GT")
print("==============================")
print(f"Y_DE      | PSNR={psnr(Y_DE, YH):.3f} dB | SAM={sam_deg(Y_DE, YH):.3f} deg")
print(f"YH_star   | PSNR={psnr(YH_star_raw, YH):.3f} dB | SAM={sam_deg(YH_star_raw, YH):.3f} deg")

print("\n==============================")
print("CNMF check")
print("==============================")
YH_raw_flat = YH_star_raw.unsqueeze(0).reshape(1, 172, -1)
YDE_flat = Y_DE.unsqueeze(0).reshape(1, 172, -1)

if mode == 'r':
    YS_10m = torch.stack([YS[1], YS[2], YS[3], YS[7]], dim=0).unsqueeze(0)
else:
    YS_10m = torch.stack([YS[1], YS[2], YS[3], YS[7]], dim=0).unsqueeze(0)

YS_10m_flat = YS_10m.reshape(1, 4, -1)

D_cpu = D_wave.detach().cpu()
Ym_check = torch.bmm(D_cpu, YH_raw_flat)

print(f"mse(YH_raw, Y_DE)       = {torch.mean((YH_star_raw - Y_DE) ** 2).item():.8f}")
print(f"mse(D @ YH_raw, YS_10m) = {torch.mean((Ym_check - YS_10m_flat) ** 2).item():.8f}")
print("D_wave row sum:", D_cpu.sum(dim=2).numpy())
print("YH_raw has NaN:", torch.isnan(YH_star_raw).any().item())
print("YH_raw has Inf:", torch.isinf(YH_star_raw).any().item())

print("\n=== Bad band check ===")
bad_bands = []

for b in range(172):
    gt_std = YH[b].std().item()
    de_std = Y_DE[b].std().item()
    raw_std = YH_star_raw[b].std().item()

    if gt_std > 1e-3 and de_std < 1e-6:
        bad_bands.append((b, gt_std, de_std, raw_std))

print("Bad bands where GT has variation but Y_DE is flat:")
for b, gt_std, de_std, raw_std in bad_bands:
    print(f"band {b:03d}: GT std={gt_std:.6f}, Y_DE std={de_std:.6f}, YH_raw std={raw_std:.6f}")

print("Number of bad bands:", len(bad_bands))


plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(rgb_S.numpy())
plt.title("Input S2")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(rgb_GT.numpy())
plt.title("GT AVIRIS")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(rgb_DE.numpy())
plt.title(f"Y_DE")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(rgb_star.numpy())
plt.title(f"YH_star")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.show()