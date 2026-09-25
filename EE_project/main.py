import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.algo import build_fixed_D 
from src.COS2A import COS2A
from src.dataset import load_mat_file
from src.psnr import MPSNR,  SAM

class loadDataset(Dataset):
    def __init__(self, ys_dir, yh_dir, file_list, patch_size=64, stride=64):
        self.ys_dir = ys_dir
        self.yh_dir = yh_dir
        self.files = list(file_list)
        self.patch_size = patch_size
        self.stride = stride

        self.index = []

        # cache 目前檔案
        self.cached_file_id = None
        self.cached_YS = None
        self.cached_YH = None

        for file_id, fname in enumerate(self.files):
            ys_path = os.path.join(self.ys_dir, fname)

            ys = load_mat_file(ys_path).astype("float32")

            if ys.shape[0] != 12 and ys.shape[-1] == 12:
                ys = np.transpose(ys, (2, 0, 1))

            _, H, W = ys.shape

            for x in range(0, H - patch_size + 1, stride):
                for y in range(0, W - patch_size + 1, stride):
                    self.index.append((file_id, x, y))

        print(f"Total patches: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def load_one_file(self, file_id):
        if self.cached_file_id == file_id:
            return self.cached_YS, self.cached_YH

        fname = self.files[file_id]

        ys_path = os.path.join(self.ys_dir, fname)
        yh_path = os.path.join(self.yh_dir, fname)

        YS = load_mat_file(ys_path).astype("float32")
        YH = load_mat_file(yh_path).astype("float32")

        if YS.shape[0] != 12 and YS.shape[-1] == 12:
            YS = np.transpose(YS, (2, 0, 1))

        if YH.shape[0] != 172 and YH.shape[-1] == 172:
            YH = np.transpose(YH, (2, 0, 1))

        self.cached_file_id = file_id
        self.cached_YS = YS
        self.cached_YH = YH

        return YS, YH

    def __getitem__(self, idx):
        file_id, x, y = self.index[idx]

        YS, YH = self.load_one_file(file_id)

        ps = self.patch_size

        YS_patch = YS[:, x:x + ps, y:y + ps]
        YH_patch = YH[:, x:x + ps, y:y + ps]

        if np.random.rand() < 0.5:
            YS_patch = YS_patch[:, :, ::-1].copy()
            YH_patch = YH_patch[:, :, ::-1].copy()

        if np.random.rand() < 0.5:
            YS_patch = YS_patch[:, ::-1, :].copy()
            YH_patch = YH_patch[:, ::-1, :].copy()

        k = np.random.randint(0, 4)
        YS_patch = np.rot90(YS_patch, k, axes=(1, 2)).copy()
        YH_patch = np.rot90(YH_patch, k, axes=(1, 2)).copy()

        return (
            torch.from_numpy(YS_patch.copy()).float(),
            torch.from_numpy(YH_patch.copy()).float())
    
def read_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def calc_batch_psnr(pred, target):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    vals = []
    for b in range(pred.shape[0]):
        p = pred[b].permute(1, 2, 0).numpy()   # (H, W, C)
        t = target[b].permute(1, 2, 0).numpy()
        vals.append(MPSNR(p, t, data_range=1.0))
    return float(np.mean(vals))

def calc_batch_sam(pred, target):
    pred = pred.detach()
    target = target.detach()

    p = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    t = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    sam = SAM(p, t)  # radians
    sam_deg = sam.mean().item() * 180.0 / np.pi
    return sam_deg

def spectral_angle_loss(pred, target, eps=1e-8):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    # (B, C, H, W) -> (B*H*W, C)
    p = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    t = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    p_norm = torch.norm(p, dim=1, keepdim=True).clamp_min(eps)
    t_norm = torch.norm(t, dim=1, keepdim=True).clamp_min(eps)

    cos_sim = (p * t).sum(dim=1, keepdim=True) / (p_norm * t_norm)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)

    ang = torch.acos(cos_sim)   # radians
    return ang.mean()

def train(model, loader, optimizer, L1loss, device, epoch, sam_weight):
    model.train()
    total_loss = 0.0

    bar = tqdm(loader, desc=f'Epoch{epoch}', leave=True)

    for step, (YS, YH) in enumerate(bar):
        YS = torch.clamp(YS.to(device).float(), 0.0, 1.0)
        YH = torch.clamp(YH.to(device).float(), 0.0, 1.0)

        optimizer.zero_grad()
        Y_DE = model(YS)

        # ======= test 0617 =======
        # lossL1 = L1loss(torch.clamp(Y_DE, 0.0, 1.0), YH)
        # loss_sam = spectral_angle_loss(torch.clamp(Y_DE, 0.0, 1.0), YH)
        # loss = lossL1 + sam_weight * loss_sam
        # ======= test 0617 =======
        lossL1 = L1loss(Y_DE, YH)
        loss_sam = spectral_angle_loss(Y_DE, YH)
        range_loss = torch.relu(-Y_DE).mean() + torch.relu(Y_DE - 1).mean()
        loss = lossL1 + sam_weight * loss_sam + 0.01 *range_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        bar.set_postfix({"loss": f"{loss.item():.4f}", "l1": f"{lossL1.item():.4f}", "sam_loss": f"{loss_sam.item():.4f}"})
        if step % 100 == 0:
            rho_now = (F.softplus(model.rho) + 1e-4).item()
            print(f"[E{epoch} S{step}] rho={rho_now:.6f} loss={loss.item():.6f} l1={lossL1.item():.6f} sam_loss={loss_sam.item():.6f}")
    
    avg_loss = total_loss/len(loader)
    print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

    return avg_loss

def val(model, loader, L1loss, device, epoch, sam_weight):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    count = 0
    with torch.no_grad():
        bar = tqdm(loader, desc=f"val   Epoch {epoch}", leave=False)
        for step, (YS, YH) in enumerate(bar):
            YS = torch.clamp(YS.to(device), 0.0, 1.0)
            YH = torch.clamp(YH.to(device), 0.0, 1.0)

            Y_DE = model(YS)
            Y_DE_eval = torch.clamp(Y_DE, 0.0, 1.0)

            loss_l1 = L1loss(Y_DE_eval, YH)
            loss_sam = spectral_angle_loss(Y_DE_eval, YH)
            loss = loss_l1 + sam_weight * loss_sam

            psnr = calc_batch_psnr(Y_DE_eval, YH)
            sam = calc_batch_sam(Y_DE_eval, YH)

            total_loss += loss.item()
            total_psnr += psnr
            total_sam += sam
            count += 1

            bar.set_postfix({"val_loss": f"{loss.item():.4f}",
                                "l1": f"{loss_l1.item():.4f}",
                                "sam_loss": f"{loss_sam.item():.4f}",
                                "psnr": f"{psnr:.2f}",
                                "sam": f"{sam:.2f}"})
    return (total_loss / max(1, count),
            total_psnr / max(1, count),
            total_sam / max(1, count))
# -------------------- Main --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = 's'          # [Mode : real or simulation]

    parser = argparse.ArgumentParser(description='PyTorch COS2A HSI Reconstruction')
    parser.add_argument('--batchSize', type=int, default=16)         # [BatchSize]
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=12)
    parser.add_argument('--out_channels', type=int, default=172)
    parser.add_argument('--nEpochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sam_weight', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=123)
    if mode == 'r':
        parser.add_argument('--save_folder', default='checkpoints_realS2_test')  # [Real or Simulation]
        parser.add_argument('--ys_dir', type=str, default="crop_S_r")       # [Real or Simulation]
        parser.add_argument('--yh_dir', type=str, default="crop_A_r")
        parser.add_argument('--split_dir', type=str, default="split_real")
    else:
        parser.add_argument('--save_folder', default='checkpoints_test')         # [Real or Simulation]
        parser.add_argument('--ys_dir', type=str, default="crop_S")         # [Real or Simulation]
        parser.add_argument('--yh_dir', type=str, default="crop_A")
        parser.add_argument('--split_dir', type=str, default="split_simulation")
    opt = parser.parse_args()
    
    print(opt)
    print("Device:", device)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    train_files = read_split_file(os.path.join(opt.split_dir, "train.txt"))
    val_files = read_split_file(os.path.join(opt.split_dir, "val.txt"))

    print("[Using file-level split]")
    print("Train files:", len(train_files))
    print("Val files:", len(val_files))

    overlap = set(train_files) & set(val_files)
    print("Train/Val overlap:", len(overlap))

    if len(overlap) > 0:
        raise RuntimeError("Train and val split overlap!")

    train_set = loadDataset(opt.ys_dir, opt.yh_dir, file_list=train_files, patch_size=opt.patch_size, stride=opt.patch_size)
    val_set = loadDataset(opt.ys_dir, opt.yh_dir, file_list=val_files, patch_size=opt.patch_size, stride=opt.patch_size)
 
    D_init = build_fixed_D()
    model = COS2A(D_init).to(device)

    # ===== Loss & Optimizer =====
    L1loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    best_val_loss = float("inf")
    
    train_loader = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    
    YS, YH = next(iter(train_loader))
    YS = torch.clamp(YS.to(device), 0.0, 1.0)
    YH = torch.clamp(YH.to(device), 0.0, 1.0)

    with torch.no_grad():
        Y_DE = model(YS)
        rho = (F.softplus(model.rho) + 1e-4).item()

        print("\n[DEBUG Batch]")
        print("YS min/max:", YS.min().item(), YS.max().item())
        print("YH min/max:", YH.min().item(), YH.max().item())
        print("rho:", rho)
        print("Y_DE range:", Y_DE.min().item(), Y_DE.max().item())
        print("Y_DE mean:", Y_DE.mean().item())
        print("L1(Y_DE_clamp, YH):", L1loss(torch.clamp(Y_DE, 0.0, 1.0), YH).item())

    """
    Input  YS     : torch.Size([8, 12, 64, 64])
    Output YH_star: torch.Size([8, 172, 64, 64])
    Output Ym_hat : torch.Size([8, 12, 64, 64])
    Output Y_DE   : torch.Size([8, 172, 64, 64])
    Output D_wave : torch.Size([8, 12, 172])
    """
    del YS, YH, Y_DE
    torch.cuda.empty_cache()
    
    os.makedirs(opt.save_folder, exist_ok=True)
    for epoch in range(opt.nEpochs):
        train_loss = train(model, train_loader, optimizer, L1loss, device, epoch, opt.sam_weight)
        val_loss, val_psnr, val_sam = val(model, val_loader, L1loss, device, epoch, opt.sam_weight)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  "
              f"val_psnr={val_psnr:.2f}  "
              f"val_sam={val_sam:.2f}" )
        
        torch.save(model.state_dict(), os.path.join(opt.save_folder, "latest.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "best_model.pth"))
            print("Saved best model.")

    print("✅ Training Finished")