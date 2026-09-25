import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scipy.io as sio
import numpy as np
import h5py

# ---------- 讀 .mat ----------
def load_mat_file(path):
    try:
        # 一般 MATLAB .mat，用 scipy 讀
        data = sio.loadmat(path)

        if 'patch_A' in data:
            arr = data['patch_A']
        elif 'patch_S' in data:
            arr = data['patch_S']
        else:
            raise ValueError(f"Unknown key in {path}")

    except NotImplementedError:
        # MATLAB -v7.3 .mat，用 h5py 讀
        with h5py.File(path, 'r') as f:
            if 'patch_A' in f:
                arr = np.array(f['patch_A'])
            elif 'patch_S' in f:
                arr = np.array(f['patch_S'])
            else:
                raise ValueError(f"Unknown key in {path}. Keys: {list(f.keys())}")

            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 1, 0))

    arr = arr.astype(np.float32)

    # H W C → C H W
    if arr.shape[0] != 12 and arr.shape[0] != 172:
        arr = np.transpose(arr, (2, 0, 1))

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


# ---------- 切 patch ----------
def extract_patches(img, patch_size, stride):
    # img: C H W
    C, H, W = img.shape
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    patches = np.stack(patches, axis=0)  # N C H W
    return torch.tensor(patches)

if __name__ == '__main__':

    # ===== 路徑 =====
    ys_dir = "crop_S"
    yh_dir = "crop_A"

    # ===== 找檔案 =====
    ys_files = sorted([f for f in os.listdir(ys_dir) if f.endswith(".mat")])
    yh_files = sorted([f for f in os.listdir(yh_dir) if f.endswith(".mat")])

    print(f"YS files: {len(ys_files)}")
    print(f"YH files: {len(yh_files)}")

    assert len(ys_files) == len(yh_files), "數量不一致"

    # ===== 測試單張讀取 =====
    ys_path = os.path.join(ys_dir, ys_files[80])
    yh_path = os.path.join(yh_dir, yh_files[80])

    YS = load_mat_file(ys_path)
    YH = load_mat_file(yh_path)

    print("\n=== Single File Test ===")
    print("YS shape:", YS.shape)
    print("YH shape:", YH.shape)

    assert YS.shape[0] == 12, "YS band 錯誤"
    assert YH.shape[0] == 172, "YH band 錯誤"

    # ===== 測試 patch extraction =====
    patch_size = 64
    stride = 32

    YS_patches = extract_patches(YS, patch_size, stride)
    YH_patches = extract_patches(YH, patch_size, stride)

    print("\n=== Patch Test ===")
    print("YS patches:", YS_patches.shape)
    print("YH patches:", YH_patches.shape)

    assert YS_patches.shape[0] == YH_patches.shape[0], "patch數量不一致"

    # ===== 顯示一張 =====
    def normalize(x):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x
    
    YH = torch.tensor(YH)
    YS = torch.tensor(YS)
    # AVIRIS RGB
    rgb_A = torch.stack([
        YH[30],
        YH[60],
        YH[90]
    ], dim=0).permute(1,2,0)

    # Sentinel RGB
    rgb_S = torch.stack([
        YS[3],
        YS[2],
        YS[1]
    ], dim=0).permute(1,2,0)

    rgb_A = normalize(rgb_A)
    rgb_S = normalize(rgb_S)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(rgb_A.numpy())
    plt.title("AVIRIS")

    plt.subplot(1,2,2)
    plt.imshow(rgb_S.numpy())
    plt.title("Sentinel-2")

    plt.show()

    # ===== 測試 DataLoader =====
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, ys_dir, yh_dir):
            self.ys_files = sorted([os.path.join(ys_dir,f) for f in os.listdir(ys_dir) if f.endswith(".mat")])
            self.yh_files = sorted([os.path.join(yh_dir,f) for f in os.listdir(yh_dir) if f.endswith(".mat")])

        def __len__(self):
            return len(self.ys_files)

        def __getitem__(self, idx):
            YS = load_mat_file(self.ys_files[idx])
            YH = load_mat_file(self.yh_files[idx])

            YS = torch.tensor(YS)
            YH = torch.tensor(YH)

            return YS, YH

    dataset = SimpleDataset(ys_dir, yh_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    print("\n=== DataLoader Test ===")
    for YS_batch, YH_batch in loader:
        print("Batch YS:", YS_batch.shape)
        print("Batch YH:", YH_batch.shape)
        break

    print("\n✅ dataset.py 測試完成！")