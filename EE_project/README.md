# 基於 COS2A 的 Sentinel-2 高光譜影像重建

本專題以 12 波段 Sentinel-2 多光譜影像預測 172 波段高光譜影像。`COS2A` 深度展開網路先產生粗略重建結果 `Y_DE`；`algo1` 再結合光譜響應估計及非負矩陣分解後處理，產生 `YH_star`。研究報告比較模擬 Sentinel-2 與真實 Sentinel-2 配對資料在不同地貌上的 PSNR、SAM、SSIM。本資料夾包含研究用程式和兩組權重，**尚未包含訓練或推論所需的影像／patch 資料**，因此不能直接重現報告數值。

## 資料夾

```text
EE_project/
├── main.py                    # 訓練 COS2A 深度展開網路
├── inference.py               # 單張配對 patch 推論及視覺比較
├── README.md
├── requirements.txt
├── src/
│   ├── COS2A.py, models.py    # 深度展開模型及 stage
│   ├── algo.py, fast_convex.py # 光譜響應估計與 CNMF 後處理
│   ├── dataset.py, psnr.py    # .mat 讀取及評估函式
│   ├── make_split.py          # 按原始影像分組建立資料分割
│   └── Fast_convex_test.py    # 研究中的替代 CNMF 測試程式
├── dataset/
│   ├── crop_A/, crop_S/       # 模擬組：AVIRIS / Sentinel-2 .mat
│   └── crop_A_r/, crop_S_r/   # 真實組：AVIRIS / Sentinel-2 .mat 
├── split_simulation/          # 原有 train/val/test 檔名清單
├── split_real/                # 原有 train/val/test 檔名清單
├── checkpoints/
│   ├── simulationS2/          # 原有 best_model.pth、latest.pth
│   └── realS2/                # 原有 best_model.pth、latest.pth
├── Preprocessing/             # MATLAB 裁切、人工檢查與篩選
└── doc/                       # 專題研究成果報告
```

壓縮檔原有的根目錄模型程式與 `src/models/` 內容相同，兩處權重檔也逐一相同；整理版只保留一套模型程式與每種設定各一套權重。`crop_*` 原本是空資料夾，這裡保留空位供補資料。

## 安裝

研究報告使用 Python 3.9.21 與 MATLAB R2024a。Python 程式的套件列於 `requirements.txt`；PyTorch 的 CPU 或 CUDA 安裝版本需配合執行電腦。

```bash
cd EE_project
python -m pip install -r requirements.txt
```

`Preprocessing/*.m` 需要 MATLAB。ENVI 與 QGIS 是原研究使用的資料處理工具，並非 Python 套件。

## 補入資料

若已有處理好的配對 `.mat`，請將模擬組放進 `dataset/crop_S/`（輸入 Sentinel-2）與 `dataset/crop_A/`（AVIRIS 真值）；真實組放進 `dataset/crop_S_r/` 和 `dataset/crop_A_r/`。每對檔案須同名，例如兩邊都叫 `patch_00033.mat`。程式分別讀取 `.mat` 內的 `patch_S`（12 波段）與 `patch_A`（172 波段），使用時轉為 `(波段, 高, 寬)`。原有分割名單也必須能在這些資料夾找到相應檔名。

若要重新建立分割，配對檔案還須有一致的 `source_file` 欄位；程式依原始影像分組，避免同一來源同時出現在不同集合，且至少需要三個來源群組。`split_real/` 與 `split_simulation/` 已有名單；重新執行分割程式會改寫對應資料夾中的文字檔，請先保存原名單。

若從 ENVI 原始資料開始，將 AVIRIS 與配準後的真實 S2 `.dat`、`.hdr` 分別放在 `dataset/raw/aviris/`、`dataset/raw/s2/`。`Preprocessing/crop.m` 的真實資料模式另需 `dataset/raw/s2_pair_map.csv`，至少包含 `aviris_file` 和 `s2_file` 欄。請在專案根目錄設定 MATLAB 工作目錄再執行，並核對 `crop.m` 的 `mode` 與資料配準設定；模擬模式只從 AVIRIS 建立輸入。

## 訓練及推論

在專案根目錄執行：

```bash
python main.py --mode simulation
python main.py --mode real
```

`main.py` 讀取對應的 `split_simulation/` 或 `split_real/`，預設 64 × 64 patch、30 個 epoch。它訓練 COS2A，使用 L1、光譜角及輸出範圍損失；新權重預設寫入 `checkpoints/training_simulationS2/`、`checkpoints/training_realS2/`，保留原有權重。`--help` 可查看批次大小和其他參數。

對單張配對 patch 進行推論與繪圖：

```bash
python inference.py --mode simulation --sample patch_00033.mat
python inference.py --mode real --sample patch_00451.mat
```

這些是原程式的示例檔名；請依補入資料的檔名替換。預設使用 `checkpoints/simulationS2/best_model.pth` 或 `checkpoints/realS2/best_model.pth`，也可以使用 `--checkpoint` 指定其他權重。此分析腳本需要同名的 AVIRIS 真值，會顯示 Sentinel-2 輸入、真值、`Y_DE` 及後處理結果 `YH_star`，並輸出單張 PSNR、SAM 等資訊；它不會自動計算報告的 20 張影像統計表。

需要以新增資料重建 train/val/test 名單時，可執行：

```bash
python -m src.make_split --mode simulation
python -m src.make_split --mode real
```

預設依 `source_file` 群組，以約 80% / 10% / 10% 切分並寫入既有 `split_*` 資料夾。分組數少時，整數切分結果不一定精確等於上述比例。

## 已知限制

- 原有兩組 checkpoint 及分割名單已附，但缺少 `.mat` 影像；須補入相符檔名與預期波段數後才能進一步驗證。
- 本次修改處理資料夾搬移造成的模組匯入及資料路徑，保留原模型架構與已訓練權重，沒有重新訓練。
- MATLAB 檢查與篩選腳本可能依賴人工操作或未附的篩選紀錄；請在執行刪除或改寫資料的腳本前核對其輸入。
- 程式語法及靜態路徑已檢查；尚未以完整影像資料在 PyTorch/MATLAB 環境執行整套流程。