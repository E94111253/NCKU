# Image Processing HW1 — Color Image Histogram Equalization

## 1. 簡介

本專案為影像處理作業一，主要目的是**對彩色影像實作 Histogram Equalization（直方圖均衡化）**。

作業重點在於**不依賴 OpenCV 內建的均衡化函式**，而是自己實作以下流程：

1. 將 RGB 影像轉換為 HSV 顏色空間
2. 對亮度通道（Value）計算累積分布函數（CDF）
3. 對亮度通道進行 Histogram Equalization
4. 將 HSV 影像轉換回 RGB 顏色空間

此外，本專案也使用 **2D Histogram** 分析色相（Hue）與飽和度（Saturation）之間的聯合分布關係。

## 2. 專案架構

```
.
├── IP_hw1.py              # 主程式
├── image/
│   ├── image1.png         # 測試影像 1（較小）
│   └── image2.png         # 測試影像 2（較大）
│   ├── report/
│   │    └── Image Processing HW1.pdf               # 成果報告
├── requirements.txt       # 環境需求
└── README.md
```

## 3. 環境需求

本專案以 Python 開發，需要的套件列於 `requirements.txt`。安裝方式：

```bash
pip install -r requirements.txt
```

主要依賴套件：

| 套件 | 用途 |
| --- | --- |
| `numpy` | 數值運算、陣列處理 |
| `matplotlib` | 顯示影像、繪製 PDF 與 2D Histogram |
| `opencv-python` | 讀取影像、BGR→RGB 顏色通道轉換 |

> 注意：OpenCV **僅用於讀取影像與顏色通道順序轉換**，RGB↔HSV 轉換、CDF、Histogram Equalization 皆為自行實作。

## 4. 如何使用

1. 將測試影像放入 `image/` 資料夾，命名為 `image1.png` 或 `image2.png`。
2. 修改 `IP_hw1.py` 第 5 行的檔案路徑：

   ```python
   Image = cv2.imread('image/image2.png')      # image1 or image2
   ```

3. 執行程式：

   ```bash
   python IP_hw1.py
   ```

## 5. 輸出範例

程式執行後會依序顯示：

1. **H–S 二維直方圖**：顯示色相與飽和度的聯合分布。
   - 因原始影像整體偏暗，H–S 分布會集中在特定區域，故程式中設有頻率上限 `hist[hist > 25000] = 25000` 以提升低頻區域的可見度。

2. **原始影像 vs. 均衡化後影像**：左右對照。

3. **原始 PDF vs. 均衡化後 PDF**：顯示亮度分布的前後變化。

## 6. 實作流程說明

### Step 1：RGB → HSV

從 RGB 影像取出 R、G、B 三個通道後，依公式計算：

- **H（Hue）**：依最大值落在 R、G、B 哪個通道，套用不同公式，範圍 0～360
- **S（Saturation）**：`S = 1 - m / M`（M 為最大值、m 為最小值），範圍 0～1
- **V（Value）**：`V = M`，範圍 0～1

### Step 2：對 V 通道做 Histogram Equalization

1. 計算 V 通道的 PDF（機率密度函數）
2. 計算 CDF（累積分布函數）
3. 將 CDF 正規化到 0～255，得到新的灰階值對應表
4. 建立 `mapping`，把原圖每個像素值替換為均衡化後的值

### Step 3：HSV → RGB

使用 HSV 轉 RGB 的標準公式：

- 先計算 `hi = floor(h / 60)`、`f = h/60 - hi`
- 再計算 `p = v(1-s)`、`q = v(1-fs)`、`t = v(1-(1-f)s)`
- 依 `hi` 的值（0～5）分別填入 (r, g, b)

### Step 4：視覺化

- 使用 `np.histogram2d` 繪製 H–S 二維直方圖
- 使用 `matplotlib` 顯示原圖、均衡化圖、原始 PDF、均衡化 PDF

## 7. 結果討論

- **測試影像 1（225×225）**：均衡化後對比度明顯提升，但因影像較小，色調還原不夠鮮豔。
- **測試影像 2（1080×1920）**：均衡化後色彩鮮豔許多，影像左側亮區也對應到 PDF 右側的高峰。
- **2D Histogram**：原始影像偏暗，分布集中，因此設定頻率上限以避免低頻區域被淹沒。

## 8. 結論

1. Histogram Equalization 可有效擴展亮度分布並提升影像對比度。
2. 小尺寸影像的均衡化效果普通，色調還原不鮮豔。
3. 大尺寸影像的均衡化效果較好，色彩更加鮮豔。
4. 2D Histogram 設定頻率上限可解決資料集中於某區域的問題。