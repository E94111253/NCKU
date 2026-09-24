# 魔術方塊啟發式搜尋與自動還原專題

本專題以魔術方塊的角塊與邊塊狀態為基礎，使用預先計算的搜尋表、啟發值及分階段搜尋產生轉動序列，並結合相機色彩辨識與樹莓派 GPIO 馬達控制。研究重點是觀察啟發式搜尋、剪枝與多候選解選擇如何影響求解步數和搜尋時間；樹莓派是實作平台與資源限制。

**目前程式為研究原型。** 第二階段會從不同起始轉動啟動平行搜尋，預設收集最多三組回傳解，選其中步數最少者；這不是全部魔術方塊解法中的全域最短解證明。README 依現有程式描述，沒有把報告中的實驗統計當作本壓縮檔可直接重現的結果。

## 專案結構

```text
ES_project/
├── README.md
├── requirements.txt             # 桌面環境的 Python 套件
├── main.py                      # 相機／預設色彩 → 狀態建構 → 求解 → 馬達
├── project/                     # 魔術方塊狀態、編碼、搜尋及求解器
│   ├── cube_state.py            # 角塊與邊塊狀態、轉動操作
│   ├── build.py                 # 六面顏色轉成角塊／邊塊編碼
│   ├── Database.py              # 狀態編碼與搜尋表建立程式
│   ├── heuristic.py             # 載入搜尋表、計算兩階段啟發值
│   ├── search.py                # 第一階段平行搜尋；另有獨立示範入口
│   ├── IDA.py                   # 第二階段平行 IDA* 與候選解選擇
│   ├── IDAs.py                  # 第二階段單程序後備搜尋
│   └── solver.py                # 串接第一、第二階段
├── vision/
│   ├── camara.py                # Picamera2 六面拍攝（沿用原檔名）
│   ├── color.py                 # HSV 顏色分類與單面視覺化
│   └── scan_face.py             # 讀取六面照片並取樣九宮格
├── hardware/
│   ├── servo.py                 # 單馬達／多馬達校正腳本
│   └── sevros.py                # 固定轉動序列的馬達測試腳本
└── data/
    ├── pattern_tables/         # 附帶的 .npy 搜尋表
    ├── sample_photos/          # 附帶的六面測試照片
    └── captures/               # 相機執行時輸出的照片


## 求解流程

1. `vision/camara.py` 拍攝 U、D、F、B、L、R 六面，照片寫入 `data/captures/`；也可用現成的顏色矩陣略過相機。
2. `vision/scan_face.py` 以固定九宮格座標及 HSV 規則辨識貼紙顏色。拍攝位置、照明與 HSV 門檻都需要依實際環境校正。
3. `project/build.py` 依六面中心色及各位置貼紙建立角塊排列 `cp`、角塊方向 `co`、邊塊排列 `ep`、邊塊方向 `eo`。
4. `project/search.py` 在深度限制內使用啟發值與平行 DFS 尋找**第一組**第一階段結果；`project/heuristic.py` 從 `.npy` 表取得啟發值。
5. `project/IDA.py` 以不同起始轉動開啟第二階段平行 IDA* 搜尋，預設收集三組回傳解並以 `min(results, key=len)` 選擇步數最少的一組。不足三組時會在等待逾時後從已收集的解中挑選；若沒有解，會嘗試 `IDAs.py` 的單程序後備搜尋。
6. `project/solver.py` 合併兩階段轉動序列。序列元素是 `(face, turns)`；面編號為 `0=U, 1=D, 2=F, 3=B, 4=L, 5=R`，`turns` 為 1、2 或 3。
7. `test.py` 等待使用者按 Enter，再以 GPIO/PWM 發送轉動命令。馬達腳位與機構面位對應需要依實際接線檢查。

> 注意：這份專題與標準 Kociemba 演算法的完整條件並不完全等同。現有第一階段目標僅檢查角塊和邊塊方向，搜尋迴圈也未遍歷面 `0`；它不能據此宣稱完整搜尋或保證對所有狀態成功。第二階段的「三組」是成功回傳並收集到的候選解上限，不保證每次恰好有三組，也不保證候選解裡有最佳解。

## 環境與安裝

- 建議 Python 3.11（壓縮檔原始快取顯示曾使用 3.11；未附完整版本鎖定）。
- 核心搜尋需 NumPy；影像辨識需 OpenCV、Matplotlib；樹莓派拍照需 `picamera2`；GPIO 控制需 `RPi.GPIO` 及已接妥的硬體。
- 六個 `.npy` 搜尋表已附在 `data/pattern_tables/`。以下指令請從專案根目錄執行，以便 Python 找到 project 和 vision 套件。

```bash
cd ES_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

requirements.txt 列出可由一般 Python 套件環境安裝的 NumPy、OpenCV、Matplotlib。樹莓派專用的 picamera2、RPi.GPIO 仍須依 Raspberry Pi OS 與相機/GPIO 設定安裝及測試；main.py 即使使用預設色彩矩陣，也會在啟動時匯入這兩個樹莓派套件。桌面電腦可單獨執行 project.solver，無法直接操作相機及馬達。

## 執行方式

**只測試演算法（不連接 GPIO）：**

```bash
cd ES_project
python -m project.solver
```

此入口會使用 `project/heuristic.py` 內固定的打亂序列。搜尋程式可能長時間執行或等待程序回傳；若沒有找到解，會回報失敗。

**檢查單面色彩取樣：**

```bash
python -m vision.color
```

上面的程式讀取 `data/sample_photos/photo_d.jpg`，顯示九宮格及 HSV 值；需桌面顯示環境。若要掃描六面已拍攝照片，可將照片以 `photo_u.jpg`、`photo_d.jpg`、`photo_f.jpg`、`photo_b.jpg`、`photo_l.jpg`、`photo_r.jpg` 命名，放入 `data/captures/`，再執行：

```bash
python -m vision.scan_face
```

**樹莓派整合流程（會啟動馬達）：** `main.py` 的預設 `test(mode='1')` 使用內建六面顏色矩陣，不拍照；若改為 `mode='2'`，會先拍攝六面並辨識照片。兩種模式在求解後都會等待 Enter，之後操作 GPIO 馬達：

```bash
python main.py
```

首次執行前請逐一核對 `main.py` 中的 `motor_pins`、`DEGREE_TIME`、轉動面編號及機構方向。`hardware/` 中的三個檔案是獨立硬體測試，部分在執行或匯入時就會驅動腳位，不是整合流程的必要入口。

若要重新產生搜尋表，可在專案根目錄執行 `python -m project.Database`。這會**覆寫**附帶的表，建議先備份；建立過程可能耗時且耗用記憶體。

## 已知限制與後續驗證

- 現有 `.npy` 表並非全部座標都有距離值；例如 `ep1_pattern_db.npy` 有大量 `-1`。目前啟發值直接讀取表值，尚未處理未覆蓋座標，因此不可將所有啟發值解釋為有效的最低剩餘步數。
- `build.py` 依顏色組合建立塊狀態，沒有完整檢查照片辨識結果是否構成合法魔術方塊。
- 目前未附第一組、第二組、第三組候選解的逐筆實驗紀錄；若要實證比較三候選策略與單解策略，需額外記錄每個候選解步數、累計搜尋時間及失敗案例。
- 本整理版已做 Python 語法編譯、搜尋表載入及已還原狀態的啟發值檢查；未在實機上驗證拍攝、GPIO 和完整還原流程。
