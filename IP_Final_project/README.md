Image Processing Final Project

## 1.簡介 : 跟據已知的Dataset將芒果分別做**品種分類**與**品質分群**，最後會直接顯示出全部分群完的結果。
## 2.架構 :
    IP_Final_project
        ├── code/
        │    ├── classification.py          # 品種分群
        │    ├── grading.py                 # 品質分群
        │    └── segmentation.py            # 主要特徵萃取
        ├── Dataset/
        |    ├── Classification_dataset/    # 品種資料集（需自行下載）
        │    └── Grading_dataset/           # 品質資料集（需自行下載）
        ├── report/
        │    └── Image Processing Final Project.pdf                 # 成果報告
        ├── requirements.txt                # 環境需求
        └── README.md

## 3. 環境需求

本專案以 Python 開發，需要的套件列於 `requirements.txt`。安裝方式：

    ```bash
    pip install -r requirements.txt
    ```

## 4. 資料集下載

本專案所使用的芒果影像資料集**並未隨專案附上**，請自行前往 Kaggle 下載：

- **Mango Varieties Classification**
  <https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification>

下載後請解壓縮，並依照下列結構放置：
Dataset/
├── Classification_dataset/ # 品種資料集
└── Grading_dataset/ # 品質資料集


## 5.如何使用 :
    主要有三個程式碼，其中
        segmentation.py     : 只有特徵萃取功能，不會有輸出，裡面定義兩個函數extract_features()與extract_quality_features()提供給另外兩個程式碼所使用。
        classification.py   : 主要實現K Means演算法，輸出為最後Classification_dataset的分群結果
        grading.py          : 主要實現K Means演算法，輸出為最後Grading_dataset的分群結果
        
            (如果執行發生錯誤，可以檢查第12行資料路徑有沒有正確)
            
## 6.輸出範例 :
    grading.py :

        群集 0:
            包含 Class_I: 100 張
            包含 Class_II: 25 張
            包含 Extra_Class: 10 張

