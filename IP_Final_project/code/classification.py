from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from segmentation import extract_features
from collections import defaultdict
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "Dataset/Classification_dataset"
categories = ["Anwar Ratool", "Chaunsa (Black)", "Chaunsa (Summer Bahisht)", "Chaunsa (White)", "Dosehri", "Fajri", "Langra", "Sindhri", ]  # 對應資料夾名稱

all_features = []
all_labels = []

for label, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    
    # 讀取資料夾內所有影像
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
    
        if img_name.lower().endswith(('.jpg')):
            img = cv2.imread(img_path)
           
            # 提取特徵
            features = extract_features(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(label)  # 使用數字標籤 (0:A, 1:B, 2:C...)

# 轉換為 numpy 陣列
X = np.array(all_features)
y = np.array(all_labels)

# 檢查是否有數據
if len(X) == 0:
    print("錯誤：沒有讀取到任何有效的影像數據")
    exit()

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 K-Means 分群 (分8群)
kmeans = KMeans(n_clusters = 8, random_state = 42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# 分析分群結果
cluster_analysis = defaultdict(list)

for true_label, cluster_label in zip(y, cluster_labels):
    cluster_analysis[cluster_label].append(true_label)

max_len = max(len(cat) for cat in categories)
print("\n分群結果分析:")
for cluster, true_labels in cluster_analysis.items():
    print(f"群集 {cluster}:")
    for i, cat in enumerate(categories):
        print(f"  包含 {cat.ljust(max_len)}: {true_labels.count(i)} 張")

print("KMeans Inertia:", kmeans.inertia_)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for cluster in range(8):
    plt.scatter(X_pca[cluster_labels==cluster, 0], X_pca[cluster_labels==cluster, 1], label=f'Cluster {cluster}')
plt.legend()
plt.title('K-Means Cluster Visualization (PCA)')
plt.show()
#---------------------------------------------------------------------------------
"""
異常點檢查
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=42)
outlier_flags = iso.fit_predict(X_scaled)  # -1: outlier, 1: inlier


X_clean = X_scaled[outlier_flags == 1]
y_clean = y[outlier_flags == 1]

# 重新分群
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X_clean)
cluster_labels = kmeans.labels_

cluster_analysis = defaultdict(list)
for true_label, cluster_label in zip(y_clean, cluster_labels):
    cluster_analysis[cluster_label].append(true_label)

max_len = max(len(cat) for cat in categories)
print("\n分群結果分析:")
for cluster, true_labels in cluster_analysis.items():
    print(f"群集 {cluster}:")
    for i, cat in enumerate(categories):
        print(f"  包含 {cat.ljust(max_len)}: {true_labels.count(i)} 張")

print("KMeans Inertia:", kmeans.inertia_)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)
plt.figure(figsize=(8,6))
for cluster in range(8):
    plt.scatter(X_pca[cluster_labels==cluster, 0], X_pca[cluster_labels==cluster, 1], label=f'Cluster {cluster}')
plt.legend()
plt.title('K-Means Cluster Visualization (PCA)')
plt.show()

