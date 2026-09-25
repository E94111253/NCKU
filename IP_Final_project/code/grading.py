import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from segmentation import extract_quality_features

dataset_path = "Dataset/Grading_dataset"
categories = ["Class_I", "Class_II", "Extra_Class"]  

features_list = []
labels_list = []
image_paths = []

for label, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    for fname in os.listdir(category_path):
        if fname.lower().endswith(('.jpg')):
            img = cv2.imread(os.path.join(category_path, fname))
            if img is not None:
                feat = extract_quality_features(img)
                features_list.append(feat)
                labels_list.append(label)
                image_paths.append(category_path)

X = np.array(features_list)
y = np.array(labels_list)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 分群
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
predicted_clusters = kmeans.labels_

# 分群統計分析
cluster_summary = defaultdict(list)
for true, pred in zip(y, predicted_clusters):
    cluster_summary[pred].append(true)

print("\n分群結果分析:")
for cluster, true_labels in cluster_summary.items():
    print(f"群集 {cluster}:")
    for i, cat in enumerate(categories):
        print(f"  包含 {cat}:{true_labels.count(i)} 張")

# 視覺化
plt.figure(figsize=(12, 6))
for i in range(9):
    if i >= len(features_list): break
    idx = i
    category = categories[y[idx]]
    cluster = predicted_clusters[idx]
    img_path = os.path.join(dataset_path, category, os.listdir(os.path.join(dataset_path, category))[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"actural: {category}\ncluster: {cluster}")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
for i in range(9):
    if i >= len(features_list): break
    idx = i
    category = categories[y[idx]]
    cluster = predicted_clusters[idx]
    img_path = os.path.join(dataset_path, category, os.listdir(os.path.join(dataset_path, category))[1])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"actural: {category}\ncluster: {cluster}")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
for i in range(9):
    if i >= len(features_list): break
    idx = i
    category = categories[y[idx]]
    cluster = predicted_clusters[idx]
    img_path = os.path.join(dataset_path, category, os.listdir(os.path.join(dataset_path, category))[2])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"actural: {category}\ncluster: {cluster}")
    plt.axis('off')
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 K-Means 分群 (假設分3群)
kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for cluster in range(3):
    plt.scatter(X_pca[cluster_labels==cluster, 0], X_pca[cluster_labels==cluster, 1], label=f'Cluster {cluster}')
plt.legend()
plt.title('K-Means Cluster Visualization (PCA)')
plt.show()

