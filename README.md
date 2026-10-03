# NCKU Projects & Technical Portfolio

本作品集彙整了我在國立成功大學四年期間所完成之精選專案、技術課程及程式設計認證。

我的研究範疇主要涵蓋 **影像處理、電腦視覺、深度學習、最佳化、搜尋演算法、科學計算，以及嵌入式／系統整合**。

My work mainly covers **image processing, computer vision, deep learning, optimization, search algorithms, scientific computing, and embedded/system integration**.

---

## Featured Projects

### 1. COS2A Hyperspectral Reconstruction

**Topic:** 基於COS2A之 Sentinel-2影像高光譜重建與實驗分析

**Main work**
- Preprocessed and paired AVIRIS and Sentinel-2 remote-sensing data.
- Implemented the COS2A reconstruction pipeline with **Deep Unfolding** and **CNMF**.
- Trained and evaluated models on both simulated and real Sentinel-2 data.
- Investigated spectral-band instability and modified the training loss to improve spectral consistency.
- Evaluated reconstruction quality using **PSNR, SAM, and SSIM**.

**Tools / Methods:** Python, MATLAB, ENVI, QGIS, Deep Unfolding, CNMF, Convex Optimization, Remote Sensing


---

### 2. Automatic Rubik's Cube Solving System

**Topic:** 啟發函數與剪枝策略對魔術方塊求解效率之理論與實證分析

**Main work**
- Implemented a Kociemba-based solver using **IDA\\***, DFS, heuristic search, and pruning tables.
- Built a computer-vision pipeline for cube-face color recognition using OpenCV and HSV color space.
- Converted recognized cube states into solver-compatible representations.
- Integrated the solver with Raspberry Pi, camera, servo motors, and a custom mechanical structure.
- Evaluated solving efficiency using 200 randomly generated cube states.

**Tools / Methods:** Python, OpenCV, Raspberry Pi, Kociemba Algorithm, IDA\\*, DFS, Pruning, Computer Vision, System Integration


---

## Additional Projects

### Binary Semantic Segmentation
Implemented semantic-segmentation models using **U-Net** and **ResNet34-U-Net**, including model training, validation, and qualitative result comparison.

**Tools / Methods:** PyTorch, U-Net, ResNet34, Image Segmentation

### Value-Based Reinforcement Learning
Implemented and compared value-based reinforcement-learning methods including **DQN, Double DQN, Prioritized Experience Replay, and N-step Return**.

**Tools / Methods:** PyTorch, Gymnasium, DQN, Double DQN, PER, N-step Return

### Generative Models
Implemented a conditional diffusion model and explored techniques including **Conditional DDPM, Classifier-Free Guidance, Attention, and EMA**.

**Tools / Methods:** PyTorch, DDPM, CFG, Attention, EMA

### Color Image Histogram Equalization
Applied histogram-based image-enhancement techniques to color images and compared the effects of different processing approaches.

**Tools / Methods:** Python, OpenCV, Histogram Equalization

### Morphological Processing & Fourier Transform
Implemented common digital image-processing operations including **morphological processing** and **frequency-domain analysis using Fourier Transform**.

**Tools / Methods:** Python, OpenCV, Morphology, Fourier Transform

### Mango Image Analysis
Built an image-analysis workflow using handcrafted features and clustering methods for mango images.

**Tools / Methods:** Python, OpenCV, K-Means, LBP, HSV, Morphology

### Ultimate Tic-Tac-Toe
Implemented an Ultimate Tic-Tac-Toe game on **PYNQ-Z2**, including logic design and LED-matrix control.

**Tools / Methods:** PYNQ-Z2, FPGA, Logic Design, LED Matrix Control

---

## Technical Skills

- **Programming / Scientific Computing:** Python, MATLAB, NumPy, Pandas
- **Deep Learning:** PyTorch, CNN, U-Net, ResNet, DDPM
- **Computer Vision / Image Processing:** OpenCV, HSV, Histogram Equalization, Morphology, Fourier Transform
- **Optimization / Algorithms:** Convex Optimization, CNMF, Deep Unfolding, IDA*, DFS, Pruning
- **Remote Sensing:** ENVI, QGIS, Sentinel-2, AVIRIS
- **System / Hardware:** Raspberry Pi, Camera Module, Servo Motor Control, PYNQ-Z2, FPGA

---

## Certifications

| Platform / Institution | Certificate | Completion | Key Skills |
|---|---|---:|---|
| Harvard University / CS50 | CS50's Introduction to Programming with Python (CS50P) | 2026 | Python programming, file I/O, exception handling, testing, regular expressions, final project |
| MathWorks | Core MATLAB Skills | 2026-09-10 | MATLAB desktop tools, debugging, matrices, vectors, calculations, plotting |
| Kaggle Learn | Python | 2024-06-24 | Python fundamentals, functions, control flow, data structures |
| Kaggle Learn | Pandas | 2024-07-11 | Data cleaning, DataFrame manipulation, exploratory data analysis |
| Kaggle Learn | Intro to Machine Learning | 2024-07-23 | Decision trees, random forests, model validation |
| Kaggle Learn | Intermediate Machine Learning | 2024-08-01 | Pipelines, cross-validation, feature engineering, XGBoost |
| Kaggle Learn | Computer Vision | 2025-07-08 | Image classification, convolutional neural networks, data augmentation |
| Kaggle Learn | Geospatial Analysis | 2025-07-21 | Geospatial data processing, mapping, spatial visualization |

---

## GitHub

GitHub profile: [github.com/E94111253](https://github.com/E94111253/NCKU)

For detailed implementation, source code, experimental results, and project-specific documentation, please refer to the corresponding repositories.
