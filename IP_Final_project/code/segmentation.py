import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_features(img):
    try:
        # 色彩轉換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # 邊緣與輪廓
        edges = cv2.Canny(blur, 50, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:
            return None

        # --------- 形狀特徵 ----------
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            return None

        aspect_ratio = w / h
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        shape_features = [aspect_ratio, circularity, convexity]

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # --------- HSV顏色特徵 ----------
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        mean_hue = cv2.mean(hue, mask=mask)[0]
        mean_sat = cv2.mean(sat, mask=mask)[0]
        mean_val = cv2.mean(val, mask=mask)[0]

        # 色彩變異（標準差）
        std_sat = np.std(sat[mask == 255])
        std_val = np.std(val[mask == 255])

        color_features = [mean_hue, mean_sat, mean_val, std_sat, std_val]

        # --------- 紋理特徵：LBP ----------
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_masked = lbp[mask == 255]
        hist_lbp, _ = np.histogram(lbp_masked, bins=10, range=(0, 10))
        hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-6)

        texture_features = list(hist_lbp)

        # --------- 合併所有特徵 ----------
        return np.concatenate([shape_features, color_features, texture_features])
    
    except Exception as e:
        print(f"[錯誤] 特徵擷取失敗：{e}")
        return None

def extract_quality_features(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[:, int(0.1*w):int(0.9*w)] = 255  # 保留中間 80%

    # 黑斑範圍（深褐區）
    dark_mask = cv2.inRange(hsv, (0, 50, 0), (50, 255, 100))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # 去除邊界小圓形（疑似蒂頭）
    # dark_mask[:, :int(0.08*w)] = 0
    # dark_mask[:, -int(0.08*w):] = 0

    dark_area_ratio = np.sum(dark_mask > 0) / (h*w)

    # 飽和度與亮度平均值
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    avg_s = np.mean(sat[mask_center == 255])
    avg_v = np.mean(val[mask_center == 255])

    return np.array([
        dark_area_ratio,        # 黑斑比例
        avg_s / 255.0,          # 飽和度 (0~1)
        avg_v / 255.0           # 亮度 (0~1)
    ])
