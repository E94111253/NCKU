import numpy as np
import cv2
import matplotlib.pyplot as plt


path_set = ['u', 'd', 'f', 'b', 'l', 'r']
path = 'data/sample_photos/photo_d.jpg'
image = cv2.imread(path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#--------------------- 調整格子 ---------------------
cell_size = 300             # 每格大小
margin = 400                # 格子間距
grid_start_x = 340          # 格子區左上角 x 座標
grid_start_y = 120          # 格子區左上角 y 座標
#--------------------- 調整格子 ---------------------

manual_hues = []
manual_names = []
manual_hsv_values = []
fig, ax = plt.subplots()
ax.imshow(image_rgb)

# HSV主色擷取（使用Hue通道直方圖＋遮罩）
def extract_dominant_hsv_color(hsv_patch):
    h, s, v = cv2.split(hsv_patch)
    mask = cv2.inRange(hsv_patch, (0, 30, 30), (180, 255, 255))
    hist = cv2.calcHist([h], [0], mask, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    s_mean = np.mean(s[mask > 0]) / 255
    v_mean = np.mean(v[mask > 0]) / 255
    return dominant_hue, s_mean, v_mean

# 對應顏色名稱
def color_name(h, s, v):
    
    if s < 0.15:
        return 'W'
        
    if 179>= h >= 170 :
        if v < 0.72:
            return 'R'
        else:
            return 'O'
    elif h == 0:
        if v > 0.75:
            return 'O'
        elif 0.75 > v > 0.4:
            return 'R'
        else:
            return "w"
    elif 0< h < 20 :
        return 'O'
    elif h < 35:
        return 'Y'
    elif h < 85:
        return 'G'
    elif h < 140:
        if s > 0.5:
            return 'B'
        else:
            return 'W'
    
    else:
        return 'W'


if __name__ == "__main__":
    # 掃描九宮格
    for row in range(3):
        for col in range(3):
            x1 = grid_start_x + col * (cell_size + margin)
            y1 = grid_start_y + row * (cell_size + margin)
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            patch = hsv[y1:y2, x1:x2]
            h, s, v = extract_dominant_hsv_color(patch)
            cname = color_name(h, s, v)
            manual_hues.append(h)
            manual_names.append(cname)
            manual_hsv_values.append((h, s, v))

            # 畫格子與顯示Hue值
            rect = plt.Rectangle((x1, y1), cell_size, cell_size, linewidth=1.5, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1 + 3, y1 + 20, f"H:{h} {cname}", color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.6))
            
    # 顯示終端九宮格結果
    print("九宮格辨識結果：")
    for i in range(9):
        print(manual_names[i], end=' ')
        if (i + 1) % 3 == 0:
            print()

    print("="*20)
    print("\nHSV 值矩陣格式：")
    print("H 值矩陣:")
    for i in range(0, 9, 3):
        print(f"       {manual_hsv_values[i][0]:3d}  {manual_hsv_values[i+1][0]:3d}  {manual_hsv_values[i+2][0]:3d}")

    print("\nS 值矩陣:")
    for i in range(0, 9, 3):
        print(f"       {manual_hsv_values[i][1]:3f}  {manual_hsv_values[i+1][1]:3f}  {manual_hsv_values[i+2][1]:3f}")

    print("\nV 值矩陣:")
    for i in range(0, 9, 3):
        print(f"       {manual_hsv_values[i][2]:3f}  {manual_hsv_values[i+1][2]:3f}  {manual_hsv_values[i+2][2]:3f}")

    print("\n顏色名稱矩陣:")
    for i in range(0, 9, 3):
        print(f"       {manual_names[i]}   {manual_names[i+1]}   {manual_names[i+2]}")
    ax.set_title("Manual Grid (Tight) with HSV Hue Detection")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



