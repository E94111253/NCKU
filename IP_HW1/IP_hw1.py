import numpy as np
import matplotlib.pyplot as plt
import cv2

Image = cv2.imread('image/image2.png')      # image1 or image2
image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
image = image / 255
R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
print(image.size)
print(image.shape)      # (225, 225, 3)

M = np.maximum(R, np.maximum(G, B))
m = np.minimum(R, np.minimum(G, B))
temp = M - m

def to_HSV(R, G, B, M, m, temp):
  
    H = np.zeros_like(M)
    max_R1 = (M == R) & (G >= B) & (temp != 0)
    max_R2 = (M == R) & (G < B) & (temp != 0)
    max_G = (M == G) & (temp != 0)
    max_B = (M == B) & (temp != 0)
    H[max_R1] = 60 * (G[max_R1] - B[max_R1])/temp[max_R1]
    H[max_R2] = 60 * (G[max_R2] - B[max_R2])/temp[max_R2] + 360
    H[max_G] = 60 * (B[max_G] - R[max_G])/temp[max_G] + 120
    H[max_B] = 60 * (R[max_B] - G[max_B])/temp[max_B] + 240

    S = np.zeros_like(M)
    S[M != 0] = 1 - (m[M != 0] / M[M != 0])
    
    V = M

    return H, S, V
# H : 0~360, S : 0~1, V : 0~1
H, S, V = to_HSV(R, G, B, M, m, temp)

show_H, show_S, show_V =H , S * 255, V * 255

def probability(v):
    unique_values, counts = np.unique(v, return_counts=True)
    total_count = np.sum(counts)
    
    # for i in range(len(unique_values)): 
    p = counts / total_count
    
    return unique_values, counts, p

U, C, P = probability(show_V)

U_max, U_min = int(max(U)), int(min(U))
C_max, C_min = int(max(C)), int(min(C))

def cdf(c):
    tempC = np.zeros_like(C)
    tempC[0] = c[0]
    for i in range(1, len(c)): 
        tempC[i] = c[i] + tempC[i-1]
    return tempC
CDF = cdf(C)        # 累計出現次數

def h(CDF):
    h = np.round((CDF - min(CDF))/(max(CDF)-min(CDF)) * (255))
    return h
newU = h(CDF)               # 新單一灰階值

# 新灰階值
def eq(show_V, U, newU):
    eq_V = np.zeros_like(V)
    mapping = {u: new_u for u, new_u in zip(U, newU)}
    # 遍歷 show_V 中的每個像素值
    for i in range(show_V.shape[0]):
        for j in range(show_V.shape[1]):
            # 獲取當前像素值
            pixel_value = show_V[i, j]
            eq_V[i, j] = mapping.get(pixel_value, pixel_value)
    return eq_V
eq_V = eq(show_V, U, newU)    
eq_U, eq_C, eq_P = probability(eq_V)

def to_RGB(h, s, v):
    
    hi = np.floor(h / 60) 
    f = h / 60 - hi
    p = v * (1 - s)
    q = v * ( 1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.zeros_like(H)
    g = np.zeros_like(H)
    b = np.zeros_like(H)
    
    r[hi == 0], g[hi == 0], b[hi == 0] = v[hi == 0], t[hi == 0], p[hi == 0]
    r[hi == 1], g[hi == 1], b[hi == 1] = q[hi == 1], v[hi == 1], p[hi == 1]
    r[hi == 2], g[hi == 2], b[hi == 2] = p[hi == 2], v[hi == 2], t[hi == 2]
    r[hi == 3], g[hi == 3], b[hi == 3] = p[hi == 3], q[hi == 3], v[hi == 3]
    r[hi == 4], g[hi == 4], b[hi == 4] = t[hi == 4], p[hi == 4], v[hi == 4]
    r[hi == 5], g[hi == 5], b[hi == 5] = v[hi == 5], p[hi == 5], q[hi == 5]
    
    return r, g, b
eq_r, eq_g, eq_b = to_RGB(H, S, eq_V/255)
eq_image = np.zeros_like(image)
eq_image[:, :, 0] = eq_r  # 紅色通道
eq_image[:, :, 1] = eq_g  # 綠色通道
eq_image[:, :, 2] = eq_b  # 藍色通道

gray_image = (show_V).astype(np.uint8)

U = np.append(U, np.arange(U_max + 1, 256))
P = np.append(P, np.zeros(255 - U_max))
eq_U = np.append(eq_U, np.arange(int(max(eq_U)) + 1, 256))
eq_P = np.append(eq_P, np.zeros(255 - int(max(eq_U))))

x, y = np.indices(R.shape)

#-------------------------------------------------------------
# 顯示 S (飽和度) 與 H (色相) 的關係
hist, xedges, yedges = np.histogram2d(show_H.flatten(), show_S.flatten(), bins=(180, 256))
hist[hist > 25000] = 25000  # 設置頻率上限為 25000

plt.figure(figsize=(8, 6))
plt.imshow(hist.T, extent=[0, 360, 0, 256], origin='lower', cmap='hot')
plt.colorbar(label='Frequency')
plt.xlabel('Hue (H)')
plt.ylabel('Saturation (S)')
plt.title('2D Histogram of H and S channels')
plt.show()
#-------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.title('orinigonal image')
plt.imshow(image)  # 使用灰度顯示圖片

plt.subplot(223)
plt.title('Equalized image')
plt.imshow(eq_image)
#-------------------------------------------------------------
# 繪製折線圖
plt.subplot(222)
plt.title('orinigonal PDF')
plt.plot(U, P, linestyle='-', color='steelblue', label='Original PDF')
plt.xlim([-10, 260])

plt.subplot(224)
plt.title('Equalized PDF')
plt.plot(eq_U, eq_P, linestyle='-', color='steelblue', label='Original PDF')
plt.xlim([-10, 260])

plt.show()