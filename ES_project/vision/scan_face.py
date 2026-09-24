import cv2
import time
from color import color_name, extract_dominant_hsv_color


#--------------------- 調整格子 ---------------------
cell_size = 300             # 每格大小
margin = 380                # 格子間距
grid_start_x = 400          # 格子區左上角 x 座標
grid_start_y = 150          # 格子區左上角 y 座標
#--------------------- 調整格子 ---------------------

def scan_face(hsv):
    face_colors = []

    for row in range(3):
        for col in range(3):
            x1 = grid_start_x + col * (cell_size + margin)
            y1 = grid_start_y + row * (cell_size + margin)
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            patch = hsv[y1:y2, x1:x2]
            h, s, v = extract_dominant_hsv_color(patch)
            cname = color_name(h, s, v)
            face_colors.append(cname)

    return face_colors  # 回傳九個顏色（長度=9）

def make_list(set):
    colors = [["" for _ in range(9)] for _ in range(6)]
    for face in range(6):
        path = f"photo_{set[face]}.jpg"
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(f"正在掃描第 {face+1} 面...")

        # 掃描一面
        face_result = scan_face(hsv)

        # 存進矩陣
        colors[face] = face_result

        print("結果：", face_result)
        print()

    return colors

# -------------------- 主程式 --------------------
if __name__ == "__main__":
    # 6 面，每面 9 個顏色
    set = ['u', 'd', 'f', 'b', 'l', 'r']
    colors = make_list(set)
    # 最後顯示所有顏色矩陣
    print("六面顏色矩陣：")
    for f in range(6):
        print(f"Face {set[f]}: {colors[f]}")

"""
# -------------------- 主程式 --------------------
if __name__ == "__main__":
    # 6 面，每面 9 個顏色
    colors = [["" for _ in range(9)] for _ in range(6)]

    # 打開樹莓派相機（一般用 0, 如果你用 USB 攝影機，可能是 1)
    cap = cv2.VideoCapture(0)


    for face in range(6):
        print(f"請把第 {face+1} 面對準鏡頭, 3 秒後拍照...")

        # 倒數計時
        for t in range(3, 0, -1):
            print(t, end=" ", flush=True)
            time.sleep(1)
        print("拍照！")

        # 拍攝一張
        ret, frame = cap.read()

        # 轉換成 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        face_result = scan_face(hsv)  # 你自己定義的函式
        colors[face] = face_result

        print("結果：", face_result)
        print()

        # 顯示拍攝的畫面（可選）
        cv2.imshow("Captured Face", frame)
        cv2.waitKey(1000)  # 顯示 1 秒

    cap.release()
    cv2.destroyAllWindows()

    # 最後顯示所有顏色矩陣
    print("六面顏色矩陣：")
    for f in range(6):
        print(f"Face {f+1}: {colors[f]}")

"""