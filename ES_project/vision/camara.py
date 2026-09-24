from picamera2 import Picamera2
import time

def take_photo(face):
    picam2 = Picamera2()
    
    print(f"準備開始拍攝{face}面")
    time.sleep(1)
    for i in range(5):
        print(f"倒數{5-i}秒...")
        time.sleep(1)
    # 建立拍照設定
    config = picam2.create_still_configuration()
    picam2.configure(config)

    # 啟動相機
    picam2.start()
    time.sleep(2)  # 等待曝光穩定

    # 拍照並儲存檔案
    filename = f"data/captures/photo_{face}.jpg"
    picam2.capture_file(filename)
    print(f"拍照完成：{filename}")
    
    # 關閉相機
    picam2.stop()
    picam2.close()


if __name__ == "__main__":
    face = ['u', 'd', 'f', 'b', 'l', 'r']
    for i in face :
        take_photo(i)
