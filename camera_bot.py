import cv2
import time
import numpy as np
import requests
import telebot
import os
import datetime
import sys

# ========== GHI LOG CHUẨN CHO NSSM ==========
LOG_DIR = "C:\\CNTTK30\\python\\CameraBot"
LOG_FILE = os.path.join(LOG_DIR, "log.txt")

# Đảm bảo thư mục tồn tại
os.makedirs(LOG_DIR, exist_ok=True)

# Ghi log cả ra file và hiển thị cho NSSM
sys.stdout = open(os.path.join(LOG_DIR, "service_output.log"), "a", encoding="utf-8")
sys.stderr = open(os.path.join(LOG_DIR, "service_error.log"), "a", encoding="utf-8")

def log(message):
    text = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(text, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ==================== CẤU HÌNH ====================
BOT_TOKEN = "7798299471:AAFI3etpoybCX6eVS1phUfaLvLQz5TDBIi0"
CHAT_ID = "5482073289"
RTSP_URL = "rtsp://aefk:Hien@1977@192.168.1.5:554/live/ch00_0"

MOTION_COOLDOWN_SEC = 20
RECORD_DURATION = 5  # số giây ghi video
MODEL_DIR = os.path.join(os.getcwd(), "mobilenet_ssd")

bot = telebot.TeleBot(BOT_TOKEN)

# ==================== HÀM TẢI MODEL ====================
def ensure_model_files():
    proto = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
    model = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
    if not os.path.exists(proto):
        raise FileNotFoundError(f"❌ Thiếu file: {proto}")
    if not os.path.exists(model):
        raise FileNotFoundError(f"❌ Thiếu file: {model}")
    return proto, model


def load_model():
    log("[✅] Đang tải model...")
    proto, model = ensure_model_files()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    log("[✅] Model đã sẵn sàng.")
    return net

# ==================== HÀM GỬI TELEGRAM ====================
def send_telegram_photo(image_path, message="🚨 Phát hiện người trước camera!"):
    try:
        with open(image_path, "rb") as img:
            bot.send_photo(CHAT_ID, img, caption=message)
        log("[📤] Đã gửi ảnh cảnh báo Telegram.")
    except Exception as e:
        log(f"[⚠️] Lỗi gửi ảnh Telegram: {e}")


def send_telegram_video(video_path, message="🎥 Video ghi lại chuyển động"):
    try:
        with open(video_path, "rb") as vid:
            bot.send_video(CHAT_ID, vid, caption=message)
        log("[📤] Đã gửi video cảnh báo Telegram.")
    except Exception as e:
        log(f"[⚠️] Lỗi gửi video Telegram: {e}")

# ==================== HÀM CHÍNH ====================
def detect_human():
    log("[🟢] CameraBot đang chạy ngầm... (bạn có thể đóng cửa sổ này)")
    log("[🚀] Đang khởi động giám sát camera...")

    try:
        net = load_model()
        cap = cv2.VideoCapture(RTSP_URL)
    except Exception as e:
        log(f"[❌] Không thể khởi động camera hoặc model: {e}")
        return

    if not cap.isOpened():
        log("[❌] Không kết nối được camera.")
        return

    last_alert_time = 0
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            log(f"[⚠️] Mất tín hiệu ({fail_count}/5)...")
            time.sleep(3)
            if fail_count > 5:
                log("[❌] Camera mất kết nối. Dừng chương trình.")
                break
            continue
        fail_count = 0

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            if confidence > 0.5 and idx == 15:  # 15 = người
                now = time.time()
                if now - last_alert_time > MOTION_COOLDOWN_SEC:
                    img_path = os.path.join(LOG_DIR, "alert.jpg")
                    cv2.imwrite(img_path, frame)
                    send_telegram_photo(img_path)

                    video_path = os.path.join(LOG_DIR, "alert.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 20
                    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

                    log("[🎥] Đang ghi video...")
                    start_time = time.time()
                    while time.time() - start_time < RECORD_DURATION:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            break
                        out.write(frame2)
                        time.sleep(1 / fps)
                    out.release()
                    log("[✅] Đã ghi xong video.")

                    send_telegram_video(video_path)

                    try:
                        os.remove(video_path)
                    except:
                        pass

                    last_alert_time = now
                    log("[🚨] Đã gửi cảnh báo Telegram.")
                break

        time.sleep(0.1)

    cap.release()
    log("[👋] Dừng hệ thống.")

# ==================== CHẠY ====================
if __name__ == "__main__":
    try:
        detect_human()
    except KeyboardInterrupt:
        log("👋 Dừng thủ công.")
    except Exception as e:
        log(f"[❌ Lỗi]: {e}")
