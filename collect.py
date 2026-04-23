import cv2
import os
from datetime import datetime, time as dt_time
import time

# ------------------ НАСТРОЙКИ ------------------
CAMERA_URL = "rtsp://admin:Ancestral123@192.168.1.2:554/stream1"
SAVE_DIR = "photos"
FULL_SAVE_DIR = "photo-full"  # Папка для полных кадров

# Интервалы в минутах
FREQUENT_INTERVAL_MINUTES = 1440    # Для 17:00–18:00
RARE_INTERVAL_MINUTES = 1440       # Для остального времени

# Временные границы частой записи (включительно: [start, end))
FREQUENT_START = dt_time(17, 0)  # 17:00
FREQUENT_END = dt_time(18, 0)    # 18:00

ROI = (1440, 340, 1740, 480)
# ------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FULL_SAVE_DIR, exist_ok=True)  # Создаём папку для полных кадров

def capture_snapshot():
    """Сохраняет кадр с вырезанным ROI (без изменений оригинального функционала)"""
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Не удалось открыть поток")
        return False

    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Не удалось получить кадр")
        return False

    x1, y1, x2, y2 = ROI
    roi = frame[y1:y2, x1:x2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
    cv2.imwrite(filename, roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"✅ Сохранено: {filename}")
    return True

def capture_full_frame():
    """Сохраняет полный кадр без обрезки ROI в папку photo-full"""
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Не удалось открыть поток для полного кадра")
        return False

    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Не удалось получить полный кадр")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(FULL_SAVE_DIR, f"{timestamp}_full.jpg")
    cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"🌙 Сохранён полный кадр: {filename}")
    return True

def is_frequent_time():
    now = datetime.now().time()
    return FREQUENT_START <= now < FREQUENT_END

# ------------------------------------------------
# Инициализация: запоминаем дату последней записи полного кадра
last_full_capture_date = datetime.now().date()

print(f"Частая запись с {FREQUENT_START} до {FREQUENT_END}: каждые {FREQUENT_INTERVAL_MINUTES} мин")
print(f"Редкая запись: каждые {RARE_INTERVAL_MINUTES} мин")
print(f"ROI = {ROI}")
print(f"Полные кадры сохраняются в '{FULL_SAVE_DIR}' при смене даты (после полуночи)\n")

try:
    while True:
        # Основная съёмка ROI
        capture_snapshot()

        # Проверка наступления новой даты (после полуночи)
        current_date = datetime.now().date()
        if current_date != last_full_capture_date:
            capture_full_frame()  # Сохраняем полный кадр один раз в сутки
            last_full_capture_date = current_date

        # Определение интервала до следующей съёмки ROI
        if is_frequent_time():
            sleep_time = FREQUENT_INTERVAL_MINUTES * 60
        else:
            sleep_time = RARE_INTERVAL_MINUTES * 60

        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nОстановлено пользователем.")
