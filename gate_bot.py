#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import requests
import time
import threading
import tempfile
from datetime import datetime
from tflite_runtime.interpreter import Interpreter


# === Настройки ===
RTSP_URL = "rtsp://admin:Ancestral123@192.168.1.2:554/stream1"
MODEL_PATH = "gate_model_v214.tflite"
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set")
NOTIFY_CHAT_ID = os.environ.get("TG_CHAT_ID")
if not TOKEN:
    raise RuntimeError("TG_CHAT_ID is not set")
CHECK_INTERVAL = 30
COMMAND_POLL_INTERVAL = 1.0
ROI = (1440, 340, 1740, 480)

# === Глобальное состояние ===
last_status = None
latest_full_frame = None
latest_frame_timestamp = 0
latest_frame_lock = threading.Lock()
last_update_id = None

try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Используем tflite-runtime")
except ImportError:
    print("❌ tflite-runtime не установлен")
    sys.exit(1)

def normalize_command(text):
    """Обрабатывает команды вида /photo@botname"""
    text = text.strip()
    if text.startswith('/'):
        return text.split('@')[0].lower()
    return text.lower()

def check_telegram_commands():
    global last_update_id, latest_full_frame, latest_frame_timestamp
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {'timeout': 5}
        if last_update_id is not None:
            params['offset'] = last_update_id + 1

        resp = requests.get(url, params=params, timeout=8)  # увеличен таймаут
        if resp.status_code != 200:
            return

        data = resp.json()
        for update in data.get('result', []):
            last_update_id = update['update_id']
            if 'message' in update:
                msg = update['message']
                chat_id = msg['chat']['id']
                text = msg.get('text', '')
                cmd = normalize_command(text)
                
                # Отладка: логируем ВСЕ входящие сообщения
                print(f"📥 Чат {chat_id}: '{text}' → команда '{cmd}'")
                
                if chat_id == NOTIFY_CHAT_ID:
                    continue
                
                if cmd == '/photo':
                    send_frame_to_user(chat_id, full=True)
                elif cmd in ('/gate', 'отправить фото ворот'):
                    send_frame_to_user(chat_id, full=False)
                    
    except requests.exceptions.ConnectionError:
        print("⚠️ Нет интернета для опроса команд Telegram")
    except Exception as e:
        print(f"⚠️ Ошибка опроса команд: {e}")

def send_frame_to_user(chat_id, full=True):
    global latest_full_frame, latest_frame_timestamp
    
    with latest_frame_lock:
        if latest_full_frame is None or latest_full_frame.size == 0:
            send_telegram_text(chat_id, "⏳ Камера не готова. Подождите 10 секунд.")
            return
        frame = latest_full_frame.copy()
        age = int(time.time() - latest_frame_timestamp)
    
    try:
        if full:
            # Уменьшаем размер полного кадра для надёжной отправки
            h, w = frame.shape[:2]
            scale = min(1.0, 1280 / max(w, h))  # Макс 1280px по большей стороне
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            caption = f"📸 Полный кадр ({age}с назад, {frame.shape[1]}x{frame.shape[0]})"
        else:
            frame = apply_roi(frame)
            caption = f"📸 ROI ворот ({age}с назад)"
        
        # Сохраняем с сильным сжатием для надёжности
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
            temp_path = tmp.name
        cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])  # 75% качество
        
        # Повторная отправка при обрыве связи
        for attempt in range(3):
            try:
                send_telegram_photo(chat_id, temp_path, caption)
                print(f"✅ Отправлено {'полное' if full else 'ROI'} фото ({os.path.getsize(temp_path)//1024} КБ) в чат {chat_id}")
                break
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    print(f"🔄 Попытка {attempt+2}/3 отправки фото после обрыва сети...")
                    time.sleep(1.5)
                else:
                    send_telegram_text(chat_id, "❌ Не удалось отправить фото: нет интернета")
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Ошибка отправки фото: {e}")
        send_telegram_text(chat_id, f"❌ Ошибка: {str(e)[:60]}")

def command_polling_thread():
    print(f"📡 Поток команд запущен (интервал {COMMAND_POLL_INTERVAL}с)")
    while True:
        check_telegram_commands()
        time.sleep(COMMAND_POLL_INTERVAL)

def send_telegram_text(chat_id, text):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=8)
    except:
        pass

def send_telegram_photo(chat_id, photo_path, caption):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(photo_path, 'rb') as f:
            requests.post(url, files={'photo': f}, data={'chat_id': chat_id, 'caption': caption}, timeout=20)
    except requests.exceptions.ConnectionError:
        raise  # пробрасываем для повторной попытки
    except:
        pass

def capture_frame():
    for attempt in range(3):
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.7)
    raise Exception("Не удалось получить кадр с камеры")

def apply_roi(frame):
    x1, y1, x2, y2 = ROI
    h, w = frame.shape[:2]
    if x2 > w or y2 > h:
        raise ValueError(f"ROI выходит за границы ({w}x{h})")
    return frame[y1:y2, x1:x2]

def predict_status(roi_img):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.resize(roi_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return ("open" if output > 0.5 else "closed"), (float(output) if output > 0.5 else float(1 - output))

def send_alert_to_group(status, confidence, photo_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    human_status = "открыты" if status == "open" else "закрыты"
    caption = f"🚪 Ворота: {human_status}\n🕒 Время: {now}\n🔍 Уверенность: {confidence:.0%}"
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(photo_path, 'rb') as f:
            requests.post(url, files={'photo': f}, data={'chat_id': NOTIFY_CHAT_ID, 'caption': caption}, timeout=15)
    except:
        pass

def set_bot_commands():
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/setMyCommands"
        commands = [
            {"command": "photo", "description": "📸 Полный кадр"},
            {"command": "gate", "description": "🚪 ROI ворот"}
        ]
        requests.post(url, json={"commands": commands}, timeout=10)
    except:
        pass

def main():
    global last_status, latest_full_frame, latest_frame_timestamp
    print("🚀 Бот запущен. Инициализация...")
    set_bot_commands()
    threading.Thread(target=command_polling_thread, daemon=True).start()
    time.sleep(3)

    while True:
        try:
            frame = capture_frame()
            with latest_frame_lock:
                latest_full_frame = frame.copy()
                latest_frame_timestamp = time.time()

            roi_img = apply_roi(frame)
            status, conf = predict_status(roi_img)
            human_status = "открыты" if status == "open" else "закрыты"
            print(f"🕒 [{datetime.now().strftime('%H:%M:%S')}] {human_status} ({conf:.1%})")

            if last_status is None:
                last_status = status
                print("ℹ️ Начальное состояние установлено")
            elif status != last_status and conf > 0.95:
                print(f"🚪❗ СОСТОЯНИЕ ИЗМЕНИЛОСЬ → {human_status}")
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
                    temp_path = tmp.name
                cv2.imwrite(temp_path, roi_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                send_alert_to_group(status, conf, temp_path)
                os.unlink(temp_path)
                last_status = status

        except Exception as e:
            print(f"❌ Ошибка цикла: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
