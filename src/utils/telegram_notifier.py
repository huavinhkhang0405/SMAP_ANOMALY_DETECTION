import requests
import threading
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_TOKEN = os.getenv("BOT_API_TOKEN", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

def send_alert_sync(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": ADMIN_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown" 
    }
    try:
        print("Đang gọi API Telegram gửi cảnh báo...")
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            print("✅ Đã gửi cảnh báo Telegram thành công!")
        else:
            print(f"❌ Telegram từ chối gửi. Mã lỗi: {response.status_code} - Lời giải thích: {response.text}")
            
    except Exception as e:
        print(f"⚠️ Không thể kết nối tới máy chủ Telegram: {e}")

def notify_admin_async(filename: str, anomaly_count: int, max_residual: float):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "":
        print("⚠️ Cảnh báo: Chưa cấu hình TELEGRAM_TOKEN trong file .env")
        return

    print(f"🚨 Phát hiện {anomaly_count} lỗi tại file {filename}. Kích hoạt Telegram Bot...")

    msg = (
        "🚨 *CẢNH BÁO BẤT THƯỜNG VIỄN TRẮC* 🚨\n"
        "------------------------------------\n"
        f"📡 *Nguồn cấp:* `{filename}`\n"
        f"⚠️ *Số điểm vi phạm:* `{anomaly_count}`\n"
        f"📈 *Độ lệch cực đại:* `{max_residual:.4f}`\n"
        "🤖 *Mô hình nhận diện:* `LSTM + Isolation Forest`\n"
        "------------------------------------\n"
        "Tình trạng khẩn cấp. Yêu cầu kiểm tra mặt đất ngay lập tức!"
    )

    # Chạy ngầm trong một luồng (thread) riêng biệt
    thread = threading.Thread(target=send_alert_sync, args=(msg,))
    thread.start()