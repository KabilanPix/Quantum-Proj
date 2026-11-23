import time
import requests
from typing import Dict, Any, Optional
import config
import db

# In-memory cooldown tracking: {machine_id: last_alert_time}
_LAST_ALERT_TIMES: Dict[str, float] = {}

def send_telegram_alert(message: str) -> bool:
    """Sends Telegram message if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."""
    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"🚨 [Smart Factory Alert]\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, json=payload, timeout=4.0)
        return r.status_code == 200
    except Exception as e:
        print(f"[Alerts] Telegram send failed: {e}")
        return False

def check_and_trigger_alerts(machine_id: str, risk: float, top_factor: str = "") -> Optional[Dict[str, Any]]:
    """
    Checks if machine risk crosses RISK_HIGH and triggers alert respecting 5-minute cooldown.
    Writes to SQLite alerts table and optionally sends Telegram notification.
    """
    now = time.time()

    if risk < config.RISK_HIGH:
        return None

    last_time = _LAST_ALERT_TIMES.get(machine_id, 0.0)
    if (now - last_time) < config.ALERT_COOLDOWN_SEC:
        # Cooldown active, skip alert triggering
        return None

    _LAST_ALERT_TIMES[machine_id] = now
    level = "CRITICAL" if risk >= 0.85 else "HIGH"
    message = f"Machine {machine_id} risk escalated to {risk:.2f} ({level}). Top contributing factor: {top_factor}."

    print(f"\n[ALERT TRIGGERED] >>> {message} <<<")
    sent_telegram = 1 if send_telegram_alert(message) else 0

    try:
        alert_id = db.insert_alert(machine_id, now, level, risk, message, sent_telegram=sent_telegram)
    except Exception as e:
        print(f"[Alerts] DB insert failed: {e}")
        alert_id = -1

    return {
        "id": alert_id,
        "machine_id": machine_id,
        "timestamp": now,
        "level": level,
        "risk": risk,
        "message": message,
        "sent_telegram": sent_telegram
    }
