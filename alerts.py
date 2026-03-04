import os
import time
import json
import smtplib
from email.message import EmailMessage

try:
    import winsound
except Exception:
    winsound = None

try:
    import requests
except Exception:
    requests = None


class AlertManager:
    """Send alerts via console, sound, email, Telegram, or dashboard POST.

    Configuration is read from environment variables (optional):
      ALERT_SOUND=1
      EMAIL_SMTP, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_TO
      TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
      DASHBOARD_URL
    """

    def __init__(self, cooldown: float = 10.0):
        self.cooldown = cooldown
        self._last_sent = 0.0

        # read config
        self.sound_enabled = os.getenv("ALERT_SOUND", "1") != "0"

        self.email_cfg = {
            "smtp": os.getenv("EMAIL_SMTP"),
            "port": int(os.getenv("EMAIL_PORT", "0")) if os.getenv("EMAIL_PORT") else None,
            "user": os.getenv("EMAIL_USER"),
            "pass": os.getenv("EMAIL_PASS"),
            "to": os.getenv("EMAIL_TO"),
        }

        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

        self.dashboard_url = os.getenv("DASHBOARD_URL")

    # -------------------- low-level channels -----------------------
    def _console(self, msg: str):
        print(f"⚠ ALERT: {msg}")

    def _sound(self):
        if not self.sound_enabled or winsound is None:
            return
        try:
            winsound.Beep(1000, 500)
        except Exception:
            pass

    def _email(self, subject: str, body: str):
        cfg = self.email_cfg
        if not cfg["smtp"] or not cfg["to"] or not cfg["user"] or not cfg["pass"]:
            return
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = cfg["user"]
            msg["To"] = cfg["to"]

            port = cfg["port"] or 465
            if port == 465:
                with smtplib.SMTP_SSL(cfg["smtp"], port) as s:
                    s.login(cfg["user"], cfg["pass"])
                    s.send_message(msg)
            else:
                with smtplib.SMTP(cfg["smtp"], port) as s:
                    s.starttls()
                    s.login(cfg["user"], cfg["pass"])
                    s.send_message(msg)
        except Exception:
            pass

    def _telegram(self, text: str):
        if not self.telegram_token or not self.telegram_chat or requests is None:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            resp = requests.post(url, json={"chat_id": self.telegram_chat, "text": text})
            return resp.ok
        except Exception:
            return False

    def _dashboard(self, text: str):
        if not self.dashboard_url or requests is None:
            return
        try:
            requests.post(self.dashboard_url, json={"alert": text, "ts": time.time()})
        except Exception:
            pass

    # -------------------- public API -------------------------------
    def send_alerts(self, alerts):
        """Send alerts if cooldown passed. `alerts` is a list of strings."""
        if not alerts:
            return
        now = time.time()
        if now - self._last_sent < self.cooldown:
            return
        self._last_sent = now

        payload = "; ".join(alerts)

        # console + sound always (console is useful)
        self._console(payload)
        if self.sound_enabled:
            self._sound()

        # async / best-effort channels
        # Email
        try:
            self._email("Security Alert", payload)
        except Exception:
            pass

        # Telegram
        try:
            self._telegram(payload)
        except Exception:
            pass

        # Dashboard
        try:
            self._dashboard(payload)
        except Exception:
            pass
