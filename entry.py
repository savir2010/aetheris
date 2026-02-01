"""Single entry: start Flask (POST /predict, GET /health), optional file watcher that triggers prediction."""
import os
import sys
import time
import threading

# Shared config
try:
    from config import SCREENSHARE_IMAGE, FEEDBACK_PORT
except ImportError:
    _root = os.path.dirname(os.path.abspath(__file__))
    SCREENSHARE_IMAGE = os.environ.get("SCREENSHARE_IMAGE") or os.path.join(_root, "weave3", "server", "screenshare_latest.png")
    FEEDBACK_PORT = int(os.environ.get("FEEDBACK_PORT", "3001"))

from rl import app, run_feedback_server

DEFAULT_PROMPT = os.environ.get("PREDICT_PROMPT", "What button on this screen should I click next? ")
WATCH_INTERVAL = float(os.environ.get("WATCH_INTERVAL", "1.0"))


def _run_flask():
    """Run Flask in this thread (blocking)."""
    app.run(host="0.0.0.0", port=FEEDBACK_PORT, debug=False, use_reloader=False)


def _post_predict(prompt: str = DEFAULT_PROMPT):
    import requests
    try:
        r = requests.post(
            f"http://127.0.0.1:{FEEDBACK_PORT}/predict",
            json={"prompt": prompt},
            timeout=60,
        )
        if r.ok:
            print("[entry] Prediction done:", r.json().get("final_coords"), r.json().get("button_label"))
        else:
            print("[entry] Predict failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("[entry] Predict request error:", e)


def main():
    # Start Flask in background
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()
    time.sleep(1.0)  # let Flask bind
    print(f"[entry] Flask running on port {FEEDBACK_PORT} — POST /predict, GET /health")
    print(f"[entry] File watcher: {SCREENSHARE_IMAGE} (trigger every {WATCH_INTERVAL}s)")
    print("Optional: run bot separately: cd weave3/server && uv run bot.py -t daily\n")

    last_mtime = None
    while True:
        time.sleep(WATCH_INTERVAL)
        try:
            if not os.path.isfile(SCREENSHARE_IMAGE):
                last_mtime = None
                continue
            mtime = os.path.getmtime(SCREENSHARE_IMAGE)
            if last_mtime is not None and mtime != last_mtime:
                print("[entry] Screenshot changed, triggering prediction...")
                _post_predict()
            last_mtime = mtime
        except Exception as e:
            print("[entry] Watcher error:", e)


if __name__ == "__main__":
    main()
