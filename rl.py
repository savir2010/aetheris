import os
import io
import re
import base64
import argparse
import sqlite3
import hashlib
import math
import time
from datetime import datetime, timedelta
from PIL import Image
from huggingface_hub import InferenceClient
from flask import Flask, request, jsonify
import threading
import json
import requests

# Config
try:
    from config import SCREENSHARE_IMAGE, DRAW_URL, FEEDBACK_PORT
except ImportError:
    _root = os.path.dirname(os.path.abspath(__file__))
    SCREENSHARE_IMAGE = os.environ.get("SCREENSHARE_IMAGE") or os.path.join(_root, "weave3", "server", "screenshare_latest.png")
    DRAW_URL = os.environ.get("DRAW_URL", "http://localhost:3000/draw")
    FEEDBACK_PORT = int(os.environ.get("FEEDBACK_PORT", "3001"))

HF_API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
DEFAULT_IMAGE = "screenshot.png"
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "1280"))  # max dimension before encode (speed)

# Simulated coordinate prediction (no Florence/SAM/MoLMo)
_x, _y = os.environ.get("SIMULATE_COORD_X"), os.environ.get("SIMULATE_COORD_Y")
HARDCODED_COORDS = (int(_x), int(_y)) if (_x is not None and _y is not None) else (735, 478)
SIMULATE_DELAY_SECS = float(os.environ.get("SIMULATE_COORD_DELAY_SECS", "1.0"))

# 3-step IAM flow: screenshot paths and hardcoded coords per step
_root = os.path.dirname(os.path.abspath(__file__))
THREE_STEP_IMAGE_PATHS = [
    os.environ.get("THREE_STEP_SS1") or os.path.join(_root, "ss1.png"),
    os.environ.get("THREE_STEP_SS2") or os.path.join(_root, "ss2.png"),
    os.environ.get("THREE_STEP_SS3") or os.path.join(_root, "ss3.png"),
]
THREE_STEP_COORDINATES = [(630, 700),(890, 560),(800, 330)]

# Optional: last prediction (for minimal /click if used later)
last_prediction = None

# Flask app for feedback
app = Flask(__name__)

class ClickDatabase:
    """Store and analyze click feedback for continuous improvement"""
    
    def __init__(self, db_path="click_feedback.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        
    def _init_tables(self):
        """Initialize database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                button_label TEXT,
                pred_x INTEGER,
                pred_y INTEGER,
                actual_x INTEGER,
                actual_y INTEGER,
                error_pixels REAL,
                image_hash TEXT,
                screen_width INTEGER,
                screen_height INTEGER,
                qwen_response TEXT,
                florence_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        # Migration: rename molmo_response -> florence_response if old column exists (SQLite 3.25+)
        try:
            cursor = self.conn.execute("PRAGMA table_info(clicks)")
            columns = [row[1] for row in cursor.fetchall()]
            if "molmo_response" in columns and "florence_response" not in columns:
                self.conn.execute("ALTER TABLE clicks RENAME COLUMN molmo_response TO florence_response")
                self.conn.commit()
        except Exception:
            pass
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT,
                value REAL,
                window_size INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_click(self, button_label, pred, actual, img_hash, screen_size, qwen_resp, florence_resp):
        """Record a click feedback event"""
        error = math.sqrt((actual[0] - pred[0])**2 + (actual[1] - pred[1])**2)
        cursor = self.conn.execute("""
            INSERT INTO clicks 
            (button_label, pred_x, pred_y, actual_x, actual_y, error_pixels,
             image_hash, screen_width, screen_height, qwen_response, florence_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (button_label, pred[0], pred[1], actual[0], actual[1], error,
              img_hash, screen_size[0], screen_size[1], qwen_resp, florence_resp))
        
        self.conn.commit()
        
        # Update metrics
        self._update_metrics()
        
        return error, cursor.lastrowid
    
    def get_global_offset(self, n=20):
        """Get average offset from last n clicks (global correction)"""
        cursor = self.conn.execute("""
            SELECT actual_x - pred_x, actual_y - pred_y 
            FROM clicks 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (n,))
        
        offsets = cursor.fetchall()
        if not offsets:
            return 0, 0
        
        avg_x = sum(o[0] for o in offsets) / len(offsets)
        avg_y = sum(o[1] for o in offsets) / len(offsets)
        return avg_x, avg_y
    
    def get_label_specific_offset(self, button_label, n=10):
        """Get offset for this specific button type (more accurate)"""
        cursor = self.conn.execute("""
            SELECT actual_x - pred_x, actual_y - pred_y 
            FROM clicks 
            WHERE button_label = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (button_label, n))
        
        offsets = cursor.fetchall()
        if len(offsets) >= 3:  # Need at least 3 examples
            avg_x = sum(o[0] for o in offsets) / len(offsets)
            avg_y = sum(o[1] for o in offsets) / len(offsets)
            return avg_x, avg_y
        
        # Fall back to global offset
        return self.get_global_offset(n)
    
    def get_best_examples(self, n=5):
        """Get most accurate predictions for few-shot learning"""
        cursor = self.conn.execute("""
            SELECT button_label, actual_x, actual_y, error_pixels
            FROM clicks
            ORDER BY error_pixels ASC
            LIMIT ?
        """, (n,))
        
        return cursor.fetchall()
    
    def _update_metrics(self):
        """Calculate and store performance metrics"""
        windows = [10, 50, 100]
        
        for window in windows:
            cursor = self.conn.execute("""
                SELECT AVG(error_pixels)
                FROM (SELECT error_pixels FROM clicks ORDER BY timestamp DESC LIMIT ?)
            """, (window,))
            
            result = cursor.fetchone()
            if result and result[0] is not None:
                self.conn.execute("""
                    INSERT INTO metrics (metric_type, value, window_size)
                    VALUES (?, ?, ?)
                """, (f'avg_error', result[0], window))
        
        self.conn.commit()
    
    def get_stats(self):
        """Get comprehensive statistics"""
        # Total clicks
        cursor = self.conn.execute("SELECT COUNT(*) FROM clicks")
        total_clicks = cursor.fetchone()[0]
        
        if total_clicks == 0:
            return {
                'total_clicks': 0,
                'avg_error_all': 0,
                'avg_error_recent': 0,
                'improvement': 0,
                'status': 'No data yet'
            }
        
        # Overall average error
        cursor = self.conn.execute("SELECT AVG(error_pixels) FROM clicks")
        avg_error_all = cursor.fetchone()[0]
        
        # Recent average (last 20)
        cursor = self.conn.execute("""
            SELECT AVG(error_pixels)
            FROM (SELECT error_pixels FROM clicks ORDER BY timestamp DESC LIMIT 20)
        """)
        avg_error_recent = cursor.fetchone()[0] or avg_error_all
        
        # First 20 clicks average (baseline)
        cursor = self.conn.execute("""
            SELECT AVG(error_pixels)
            FROM (SELECT error_pixels FROM clicks ORDER BY timestamp ASC LIMIT 20)
        """)
        baseline = cursor.fetchone()[0] or avg_error_all
        
        # Calculate improvement
        improvement = ((baseline - avg_error_recent) / baseline * 100) if baseline > 0 else 0
        
        return {
            'total_clicks': total_clicks,
            'avg_error_all': round(avg_error_all, 2),
            'avg_error_recent': round(avg_error_recent, 2),
            'baseline': round(baseline, 2),
            'improvement': round(improvement, 2),
            'status': 'Learning' if total_clicks < 20 else 'Active'
        }
    
    def get_improvement_timeline(self, window_size=10):
        """Get error over time for visualization"""
        cursor = self.conn.execute("""
            SELECT id, error_pixels, timestamp
            FROM clicks
            ORDER BY timestamp ASC
        """)
        
        clicks = cursor.fetchall()
        if len(clicks) < window_size:
            return []
        
        timeline = []
        for i in range(len(clicks) - window_size + 1):
            window = clicks[i:i+window_size]
            avg_error = sum(c[1] for c in window) / window_size
            timeline.append({
                'click_num': i + window_size,
                'avg_error': round(avg_error, 2),
                'timestamp': window[-1][2]
            })
        
        return timeline


class ImprovedClickPredictor:
    """Load once, Qwen for step text; coordinates are hardcoded + time.sleep simulated (no Florence/SAM/MoLMo)."""

    def __init__(self, db=None):
        self.db = db
        self._qwen_client = InferenceClient(api_key=HF_API_KEY)

    def load_image(self, path: str, max_size: int = MAX_IMAGE_SIZE):
        """Load image, optionally resize for speed, encode once."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max_size and (w > max_size or h > max_size):
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_hash = hashlib.md5(img_bytes).hexdigest()
        return img, w, h, f"data:image/png;base64,{b64}", img_hash
    
    def step1_qwen(self, image_url: str, prompt: str) -> str:
        """Get first step from Qwen"""
        response = self._qwen_client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                                "Only give me the first step. Include the exact text label of the button or link to click. "
                                "Do not give coordinates.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            max_tokens=300,
        )
        return response.choices[0].message.content or ""
    
    def extract_button_label(self, qwen_text: str) -> str:
        """Extract button label from Qwen response"""
        text = qwen_text.strip()
        
        # Prefer quoted strings
        quoted = re.findall(r'"([^"]+)"', text) or re.findall(r"'([^']+)'", text)
        if quoted:
            for s in quoted:
                if 2 <= len(s) <= 50 and not s.endswith("."):
                    return s
            return quoted[0]
        
        # Pattern matching
        patterns = [
            r"(?:button|link)\s+(?:that\s+says?|labeled?)\s+[\"']?([^\"'.]+)[\"']?",
            r"click\s+on\s+[\"']?([^\"'.]+)[\"']?"
        ]
        
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return m.group(1).strip()
        
        # Fallback
        first_line = text.split("\n")[0].strip()
        if len(first_line) <= 60:
            return first_line
        return "IAM"

    def _simulate_coords(self) -> tuple[int, int]:
        """Simulate coordinate prediction: sleep then return hardcoded coords (no Florence/SAM/MoLMo)."""
        time.sleep(SIMULATE_DELAY_SECS)
        return HARDCODED_COORDS

    def predict(self, image_path: str, prompt=None, button_label=None):
        """Load once; Qwen (if no button_label) for step text; simulated coords (hardcoded + sleep) then POST to overlay."""
        img, w, h, image_url, img_hash = self.load_image(image_path)
        if button_label is None:
            if prompt is None:
                prompt = "Given this screenshot, what is the first button or link to click? "
            qwen_response = self.step1_qwen(image_url, prompt)
            button_label = self.extract_button_label(qwen_response)
        else:
            qwen_response = ""
        x_raw, y_raw = self._simulate_coords()
        florence_response = "simulated"
        final_coords = (x_raw, y_raw)
        result = {
            "button_label": button_label,
            "raw_coords": (x_raw, y_raw),
            "final_coords": final_coords,
            "screen_size": (w, h),
            "img_hash": img_hash,
            "qwen_response": qwen_response,
            "florence_response": florence_response,
        }
        global last_prediction
        x_draw, y_draw = final_coords
        try:
            r = requests.post(DRAW_URL, json={"x": x_draw, "y": y_draw}, timeout=2)
            if not r.ok:
                result["draw_status"] = r.status_code
        except requests.RequestException as e:
            result["draw_error"] = str(e)
        last_prediction = {
            "button_label": button_label,
            "pred_x": x_draw,
            "pred_y": y_draw,
            "img_hash": img_hash,
            "screen_size": (w, h),
            "qwen_response": qwen_response,
            "florence_response": florence_response,
        }
        return result


# Global predictor instance (db optional, not used in predict hot path)
db = ClickDatabase()
predictor = ImprovedClickPredictor(db)


@app.route("/qwen_step", methods=["POST"])
def qwen_step_route():
    """Run Qwen only: screenshot + user message → step_text and button_label (for bot to speak then simulated coords + draw)."""
    data = request.json or {}
    prompt = data.get("prompt") or "What is the first button or link the user should click on this screen? "
    image_path = data.get("image_path") or SCREENSHARE_IMAGE
    if not os.path.isfile(image_path):
        return jsonify({"error": f"File not found: {image_path}"}), 404
    try:
        _, _, _, image_url, _ = predictor.load_image(image_path)
        qwen_response = predictor.step1_qwen(image_url, prompt)
        button_label = predictor.extract_button_label(qwen_response)
        step_text = qwen_response.strip() or f"Click the {button_label} button"
        if not step_text.lower().startswith("click"):
            step_text = f"Click the {step_text}"
        return jsonify({"step_text": step_text, "button_label": button_label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/three_step_qwen", methods=["POST"])
def three_step_qwen_route():
    """Run Qwen for one step of the 3-step IAM flow: conversational prompt, return step_text + button_label."""
    data = request.json or {}
    step_number = data.get("step_number")
    user_transcript = (data.get("user_transcript") or "").strip()
    if step_number not in (1, 2, 3):
        return jsonify({"error": "step_number must be 1, 2, or 3"}), 400
    image_path = THREE_STEP_IMAGE_PATHS[step_number - 1]
    if not os.path.isfile(image_path):
        return jsonify({"error": f"File not found: {image_path}"}), 404
    prompt = (
        f"The user said: {user_transcript or 'they want to edit their role / add someone to IAM'}. "
        f"For step {step_number} of helping them edit their role or add someone to IAM, "
        "what button or link should they click on this screen? "
        "Reply in one short, friendly sentence you would say out loud (e.g. 'You need to click the IAM & Admin button.'). "
        "Do not give coordinates."
    )
    try:
        _, _, _, image_url, _ = predictor.load_image(image_path)
        qwen_response = predictor.step1_qwen(image_url, prompt)
        button_label = predictor.extract_button_label(qwen_response)
        step_text = qwen_response.strip() or f"Click the {button_label} button"
        return jsonify({"step_text": step_text, "button_label": button_label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/draw_point", methods=["POST"])
def draw_point_route():
    """Forward x,y to DRAW_URL (overlay). Used by 3-step flow to draw at hardcoded coords."""
    data = request.json or {}
    x = data.get("x")
    y = data.get("y")
    if x is None or y is None:
        return jsonify({"error": "Missing x or y"}), 400
    try:
        r = requests.post(DRAW_URL, json={"x": int(x), "y": int(y)}, timeout=5)
        return jsonify({"ok": r.ok}), r.status_code if not r.ok else 200
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """Prompt → Qwen (if needed) → simulated coords + draw; or button_label only → simulated coords + draw."""
    data = request.json or {}
    prompt = data.get("prompt")
    button_label = data.get("button_label")
    image_path = data.get("image_path") or SCREENSHARE_IMAGE
    if not os.path.isfile(image_path):
        return jsonify({"error": f"File not found: {image_path}"}), 404
    try:
        if button_label:
            result = predictor.predict(image_path, button_label=button_label)
        else:
            result = predictor.predict(
                image_path, prompt=prompt or "What button on this screen should I click next? "
            )
        if result is None:
            return jsonify({"error": "Could not get coordinates"}), 422
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


@app.route('/click', methods=['POST'])
def click_feedback():
    """Receive actual click coordinates (e.g. from Electron overlay after human clicks)."""
    global last_prediction
    data = request.json or {}

    actual_x = data.get('x')
    actual_y = data.get('y')
    pred_x = data.get('predicted_x')
    pred_y = data.get('predicted_y')
    button_label = data.get('button_label')
    img_hash = data.get('img_hash', '')
    screen_size = (data.get('screen_width', 1470), data.get('screen_height', 956))
    qwen_resp = data.get('qwen_response', '')
    florence_resp = data.get('florence_response', '')

    # If no pred in body, use last prediction (from draw sent to localhost:3000)
    if pred_x is None or pred_y is None:
        if last_prediction:
            pred_x = last_prediction['pred_x']
            pred_y = last_prediction['pred_y']
            button_label = button_label or last_prediction['button_label']
            img_hash = img_hash or last_prediction['img_hash']
            screen_size = last_prediction['screen_size']
            qwen_resp = qwen_resp or last_prediction['qwen_response']
            florence_resp = florence_resp or last_prediction['florence_response']
        else:
            return jsonify({'error': 'Missing coordinates: need x,y and either predicted_x/predicted_y or a prior /draw prediction'}), 400
    button_label = button_label or 'Unknown'

    if actual_x is None or actual_y is None:
        return jsonify({'error': 'Missing coordinates (x, y)'}), 400

    # Record feedback
    error, click_id = db.add_click(
        button_label=button_label,
        pred=(pred_x, pred_y),
        actual=(actual_x, actual_y),
        img_hash=img_hash,
        screen_size=screen_size,
        qwen_resp=qwen_resp,
        florence_resp=florence_resp
    )
    
    # Get updated stats
    stats = db.get_stats()
    
    print(f"\n{'='*60}")
    print(f"✓ FEEDBACK #{click_id} RECEIVED")
    print(f"{'='*60}")
    print(f"Button: '{button_label}'")
    print(f"Predicted: ({pred_x}, {pred_y})")
    print(f"Actual:    ({actual_x}, {actual_y})")
    print(f"Error:     {error:.1f} pixels")
    print(f"\nStats: {stats['total_clicks']} clicks, "
          f"Recent error: {stats['avg_error_recent']}px, "
          f"Improvement: {stats['improvement']:+.1f}%")
    print(f"{'='*60}\n")
    
    return jsonify({
        'status': 'received',
        'click_id': click_id,
        'error': round(error, 2),
        'stats': stats
    }), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current performance statistics"""
    stats = db.get_stats()
    timeline = db.get_improvement_timeline(window_size=10)
    
    return jsonify({
        'stats': stats,
        'timeline': timeline
    }), 200


@app.route('/export_training_data', methods=['GET'])
def export_training_data():
    """Export collected data for fine-tuning"""
    cursor = db.conn.execute("""
        SELECT button_label, pred_x, pred_y, actual_x, actual_y, 
               error_pixels, screen_width, screen_height, timestamp
        FROM clicks
        ORDER BY timestamp DESC
    """)
    
    data = []
    for row in cursor.fetchall():
        data.append({
            'label': row[0],
            'predicted': [row[1], row[2]],
            'actual': [row[3], row[4]],
            'error': row[5],
            'screen_size': [row[6], row[7]],
            'timestamp': row[8]
        })
    
    return jsonify({
        'total_examples': len(data),
        'data': data
    }), 200


def run_feedback_server():
    """Run Flask server (blocking)."""
    app.run(host="0.0.0.0", port=FEEDBACK_PORT, debug=False, use_reloader=False)


def main(prompt=None, image_path=None):
    """Run prediction. CLI: one predict and exit. --server runs Flask only. Coords are hardcoded + simulated delay."""
    parser = argparse.ArgumentParser(description="Click prediction: Qwen for step text, simulated coords, draw.")
    parser.add_argument("image", nargs="?", default=None, help="Screenshot path (default: screenshare)")
    parser.add_argument("--prompt", default="What button on this screen should I click next? ", help="Task prompt for Qwen")
    parser.add_argument("--server", action="store_true", help="Run Flask only (POST /predict, GET /health)")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()

    prompt = prompt if prompt is not None else args.prompt
    if image_path is not None:
        pass
    elif prompt is not None:
        image_path = SCREENSHARE_IMAGE
    else:
        image_path = args.image if args.image else SCREENSHARE_IMAGE

    if args.stats:
        stats = db.get_stats()
        timeline = db.get_improvement_timeline(window_size=10)
        print("\n" + "=" * 60)
        print("PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Total clicks: {stats['total_clicks']}")
        print(f"Status: {stats['status']}")
        print(f"Overall avg error: {stats['avg_error_all']} pixels")
        print(f"Recent avg error (last 20): {stats['avg_error_recent']} pixels")
        print(f"Baseline (first 20): {stats['baseline']} pixels")
        print(f"Improvement: {stats['improvement']:+.1f}%")
        if timeline:
            print("\nImprovement Timeline (10-click windows):")
            for point in timeline[-10:]:
                print(f"  Click {point['click_num']}: {point['avg_error']}px")
        print("=" * 60)
        return

    if args.server:
        print("\n" + "=" * 60)
        print(f"Flask on port {FEEDBACK_PORT} — POST /predict, GET /health")
        print("=" * 60)
        run_feedback_server()
        return

    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        return
    result = predictor.predict(image_path, prompt=prompt)
    if result:
        print(f"Predicted: {result['button_label']} at {result['final_coords']}")
    else:
        print("Could not parse coordinates")


if __name__ == "__main__":
    main()
