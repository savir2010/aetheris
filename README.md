# weave3

Voice bot (Aetheris V.O) that runs over Daily and guides users through on-screen steps. The bot uses Pipecat with Ultravox for realtime speech. A separate Flask app (`rl.py`) handles step text (Qwen) and coordinate prediction; the bot talks through steps and can trigger “draw” cues at fixed coordinates.

**What runs:**

- **weave3/server** — Pipecat bot. Connects to a Daily room, listens/speaks, and when the user asks for screen help it either does a single-step “click here” flow or a 3-step IAM flow (e.g. edit role in Google Cloud). No drawing is done in the bot; it calls the feedback service for step text and pacing.
- **weave3/client** — Web client to join the Daily room and (optionally) share screen.
- **Root** — `rl.py` is a Flask app (POST `/predict`, `/qwen_step`, `/three_step_qwen`, etc.). `entry.py` starts that Flask app and can watch a screenshot file to trigger predictions.

**Requirements:**

- Python 3.10+
- Node (for the client)
- Daily account and room URL/token for the bot
- Ultravox API key (voice)
- Optional: `OPENAI_API_KEY` or `GPT_KEY` for classifying user intent; `HF_TOKEN` or `HF_API_KEY` for Qwen in `rl.py`

---

## Setup

### Feedback API (Flask)

From repo root:

```bash
pip install -r requirements.txt
# Set HF_TOKEN or HF_API_KEY if you use Qwen endpoints
python entry.py
```

Flask listens on `FEEDBACK_PORT` (default 3001). Endpoints: `/predict`, `/qwen_step`, `/three_step_qwen`, `/draw_point`, `/health`.

### Bot (Pipecat)

```bash
cd weave3/server
uv sync
cp .env.example .env
# Edit .env: ULTRAVOX_API_KEY, and optionally OPENAI_API_KEY or GPT_KEY
uv run bot.py -t daily
```

Bot joins the Daily room; it expects the feedback API at `http://127.0.0.1:3001` unless you set `FEEDBACK_PORT` / `PREDICT_URL` etc. in env.

### Client

```bash
cd weave3/client
npm install
cp env.example .env.local
# Edit .env.local if your Pipecat/Daily endpoint is not localhost:7860
npm run dev
```

Open http://localhost:5173, create or join a Daily room, and connect. Share screen so the bot can use the screenshot path configured in the server.

---

## Env (main ones)

| Variable | Where | Purpose |
|----------|--------|---------|
| `ULTRAVOX_API_KEY` | server | Voice (required for bot) |
| `OPENAI_API_KEY` or `GPT_KEY` | server | Optional; classifies “action” vs not |
| `HF_TOKEN` or `HF_API_KEY` | root / rl.py | Qwen for step text |
| `FEEDBACK_PORT` | root, server | Flask port (default 3001) |
| `SCREENSHARE_OUTPUT_PATH` | server | Where the bot writes the latest screenshot (default `server/screenshare_latest.png`) |

---

## Repo layout

```
weave3/
├── README.md           # This file
├── .gitignore
├── config.py           # Shared config (image path, ports)
├── entry.py            # Starts Flask (rl.app) and optional file watcher
├── rl.py               # Flask app: predict, qwen_step, three_step_qwen, draw_point, etc.
├── requirements.txt    # For Flask / rl.py
├── weave3/
│   ├── server/         # Pipecat bot
│   │   ├── bot.py
│   │   ├── pyproject.toml
│   │   └── .env.example
│   └── client/         # Web UI for Daily
│       ├── src/
│       └── package.json
└── electron/           # Optional Electron app
```

---

## Docs

- [Pipecat](https://docs.pipecat.ai/)
- [Daily](https://www.daily.co/)
