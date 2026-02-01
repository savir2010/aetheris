"""Shared config for entry, rl, and bot. One source of truth for screenshare path and ports."""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Screenshare: bot writes here; predictor reads from here
SCREENSHARE_IMAGE = (
    os.environ.get("SCREENSHARE_IMAGE")
    or os.environ.get("SCREENSHARE_OUTPUT_PATH")
    or os.path.join(_REPO_ROOT, "weave3", "server", "screenshare_latest.png")
)

DRAW_URL = os.environ.get("DRAW_URL", "http://localhost:3000/draw")
FEEDBACK_PORT = int(os.environ.get("FEEDBACK_PORT", "3001"))

# GPT classification (bot only): user's OpenAI API key for action vs not-action classification.
# Set OPENAI_API_KEY or GPT_KEY in the environment. If unset, all turns are treated as non-action.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_KEY")
