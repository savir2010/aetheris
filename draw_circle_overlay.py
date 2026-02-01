import os
import re
import time
import requests

DAILY_API_KEY = "989fcbf8b8284eb5f7dfebcb8db19f24a25cf0b1dcc01563c1e3c600178c1bd6"

url = "https://api.daily.co/v1/rooms"
# Use a unique name so creating again doesn't 400 (room names are unique per domain)
room_name = f"hackathon-room-{int(time.time())}"
data = {
    "name": room_name,
    "properties": {
        "enable_screenshare": True,
    },
}
headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}

response = requests.post(url, json=data, headers=headers)
if not response.ok:
    try:
        err = response.json()
        print("Daily API error:", err)
    except Exception:
        print("Response text:", response.text)
    response.raise_for_status()
room = response.json()
room_url = room["url"]
print("Room URL:", room_url)

# Update server .env so the Pipecat agent joins this meeting
server_env = os.path.join(os.path.dirname(__file__), "weave3", "server", ".env")
if os.path.isfile(server_env):
    with open(server_env, "r") as f:
        content = f.read()
    if re.search(r"^\s*DAILY_ROOM_URL\s*=", content, re.MULTILINE):
        content = re.sub(
            r"^(\s*DAILY_ROOM_URL\s*=\s*).*$",
            r"\g<1>" + room_url,
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        content = content.replace(
            "# Optional: leave empty to create a new room\nDAILY_ROOM_URL=",
            f"# Optional: leave empty to create a new room\nDAILY_ROOM_URL={room_url}",
            1,
        )
    with open(server_env, "w") as f:
        f.write(content)
    print("Updated weave3/server/.env with DAILY_ROOM_URL")
    print("Run your Pipecat agent (Daily = real meeting, not playground): cd weave3/server && uv run bot.py -t daily")
else:
    print("Server .env not found at", server_env, "- set DAILY_ROOM_URL manually and run: cd weave3/server && uv run bot.py")
