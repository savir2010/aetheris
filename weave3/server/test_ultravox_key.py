"""Quick test: does ULTRAVOX_API_KEY work with Ultravox API? Run from server dir: uv run test_ultravox_key.py"""
import os
import sys

# Load .env from this directory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

api_key = os.getenv("ULTRAVOX_API_KEY")
if not api_key:
    print("ERROR: ULTRAVOX_API_KEY not set in .env")
    sys.exit(1)

print(f"Using API key (first 8 chars): {api_key[:8]}...")

import asyncio
import aiohttp

async def main():
    url = "https://api.ultravox.ai/api/calls"
    body = {
        "systemPrompt": "Hi",
        "maxDuration": "30s",
        "medium": {"serverWebSocket": {"inputSampleRate": 48000}},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers={"X-Api-Key": api_key},
            json=body,
        ) as resp:
            text = await resp.text()
            print(f"Status: {resp.status}")
            print(f"Response: {text[:500]}")
            if resp.status != 201:
                print("\n-> Ultravox API rejected the request. Check your key at https://app.ultravox.ai")
                sys.exit(1)
    print("-> API key works. Create-call succeeded.")

if __name__ == "__main__":
    asyncio.run(main())
