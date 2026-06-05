"""
list_gemini_models.py
======================
List the Gemini models the current GEMINI_API_KEY can use for text
generation, using the supported google-genai SDK. Helps pick a live
model name to replace the retired gemini-2.0-flash.

Run from PCIL_dev/:  python scripts/list_gemini_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PCIL_DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PCIL_DEV))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PCIL_DEV / ".env")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set")
    sys.exit(2)

from google import genai  # noqa: E402

client = genai.Client(api_key=api_key)

print("Models supporting generateContent:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(f"  {m.name}")
