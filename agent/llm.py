# agent/llm.py
from __future__ import annotations
from typing import Any, Dict, Optional
import json, logging
from openai import OpenAI

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def complete_json(system: str, user: str, schema_hint: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> Dict[str, Any]:
    """
    Calls Chat Completions and forces JSON output. Returns parsed dict.
    Requires OPENAI_API_KEY.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user}\n\nSTRICT_OUTPUT_JSON_SCHEMA:\n{schema_hint}\n\nReturn ONLY minified JSON."},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        logging.exception("Failed to parse model JSON; attempting salvage.")
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise
