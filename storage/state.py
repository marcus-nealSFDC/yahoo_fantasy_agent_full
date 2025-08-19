# storage/state.py
from __future__ import annotations
import json, os
from typing import Dict, Any

DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _path(league_id: str) -> str:
    return os.path.join(DATA_DIR, f"state_league_{league_id}.json")

def load_state(league_id: str) -> Dict[str, Any]:
    p = _path(league_id)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(league_id: str, state: Dict[str, Any]) -> None:
    p = _path(league_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
