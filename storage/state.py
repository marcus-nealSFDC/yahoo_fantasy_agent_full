# storage/state.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Iterable

STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def _coerce_jsonable(o: Any) -> Any:
    # Dicts
    if isinstance(o, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in o.items()}
    # Lists / Tuples
    if isinstance(o, (list, tuple)):
        return [_coerce_jsonable(v) for v in o]
    # Sets -> sorted lists (stable)
    if isinstance(o, set):
        try:
            return sorted(_coerce_jsonable(v) for v in o)
        except Exception:
            return list(_coerce_jsonable(v) for v in o)
    # Objects with __dict__
    if hasattr(o, "__dict__"):
        return _coerce_jsonable(o.__dict__)
    # Basic types – last resort: stringify unknowns
    try:
        json.dumps(o)
        return o
    except Exception:
        return str(o)

def load_state(league_id: str) -> dict:
    p = STATE_DIR / f"{league_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # quarantine corrupted file so the app can continue
        try:
            p.rename(p.with_suffix(".corrupt.json"))
        except Exception:
            pass
        return {}

def save_state(league_id: str, data: dict | None) -> None:
    p = STATE_DIR / f"{league_id}.json"
    tmp = p.with_suffix(".tmp")
    payload = _coerce_jsonable(data or {})
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


