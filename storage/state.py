# storage/state.py
from __future__ import annotations
import json, os, tempfile
from pathlib import Path

STATE_DIR = Path(os.getenv("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

def _state_path(key: str) -> Path:
    safe = "".join(c for c in str(key) if c.isalnum() or c in "-_.")
    return STATE_DIR / f"{safe}.json"

def load_state(key: str) -> dict:
    p = _state_path(key)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # back up the bad file and start fresh
        try: p.replace(p.with_suffix(p.suffix + ".corrupt"))
        except Exception: pass
        return {}
    except Exception:
        return {}

def save_state(key: str, data: dict) -> None:
    p = _state_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data or {}, f, indent=2, sort_keys=True)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

