# storage/ledger.py
from __future__ import annotations
import csv, os, time
from typing import Dict, Any

DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "actions_log.csv")

def log_action(action_type: str, payload: Dict[str, Any]):
    header = ["ts","type","payload"]
    row = [int(time.time()), action_type, str(payload)]
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)
