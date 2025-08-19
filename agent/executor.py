# agent/executor.py
from __future__ import annotations
from typing import List, Dict, Any
from storage.ledger import log_action

def submit_waiver_queue(sc, league, team_key: str, waiver_plan: List[Dict[str,Any]]):
    for item in waiver_plan:
        # Replace with your real Yahoo calls (add/drop with FAAB if supported)
        log_action("WAIVER_SUBMIT_DRAFT", {"team_key": team_key, **item})

def set_lineup(sc, league, team_key: str, start_sit: Dict[str,Any]):
    for move in start_sit.get("moves", []):
        # Replace with your real Yahoo roster-position API call
        log_action("LINEUP_MOVE_DRAFT", {"team_key": team_key, **move})

def send_trade_offer(sc, league, offer: Dict[str,Any]):
    # Replace with your real Yahoo propose-trade call if available
    log_action("TRADE_OFFER_DRAFT", offer)
