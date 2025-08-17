
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Light enrichment structure; caller can merge external signals
# signals = {player_id or name: {"home":1/0, "opp_def_rank": int, "game_total": float, "recent_usage": float, "ecr_delta": float, "weather_penalty": float}}

def enriched_score_row(row: pd.Series, sig: dict | None = None) -> float:
    base = float(row.get("points", 0) or 0)
    status = (row.get("status") or "").upper()
    penalty = {"IR": -100, "O": -50, "D": -25, "Q": -10}.get(status, 0)
    score = base + penalty + 10  # previous heuristic "plus 10"
    if sig:
        # Positive: higher game totals, home, usage, good ECR delta (beats ADP), weak opponent def rank
        score += 0.05 * float(sig.get("game_total", 0) or 0)           # +0.05 per total point
        score += 1.0 if sig.get("home") else 0.0                       # home bump
        score += 0.5 * float(sig.get("recent_usage", 0) or 0)          # +0.5 per 10% usage scaled (caller pre-scales 0..2)
        score += 0.2 * float(sig.get("ecr_delta", 0) or 0)             # +0.2 per tier delta
        opp_def = sig.get("opp_def_rank")
        if isinstance(opp_def, (int, float)) and opp_def:              # lower rank (1=best defense)
            score += (32 - float(opp_def)) * 0.2                       # easier opponent → higher score
        score += -1.0 * float(sig.get("weather_penalty", 0) or 0)      # subtract if poor weather
    return score

def apply_enrichment(pool_df: pd.DataFrame, signals: Dict[str, dict]) -> pd.DataFrame:
    def key_for(row):
        pid = str(row.get("player_id")) if row.get("player_id") is not None else ""
        nm  = str(row.get("name") or "").strip()
        return signals.get(pid) or signals.get(nm) or {}
    pool_df = pool_df.copy()
    pool_df["enriched_score"] = pool_df.apply(lambda r: enriched_score_row(r, key_for(r)), axis=1)
    return pool_df

def _eligible(row: pd.Series, slot: str) -> bool:
    slot = slot.upper()
    elig = (row.get("eligible_positions") or "").upper().split(",")
    if slot in ("W/R/T","WR/RB/TE","FLEX"): return any(p in elig for p in ("WR","RB","TE"))
    if slot in ("D","DEF","DST"): return "DEF" in elig
    if slot in ("K","PK"): return "K" in elig or "PK" in elig
    return slot in elig

def optimize_lineup(pool_df: pd.DataFrame, slots: List[str], exposure_caps: Dict[str, int] | None = None, score_col: str = "enriched_score") -> Tuple[List[Tuple[str, pd.Series]], pd.DataFrame]:
    """Backtracking solver with pruning; respects exposure caps by position."""
    df = pool_df.copy()
    if score_col not in df.columns:
        score_col = "score" if "score" in df.columns else "points"
        df[score_col] = df[score_col].fillna(0.0)

    # Pre-sort by score to help pruning
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    best_score, best_assign = -1e9, None
    used = set()
    caps_used = {}

    def pos_key(row):
        # pick a canonical single position for caps: first eligible in (QB,RB,WR,TE,DEF,K)
        order = ["QB","RB","WR","TE","DEF","DST","K","PK"]
        elig = (row.get("eligible_positions") or "").upper().split(",")
        for p in order:
            if p in elig:
                return "DEF" if p in ("DEF","DST") else ("K" if p in ("K","PK") else p)
        return elig[0] if elig else "UNK"

    # Precompute eligible lists to speed up
    eligible_lists = []
    for slot in slots:
        mask = df.apply(lambda r: _eligible(r, slot), axis=1)
        eligible_lists.append(df[mask].index.tolist())

    # Upper bounds for pruning (max possible if we pick top remaining for each slot)
    prefix_best = [0.0] * (len(slots)+1)
    for i in range(len(slots)-1, -1, -1):
        idxs = eligible_lists[i]
        prefix_best[i] = prefix_best[i+1] + (df.loc[idxs, score_col].max() if idxs else 0.0)

    assign: List[Tuple[str, int]] = []  # (slot, df_index)

    def backtrack(i: int, acc_score: float):
        nonlocal best_score, best_assign
        if i == len(slots):
            if acc_score > best_score:
                best_score = acc_score
                best_assign = assign.copy()
            return
        # pruning
        if acc_score + prefix_best[i] <= best_score:
            return
        slot = slots[i]
        for idx in eligible_lists[i]:
            if idx in used:
                continue
            row = df.loc[idx]
            pkey = pos_key(row)
            if exposure_caps:
                cap = exposure_caps.get(pkey)
                if cap is not None and caps_used.get(pkey, 0) >= cap:
                    continue
            used.add(idx)
            if exposure_caps:
                caps_used[pkey] = caps_used.get(pkey, 0) + 1
            assign.append((slot, idx))
            backtrack(i+1, acc_score + float(row[score_col]))
            assign.pop()
            if exposure_caps:
                caps_used[pkey] -= 1
            used.remove(idx)

    backtrack(0, 0.0)
    starters = []
    used_ids = set()
    if best_assign:
        for slot, idx in best_assign:
            row = df.loc[idx]
            starters.append((slot, row))
            used_ids.add(row.get("player_id"))
    bench = df[~df["player_id"].isin(used_ids)].sort_values(score_col, ascending=False)
    return starters, bench
