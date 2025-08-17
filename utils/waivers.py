
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd

def _pos_from_elig(elig_str: str) -> str:
    s = (elig_str or "").upper().split(",")
    for p in ["QB","RB","WR","TE","DEF","DST","K","PK"]:
        if p in s:
            return "DEF" if p in ("DEF","DST") else ("K" if p in ("K","PK") else p)
    return s[0] if s else "UNK"

def expected_delta(candidate: dict, team_df: pd.DataFrame) -> float:
    """Compare candidate to your weakest starter-ish player at that position."""
    pos = candidate.get("position") or _pos_from_elig(candidate.get("eligible_positions", ""))
    cand_pts = float(candidate.get("points") or 0.0)
    pool = team_df.copy()
    pool["pos"] = pool["eligible_positions"].apply(_pos_from_elig)
    # weakest of top N starters by simple heuristic
    starter_map = {"QB":1, "RB":2, "WR":2, "TE":1, "DEF":1, "K":1}
    N = starter_map.get(pos, 1)
    pool_pos = pool[pool["pos"] == pos].sort_values("points", ascending=False)
    if pool_pos.empty:
        baseline = 0.0
    else:
        baseline = float(pool_pos.head(N).tail(1)["points"].values[0]) if len(pool_pos) >= N else float(pool_pos.tail(1)["points"].values[0])
    return max(0.0, cand_pts - baseline)

def scarcity_factor(candidates: List[dict], pos: str, threshold: float = 8.0) -> float:
    avail = [c for c in candidates if (c.get("position") == pos and float(c.get("points") or 0) >= threshold)]
    if not avail: 
        return 1.5
    n = len(avail)
    return max(0.8, min(1.5, 1.5 - 0.05 * n))

def weeks_remaining(current_week: int, total_weeks: int = 17) -> int:
    return max(1, total_weeks - max(1, current_week))

def standing_factor(rank: int | None, teams: int | None) -> float:
    if not rank or not teams:
        return 1.0
    pct = (rank - 1) / max(1, teams - 1)
    return 0.85 + 0.30 * pct

def suggest_faab_bid(delta: float, scarcity: float, weeks_left: int, stand_factor: float, budget: int = 100, aggression: float = 0.15) -> int:
    scale = min(1.0, max(0.0, delta / 20.0))
    time_boost = 1.0 + 0.2 * (1.0 - min(1.0, weeks_left / 17.0))
    frac = aggression * scale * scarcity * time_boost * stand_factor
    return int(round(budget * frac))

def rank_waivers(candidates: List[dict], team_df: pd.DataFrame, current_week: int, settings: dict, standings_info: dict | None = None, aggression: float = 0.15) -> pd.DataFrame:
    rows = []
    faab_budget = settings.get("faab_budget") or 100
    total_weeks = int(settings.get("end_week") or 17)
    rank = (standings_info or {}).get("my_rank")
    teams = (standings_info or {}).get("num_teams")

    for c in candidates:
        d = expected_delta(c, team_df)
        pos = c.get("position") or _pos_from_elig(c.get("eligible_positions", ""))
        scarcity = scarcity_factor(candidates, pos)
        wleft = weeks_remaining(current_week, total_weeks)
        standf = standing_factor(rank, teams)
        bid = suggest_faab_bid(d, scarcity, wleft, standf, budget=faab_budget, aggression=aggression)
        rows.append({
            "name": c.get("name"), "pos": pos, "player_id": c.get("player_id"),
            "points": float(c.get("points") or 0), "delta_vs_weakest": d,
            "scarcity": scarcity, "weeks_left": wleft, "standing_factor": standf,
            "suggested_faab": bid, "score": d * scarcity
        })
    df = pd.DataFrame(rows).sort_values(["score","points"], ascending=[False, False])
    return df

def multi_claim_queue(df_ranked: pd.DataFrame, max_claims: int = 5) -> list[dict]:
    out = []
    for _, r in df_ranked.head(max_claims).iterrows():
        out.append({
            "add_player_id": str(r["player_id"]),
            "name": r["name"],
            "pos": r["pos"],
            "suggested_faab": int(r["suggested_faab"])
        })
    return out
