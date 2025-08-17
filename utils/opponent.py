
from __future__ import annotations
from typing import List, Dict

def scout_weak_spots(opponent_roster: List[dict]) -> Dict[str, dict]:
    def _pts(p): 
        if isinstance(p, dict):
            return float(p.get("points") or p.get("proj_points") or 0.0)
        return 0.0
    pos_scores: Dict[str, list] = {}
    for p in opponent_roster or []:
        for pos in (p.get("eligible_positions") or []):
            pos_scores.setdefault(pos, []).append(_pts(p))
    report = {}
    for pos, scores in pos_scores.items():
        scores = sorted(scores, reverse=True)
        depth = len(scores)
        top2 = scores[:2]
        avg_top2 = sum(top2)/max(1,len(top2))
        report[pos] = {"depth": depth, "avg_top2": avg_top2}
    weak = {pos:meta for pos, meta in report.items() if (meta["depth"] < 2 or meta["avg_top2"] < 8)}
    return {"pos_report": report, "weak": weak}

def recommend_blocks(weak: Dict[str, dict], free_agents: List[dict]) -> List[dict]:
    out = []
    weak_positions = set(weak.keys())
    for p in free_agents or []:
        if p.get("position") in weak_positions and float(p.get("points") or 0) >= 7.0:
            out.append({"name": p.get("name"), "pos": p.get("position"), "points": float(p.get("points") or 0)})
    return sorted(out, key=lambda x: x["points"], reverse=True)[:6]
