# agent/prompts.py
from __future__ import annotations
from typing import Any, Dict
from .schemas import SCHEMA_WEEKLY_PLAN, SCHEMA_START_SIT, SCHEMA_WAIVERS, SCHEMA_TRADES

ASSISTANT_SYSTEM = (
    "You are an elite Fantasy Football strategist and operations assistant. "
    "Optimize weekly win probability under the user's guardrails. "
    "Use only provided data; if required inputs are missing, return explicit gaps. "
    "Do not reveal chain-of-thought. Provide short bullet rationales only. "
    "Return ONLY valid minified JSON when asked."
)

REASONING_GUARDRAILS = (
    "- Cite usage/snap share trend, matchup difficulty, weather, injuries, bye.\n"
    "- Prefer EPAR when available; use 'n/a' if not.\n"
    "- If favorite by >5, prefer floor. If underdog by >5, prefer ceiling.\n"
    "- Enforce policy: FAAB caps, no-drop list, no-trade list, playoff emphasis.\n"
)

def weekly_plan_user_msg(week:int, league_rules: Dict[str, Any], team_key: str, record: str, opponent_name: str, playoff_weeks, policy: Dict[str, Any], roster: Any, free_agents: Any, schedule: Any, injuries: Any) -> str:
    return (
        "Context:\n"
        f"- Week: {week}\n"
        f"- League settings: {league_rules}\n"
        f"- My team key: {team_key}\n"
        f"- Record/seeding: {record}\n"
        f"- Opponent: {opponent_name}\n"
        f"- Playoff weeks: {playoff_weeks}\n"
        f"- Policy: {policy}\n\n"
        "Data:\n"
        f"- Roster: {roster}\n"
        f"- Free agents shortlist: {free_agents}\n"
        f"- Schedule & matchup: {schedule}\n"
        f"- Injuries/News: {injuries}\n\n"
        "Task:\n"
        "Produce the complete weekly plan to maximize win probability with guardrails. "
        "Return JSON exactly per schema below. Keep rationales concise (<=6 bullets). "
        "Use contingencies for questionable players. Honor FAAB caps and protection lists.\n\n"
        "SCHEMA:\n"
        f"{SCHEMA_WEEKLY_PLAN}\n\n"
        "Reasoning rules:\n"
        f"{REASONING_GUARDRAILS}\n"
        "Return ONLY JSON."
    )

def start_sit_user_msg(week:int, matchup_view: Dict[str, Any], projections_view: Any, injuries: Any, policy: Dict[str, Any], underdog_margin: float) -> str:
    return (
        f"Context: Week {week}. Underdog margin: {underdog_margin}.\n"
        f"Matchup: {matchup_view}\n"
        f"Projections: {projections_view}\n"
        f"Injuries/News: {injuries}\n"
        f"Policy: {policy}\n\n"
        "Task: Recommend start/sit moves with concise reasons and risk style matching favorite/underdog. "
        "Follow the schema exactly.\n\n"
        "SCHEMA:\n"
        f"{SCHEMA_START_SIT}\n\n"
        "Reasoning rules:\n"
        f"{REASONING_GUARDRAILS}\n"
        "Return ONLY JSON."
    )

def waivers_user_msg(week:int, needs_summary: Dict[str, Any], candidates_view: Any, policy: Dict[str, Any]) -> str:
    return (
        f"Context: Week {week}. Roster needs: {needs_summary}. Policy: {policy}.\n"
        f"Candidates (with EPAR where available): {candidates_view}\n\n"
        "Task: Build a ranked waiver/FAAB queue with at most 6 claims and short reasons. "
        "Respect weekly/per-player FAAB caps and no-drop list for drops. "
        "Use 'n/a' if EPAR missing.\n\n"
        "SCHEMA:\n"
        f"{SCHEMA_WAIVERS}\n\n"
        "Reasoning rules:\n"
        f"{REASONING_GUARDRAILS}\n"
        "Return ONLY JSON."
    )

def trades_user_msg(week:int, my_team_view: Any, other_teams_view: Any, needs_summary: Dict[str, Any], policy: Dict[str, Any]) -> str:
    return (
        f"Context: Week {week}. My needs: {needs_summary}. Policy: {policy}.\n"
        f"My team: {my_team_view}\n"
        f"Other teams: {other_teams_view}\n\n"
        "Task: Suggest up to 3 trade offers that improve my median EV with acceptable downside. "
        "Add a short message to the other manager. Respect no-trade list.\n\n"
        "SCHEMA:\n"
        f"{SCHEMA_TRADES}\n\n"
        "Reasoning rules:\n"
        f"{REASONING_GUARDRAILS}\n"
        "Return ONLY JSON."
    )
