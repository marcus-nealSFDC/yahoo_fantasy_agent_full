# agent/reasoning.py
from __future__ import annotations
from typing import Any, Dict
from .llm import complete_json
from .prompts import ASSISTANT_SYSTEM, weekly_plan_user_msg, start_sit_user_msg, waivers_user_msg, trades_user_msg
from .schemas import SCHEMA_WEEKLY_PLAN, SCHEMA_START_SIT, SCHEMA_WAIVERS, SCHEMA_TRADES

def plan_week(week:int, league_rules: Dict[str,Any], team_key: str, record: str, opponent_name: str, playoff_weeks, policy: Dict[str,Any], roster: Any, free_agents: Any, schedule: Any, injuries: Any, model: str = "gpt-4o-mini") -> Dict[str,Any]:
    user = weekly_plan_user_msg(week, league_rules, team_key, record, opponent_name, playoff_weeks, policy, roster, free_agents, schedule, injuries)
    return complete_json(ASSISTANT_SYSTEM, user, SCHEMA_WEEKLY_PLAN, model=model)

def plan_start_sit(week:int, matchup_view: Dict[str,Any], projections_view: Any, injuries: Any, policy: Dict[str,Any], underdog_margin: float, model: str = "gpt-4o-mini") -> Dict[str,Any]:
    user = start_sit_user_msg(week, matchup_view, projections_view, injuries, policy, underdog_margin)
    return complete_json(ASSISTANT_SYSTEM, user, SCHEMA_START_SIT, model=model)

def plan_waivers(week:int, needs_summary: Dict[str,Any], candidates_view: Any, policy: Dict[str,Any], model: str = "gpt-4o-mini") -> Dict[str,Any]:
    user = waivers_user_msg(week, needs_summary, candidates_view, policy)
    return complete_json(ASSISTANT_SYSTEM, user, SCHEMA_WAIVERS, model=model)

def plan_trades(week:int, my_team_view: Any, other_teams_view: Any, needs_summary: Dict[str,Any], policy: Dict[str,Any], model: str = "gpt-4o-mini") -> Dict[str,Any]:
    user = trades_user_msg(week, my_team_view, other_teams_view, needs_summary, policy)
    return complete_json(ASSISTANT_SYSTEM, user, SCHEMA_TRADES, model=model)
