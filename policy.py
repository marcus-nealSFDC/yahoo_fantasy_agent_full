# policy.py
from dataclasses import dataclass, field
from typing import Set, Dict

@dataclass
class AutopilotPolicy:
    autopilot_on: bool = True
    require_approval: bool = True              # Flip to False for full auto
    approval_threshold_epar: float = 1.5       # Auto if EPAR >= this
    max_faab_bid: int = 22                     # Per-player FAAB ceiling
    weekly_faab_cap: int = 18                  # Total weekly FAAB ceiling
    protect_players: Set[str] = field(default_factory=set)
    no_trade_list: Set[str] = field(default_factory=set)
    no_drop_list: Set[str] = field(default_factory=set)
    variance_style: str = "auto"               # "auto" | "floor" | "ceiling"
    block_opponent_when_epar_gt: float = 1.0
    playoff_weeks: Set[int] = field(default_factory=lambda: {14,15,16,17})
    tie_breaker: str = "upside"                # "median" | "upside" | "floor"

DEFAULT_POLICY = AutopilotPolicy()
