
# Win-Each-Week Addons

This folder contains drop-in utilities to extend your Yahoo Fantasy Agent with:
- **Daily Snapshot & Morning Briefs** (cron/GitHub Actions or Lambda)
- **Lineup Optimizer v1.5** (enrichment + constraint solver)
- **Waiver Engine v2** (delta vs weakest starter + FAAB curve)
- **Opponent Scouting** (weak-spot + block recommendations)

## Quick Start (Local)

1) Ensure your Yahoo OAuth token file exists (normally `oauth2.json` produced by your Streamlit auth).
2) From the project root, run:
```bash
python -m pip install yahoo-oauth yahoo-fantasy-api pandas requests boto3
python addons/scripts/daily_snapshot.py --oauth oauth2.json
```
Artifacts land in `data/daily/YYYY-MM-DD/` and are also logged to `data/logs/` (+ S3 if `AWS_REGION` & `LOG_S3_BUCKET` env vars are set).

## Streamlit Integration Snippets

### Add to your imports
```python
from utils.lineup_opt import apply_enrichment, optimize_lineup
from utils.waivers import rank_waivers, multi_claim_queue
from utils.opponent import scout_weak_spots, recommend_blocks
```

### Enrich Start/Sit tab
Replace your pool scoring with:
```python
signals = {}  # optionally load from CSV: name → dict(home, opp_def_rank, game_total, recent_usage, ecr_delta, weather_penalty)
pool = roster_df(sc, league_key)
pool = apply_enrichment(pool, signals)

starters, bench = optimize_lineup(
    pool_df=pool,
    slots=slots,  # same roster slots you already build
    exposure_caps={"RB": 3, "WR": 4, "TE": 2, "QB": 1, "DEF": 1}  # tweak per league
)
# render starters like before; bench is a DataFrame already sorted by enriched score
```

### Waivers v2 tab
```python
# get free agents with your existing L.free_agents
df_ranked = rank_waivers(cands, team_df, current_week=league_current_week(settings_full or {}), settings=settings_full, standings_info=None, aggression=float(prefs.get("aggression", 0.15)))
st.dataframe(df_ranked, use_container_width=True)
queue = multi_claim_queue(df_ranked, max_claims=5)
st.write("Queue:", queue)
# Submit via your existing execute_add_drop(...) loop; log each with log_event('waiver.approved', ...)
```

### Opponent Scouting
```python
meta = st.session_state.get("_last_snapshot_meta")
if meta:
    league_dir = Path(meta["dir"])
    opp = json.loads((league_dir / "opp_roster.json").read_text())
    fas = json.loads((league_dir / "free_agents.json").read_text())
    report = scout_weak_spots(opp)
    blocks = recommend_blocks(report["weak"], fas)
    st.write("Opponent weak spots:", report["weak"])
    st.write("Suggested block claims:", blocks)
```

## Cloud

- **GitHub Actions**: use `.github/workflows/daily_snapshot.yml`. Store your token JSON in the secret `YAHOO_OAUTH_JSON`.
- **AWS Lambda**: deploy `lambda/snapshot_handler.py` as a function. Provide env `OAUTH_FILE` (path to your oauth2.json included in the zip or fetched from Secrets Manager) and optional `OUTPUT_ROOT`. Add a CloudWatch Events rule at `rate(1 day)` ~07:00 CST.

## Notes

- Enrichment signals (def rank, totals, weather) are pluggable. Feed a CSV with columns: `name, home(0/1), opp_def_rank, game_total, recent_usage, ecr_delta, weather_penalty` and map into the `signals` dict by player `name`.
- The backtracking solver is dependency-free and supports exposure caps. For Superflex, include a `"SUPERFLEX"` slot and treat eligibility accordingly or map to FLEX + special rule.
- FAAB curve: tunable via `aggression` slider you already have in Streamlit.
