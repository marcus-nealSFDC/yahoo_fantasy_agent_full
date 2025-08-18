# 🏈 Yahoo Fantasy Agent — Live

A full-featured Streamlit app + add-on utilities for dominating your Yahoo Fantasy Football leagues.  
Connects to Yahoo Fantasy Sports (OAuth), shows your leagues & rosters, optimizes lineups,  
executes waivers/FAAB bids, scouts opponents, and (optionally) syncs logs to AWS S3.

---

## ✨ Features
- **Yahoo OAuth** — Securely connect your Yahoo account and load all leagues.
- **Roster & Start/Sit** — See your roster with projected points, apply lineup optimizer.
- **Lineup Optimizer v1.5** — Enrich with opponent defense rank, Vegas totals, weather, usage.
- **Waiver Engine v2** — Ranks free agents vs your weakest starters, suggests FAAB curves.
- **Opponent Scouting** — Detect weak spots in opponents, recommend block moves.
- **Trade Valuation v2** — Normalize values, suggest two-for-one upgrades.
- **Daily Snapshot & Morning Brief** — JSON + Markdown per league, logged for history.
- **Cloud Ready** — Run in Streamlit Cloud (UI) or schedule background tasks in GitHub Actions / AWS Lambda.

---

## ⚙️ 1. Install
```bash
git clone <this repo>
cd yahoo_fantasy_agent_full
pip3 install -r requirements.txt
```

---

## 🔑 2. Configure Environment
Create a `.env` file in the project root (see `.env.example`):

```bash
DEV_MODE=0

# Yahoo OAuth App Keys
YAHOO_CLIENT_ID=xxxx
YAHOO_CLIENT_SECRET=xxxx
YAHOO_REDIRECT_URI=https://<your-app>.streamlit.app

# OpenAI (optional: for Explain rationale)
OPENAI_API_KEY=sk-xxxx

# AWS (optional: for S3 log sync)
AWS_REGION=us-east-1
LOG_S3_BUCKET=my-fantasy-logs
```

> ⚠️ Redirect URI must exactly match your Yahoo App setup.  
> For local dev, you can use `http://localhost:8501`.

---

## ▶️ 3. Run Locally
```bash
streamlit run streamlit_app.py
```
- Click **Connect Yahoo**, approve, and your leagues appear.
- Tabs: **Roster / Start-Sit / Waivers / Opponent / Trades / Logs**

---

## 📊 4. Add-ons (Win Each Week Roadmap)

### Daily Snapshot
Runs at ~7am daily (cron, GitHub Actions, or AWS Lambda):
```bash
python addons/scripts/daily_snapshot.py --oauth oauth2.json
```
Outputs:
- `data/daily/YYYY-MM-DD/summary.json` & `.md`
- Logs to `data/logs/`

### Lineup Optimizer v1.5
```python
from utils.lineup_opt import apply_enrichment, optimize_lineup

signals = {}  # or load from CSV
pool = apply_enrichment(roster_df, signals)
starters, bench = optimize_lineup(pool, slots, exposure_caps={...})
```

### Waiver Engine v2
```python
from utils.waivers import rank_waivers, multi_claim_queue

df_ranked = rank_waivers(cands, team_df, current_week, settings, standings)
queue = multi_claim_queue(df_ranked, max_claims=5)
```

### Opponent Scouting
```python
from utils.opponent import scout_weak_spots, recommend_blocks

report = scout_weak_spots(opp_roster)
blocks = recommend_blocks(report["weak"], free_agents)
```

Upload **signals.csv** in the Opponent tab:
```csv
player_id,ecr_delta,recent_usage,opp_def_rank,game_total,weather_penalty,injury_prob,note
8479,-2.0,0.95,8,48.5,0.00,0.00,"Top CB matchup"
32173,1.5,1.10,22,51.0,0.00,0.05,"Trending up"
```

---

## ☁️ 5. Cloud Deployment

### Streamlit Cloud
- Push repo to GitHub
- Create new Streamlit Cloud app
- Add secrets (`.env` keys) in Streamlit settings

### GitHub Actions (Snapshots)
- See `.github/workflows/daily_snapshot.yml`
- Add `YAHOO_OAUTH_JSON` as secret

### AWS Lambda
- Deploy `lambda/snapshot_handler.py`
- Provide env:
  - `OAUTH_FILE=/tmp/oauth2.json`
  - `LOG_S3_BUCKET`
- Attach `CloudWatch rule` at `rate(1 day)`.

---

## 📂 Project Structure
```
yahoo_fantasy_agent_full/
│ streamlit_app.py
│ requirements.txt
│ .env.example
│ README.md
│
├── utils/
│   ├── lineup_opt.py
│   ├── waivers.py
│   └── opponent.py
│
├── addons/
│   └── scripts/
│       └── daily_snapshot.py
│
├── data/
│   ├── daily/
│   └── logs/
```

---

## 📝 Notes
- Yahoo limits some endpoints until **post-draft**.
- Pre-draft, only projections CSV is supported.
- Post-draft, all roster/waivers/trades unlock automatically.
- If S3 not configured, logs remain local in `data/`.

---

## 🚀 Roadmap
- Injury/News ingestion (scrapers, APIs)
- Trade Analyzer v2
- Full Lambda/Cloud Run split for scale
- Slack/Discord notifications for alerts
