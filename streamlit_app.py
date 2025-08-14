
import os, json, time
from pathlib import Path
from urllib.parse import urlencode

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

# Optional OpenAI rationale (safe to skip if not installed/configured)
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI()
    OPENAI_OK = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_OK = False

# Load .env for local; on Streamlit Cloud, set these as Secrets instead
load_dotenv()

CID = os.getenv("YAHOO_CLIENT_ID")
CSEC = os.getenv("YAHOO_CLIENT_SECRET")
REDIRECT = os.getenv("YAHOO_REDIRECT_URI")
OAUTH_FILE = Path("oauth2.json")  # token cache (kept out of git via .gitignore)

st.set_page_config(page_title="Yahoo Fantasy Agent — Live", layout="wide")
st.title("🏈 Yahoo Fantasy Agent — Live")

# ---------------- OAuth helpers ----------------
def authorize_url():
    # Yahoo Fantasy scope: fspt-w (write) so we can add/drop later
    return "https://api.login.yahoo.com/oauth2/request_auth?" + urlencode({
        "client_id": CID,
        "redirect_uri": REDIRECT,
        "response_type": "code",
        "scope": "fspt-w",
        "language": "en-us",
    })

def exchange_code_for_tokens(code: str) -> dict:
    r = requests.post(
        "https://api.login.yahoo.com/oauth2/get_token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": CID,
            "client_secret": CSEC,
            "redirect_uri": REDIRECT,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    try:
        return r.json()
    except Exception:
        return {"_error": "non_json", "_text": r.text, "_status": r.status_code}

def save_tokens(payload: dict):
    OAUTH_FILE.write_text(json.dumps({
        "consumer_key": CID,
        "consumer_secret": CSEC,
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token"),
        "token_time": int(time.time()),
        "expires_in": payload.get("expires_in"),
        "xoauth_yahoo_guid": payload.get("xoauth_yahoo_guid"),
        "token_type": payload.get("token_type", "bearer"),
    }))

def have_tokens() -> bool:
    if not OAUTH_FILE.exists(): return False
    try:
        d = json.loads(OAUTH_FILE.read_text())
        return bool(d.get("access_token") and d.get("refresh_token"))
    except Exception:
        return False

def get_session() -> OAuth2:
    if not have_tokens():
        raise RuntimeError("No tokens yet")
    sc = OAuth2(None, None, from_file=str(OAUTH_FILE), browser_callback=False)
    if not sc.token_is_valid():
        sc.refresh_access_token()
    return sc

# ---------------- Data helpers ----------------
def resolve_team_key(sc, league_key: str):
    L = yfa.League(sc, league_key)
    for t in L.teams():
        if t.get("is_owned_by_current_login") == 1:
            return t["team_key"]
    teams = L.teams()
    return teams[0]["team_key"] if teams else None

def roster_df(sc, league_key: str) -> pd.DataFrame:
    L = yfa.League(sc, league_key)
    team_key = resolve_team_key(sc, league_key)
    if not team_key:
        return pd.DataFrame()
    T = yfa.Team(sc, team_key)
    rows = []
    for p in T.roster():
        rows.append({
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "status": p.get("status"),
            "eligible_positions": ",".join(p.get("eligible_positions", [])),
            "selected_position": p.get("selected_position"),
            "points": p.get("points") or p.get("proj_points") or 0,
        })
    return pd.DataFrame(rows)

def free_agents(L: "yfa.League", positions=("WR","RB","TE","QB"), limit=30):
    fa = []
    for pos in positions:
        try:
            fa += L.free_agents(pos)
        except Exception:
            pass
    seen, out = set(), []
    for p in fa:
        pid = p.get("player_id")
        if pid in seen: continue
        seen.add(pid)
        pts = p.get("points") or p.get("proj_points") or 0
        out.append({"player_id": pid, "name": p.get("name"), "position": p.get("position"), "points": pts})
    out.sort(key=lambda x: x["points"] or 0, reverse=True)
    return out[:limit]

def execute_add_drop(sc, league_key: str, add_pid: str, drop_pid: str|None=None, faab_bid: int|None=None):
    """Try team add/drop; fall back to league-level add; then try waiver methods if available."""
    L = yfa.League(sc, league_key)
    team_key = resolve_team_key(sc, league_key)
    if not team_key:
        return {"status": "error", "details": "Could not resolve team_key."}
    T = yfa.Team(sc, team_key)

    last_err = None

    # 1) Team-level (immediate FA add)
    try:
        if drop_pid:
            try:
                res_add = T.add_player(add_pid)
                res_drop = T.drop_player(drop_pid)
                return {"status": "ok", "details": {"team.add": res_add, "team.drop": res_drop}}
            except Exception:
                res_drop = T.drop_player(drop_pid)
                res_add = T.add_player(add_pid)
                return {"status": "ok", "details": {"team.drop": res_drop, "team.add": res_add}}
        else:
            res_add = T.add_player(add_pid)
            return {"status": "ok", "details": {"team.add": res_add}}
    except Exception as e1:
        last_err = f"team add/drop failed: {e1}"

    # 2) League-level add (if present in this lib version)
    try:
        if hasattr(L, "add_player"):
            if drop_pid:
                res = L.add_player(add_pid, team_key=team_key, drop_player_id=drop_pid)
            else:
                res = L.add_player(add_pid, team_key=team_key)
            return {"status": "ok", "details": {"league.add_player": res}}
    except Exception as e2:
        last_err = f"{last_err} | league.add_player failed: {e2}"

    # 3) Try waiver methods (name varies across versions)
    for meth in ("waiver_add", "add_player_waiver", "place_waiver"):
        if hasattr(L, meth):
            try:
                fn = getattr(L, meth)
                res = fn(add_pid, team_key, drop_player_id=drop_pid, bid=faab_bid)
                return {"status": "ok", "details": {meth: res}}
            except Exception as e3:
                last_err = f"{last_err} | {meth} failed: {e3}"

    return {"status": "error", "details": last_err or "No supported add/waiver method found."}

# ---------------- Sidebar: Live connect ----------------
with st.sidebar:
    st.header("Yahoo Connection")
    st.caption("Redirect in env: " + (REDIRECT or "<missing>"))
    env_ok = all([CID, CSEC, REDIRECT])
    st.write("Env:", "✅" if env_ok else "⚠️ missing")

    # read query params
    try:
        q = st.query_params       # Streamlit >=1.30
    except Exception:
        q = st.experimental_get_query_params()
    code = q.get("code", [None])[0] if isinstance(q.get("code"), list) else q.get("code")

    if not have_tokens():
        if env_ok and code:
            payload = exchange_code_for_tokens(code)
            if payload.get("access_token"):
                save_tokens(payload)
                st.success("Yahoo authenticated ✅")
                try:
                    st.query_params.clear()
                except Exception:
                    st.experimental_set_query_params()
                st.rerun()
            else:
                st.error("Token exchange failed")
                st.code(str(payload), language="json")
                st.stop()
        elif env_ok:
            st.markdown(f"[🔐 Connect Yahoo]({authorize_url()})")
            st.caption("Approve in Yahoo; you'll return here with ?code=...")
            st.stop()
        else:
            st.error("Set YAHOO_* in .env or Secrets.")
            st.stop()
    else:
        st.success("Yahoo connected ✅")
def fetch_nfl_leagues(sc):
    """
    Works across yfa versions that may or may not have:
      - Game.current_game_id()
      - Game.current_season()
      - Game.leagues(season) vs leagues()
      - Game.league_ids(season)
    Returns: list[{"league_key": ..., "name": ...}] where possible.
    """
    gm = yfa.Game(sc, "nfl")

    # Determine the season/game identifier we can pass
    season = None
    if hasattr(gm, "current_game_id"):
        try: season = gm.current_game_id()
        except Exception: season = None
    if season is None and hasattr(gm, "current_season"):
        try: season = gm.current_season()
        except Exception: season = None
    if season is None and hasattr(gm, "game_id"):
        try: season = gm.game_id()
        except Exception: season = None

    # Try the most helpful, structured call first
    leagues = []
    if hasattr(gm, "leagues"):
        try:
            leagues = gm.leagues(season) if season is not None else gm.leagues()
        except Exception:
            try:
                # some versions treat leagues() as parameterless only
                leagues = gm.leagues()
            except Exception:
                pass

    # Fallback to league_ids if leagues() isn’t available/usable
    if not leagues and hasattr(gm, "league_ids"):
        try:
            ids = gm.league_ids(season) if season is not None else gm.league_ids()
            leagues = [{"league_key": lid, "name": lid} for lid in ids]
        except Exception:
            pass

    return leagues or []


# ---------------- Live data: leagues ----------------
try:
    sc = get_session()
    leagues = fetch_nfl_leagues(sc)
except Exception as e:
    st.error(f"Init error: {e}")
    st.stop()

if not leagues:
    st.warning("No NFL leagues found for this Yahoo account.")
    st.stop()

# ---------------- Tabs: Roster / Start-Sit / Waivers ----------------
tab1, tab2, tab3 = st.tabs(["Roster", "Start/Sit (heuristic)", "Waivers (Add/Drop or FAAB)"])

with tab1:
    st.subheader("Current Roster")
    df_roster = roster_df(sc, league_key)
    st.dataframe(df_roster, use_container_width=True)

with tab2:
    st.subheader("Recommended Starters (simple heuristic)")
    L = yfa.League(sc, league_key)
    settings = L.settings()

    # Build list of lineup slots
    slots = []
    for rp in settings.get("roster_positions", []):
        pos, cnt = rp.get("position"), int(rp.get("count", 0))
        if pos and cnt > 0:
            slots += [pos]*cnt

    # Score: base + points - status penalty
    def score_row(row):
        status = (row["status"] or "").upper()
        penalty = {"IR": -100, "O": -50, "D": -25, "Q": -10}.get(status, 0)
        base = 10
        return float(row.get("points", 0) or 0) + penalty + base

    if df_roster.empty:
        st.info("No roster found.")
    else:
        pool = df_roster.copy()
        pool["score"] = pool.apply(score_row, axis=1)

        def eligible(row, slot):
            slot = slot.upper()
            elig = (row["eligible_positions"] or "").upper().split(",")
            if slot in ("W/R/T","WR/RB/TE","FLEX"):
                return any(p in elig for p in ("WR","RB","TE"))
            if slot in ("D","DEF","DST"):
                return "DEF" in elig
            return slot in elig

        starters, used = [], set()
        for slot in slots:
            elig_pool = pool[~pool["player_id"].isin(used)]
            if len(elig_pool) == 0:
                continue
            elig_pool = elig_pool[elig_pool.apply(lambda r: eligible(r, slot), axis=1)]
            if not len(elig_pool):
                continue
            pick = elig_pool.sort_values("score", ascending=False).iloc[0]
            starters.append((slot, pick))
            used.add(pick["player_id"])

        bench = pool[~pool["player_id"].isin(used)].sort_values("score", ascending=False)

        st.write("### Starters")
        for slot, p in starters:
            st.write(f"- **{slot}** → {p['name']} (score: {p['score']:.1f}, status: {p['status']})")

        st.write("### Bench (top 10)")
        for _, p in bench.head(10).iterrows():
            st.write(f"- {p['name']} (score: {p['score']:.1f}, status: {p['status']})")

        if OPENAI_OK and st.button("Explain this lineup (OpenAI)"):
            try:
                msg = "You are a fantasy football assistant. Explain concisely why the chosen starters beat the bench, considering injuries (Q/O/D/IR penalties) and points."
                roster_summary = "\n".join([f"{slot}: {p['name']} ({p['score']:.1f})" for slot, p in starters])
                bench_summary = "\n".join([f"{r['name']} ({r['score']:.1f})" if 'score' in r else f"{r['name']}" for _, r in bench.head(5).iterrows()])
                resp = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":msg},
                        {"role":"user","content": f"Starters:\n{roster_summary}\nBench (top 5):\n{bench_summary}"}
                    ],
                    max_tokens=250,
                    temperature=0.2
                )
                st.info(resp.choices[0].message.content.strip())
            except Exception as e:
                st.warning(f"OpenAI explanation not available: {e}")

with tab3:
    st.subheader("Waiver targets & Approvals")
    try:
        L = yfa.League(sc, league_key)
        cands = free_agents(L)
    except Exception as e:
        st.error(f"Failed to get free agents: {e}")
        cands = []

    if not cands:
        st.info("No candidates returned right now.")
    else:
        st.write("### Top Free Agents (by points/proj)")
        for i, c in enumerate(cands):
            st.write(f"{i}. **{c['name']}** ({c['position']}) — points {c['points']} — id `{c['player_id']}`")

        st.divider()
        st.write("### Approve & Execute")
        idx = st.number_input("Index to add (0‑based)", min_value=0, max_value=max(0, len(cands)-1), value=0)
        drop_pid = st.text_input("Player ID to drop (optional)")
        faab_bid = st.number_input("FAAB bid (optional, if league uses FAAB)", min_value=0, max_value=300, value=0)

        if st.button("Approve"):
            pick = cands[int(idx)]
            try:
                result = execute_add_drop(sc, league_key, add_pid=pick["player_id"],
                                          drop_pid=drop_pid or None,
                                          faab_bid=int(faab_bid) if faab_bid else None)
                if result.get("status") == "ok":
                    st.success("Transaction submitted ✅")
                    st.json(result.get("details"))
                else:
                    st.error("Transaction failed")
                    st.json(result)
            except Exception as e:
                st.error(f"Execution error: {e}")
