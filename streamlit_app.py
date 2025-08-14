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
    if not OAUTH_FILE.exists():
        return False
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

# ---------------- HTTP helper with diagnostics ----------------
def yfs_get(sc, path):
    """
    GET Yahoo Fantasy v2 with strong diagnostics.
    Example path: "/users;use_login=1/games/leagues?format=json"
    """
    url = "https://fantasysports.yahooapis.com/fantasy/v2" + path
    r = sc.session.get(
        url,
        headers={"Accept": "application/json"},
        timeout=30,
    )

    with st.expander(f"HTTP debug: {path}"):
        st.write({"url": url, "status_code": r.status_code, "content_type": r.headers.get("Content-Type", "")})
        st.code((r.text or "")[:800])

    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized: token invalid/expired or scope missing. Click Connect again.")
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden: app not authorized for Fantasy scope or account lacks access.")
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} error from Yahoo: {r.text[:200]}")

    body = (r.text or "").strip()
    if not body:
        raise RuntimeError("Empty response from Yahoo (no body).")

    try:
        return r.json()
    except Exception:
        ct = r.headers.get("Content-Type", "")
        raise RuntimeError(f"Non‑JSON response ({r.status_code}, {ct}). First 400 chars: {body[:400]}")

# ---------------- Parsers for numeric-keyed Yahoo JSON ----------------
def parse_games_leagues_v2(raw):
    """
    Parse Yahoo JSON with numeric keys + arrays (like your debug blob).
    Returns:
      games   = [{"game_key","game_code","season","name"}]
      leagues = [{"game_key","league_key","name"}]
    """
    fc = (raw or {}).get("fantasy_content", {})
    users = fc.get("users", {})

    # Locate the user array
    user_arr = None
    if isinstance(users, dict):
        if "user" in users:
            user_arr = users["user"]
        else:
            for k, v in users.items():
                if str(k).isdigit() and isinstance(v, dict) and "user" in v:
                    user_arr = v["user"]
                    break
    if not user_arr:
        return [], []

    # Find "games" container in user array
    games_container = None
    for entry in user_arr:
        if isinstance(entry, dict) and "games" in entry:
            games_container = entry["games"]
            break
    if not games_container:
        return [], []

    games, leagues = [], []

    # Each key "0","1",... → {"game": [ meta, {"leagues": {...}}, ... ]}
    for _, game_wrapper in games_container.items():
        if not isinstance(game_wrapper, dict):
            continue
        glist = game_wrapper.get("game")
        if not isinstance(glist, list):
            continue

        # meta has code/season (or at least game_key)
        gmeta = None
        for elem in glist:
            if isinstance(elem, dict) and ("code" in elem or "game_code" in elem) and "season" in elem:
                gmeta = elem
                break
        if not gmeta:
            for elem in glist:
                if isinstance(elem, dict) and "game_key" in elem:
                    gmeta = elem
                    break
        if not gmeta:
            continue

        gk = gmeta.get("game_key")
        gc = gmeta.get("code") or gmeta.get("game_code")
        season = gmeta.get("season")
        gname = gmeta.get("name") or f"{gc} {season}"
        games.append({"game_key": gk, "game_code": gc, "season": season, "name": gname})

        # leagues for this game
        for elem in glist:
            if isinstance(elem, dict) and "leagues" in elem:
                leagues_dict = elem["leagues"]
                for _, lwrap in leagues_dict.items():
                    if not isinstance(lwrap, dict):
                        continue
                    lst = lwrap.get("league")
                    if isinstance(lst, list):
                        for lentry in lst:
                            if isinstance(lentry, dict):
                                lk = lentry.get("league_key")
                                nm = lentry.get("name") or lk
                            elif isinstance(lentry, list):
                                lk = nm = None
                                for kv in lentry:
                                    if "league_key" in kv: lk = kv["league_key"]
                                    if "name" in kv: nm = kv["name"]
                                nm = nm or lk
                            else:
                                continue
                            if lk:
                                leagues.append({"game_key": gk, "league_key": lk, "name": nm})
    return games, leagues

def filter_latest_nfl_leagues(leagues, games):
    """Keep leagues belonging to the latest NFL (game_code='nfl') season."""
    def to_int(s):
        try: return int(str(s))
        except: return -1
    nfl_games = [g for g in games if (g.get("game_code") == "nfl" and g.get("season"))]
    if not nfl_games:
        return []
    latest = max(nfl_games, key=lambda g: to_int(g["season"]))
    latest_gk = latest["game_key"]
    return [L for L in leagues if L.get("game_key") == latest_gk]

# ---------------- Data helpers ----------------
def resolve_team_key(sc, league_key: str):
    """
    Return the team_key for the logged-in user in the given league.
    1) Try yahoo_fantasy_api League.teams() (shape varies by version)
    2) Fallback to user-scoped teams endpoint and match by league_key prefix
       (team_key looks like: "<game_key>.l.<league_id>.t.<team_id>")
    """
    L = yfa.League(sc, league_key)

    # ---- Try via library first, but guard for mixed shapes
    try:
        teams = L.teams()
        def extract_team_key(entry):
            # entry might be dict, list-of-dicts, etc.
            if isinstance(entry, dict):
                return entry.get("team_key"), entry.get("is_owned_by_current_login")
            if isinstance(entry, list):
                tk = None
                owned = None
                for kv in entry:
                    if isinstance(kv, dict):
                        if tk is None and "team_key" in kv:
                            tk = kv["team_key"]
                        if owned is None and "is_owned_by_current_login" in kv:
                            owned = kv["is_owned_by_current_login"]
                return tk, owned
            return None, None

        # Prefer the one owned by current login, if that flag is present
        for t in teams:
            tk, owned = extract_team_key(t)
            if owned in (1, True, "1") and tk:
                return tk

        # Otherwise just return the first usable team_key
        for t in teams:
            tk, _ = extract_team_key(t)
            if tk:
                return tk
    except Exception:
        pass

    # ---- Fallback: user-scoped teams endpoint, match team by league_key
    try:
        data = yfs_get(sc, "/users;use_login=1/teams?format=json")
        fc = (data or {}).get("fantasy_content", {})
        users = fc.get("users", {})
        # Find the array that holds the 'user'
        user_arr = None
        if isinstance(users, dict):
            if "user" in users:
                user_arr = users["user"]
            else:
                for k, v in users.items():
                    if str(k).isdigit() and isinstance(v, dict) and "user" in v:
                        user_arr = v["user"]
                        break
        if user_arr:
            # Find the "teams" container
            teams_container = None
            for entry in user_arr:
                if isinstance(entry, dict) and "teams" in entry:
                    teams_container = entry["teams"]
                    break
            if teams_container:
                # Walk teams; shape similar to games/leagues (numeric keys)
                for _, twrap in teams_container.items():
                    if not isinstance(twrap, dict): 
                        continue
                    tlist = twrap.get("team")
                    if not isinstance(tlist, list): 
                        continue
                    # Collect fields from the list
                    tk = None
                    for el in tlist:
                        if isinstance(el, dict) and "team_key" in el:
                            tk = el["team_key"]
                            break
                        if isinstance(el, list):
                            for kv in el:
                                if isinstance(kv, dict) and "team_key" in kv:
                                    tk = kv["team_key"]
                                    break
                    if tk and tk.startswith(league_key + ".t."):
                        return tk
    except Exception:
        pass

    return None

def roster_df(sc, league_key: str) -> pd.DataFrame:
    team_key = resolve_team_key(sc, league_key)
    if not team_key:
        return pd.DataFrame()
    T = yfa.Team(sc, team_key)
    rows = []
    for p in T.roster() or []:
        name = p.get("name") if isinstance(p, dict) else None
        elig = p.get("eligible_positions") if isinstance(p, dict) else None
        if isinstance(elig, list):
            elig = ",".join(elig)
        rows.append({
            "player_id": p.get("player_id") if isinstance(p, dict) else None,
            "name": name,
            "status": p.get("status") if isinstance(p, dict) else None,
            "eligible_positions": elig or "",
            "selected_position": p.get("selected_position") if isinstance(p, dict) else None,
            "points": (p.get("points") if isinstance(p, dict) else 0) or (p.get("proj_points") if isinstance(p, dict) else 0) or 0,
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

# ---------------- Live data: leagues ----------------
try:
    sc = get_session()
    raw = yfs_get(sc, "/users;use_login=1/games/leagues?format=json")
    games, leagues_all = parse_games_leagues_v2(raw)
    nfl_leags = filter_latest_nfl_leagues(leagues_all, games)
except Exception as e:
    st.error(f"Init error while loading leagues: {e}")
    st.stop()

with st.expander("Debug: parsed games"):
    st.write(games)
with st.expander("Debug: all leagues"):
    st.write(leagues_all)
with st.expander("Debug: latest NFL leagues"):
    st.write(nfl_leags)

if not nfl_leags:
    st.warning("No NFL leagues found for this Yahoo account (latest season).")
    st.caption("If you DO have NFL leagues this season, re-check you authorized the right Yahoo ID and scope 'fspt-w'.")
    st.stop()

league_map = {f"{lg['name']} ({lg['league_key']})": lg["league_key"] for lg in nfl_leags}
choice = st.selectbox("Select a league", list(league_map.keys()))
league_key = league_map[choice]

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

