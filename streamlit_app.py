import os, json, time, math
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

# ────────────────────────────── Setup ──────────────────────────────
load_dotenv()
CID = os.getenv("YAHOO_CLIENT_ID")
CSEC = os.getenv("YAHOO_CLIENT_SECRET")
REDIRECT = os.getenv("YAHOO_REDIRECT_URI")
OAUTH_FILE = Path("oauth2.json")             # token cache (ignored by git)
PREFS_FILE = Path(".agent_prefs.json")       # simple local prefs
DATA_DIR = Path("data")

# Optional OpenAI
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI()
    OPENAI_OK = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_OK = False

st.set_page_config(page_title="Yahoo Fantasy Agent — Live", layout="wide")
st.title("🏈 Yahoo Fantasy Agent — Live")

# Toggle to show raw Yahoo responses
show_raw = st.toggle("🔎 Show raw Yahoo API responses", value=False)

# ─────────────────────── OAuth helpers ───────────────────────
def authorize_url():
    return "https://api.login.yahoo.com/oauth2/request_auth?" + urlencode({
        "client_id": CID,
        "redirect_uri": REDIRECT,
        "response_type": "code",
        "scope": "fspt-w",  # write scope (needed for add/drop later)
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

# ───────────────── HTTP helper (diagnostics) ─────────────────
def yfs_get(sc, path):
    url = "https://fantasysports.yahooapis.com/fantasy/v2" + path
    r = sc.session.get(url, headers={"Accept": "application/json"}, timeout=30)

    if show_raw:
        with st.expander(f"HTTP debug: {path}", expanded=True):
            st.write({"url": url, "status_code": r.status_code, "content_type": r.headers.get("Content-Type", "")})
            st.code((r.text or "")[:1500])

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
        raise RuntimeError(f"Non-JSON response ({r.status_code}, {ct}). First 400 chars: {body[:400]}")

# ─────────────── Parse users→games→leagues (numeric keys) ───────────────
def parse_games_leagues_v2(raw):
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

    # Find "games"
    games_container = None
    for entry in user_arr:
        if isinstance(entry, dict) and "games" in entry:
            games_container = entry["games"]
            break
    if not games_container:
        return [], []

    games, leagues = [], []
    for _, game_wrapper in games_container.items():
        if not isinstance(game_wrapper, dict):
            continue
        glist = game_wrapper.get("game")
        if not isinstance(glist, list):
            continue

        # game meta
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
    def to_int(s):
        try: return int(str(s))
        except: return -1
    nfl_games = [g for g in games if (g.get("game_code") == "nfl" and g.get("season"))]
    if not nfl_games:
        return []
    latest = max(nfl_games, key=lambda g: to_int(g["season"]))
    latest_gk = latest["game_key"]
    return [L for L in leagues if L.get("game_key") == latest_gk]

# ───────────────────── Data helpers ─────────────────────
def resolve_team_key(sc, league_key: str):
    """Return your team_key in a league; logs attempts for clarity."""
    L = yfa.League(sc, league_key)

    # Try library first
    try:
        teams = L.teams()
        st.caption("resolve_team_key: L.teams() returned "
                   + (str(len(teams)) if hasattr(teams, "__len__") else "unknown length"))
        def extract(entry):
            if isinstance(entry, dict):
                return entry.get("team_key"), entry.get("is_owned_by_current_login"), entry.get("name")
            if isinstance(entry, list):
                tk = owned = name = None
                for kv in entry:
                    if isinstance(kv, dict):
                        if tk is None and "team_key" in kv: tk = kv["team_key"]
                        if owned is None and "is_owned_by_current_login" in kv: owned = kv["is_owned_by_current_login"]
                        if name is None and "name" in kv: name = kv["name"]
                return tk, owned, name
            return None, None, None

        for t in teams:
            tk, owned, name = extract(t)
            if owned in (1, True, "1") and tk:
                st.caption(f"resolve_team_key: picking owned team {name} ({tk})")
                return tk
        for t in teams:
            tk, owned, name = extract(t)
            if tk:
                st.caption(f"resolve_team_key: falling back to first team {name} ({tk})")
                return tk
    except Exception as e:
        st.caption(f"resolve_team_key: L.teams() failed: {e}")

    # Fallback via /users;use_login=1/teams
    try:
        data = yfs_get(sc, "/users;use_login=1/teams?format=json")
        fc = (data or {}).get("fantasy_content", {})
        users = fc.get("users", {})
        user_arr = users.get("user") if "user" in users else None
        if not user_arr:
            for k, v in users.items():
                if str(k).isdigit() and isinstance(v, dict) and "user" in v:
                    user_arr = v["user"]; break
        if user_arr:
            teams_container = None
            for entry in user_arr:
                if isinstance(entry, dict) and "teams" in entry:
                    teams_container = entry["teams"]; break
            if teams_container:
                for _, twrap in teams_container.items():
                    if not isinstance(twrap, dict): continue
                    tlist = twrap.get("team")
                    if not isinstance(tlist, list): continue
                    tk = None; name = None
                    for el in tlist:
                        if isinstance(el, dict):
                            if "team_key" in el: tk = el["team_key"]
                            if "name" in el and not name: name = el["name"]
                        if isinstance(el, list):
                            for kv in el:
                                if isinstance(kv, dict):
                                    if "team_key" in kv: tk = kv["team_key"]
                                    if "name" in kv and not name: name = kv["name"]
                    if tk and tk.startswith(league_key + ".t."):
                        st.caption(f"resolve_team_key: matched via /users teams → {name} ({tk})")
                        return tk
        st.caption("resolve_team_key: no match in /users;use_login=1/teams")
    except Exception as e:
        st.caption(f"resolve_team_key: users/teams fallback failed: {e}")

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
        if isinstance(elig, list): elig = ",".join(elig)
        rows.append({
            "player_id": p.get("player_id") if isinstance(p, dict) else None,
            "name": name,
            "status": p.get("status") if isinstance(p, dict) else None,
            "eligible_positions": elig or "",
            "selected_position": p.get("selected_position") if isinstance(p, dict) else None,
            "points": (p.get("points") if isinstance(p, dict) else 0) or (p.get("proj_points") if isinstance(p, dict) else 0) or 0,
        })
    return pd.DataFrame(rows)

def league_settings(sc, league_key: str) -> dict:
    try:
        return yfa.League(sc, league_key).settings() or {}
    except Exception:
        return {}

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
    L = yfa.League(sc, league_key)
    team_key = resolve_team_key(sc, league_key)
    if not team_key:
        return {"status": "error", "details": "Could not resolve team_key."}
    T = yfa.Team(sc, team_key)
    last_err = None

    try:
        if drop_pid:
            try:
                res_add = T.add_player(add_pid); res_drop = T.drop_player(drop_pid)
                return {"status": "ok", "details": {"team.add": res_add, "team.drop": res_drop}}
            except Exception:
                res_drop = T.drop_player(drop_pid); res_add = T.add_player(add_pid)
                return {"status": "ok", "details": {"team.drop": res_drop, "team.add": res_add}}
        else:
            res_add = T.add_player(add_pid)
            return {"status": "ok", "details": {"team.add": res_add}}
    except Exception as e1:
        last_err = f"team add/drop failed: {e1}"

    try:
        if hasattr(L, "add_player"):
            if drop_pid:
                res = L.add_player(add_pid, team_key=team_key, drop_player_id=drop_pid)
            else:
                res = L.add_player(add_pid, team_key=team_key)
            return {"status": "ok", "details": {"league.add_player": res}}
    except Exception as e2:
        last_err = f"{last_err} | league.add_player failed: {e2}"

    for meth in ("waiver_add", "add_player_waiver", "place_waiver"):
        if hasattr(L, meth):
            try:
                fn = getattr(L, meth)
                res = fn(add_pid, team_key, drop_player_id=drop_pid, bid=faab_bid)
                return {"status": "ok", "details": {meth: res}}
            except Exception as e3:
                last_err = f"{last_err} | {meth} failed: {e3}"

    return {"status": "error", "details": last_err or "No supported add/waiver method found."}

# ───────── Extra diag: list all user teams and highlight match ─────────
def render_user_teams(sc, league_key: str):
    """Show all teams for the logged-in user and highlight the one in this league."""
    try:
        data = yfs_get(sc, "/users;use_login=1/teams?format=json")
    except Exception as e:
        st.error(f"/users;use_login=1/teams failed: {e}")
        return

    fc = (data or {}).get("fantasy_content", {})
    users = fc.get("users", {})
    user_arr = users.get("user") if "user" in users else None
    if not user_arr:
        for k, v in users.items():
            if str(k).isdigit() and isinstance(v, dict) and "user" in v:
                user_arr = v["user"]; break

    rows = []
    if user_arr:
        teams_container = None
        for entry in user_arr:
            if isinstance(entry, dict) and "teams" in entry:
                teams_container = entry["teams"]; break
        if teams_container:
            for _, twrap in teams_container.items():
                if not isinstance(twrap, dict):
                    continue
                tlist = twrap.get("team")
                if not isinstance(tlist, list):
                    continue
                info = {"team_key": None, "team_name": None, "league_key_guess": None, "manager": None}
                for el in tlist:
                    if isinstance(el, dict):
                        if "team_key" in el:
                            info["team_key"] = el["team_key"]
                            parts = el["team_key"].split(".t.")[0] if ".t." in el["team_key"] else None
                            info["league_key_guess"] = parts
                        if "name" in el: info["team_name"] = el["name"]
                        if "managers" in el and isinstance(el["managers"], dict):
                            mgr = el["managers"].get("manager", [{}])[0]
                            if isinstance(mgr, dict) and "nickname" in mgr:
                                info["manager"] = mgr["nickname"]
                if info["team_key"]:
                    rows.append(info)

    if not rows:
        st.info("No teams found in /users;use_login=1/teams (this is normal if all your leagues are pre-draft).")
        return

    df = pd.DataFrame(rows)
    df["matches_selected_league"] = df["league_key_guess"] == league_key
    st.write("#### User Teams (from /users;use_login=1/teams)")
    st.dataframe(df, use_container_width=True)

    match = df[df["matches_selected_league"]]
    if match.empty:
        st.warning("No team matches the selected league yet. If you haven’t drafted, this is expected.")
    else:
        st.success(f"Found your team for this league: {match.iloc[0]['team_key']}")

# ─────────── AI Snapshot + Summaries + Retrieval ───────────
def league_current_week(settings_dict: dict) -> int:
    for k in ("current_week", "week", "standings_week"):
        v = settings_dict.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    return 1

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_scoreboard(sc, league_key: str, week: int|None=None):
    try:
        wk_q = f";week={week}" if week else ""
        return yfs_get(sc, f"/league/{league_key}/scoreboard{wk_q}?format=json")
    except Exception:
        return None

def team_roster_raw(sc, team_key: str):
    try:
        T = yfa.Team(sc, team_key)
        return T.roster() or []
    except Exception:
        return []

def snapshot_now(sc, league_key: str, settings: dict, week_override: int|None=None) -> dict:
    wk = week_override or league_current_week(settings or {})
    league_dir = DATA_DIR / "league" / league_key / f"week_{wk}"
    ensure_dir(league_dir)

    # Save settings
    try:
        (league_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    except Exception:
        pass

    # Scoreboard + find opponent
    sb = get_scoreboard(sc, league_key, wk) or {}
    try:
        (league_dir / "scoreboard.json").write_text(json.dumps(sb, indent=2))
    except Exception:
        pass

    you_tk = resolve_team_key(sc, league_key)
    opp_tk = None
    your_name = None
    opp_name = None
    try:
        # Parse scoreboard to locate opponent
        matchups = []
        fc = sb.get("fantasy_content", {})
        league = fc.get("league")
        # league can be list-shaped; find "scoreboard"
        scoreboard = None
        if isinstance(league, list):
            for el in league:
                if isinstance(el, dict) and "scoreboard" in el:
                    scoreboard = el["scoreboard"]
                    break
        elif isinstance(league, dict):
            scoreboard = league.get("scoreboard")

        if scoreboard:
            ms = scoreboard.get("matchups", {})
            for _, wrap in ms.items():
                if isinstance(wrap, dict) and "matchup" in wrap:
                    m = wrap["matchup"]
                    # each matchup has "teams"
                    teams_c = None
                    if isinstance(m, list):
                        for x in m:
                            if isinstance(x, dict) and "teams" in x:
                                teams_c = x["teams"]; break
                    elif isinstance(m, dict):
                        teams_c = m.get("teams")
                    if teams_c:
                        tkeys = []
                        tnames = []
                        for _, tw in teams_c.items():
                            if isinstance(tw, dict) and "team" in tw:
                                team = tw["team"]
                                tk = None; nm = None
                                if isinstance(team, list):
                                    for kv in team:
                                        if isinstance(kv, dict):
                                            if "team_key" in kv and not tk: tk = kv["team_key"]
                                            if "name" in kv and not nm: nm = kv["name"]
                                elif isinstance(team, dict):
                                    tk = team.get("team_key") or None
                                    nm = team.get("name") or None
                                if tk:
                                    tkeys.append(tk)
                                    tnames.append(nm or tk)
                        if you_tk and you_tk in tkeys and len(tkeys) == 2:
                            idx = tkeys.index(you_tk)
                            opp_tk = tkeys[1-idx]
                            your_name = tnames[idx]
                            opp_name = tnames[1-idx]
                            break
    except Exception:
        pass

    # Your roster + opponent roster
    your_roster = team_roster_raw(sc, you_tk) if you_tk else []
    opp_roster = team_roster_raw(sc, opp_tk) if opp_tk else []

    try:
        (league_dir / "your_roster.json").write_text(json.dumps(your_roster, indent=2))
    except Exception:
        pass
    try:
        (league_dir / "opp_roster.json").write_text(json.dumps(opp_roster, indent=2))
    except Exception:
        pass

    # Transactions (best-effort)
    try:
        tx = yfs_get(sc, f"/league/{league_key}/transactions?format=json")
        (league_dir / "transactions.json").write_text(json.dumps(tx, indent=2))
    except Exception:
        tx = {}

    # Free agents snapshot (top N)
    try:
        L = yfa.League(sc, league_key)
        fas = free_agents(L, positions=("WR","RB","TE","QB"), limit=50)
        (league_dir / "free_agents.json").write_text(json.dumps(fas, indent=2))
    except Exception:
        fas = []

    return {
        "dir": str(league_dir),
        "week": wk,
        "you_team_key": you_tk,
        "opp_team_key": opp_tk,
        "you_team_name": your_name,
        "opp_team_name": opp_name,
        "files": [p.name for p in league_dir.glob("*.json")],
    }

def _name(p): return p.get("name") if isinstance(p, dict) else None
def _pts(p): 
    if isinstance(p, dict):
        return float(p.get("points") or p.get("proj_points") or 0.0)
    return 0.0

def summarize_roster(roster_list, title):
    if not roster_list:
        return f"{title}: (no players)"
    ps = sorted(
        [f"{_name(p)} ({','.join(p.get('eligible_positions', []))}) {_pts(p):.1f}" for p in roster_list if isinstance(p, dict)],
        key=lambda s: float(s.split()[-1]) if s.split() else 0.0,
        reverse=True
    )[:10]
    return f"{title}: top players → " + "; ".join(ps)

def summarize_transactions(tx_json):
    try:
        fc = (tx_json or {}).get("fantasy_content", {})
        league = fc.get("league")
        txs = []
        if isinstance(league, list):
            for el in league:
                if isinstance(el, dict) and "transactions" in el:
                    txs = list(el["transactions"].values())
                    break
        elif isinstance(league, dict):
            txs = list(league.get("transactions", {}).values())
        names = []
        for w in txs:
            if not isinstance(w, dict): 
                continue
            t = w.get("transaction")
            if isinstance(t, list):
                for kv in t:
                    if isinstance(kv, dict) and "players" in kv:
                        for _, pw in kv["players"].items():
                            if isinstance(pw, dict) and "player" in pw:
                                pl = pw["player"]
                                nm = None
                                if isinstance(pl, list):
                                    for z in pl:
                                        if isinstance(z, dict) and "name" in z:
                                            nm = z["name"].get("full")
                                elif isinstance(pl, dict):
                                    nm = pl.get("name", {}).get("full")
                                if nm:
                                    names.append(nm)
        if not names:
            return "Recent transactions: none found."
        uniq = []
        for n in names:
            if n not in uniq: uniq.append(n)
        return f"Recent transactions (sample): " + ", ".join(uniq[:10])
    except Exception:
        return "Recent transactions: (unavailable)"

def build_context_summaries(snapshot_meta: dict) -> list[str]:
    league_dir = Path(snapshot_meta["dir"])
    def read_json(name):
        p = league_dir / name
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: return {}
        return {}

    your = read_json("your_roster.json")
    opp = read_json("opp_roster.json")
    tx = read_json("transactions.json")
    fas = read_json("free_agents.json")

    you_sum = summarize_roster(your, "Your roster")
    opp_sum = summarize_roster(opp, f"Opponent roster ({snapshot_meta.get('opp_team_name') or 'unknown'})")
    tx_sum = summarize_transactions(tx)
    fa_sum = f"Top free agents snapshot: " + "; ".join([f"{p.get('name')} {float(p.get('points') or 0):.1f}" for p in fas[:10]])

    return [you_sum, opp_sum, tx_sum, fa_sum]

def select_context(query: str, summaries: list[str], k: int = 3) -> list[str]:
    if not query:
        return summaries[:k]
    q = query.lower().split()
    scored = []
    for s in summaries:
        score = sum(s.lower().count(tok) for tok in q)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [s for sc, s in scored if sc > 0][:k]
    return out or summaries[:k]

def answer_with_ai(question: str, summaries: list[str], structured: dict) -> str:
    if not OPENAI_OK:
        return "OpenAI key not configured; cannot run the AI answer. (Set OPENAI_API_KEY and rerun.)"
    sys = "You are a concise fantasy football assistant. Use the provided CONTEXT to ground your answer. When uncertain, say what additional data you need."
    ctx = "\n".join(f"- {s}" for s in summaries)
    struct_txt = json.dumps(structured, ensure_ascii=False)
    prompt = f"""CONTEXT_SUMMARIES:
{ctx}

STRUCTURED_SNAPSHOT (JSON):
{struct_txt}

LIVE_CHECKS:
- Use logic to verify roster eligibility and positions.
- Prefer safer picks if injury status is questionable.

QUESTION:
{question}

ANSWER STYLE:
- Bullet points first (1–4 bullets), then a short 1–2 sentence recommendation.
- Be specific and reference players/positions from context when possible."""
    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ───────────────── Sidebar: Connect ─────────────────
with st.sidebar:
    st.header("Yahoo Connection")
    st.caption("Redirect in env: " + (REDIRECT or "<missing>"))
    env_ok = all([CID, CSEC, REDIRECT])
    st.write("Env:", "✅" if env_ok else "⚠️ missing")

    try:
        q = st.query_params
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

# ───────────────── Fetch latest NFL leagues ─────────────────
try:
    sc = get_session()
    raw = yfs_get(sc, "/users;use_login=1/games/leagues?format=json")
    games, leagues_all = parse_games_leagues_v2(raw)
    nfl_leags = filter_latest_nfl_leagues(leagues_all, games)
except Exception as e:
    st.error(f"Init error while loading leagues: {e}")
    st.stop()

if show_raw:
    with st.expander("Debug: parsed games", expanded=False): st.write(games)
    with st.expander("Debug: all leagues", expanded=False): st.write(leagues_all)
    with st.expander("Debug: latest NFL leagues", expanded=False): st.write(nfl_leags)

if not nfl_leags:
    st.warning("No NFL leagues found for this Yahoo account (latest season).")
    st.caption("If you DO have NFL leagues this season, re-check you authorized the right Yahoo ID and scope 'fspt-w'.")
    st.stop()

league_map = {f"{lg['name']} ({lg['league_key']})": lg["league_key"] for lg in nfl_leags}
choice = st.selectbox("Select a league", list(league_map.keys()))
league_key = league_map[choice]

# Show user teams table to clarify whether a team exists for this league
render_user_teams(sc, league_key)

# ─────────────── League status + pre-draft gate (robust) ───────────────
def get_league_settings_safe(sc, league_key: str) -> dict:
    """Robust settings fetch (library first, then raw API)."""
    # 1) Library
    try:
        s = yfa.League(sc, league_key).settings() or {}
        if isinstance(s, list):
            flat = {}
            for it in s:
                if isinstance(it, dict):
                    flat.update(it)
            s = flat
        return s
    except Exception:
        pass
    # 2) Raw API
    try:
        raw_settings = yfs_get(sc, f"/league/{league_key}/settings?format=json")
        fc = (raw_settings or {}).get("fantasy_content", {})
        league = fc.get("league")
        if isinstance(league, list):
            for el in league:
                if isinstance(el, dict) and "settings" in el:
                    return el["settings"] or {}
        if isinstance(league, dict) and "settings" in league:
            return league["settings"] or {}
    except Exception:
        pass
    return {}

def coerce_draft_status(val) -> str:
    v = str(val or "unknown").lower()
    if "pre" in v:  return "predraft"
    if "post" in v: return "postdraft"
    if v in ("predraft","postdraft","inseason"): return v
    return "unknown"

settings = get_league_settings_safe(sc, league_key)
draft_status = coerce_draft_status(settings.get("draft_status"))

# BIG visible status line
badge = {"predraft":"🟡", "postdraft":"🟢", "inseason":"🟢", "unknown":"⚠️"}.get(draft_status, "⚠️")
st.markdown(f"### League status: {badge} **{draft_status}**")

with st.expander("League info", expanded=False):
    st.write({
        "league_key": league_key,
        "name": settings.get("name"),
        "season": settings.get("season"),
        "draft_status": draft_status,
        "num_teams": settings.get("num_teams"),
        "scoring_type": settings.get("scoring_type"),
        "waiver_type": settings.get("waiver_type"),
        "faab_budget": settings.get("faab_budget"),
    })

# Clear “why no data” explainer
empty_reasons = []
if not settings:
    empty_reasons.append("League settings could not be read (Yahoo may be rate-limiting or the league is not finalized).")
if draft_status in ("predraft", "unknown"):
    empty_reasons.append("League shows as pre-draft or unknown; teams/rosters aren’t published yet.")
# Probe teams/roster to refine message
try:
    L_probe = yfa.League(sc, league_key)
    teams_probe = L_probe.teams()
    if not teams_probe:
        empty_reasons.append("No teams returned by Yahoo for this league yet.")
    else:
        tk_probe = resolve_team_key(sc, league_key)
        if tk_probe:
            try:
                T_probe = yfa.Team(sc, tk_probe)
                roster_probe = T_probe.roster()
                if not roster_probe:
                    empty_reasons.append("Your roster is empty right now (common pre-draft).")
            except Exception:
                empty_reasons.append("Could not fetch your roster (authentication/scope or timing).")
        else:
            empty_reasons.append("Could not match your team_key in this league yet.")
except Exception:
    empty_reasons.append("League probe failed (network/auth).")

if empty_reasons:
    st.warning("ℹ️ **Why you may not see data yet:**\n\n- " + "\n- ".join(empty_reasons))
    st.caption("Tip: toggle **‘Show raw Yahoo API responses’** above to inspect raw JSON and confirm status.")

# Optional raw league endpoints peek
if show_raw:
    with st.expander("Raw league endpoints (sanity check)", expanded=False):
        try:
            raw_settings = yfs_get(sc, f"/league/{league_key}/settings?format=json")
            st.write("**/league/.../settings**:", raw_settings)
        except Exception as e:
            st.error(f"settings error: {e}")
        try:
            raw_standings = yfs_get(sc, f"/league/{league_key}/standings?format=json")
            st.write("**/league/.../standings**:", raw_standings)
        except Exception as e:
            st.error(f"standings error: {e}")
        try:
            raw_teams = yfs_get(sc, f"/league/{league_key}/teams?format=json")
            st.write("**/league/.../teams**:", raw_teams)
        except Exception as e:
            st.error(f"teams error: {e}")

# PRE-DRAFT → show only Draft Assistant and STOP
if draft_status != "postdraft":
    st.info("📝 **Pre-draft** detected. Roster, Start/Sit, Waivers, Trades, and Scheduler unlock after your draft.")
    st.subheader("Draft Assistant (Pre-draft)")
    st.caption("Upload projections/ADP CSV. Required: `name`, `position`, `proj_points`. Optional: `team`, `adp`, `ecr`.")
    uploaded_predraft = st.file_uploader("Upload projections CSV", type=["csv"], key="draft_upload_predraft")
    if uploaded_predraft:
        try:
            proj = pd.read_csv(uploaded_predraft)
            cols = {c.lower().strip(): c for c in proj.columns}
            need = ["name","position","proj_points"]
            if not all(k in cols for k in need):
                st.error(f"CSV must include: {need}. Found: {list(proj.columns)}")
            else:
                dfp = proj.rename(columns={
                    cols["name"]:"name",
                    cols["position"]:"position",
                    cols["proj_points"]:"proj_points"
                })
                for opt in ("adp","ecr","team"):
                    if opt in cols:
                        dfp[opt] = proj[cols[opt]]

                tiers = []
                for pos, grp in dfp.groupby(dfp["position"].str.upper()):
                    g = grp.copy()
                    top = 12 if pos in ("WR","RB") else 6 if pos in ("QB","TE") else max(4, len(g)//5)
                    top_sorted = g.sort_values("proj_points", ascending=False)
                    t1 = set(top_sorted.head(top).index)
                    t2 = set(top_sorted.iloc[top:top*2].index)
                    labels = []
                    for i in g.index:
                        labels.append("T1" if i in t1 else "T2" if i in t2 else "T3+")
                    tiers.append(g.assign(tier=labels))
                tiers_df = pd.concat(tiers).sort_values(
                    ["position","tier","proj_points"], ascending=[True, True, False]
                )
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)
                st.caption("Heuristic tiers. Sort/filter to build your queue.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Tip: export projections from your favorite site and upload here.")
    st.stop()  # hide everything else until postdraft

# POSTDRAFT → show full tabs (added AI Assistant tab)
tabs = st.tabs(["Draft Assistant", "Roster", "Start/Sit", "Waivers", "Trades", "AI Assistant", "Scheduler"])

# ───────────────────── Draft Assistant (postdraft tab) ─────────────────────
with tabs[0]:
    st.subheader("Draft Assistant")
    uploaded = st.file_uploader("Upload projections CSV", type=["csv"], key="draft_upload_post")
    if uploaded:
        try:
            proj = pd.read_csv(uploaded)
            cols = {c.lower().strip(): c for c in proj.columns}
            need = ["name","position","proj_points"]
            if not all(k in cols for k in need):
                st.error(f"CSV must include: {need}. Found: {list(proj.columns)}")
            else:
                dfp = proj.rename(columns={
                    cols["name"]:"name",
                    cols["position"]:"position",
                    cols["proj_points"]:"proj_points"
                })
                for opt in ("adp","ecr","team"):
                    if opt in cols:
                        dfp[opt] = proj[cols[opt]]
                tiers = []
                for pos, grp in dfp.groupby(dfp["position"].str.upper()):
                    g = grp.copy()
                    top = 12 if pos in ("WR","RB") else 6 if pos in ("QB","TE") else max(4, len(g)//5)
                    top_sorted = g.sort_values("proj_points", ascending=False)
                    t1 = set(top_sorted.head(top).index)
                    t2 = set(top_sorted.iloc[top:top*2].index)
                    labels = []
                    for i in g.index:
                        labels.append("T1" if i in t1 else "T2" if i in t2 else "T3+")
                    tiers.append(g.assign(tier=labels))
                tiers_df = pd.concat(tiers).sort_values(
                    ["position","tier","proj_points"], ascending=[True, True, False]
                )
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.caption("Upload projections/ADP to populate tiers.")


# 1) Roster
with tabs[1]:
    st.subheader("Current Roster")
    df_roster = roster_df(sc, league_key)
    if df_roster.empty:
        st.info("No roster found. If your draft just ended, give Yahoo a little time then refresh.")
    else:
        st.dataframe(df_roster, use_container_width=True)

# 2) Start/Sit
with tabs[2]:
    st.subheader("Recommended Starters (simple heuristic)")
    settings_full = settings or league_settings(sc, league_key)

    # lineup slots
    slots = []
    for rp in settings_full.get("roster_positions", []):
        pos, cnt = rp.get("position"), int(rp.get("count", 0))
        if pos and cnt > 0:
            slots += [pos]*cnt

    def score_row(row):
        status = (row["status"] or "").upper()
        penalty = {"IR": -100, "O": -50, "D": -25, "Q": -10}.get(status, 0)
        return float(row.get("points", 0) or 0) + penalty + 10

    pool = roster_df(sc, league_key)
    if pool.empty:
        st.info("No roster found (Yahoo may still be finalizing teams).")
    else:
        pool["score"] = pool.apply(score_row, axis=1)

        def eligible(row, slot):
            slot = slot.upper()
            elig = (row["eligible_positions"] or "").upper().split(",")
            if slot in ("W/R/T","WR/RB/TE","FLEX"): return any(p in elig for p in ("WR","RB","TE"))
            if slot in ("D","DEF","DST"): return "DEF" in elig
            return slot in elig

        starters, used = [], set()
        for slot in slots:
            elig_pool = pool[~pool["player_id"].isin(used)]
            elig_pool = elig_pool[elig_pool.apply(lambda r: eligible(r, slot), axis=1)]
            if not len(elig_pool): continue
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

        if OPENAI_OK and st.button("Explain lineup (OpenAI)"):
            try:
                roster_summary = "\n".join([f"{slot}: {p['name']} ({p['score']:.1f})" for slot, p in starters])
                bench_summary = "\n".join([f"{r['name']} ({r['score']:.1f})" for _, r in bench.head(5).iterrows()])
                resp = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are a fantasy football assistant. Be concise."},
                        {"role":"user","content": f"Starters:\n{roster_summary}\nBench (top 5):\n{bench_summary}"}
                    ],
                    max_tokens=250, temperature=0.2
                )
                st.info(resp.choices[0].message.content.strip())
            except Exception as e:
                st.warning(f"OpenAI explanation not available: {e}")

# 3) Waivers
with tabs[3]:
    st.subheader("Waiver targets & Approvals")
    try:
        L = yfa.League(sc, league_key)
        cands = free_agents(L)
    except Exception as e:
        st.error(f"Failed to get free agents: {e}")
        cands = []

    team_df = roster_df(sc, league_key)

    def positional_needs(roster_df: pd.DataFrame):
        need = {"QB":0,"RB":0,"WR":0,"TE":0,"DEF":0}
        for _, r in roster_df.iterrows():
            for p in str(r.get("eligible_positions","")).split(","):
                p = p.strip().upper()
                if p in need: need[p] += 1
        return need

    def suggest_faab(points_gain, budget=100, aggression=0.15):
        # very simple: value 0–1 → FAAB percent
        scale = min(max((points_gain or 0)/20.0, 0), 1)  # assume 20 pts = max impact
        return int(round(budget * (aggression * scale)))

    if not cands:
        st.info("No free agents returned right now (or Yahoo didn’t surface any for these positions).")
    else:
        st.write("### Suggested priorities")
        needs = positional_needs(team_df) if not team_df.empty else {}
        faab_budget = settings.get("faab_budget") or 100
        shown = []
        for c in cands:
            need_boost = 4 if needs.get(c["position"], 0) <= 1 else 0
            score = (c["points"] or 0) + need_boost
            bid = suggest_faab(points_gain=c["points"] or 0, budget=faab_budget)
            shown.append({"name": c["name"], "pos": c["position"], "points": c["points"], "score": score, "suggested_faab": bid, "player_id": c["player_id"]})
        dfw = pd.DataFrame(shown).sort_values("score", ascending=False)
        if dfw.empty:
            st.info("No waiver suggestions at the moment.")
        else:
            st.dataframe(dfw, use_container_width=True)

            st.divider()
            st.write("### Approve & Execute")
            idx = st.number_input("Row to add (0-based from table above)", min_value=0, max_value=max(0, len(dfw)-1), value=0)
            pick = dfw.iloc[int(idx)] if not dfw.empty else None
            drop_pid = st.text_input("Player ID to drop (optional)")
            default_bid = int(pick["suggested_faab"]) if pick is not None else 0
            faab_bid = st.number_input("FAAB bid (optional)", min_value=0, max_value=300, value=default_bid)
            if st.button("Approve transaction") and pick is not None:
                try:
                    result = execute_add_drop(sc, league_key, add_pid=str(pick["player_id"]),
                                              drop_pid=drop_pid or None,
                                              faab_bid=int(faab_bid) if faab_bid else None)
                    if result.get("status") == "ok":
                        st.success("Transaction submitted ✅")
                        st.json(result.get("details"))
                    else:
                        st.error("Transaction failed"); st.json(result)
                except Exception as e:
                    st.error(f"Execution error: {e}")

# 4) Trades (Evaluator)
with tabs[4]:
    st.subheader("Trade Evaluator")
    st.caption("Pick players to offer/request. We estimate value using current points/projections. For better accuracy, upload projections in Draft Assistant.")
    league = yfa.League(sc, league_key)
    team_key_you = resolve_team_key(sc, league_key)
    teams = []
    try:
        for t in league.teams():
            tk = None; name = None
            if isinstance(t, dict):
                tk = t.get("team_key"); name = t.get("name") or tk
            elif isinstance(t, list):
                for kv in t:
                    if isinstance(kv, dict):
                        if "team_key" in kv: tk = kv["team_key"]
                        if "name" in kv and not name: name = kv["name"]
            if tk: teams.append({"team_key": tk, "name": name or tk})
    except Exception:
        st.error("Could not list teams.")
        teams = []

    name_map = { (("YOU: " if t["team_key"]==team_key_you else "") + (t["name"] or t["team_key"])): t["team_key"] for t in teams }
    if not name_map:
        st.info("No teams yet.")
    else:
        left = st.selectbox("Your team", [k for k in name_map.keys() if k.startswith("YOU:")] or list(name_map.keys()))
        right = st.selectbox("Other team", [k for k in name_map.keys() if k != left])

        def team_roster_df(tk):
            T = yfa.Team(sc, tk)
            rows=[]
            for p in T.roster() or []:
                pts = p.get("points") or p.get("proj_points") or 0
                rows.append({"player_id": p.get("player_id"), "name": p.get("name"), "pos": ",".join(p.get("eligible_positions", [])), "points": float(pts)})
            return pd.DataFrame(rows)

        df_left = team_roster_df(name_map[left])
        df_right = team_roster_df(name_map[right])

        col1, col2 = st.columns(2)
        with col1:
            st.write("Your roster")
            if df_left.empty:
                st.info("Your roster is empty (pre-draft or Yahoo hasn’t posted teams yet).")
            else:
                st.dataframe(df_left, use_container_width=True)
            offer_ids = st.multiselect("Offer player IDs", df_left["player_id"].tolist())
        with col2:
            st.write("Their roster")
            if df_right.empty:
                st.info("That roster is empty (pre-draft or no players yet).")
            else:
                st.dataframe(df_right, use_container_width=True)
            request_ids = st.multiselect("Request player IDs", df_right["player_id"].tolist())

        def value(df, ids):
            return float(df[df["player_id"].isin(ids)].points.sum())

        v_you_out = value(df_left, offer_ids)
        v_you_in  = value(df_right, request_ids)
        delta_you = v_you_in - v_you_out

        v_them_out = value(df_right, request_ids)
        v_them_in  = value(df_left, offer_ids)
        delta_them = v_them_in - v_them_out

        st.write(f"**Your Δ value:** {delta_you:+.1f}  |  **Their Δ value:** {delta_them:+.1f}")

        if OPENAI_OK and st.button("Draft a trade pitch (OpenAI)"):
            try:
                msg = f"We propose trading {offer_ids} for {request_ids}. Our delta {delta_you:+.1f}, their delta {delta_them:+.1f}. Write a friendly, 3-sentence pitch."
                resp = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":msg}],
                    max_tokens=150, temperature=0.6
                )
                st.info(resp.choices[0].message.content.strip())
            except Exception as e:
                st.warning(f"OpenAI not available: {e}")

# 5) AI Assistant (RAG-lite)
with tabs[5]:
    st.subheader("AI Assistant (snapshot → summarize → retrieve → answer)")

    wk_guess = league_current_week(settings or {})
    wk = st.number_input("Week to snapshot", min_value=1, max_value=25, value=wk_guess, step=1)
    col_a, col_b = st.columns([1,2])

    with col_a:
        if st.button("📸 Snapshot league data now"):
            try:
                meta = snapshot_now(sc, league_key, settings, week_override=int(wk))
                st.success(f"Snapshot saved under: {meta['dir']}")
                st.json(meta)
                st.session_state["_last_snapshot_meta"] = meta
            except Exception as e:
                st.error(f"Snapshot failed: {e}")

    with col_b:
        meta = st.session_state.get("_last_snapshot_meta")
        if meta:
            st.caption(f"Using snapshot dir: {meta['dir']}")
            summaries = build_context_summaries(meta)
            st.write("**Context summaries (top of mind):**")
            for s in summaries:
                st.write("- " + s)

            q = st.text_area("Ask a question (e.g., 'Who should I flex in PPR this week?')", height=80)
            if st.button("🤖 Ask AI"):
                chosen = select_context(q, summaries, k=3)
                structured = {
                    "league_key": league_key,
                    "week": meta.get("week"),
                    "you_team_key": meta.get("you_team_key"),
                    "opp_team_key": meta.get("opp_team_key"),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                ans = answer_with_ai(q, chosen, structured)
                st.markdown("### Answer")
                st.write(ans)
                with st.expander("Context used"):
                    st.write(chosen)
        else:
            st.info("Take a snapshot first to build context.")

# 6) Scheduler / Automation
with tabs[6]:
    st.subheader("Scheduler / Automation")
    st.caption("These are preferences the agent will use. To fully automate, trigger a daily script (cron, GitHub Action, or Cloud Scheduler) that reads this file and calls the same functions.")
    prefs = {"auto_lineup": False, "auto_waivers": False, "waiver_budget_cap": 25, "aggression": 0.15}
    if PREFS_FILE.exists():
        try:
            prefs.update(json.loads(PREFS_FILE.read_text()))
        except Exception:
            pass

    prefs["auto_lineup"] = st.toggle("Auto set best lineup (Thu/Sun)", value=prefs["auto_lineup"])
    prefs["auto_waivers"] = st.toggle("Auto submit waivers", value=prefs["auto_waivers"])
    prefs["waiver_budget_cap"] = st.slider("Max FAAB per week", 0, 100, value=int(prefs["waiver_budget_cap"]))
    prefs["aggression"] = st.slider("Waiver aggression (0.05–0.35)", 0.05, 0.35, value=float(prefs["aggression"]), step=0.01)

    if st.button("Save preferences"):
        PREFS_FILE.write_text(json.dumps(prefs, indent=2))
        st.success("Saved.")

    st.divider()
    st.write("**Run now (manual):**")
    if st.button("Simulate weekly waivers now"):
        st.info("This will compute waiver suggestions using current free agents and preferences. (Submission still requires clicking Approve in Waivers tab.)")


