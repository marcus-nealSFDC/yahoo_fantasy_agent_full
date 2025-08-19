# streamlit_app.py
import os, json, time
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

# Utils (your new modules)
from utils.lineup_opt import apply_enrichment, optimize_lineup  # optional enrich + solver
from utils.waivers import rank_waivers, multi_claim_queue       # waiver scoring + queue
from utils.opponent import scout_weak_spots, recommend_blocks   # opponent scouting

from policy import AutopilotPolicy, DEFAULT_POLICY
from storage.state import load_state, save_state
from agent.reasoning import plan_week, plan_start_sit, plan_waivers, plan_trades
from agent.executor import submit_waiver_queue, set_lineup, send_trade_offer


# ───────────────── Setup base dirs & logging ─────────────────
load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(os.getenv("LOG_DIR", DATA_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Optional S3 mirroring for logs
LOG_S3_BUCKET = os.getenv("LOG_S3_BUCKET", os.getenv("AWS_S3_BUCKET", ""))  
LOG_S3_PREFIX = os.getenv("LOG_S3_PREFIX", "logs/")

CID = os.getenv("YAHOO_CLIENT_ID")
CSEC = os.getenv("YAHOO_CLIENT_SECRET")
REDIRECT = os.getenv("YAHOO_REDIRECT_URI")
OAUTH_FILE = Path("oauth2.json")       # token cache (ignored by git)
PREFS_FILE = Path(".agent_prefs.json") # simple local prefs

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
def yfs_get(sc, path, *, max_retries: int = 3):
    """
    GET Yahoo Fantasy v2 with JSON accept header, friendly UA, and retry on throttle.
    Retries on 999/429/502/503 with exponential backoff.
    """
    import math, random

    url = "https://fantasysports.yahooapis.com/fantasy/v2" + path
    headers = {
        "Accept": "application/json",
        # A polite UA helps in some edge cases; keep it short and non-misleading
        "User-Agent": "yahoo-fantasy-agent/1.0 (+streamlit)",
    }

    last_err_txt = None
    for attempt in range(max_retries + 1):
        r = sc.session.get(url, headers=headers, timeout=30)

        if show_raw:
            with st.expander(f"HTTP debug: {path}", expanded=True):
                st.write({"url": url, "status_code": r.status_code, "content_type": r.headers.get("Content-Type", "")})
                st.code((r.text or "")[:1500])

        # OK path
        if 200 <= r.status_code < 300:
            body = (r.text or "").strip()
            if not body:
                raise RuntimeError("Empty response from Yahoo (no body).")
            try:
                return r.json()
            except Exception:
                ct = r.headers.get("Content-Type", "")
                raise RuntimeError(f"Non-JSON response ({r.status_code}, {ct}). First 400 chars: {body[:400]}")

        # Hard auth errors → no retry
        if r.status_code == 401:
            raise RuntimeError("401 Unauthorized: token invalid/expired or scope missing. Click Connect again.")
        if r.status_code == 403:
            raise RuntimeError("403 Forbidden: app not authorized for Fantasy scope or account lacks access.")

        # Throttle / transient → retry with backoff
        if r.status_code in (999, 429, 502, 503):
            last_err_txt = (r.text or "")[:200]
            if attempt < max_retries:
                # exponential backoff with jitter: 0.75s, 1.5s, 3.0s (approx)
                delay = (0.75 * (2 ** attempt)) * (0.8 + 0.4 * random.random())
                time.sleep(delay)
                continue
            else:
                raise RuntimeError(f"{r.status_code} error from Yahoo (after retries): {last_err_txt}")

        # Other 4xx/5xx → fail fast
        raise RuntimeError(f"{r.status_code} error from Yahoo: {(r.text or '')[:200]}")


# ─────────────── Parse users→games→leagues (numeric keys) ───────────────
def parse_games_leagues_v2(raw):
    fc = (raw or {}).get("fantasy_content", {})
    users = fc.get("users", {})

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
    L = None
    try:
        L = yfa.League(sc, league_key)
    except Exception as e:
        st.caption(f"resolve_team_key: League init failed, falling back to /users teams. ({e})")

    # Try library first, if League constructed
    if L is not None:
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


 # Fallback via /users;use_login=_

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

def _positions_from_settings_for_fa(settings: dict) -> List[str]:
    # Build FA queries from *starting* roster slots (exclude bench/IR)
    want = []
    for rp in (settings or {}).get("roster_positions", []):
        pos = (rp.get("position") or "").upper()
        cnt = int(rp.get("count") or 0)
        if cnt <= 0: 
            continue
        if pos in ("BN","IR"): 
            continue
        # FLEX slot → ask Yahoo by its literal name and also core eligibles
        if pos in ("W/R/T","WR/RB/TE","FLEX"):
            want += ["WR","RB","TE"]
        elif pos in ("SUPERFLEX","Q/W/R/T","Q/W/R/T"):
            want += ["QB","WR","RB","TE"]
        else:
            want.append(pos)

    # Yahoo FA endpoint understands these tokens. Dedup and keep only supported.
    supported = {"QB","RB","WR","TE","K","DEF","DL","LB","DB"}
    out = [p for p in dict.fromkeys(want) if p in supported]
    return out or ["QB","RB","WR","TE","K","DEF","DL","LB","DB"]

def free_agents(L: "yfa.League", positions: Optional[List[str]]=None, limit=60):
    if positions is None:
        try:
            s = L.settings() or {}
        except Exception:
            s = {}
        positions = _positions_from_settings_for_fa(s)

    fa = []
    for pos in positions:
        try:
            fa += L.free_agents(pos)
        except Exception:
            pass

    seen, out = set(), []
    for p in fa or []:
        pid = p.get("player_id")
        if not pid or pid in seen: 
            continue
        seen.add(pid)
        pts = p.get("points") or p.get("proj_points") or 0
        out.append({
            "player_id": pid,
            "name": p.get("name"),
            "position": (p.get("position") or "").upper(),
            "points": float(pts) if pts is not None else 0.0
        })
    out.sort(key=lambda x: x["points"], reverse=True)
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

# ───────────────── Logging helpers ─────────────────
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _json_default(o):
    try:
        if hasattr(o, "item"):
            return o.item()
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        return None

def _safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(d, default=_json_default))
    except Exception:
        out = {}
        for k, v in (d or {}).items():
            try:
                json.dumps(v, default=_json_default)
                out[k] = v
            except Exception:
                out[k] = str(v)
        return out

def _local_log_path_for_day(day: Optional[str] = None) -> Path:
    day = day or datetime.utcnow().strftime("%Y-%m-%d")
    return LOG_DIR / f"{day}.jsonl"

def _append_jsonl_local(record: Dict[str, Any]):
    p = _local_log_path_for_day()
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")

# Optional S3
AWS_OK, aws_err = False, None
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_REGION = os.getenv("AWS_REGION")
    S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    if AWS_REGION and S3_BUCKET:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.head_bucket(Bucket=S3_BUCKET)
        AWS_OK = True
except Exception as e:
    aws_err = str(e)

def _put_s3_log_record(record: Dict[str, Any]):
    if not (AWS_OK and LOG_S3_BUCKET):
        return
    try:
        ts = record.get("ts") or datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        day = (record.get("ts") or "").split("T")[0] or datetime.utcnow().strftime("%Y-%m-%d")
        rid = record.get("id") or str(uuid.uuid4())
        key = f"{LOG_S3_PREFIX}{day}/{ts}-{rid}.json"
        boto3.client("s3", region_name=AWS_REGION).put_object(
            Bucket=LOG_S3_BUCKET, Key=key,
            Body=json.dumps(record, default=_json_default).encode("utf-8"),
            ContentType="application/json"
        )
    except Exception as e:
        st.caption(f"S3 log mirror failed: {e}")

def log_event(kind: str,
              league_key: Optional[str] = None,
              week: Optional[int] = None,
              data: Optional[Dict[str, Any]] = None,
              ai: Optional[Dict[str, Any]] = None):
    record = {
        "id": str(uuid.uuid4()),
        "ts": _utc_now_iso(),
        "kind": kind,
        "league_key": league_key,
        "week": week,
        "data": _safe_dict(data or {}),
        "ai": _safe_dict(ai or {}),
        "app_version": "live-0.1",
    }
    try:
        _append_jsonl_local(record)
    except Exception as e:
        st.caption(f"Local log write failed: {e}")
    _put_s3_log_record(record)

def log_ai_qa(league_key: Optional[str],
              question: str,
              context: List[str],
              answer: str,
              structured: Dict[str, Any],
              model: str = "gpt-4o-mini",
              mode: str = "assistant"):
    ai = {
        "mode": mode,
        "model": model,
        "question": question,
        "context": context,
        "answer": answer,
        "structured": _safe_dict(structured),
    }
    log_event(kind="ai.qa", league_key=league_key, week=structured.get("week"), ai=ai)

# ───────────────── Snapshot (for AI/RAG & morning brief) ─────────────────
def league_current_week(settings_dict: dict) -> int:
    for k in ("current_week", "week", "standings_week"):
        v = settings_dict.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    return 1

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

    # Scoreboard
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
        fc = sb.get("fantasy_content", {})
        league = fc.get("league")
        scoreboard = None
        if isinstance(league, list):
            for el in league:
                if isinstance(el, dict) and "scoreboard" in el:
                    scoreboard = el["scoreboard"]; break
        elif isinstance(league, dict):
            scoreboard = league.get("scoreboard")

        if scoreboard:
            ms = scoreboard.get("matchups", {})
            for _, wrap in ms.items():
                if isinstance(wrap, dict) and "matchup" in wrap:
                    m = wrap["matchup"]
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

    # Transactions
    try:
        tx = yfs_get(sc, f"/league/{league_key}/transactions?format=json")
        (league_dir / "transactions.json").write_text(json.dumps(tx, indent=2))
    except Exception:
        tx = {}

    # Free agents snapshot
    try:
        L = yfa.League(sc, league_key)
        fas = free_agents(L, positions=None, limit=80)
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

# ───────────────── Morning Brief ─────────────────
def _mk_league_dir(league_key: str, week: int) -> Path:
    league_dir = DATA_DIR / "league" / league_key / f"week_{week}"
    ensure_dir(league_dir)
    return league_dir

def _safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def _summarize_roster_for_brief(roster_list, n=10):
    rows = []
    for p in roster_list or []:
        name = p.get("name")
        pos = ",".join(p.get("eligible_positions", []))
        pts = _safe_float(p.get("points") or p.get("proj_points"))
        status = p.get("status") or ""
        rows.append({"name": name, "pos": pos, "pts": pts, "status": status})
    rows.sort(key=lambda r: r["pts"], reverse=True)
    return rows[:n]

def _holes_by_position(roster_list):
    need = {"QB":0,"RB":0,"WR":0,"TE":0,"DEF":0}
    for p in roster_list or []:
        for pos in (p.get("eligible_positions") or []):
            pos = pos.upper()
            if pos in need:
                need[pos] += 1
    holes = [k for k,v in need.items() if v <= 1]
    return holes

def write_morning_brief(sc, league_key: str, snapshot_meta: dict):
    wk = int(snapshot_meta.get("week") or league_current_week(league_settings(sc, league_key)))
    league_dir = _mk_league_dir(league_key, wk)

    def _read(name, fallback):
        p = Path(snapshot_meta["dir"]) / name
        return json.loads(p.read_text()) if p.exists() else fallback

    your_roster = _read("your_roster.json", [])
    opp_roster  = _read("opp_roster.json", [])
    fas         = _read("free_agents.json", [])
    # tx is read but not used in the brief at the moment
    _ = _read("transactions.json", {})

    you_top = _summarize_roster_for_brief(your_roster, 10)
    opp_top = _summarize_roster_for_brief(opp_roster, 10)
    holes   = _holes_by_position(your_roster)

    top_fa = []
    for fa in fas[:25]:
        pos = (fa.get("position") or "").upper()
        if not holes or pos in holes:
            top_fa.append({
                "name": fa.get("name"),
                "pos": pos,
                "est_pts": _safe_float(fa.get("points")),
                "player_id": fa.get("player_id"),
            })
    top_fa.sort(key=lambda r: r["est_pts"], reverse=True)
    top_fa = top_fa[:10]

    brief = {
        "league_key": league_key,
        "week": wk,
        "you_team_key": snapshot_meta.get("you_team_key"),
        "opp_team_key": snapshot_meta.get("opp_team_key"),
        "you_team_name": snapshot_meta.get("you_team_name"),
        "opp_team_name": snapshot_meta.get("opp_team_name"),
        "your_top_players": you_top,
        "opponent_top_players": opp_top,
        "roster_holes": holes,
        "top_waiver_fits": top_fa,
    }
    (league_dir / "morning_brief.json").write_text(json.dumps(brief, indent=2))

    md = []
    md.append(f"# Morning Brief — Week {wk}")
    md.append(f"- **You:** {snapshot_meta.get('you_team_name') or snapshot_meta.get('you_team_key')}")
    md.append(f"- **Opponent:** {snapshot_meta.get('opp_team_name') or snapshot_meta.get('opp_team_key')}")
    md.append("")
    md.append("## Your Top Players")
    for r in you_top:
        md.append(f"- {r['name']} ({r['pos']}) — {r['pts']:.1f} pts  {('['+r['status']+']' if r.get('status') else '')}")
    md.append("")
    md.append("## Opponent Top Players")
    for r in opp_top:
        md.append(f"- {r['name']} ({r['pos']}) — {r['pts']:.1f} pts")
    md.append("")
    md.append("## Roster Holes")
    md.append("- " + (", ".join(holes) if holes else "No obvious holes"))
    md.append("")
    md.append("## Top Waiver Fits")
    if not top_fa:
        md.append("- (none)")
    else:
        for r in top_fa:
            md.append(f"- {r['name']} ({r['pos']}) — {r['est_pts']:.1f} pts (id={r['player_id']})")
    (league_dir / "morning_brief.md").write_text("\n".join(md))

    try:
        log_event("snapshot", league_key=league_key, week=wk,
                  data={"dir": str(league_dir), "files": ["morning_brief.json","morning_brief.md"]})
    except Exception as _e:
        st.caption(f"Brief logging failed: {_e}")

    return {"json": str(league_dir / "morning_brief.json"),
            "md": str(league_dir / "morning_brief.md")}

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

# ───────────────── AI & Cloud status ─────────────────
st.header("AI & Cloud")
st.write("OpenAI:", "✅ configured" if OPENAI_OK else "⚠️ not configured")

if OPENAI_OK and st.button("Run OpenAI smoke test"):
    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=2,
            temperature=0,
        )
        st.success(f"OpenAI OK → {resp.choices[0].message.content.strip()}")
    except Exception as e:
        st.error(f"OpenAI error: {e}")

st.write("AWS S3:", "✅ configured" if AWS_OK else "⚠️ not configured")
if not AWS_OK and aws_err:
    st.caption(aws_err)
if AWS_OK and st.button("Run S3 smoke test"):
    try:
        key = f"healthchecks/{int(time.time())}.txt"
        body = b"ok"
        import boto3 as _b
        _b.client("s3", region_name=AWS_REGION).put_object(Bucket=S3_BUCKET, Key=key, Body=body)
        st.success(f"Uploaded to s3://{S3_BUCKET}/{key}")
    except Exception as e:
        st.error(f"S3 test failed: {e}")

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
league = yfa.League(sc, league_key)  # ← EDIT 1: create reusable league object

# Show user teams table to clarify whether a team exists for this league
def render_user_teams(sc, league_key: str):
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
                if not isinstance(twrap, dict): continue
                tlist = twrap.get("team")
                if not isinstance(tlist, list): continue
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
        st.info("No teams found in /users;use_login=1/teams (normal if all leagues are pre-draft).")
        return

    df = pd.DataFrame(rows)
    df["matches_selected_league"] = df["league_key_guess"] == league_key
    st.write("#### User Teams (from /users;use_login=1/teams)")
    st.dataframe(df, use_container_width=True)

render_user_teams(sc, league_key)

# ─────────────── League status + pre-draft gate (robust) ───────────────
def get_league_settings_safe(sc, league_key: str) -> dict:
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
    
# ───────────────── Team mapping cache ─────────────────
if "team_key" not in st.session_state:
    st.session_state["team_key"] = None

if draft_status in ("postdraft","inseason"):
    # try to resolve and cache once per league selection
    resolved_team = resolve_team_key(sc, league_key)
    if resolved_team:
        if st.session_state.get("team_key") != resolved_team:
            st.session_state["team_key"] = resolved_team
        st.success(f"✅ Team mapped: {resolved_team}")
    else:
        st.warning("⚠️ Could not map a team for this league yet. If you just finished your draft, try re-authenticating or refresh in a few minutes.")
team_key = st.session_state.get("team_key")
# ───────────────── Quick Snapshot (always available) ─────────────────
st.markdown("### Quick Snapshot")
wk_guess = league_current_week(settings or {})
wk_input_cols = st.columns([1,1,2])
with wk_input_cols[0]:
    wk = st.number_input("Week", min_value=1, max_value=25, value=int(wk_guess or 1), step=1)
with wk_input_cols[1]:
    if st.button("📸 Snapshot league data now", key="snap_always"):
        try:
            meta = snapshot_now(sc, league_key, settings, week_override=int(wk))
            st.session_state["_last_snapshot_meta"] = meta
            st.success(f"Snapshot saved under: {meta['dir']}")
            st.json(meta)
        except Exception as e:
            st.error(f"Snapshot failed: {e}")

# If a snapshot exists, show where it lives (handy visual)
if "_last_snapshot_meta" in st.session_state:
    meta = st.session_state["_last_snapshot_meta"]
    with st.expander("Latest snapshot meta"):
        st.json(meta)


# Pre-draft gate
empty_reasons = []
if not settings:
    empty_reasons.append("League settings could not be read (rate limit or league not finalized).")
if draft_status in ("predraft", "unknown"):
    empty_reasons.append("League shows as pre-draft/unknown; teams/rosters aren’t published yet.")

try:
    L_probe = yfa.League(sc, league_key)
    teams_probe = L_probe.teams()
    if not teams_probe:
        empty_reasons.append("No teams returned by Yahoo yet.")
    else:
        tk_probe = resolve_team_key(sc, league_key)
        if tk_probe:
            try:
                T_probe = yfa.Team(sc, tk_probe)
                roster_probe = T_probe.roster()
                if not roster_probe:
                    empty_reasons.append("Your roster is empty right now (common pre-draft).")
            except Exception:
                empty_reasons.append("Could not fetch your roster (auth/scope or timing).")
        else:
            empty_reasons.append("Could not match your team_key in this league yet.")
except Exception:
    empty_reasons.append("League probe failed (network/auth).")

if empty_reasons:
    st.warning("ℹ️ **Why you may not see data yet:**\n\n- " + "\n- ".join(empty_reasons))
    st.caption("Tip: toggle **‘Show raw Yahoo API responses’** above to inspect raw JSON and confirm status.")

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

# --- AUTOPILOT PANEL (paste after sc, league, team_key are set) ---
import json
import pandas as pd
st.subheader("🤖 Autopilot")

# ensure a League object exists here (it's fine to instantiate again)
league = yfa.League(sc, league_key)

# league_id is a property, not a callable
# Use the selected Yahoo league_key as the stable ID for state files.
# --- Autopilot ID + state (drop-in replacement) ---

def _league_id_for_state(league_key: str) -> str:
    """
    Make a stable, file-safe id from a Yahoo league_key.
    Example: '423.l.12345' -> '12345'; else fall back to the full key.
    """
    if ".l." in league_key:
        tail = league_key.split(".l.", 1)[1]
        # if a team suffix exists (e.g., ".t.8"), strip it
        return tail.split(".", 1)[0]
    return league_key

league_id = _league_id_for_state(league_key)

state = load_state(league_id) or {}

# Build / edit policy in the UI
policy_dict = state.get("policy", DEFAULT_POLICY.__dict__)
policy = AutopilotPolicy(**policy_dict)

c1, c2, c3 = st.columns(3)
with c1:
    policy.autopilot_on = st.toggle("Enable Autopilot", policy.autopilot_on)
with c2:
    policy.require_approval = st.toggle("Require Approval", policy.require_approval)
with c3:
    policy.max_faab_bid = st.number_input("Max FAAB per player", 0, 100, policy.max_faab_bid)

policy.approval_threshold_epar = st.slider("Auto-approve if EPAR ≥", 0.0, 5.0, policy.approval_threshold_epar, 0.1)
policy.variance_style = st.selectbox("Lineup risk style", ["auto","floor","ceiling"],
                                     index=["auto","floor","ceiling"].index(policy.variance_style))

# IMPORTANT: coerce non-JSON types *before* saving
safe_policy = dict(policy.__dict__)
if isinstance(safe_policy.get("playoff_weeks"), set):
    safe_policy["playoff_weeks"] = sorted(safe_policy["playoff_weeks"])

state["policy"] = safe_policy
save_state(league_id, state)


# --- Build inputs (swap these stubs with your real helpers) ---
# --- Build inputs (wired to your utils + Yahoo) ---

def _safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def build_roster_view(league: "yfa.League", team_key: str) -> list[dict]:
    # Use your roster_df helper, then enrich with your signals/scoring if available
    df = roster_df(sc, league_key)
    if df.empty:
        return []
    try:
        enriched = apply_enrichment(df.copy(), signals={})  # thread real signals later if you want
    except Exception:
        enriched = df.copy()

    # Ensure required columns
    if "proj_points" not in enriched.columns:
        enriched["proj_points"] = enriched.get("points", pd.Series([0]*len(enriched))).astype(float)
    if "floor" not in enriched.columns:
        enriched["floor"] = enriched["proj_points"] * 0.8
    if "ceiling" not in enriched.columns:
        enriched["ceiling"] = enriched["proj_points"] * 1.25
    if "eligible_positions" not in enriched.columns:
        enriched["eligible_positions"] = ""

    rows = []
    for _, r in enriched.iterrows():
        rows.append({
            "player_id": str(r.get("player_id")),
            "player_name": r.get("name"),
            "pos": (r.get("eligible_positions") or "").split(",")[0] if r.get("eligible_positions") else "",
            "proj_points": _safe_float(r.get("proj_points")),
            "floor": _safe_float(r.get("floor")),
            "ceiling": _safe_float(r.get("ceiling")),
            "bye": r.get("bye") or None,
            "injury_status": r.get("status") or "",
            "tags": {
                "eligible": r.get("eligible_positions"),
                "selected": r.get("selected_position"),
            }
        })
    return rows

def build_fa_view(league: "yfa.League", limit: int = 25) -> list[dict]:
    # Use your free_agents + rank_waivers to produce a shortlist with EPAR
    try:
        settings_full = settings or league_settings(sc, league_key)
        team = roster_df(sc, league_key)
        pool = free_agents(league, positions=None, limit=limit*3)  # take extra, then trim
        df_ranked = rank_waivers(pool, team, league_current_week(settings_full), settings_full)
        if not isinstance(df_ranked, pd.DataFrame) or df_ranked.empty:
            return []
        df_ranked = df_ranked.head(limit).copy()

        def _pick_epar(row):
            for k in ("epar", "EPAR", "epar_3wk", "epar_next", "epar_ros", "points_gain"):
                if k in row and pd.notnull(row[k]):
                    return _safe_float(row[k])
            return 0.0

        out = []
        for _, r in df_ranked.iterrows():
            out.append({
                "player_id": str(r.get("player_id")),
                "player_name": r.get("name") or r.get("player_name"),
                "position": (r.get("position") or r.get("pos") or "").upper(),
                "epar": _pick_epar(r),
                "availability": "FA/Waivers",
                "reason": r.get("why") or r.get("notes") or "",
            })
        return out
    except Exception:
        return []

def build_schedule_view(league: "yfa.League", team_key: str) -> tuple[dict, str]:
    # Return (schedule_view, opponent_name) for current week
    wk = league_current_week(settings or {})
    sb = get_scoreboard(sc, league_key, wk) or {}
    you_tk = resolve_team_key(sc, league_key)
    opp_name = "TBD"
    opp_tk = None
    try:
        fc = (sb or {}).get("fantasy_content", {})
        lg = fc.get("league")
        scoreboard = lg.get("scoreboard") if isinstance(lg, dict) else None
        if not scoreboard and isinstance(lg, list):
            for el in lg:
                if isinstance(el, dict) and "scoreboard" in el:
                    scoreboard = el["scoreboard"]; break
        if scoreboard:
            ms = scoreboard.get("matchups", {})
            for _, wrap in ms.items():
                m = wrap.get("matchup") if isinstance(wrap, dict) else None
                teams_c = None
                if isinstance(m, dict): teams_c = m.get("teams")
                if not teams_c and isinstance(m, list):
                    for z in m:
                        if isinstance(z, dict) and "teams" in z:
                            teams_c = z["teams"]; break
                if teams_c:
                    tkeys, tnames = [], []
                    for _, tw in teams_c.items():
                        if isinstance(tw, dict) and "team" in tw:
                            team = tw["team"]
                            tk = nm = None
                            if isinstance(team, dict):
                                tk = team.get("team_key"); nm = team.get("name")
                            elif isinstance(team, list):
                                for kv in team:
                                    if isinstance(kv, dict):
                                        if "team_key" in kv and not tk: tk = kv["team_key"]
                                        if "name" in kv and not nm: nm = kv["name"]
                            if tk:
                                tkeys.append(tk); tnames.append(nm or tk)
                    if you_tk and you_tk in tkeys and len(tkeys) == 2:
                        idx = tkeys.index(you_tk)
                        opp_tk = tkeys[1-idx]; opp_name = tnames[1-idx]
    except Exception:
        pass
    return ({"week": wk, "opponent_team_key": opp_tk, "opponent_name": opp_name}, opp_name)

def build_injuries_view(league: "yfa.League") -> dict:
    try:
        inj = league.injuries()
        return inj if isinstance(inj, (dict, list)) else {}
    except Exception:
        return {}

week = st.session_state.get("week") or 1
league_rules = {"scoring": "half-ppr"}  # replace with league.settings() if available
record = "0-0"  # fill from standings if you have it
schedule_view, opponent_name = build_schedule_view(league, team_key)
roster_view = build_roster_view(league, team_key)
fa_view = build_fa_view(league)
injuries_view = build_injuries_view(league)

if st.button("Generate Weekly AI Plan"):
    plan = plan_week(
        week=week,
        league_rules=league_rules,
        team_key=team_key,
        record=record,
        opponent_name=opponent_name,
        playoff_weeks=sorted(list(policy.playoff_weeks)),
        policy=policy.__dict__,
        roster=roster_view,
        free_agents=fa_view,
        schedule=schedule_view,
        injuries=injuries_view,
    )
    st.session_state["pending_plan"] = plan
    st.success("Weekly plan generated.")
    with st.expander("View plan JSON"):
        st.json(plan)
    if "start_sit" in plan:
        st.markdown("**Proposed Start/Sit**")
        st.dataframe(pd.DataFrame(plan["start_sit"].get("moves", [])))
    if "waivers" in plan:
        st.markdown("**Waiver Claims**")
        st.dataframe(pd.DataFrame(plan["waivers"]))
    if "trades" in plan and plan["trades"]:
        st.markdown("**Trade Ideas**")
        st.dataframe(pd.DataFrame(plan["trades"]))

if st.button("Approve & Execute Plan"):
    plan = st.session_state.get("pending_plan")
    if not plan:
        st.warning("Generate a plan first.")
    else:
        submit_waiver_queue(sc, league, team_key, plan.get("waivers", []))
        if "start_sit" in plan:
            set_lineup(sc, league, team_key, plan["start_sit"])
        for offer in plan.get("trades", []):
            send_trade_offer(sc, league, offer)
        st.success("Draft transactions logged. Swap executor stubs for real Yahoo calls when ready.")


# PRE-DRAFT → Draft Assistant only
# Allow both postdraft and inseason to pass through
if draft_status not in ("postdraft", "inseason"):
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
                    labels = ["T1" if i in t1 else "T2" if i in t2 else "T3+" for i in g.index]
                    tiers.append(g.assign(tier=labels))
                tiers_df = pd.concat(tiers).sort_values(["position","tier","proj_points"], ascending=[True, True, False])
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)

                st.divider()
                st.subheader("AI Sandbox (works pre-draft)")
                st.caption("Ask OpenAI about this board. Uses the uploaded CSV + basic heuristics.")
                q = st.text_area("Question", "Who are the best RB values after round 6?")

                def select_context(query: str, summaries: list[str], k: int = 3) -> list[str]:
                    if not query:
                        return summaries[:k]
                    ql = query.lower().split()
                    scored = []
                    for s in summaries:
                        score = sum(s.lower().count(tok) for tok in ql)
                        scored.append((score, s))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    out = [s for sc, s in scored if sc > 0][:k]
                    return out or summaries[:k]

                if st.button("Ask OpenAI about this CSV"):
                    summaries = []
                    try:
                        for pos_name, grp in dfp.groupby(dfp["position"].str.upper()):
                            top5 = grp.sort_values("proj_points", ascending=False).head(5)
                            names = ", ".join(top5["name"].tolist())
                            summaries.append(f"Top {pos_name}: {names}")
                        if "adp" in dfp.columns:
                            sleepers = (
                                dfp[dfp["proj_points"] >= dfp["proj_points"].quantile(0.75)]
                                .sort_values("adp", ascending=False)
                                .head(8)["name"].tolist()
                            )
                            if sleepers:
                                summaries.append("Possible sleepers (high proj, late ADP): " + ", ".join(sleepers))
                    except Exception as e:
                        summaries.append(f"CSV summaries unavailable (parse issue: {e})")

                    chosen = select_context(q, summaries, k=3)
                    if OPENAI_OK:
                        try:
                            sys = "You are a concise fantasy football assistant. Be specific."
                            struct_txt = json.dumps({"mode":"predraft","rows":int(len(dfp))}, ensure_ascii=False)
                            ctx = "\n".join(f"- {s}" for s in chosen)
                            prompt = f"CONTEXT:\n{ctx}\n\nSTRUCTURED:\n{struct_txt}\n\nQUESTION:\n{q}"
                            resp = OPENAI_CLIENT.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
                                max_tokens=400, temperature=0.2
                            )
                            st.markdown("### AI Answer")
                            st.write(resp.choices[0].message.content.strip())
                        except Exception as e:
                            st.warning(f"OpenAI error: {e}")
                    else:
                        st.info("OpenAI key not configured; skipping AI.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Tip: export projections from your favorite site and upload here.")
    st.stop()

# ───────────────────────────────────────────────────────────
# POSTDRAFT → full tabs (fixed indices)
tab_draft, tab_roster, tab_start, tab_waivers, tab_opponent, tab_trades, tab_ai, tab_opp_scout, tab_sched, tab_logs = st.tabs(
    ["Draft Assistant", "Roster", "Start/Sit", "Waivers", "Opponent", "Trades", "AI Assistant", "Opponent Scouting", "Scheduler", "Logs"]
)


# ───────────────── Draft Assistant (postdraft) ─────────────────
with tab_draft:
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
                    labels = ["T1" if i in t1 else "T2" if i in t2 else "T3+" for i in g.index]
                    tiers.append(g.assign(tier=labels))
                tiers_df = pd.concat(tiers).sort_values(["position","tier","proj_points"], ascending=[True, True, False])
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.caption("Upload projections/ADP to populate tiers.")

# ───────────────── Roster ─────────────────
with tab_roster:
    st.subheader("Current Roster")
    df_roster = roster_df(sc, league_key)
    if df_roster.empty:
        st.info("No roster found. If your draft just ended, give Yahoo a little time then refresh.")
    else:
        st.dataframe(df_roster, use_container_width=True)

# ───── shared helper for lineup deltas (moved to module scope) ─────
def lineup_change_deltas(current_df: pd.DataFrame | None,
                         picks: list[tuple[str, pd.Series | dict]]):
    """
    Compare current starters vs optimizer picks.
    Returns (gained_ids, benched_ids).
    """
    # Gather desired starter ids from picks
    want = set()
    for _, p in (picks or []):
        if isinstance(p, pd.Series):
            pid = p.get("player_id") or p.get("id")
        elif isinstance(p, dict):
            pid = p.get("player_id") or p.get("id")
        else:
            pid = None
        if pid is not None:
            want.add(str(pid))

    # Iterate current starters safely (no DataFrame truthiness!)
    df = current_df if isinstance(current_df, pd.DataFrame) else pd.DataFrame()
    now_starters = set()
    for _, r in df.iterrows():
        sel = str((r.get("selected_position") if hasattr(r, "get") else r["selected_position"]) or "").upper()
        if sel and sel not in ("BN", "IR", "NA"):
            pid = r.get("player_id") if hasattr(r, "get") else r["player_id"]
            if pid is not None:
                now_starters.add(str(pid))

    return list(want - now_starters), list(now_starters - want)


# ───────────────── Start/Sit (Optimizer v1.5 + optional signals) ─────────────────
with tab_start:
    st.subheader("Recommended Starters (optimizer v1.5)")

    settings_full = settings or league_settings(sc, league_key)

    # slots from league settings
    slots = []
    for rp in settings_full.get("roster_positions", []):
        pos, cnt = rp.get("position"), int(rp.get("count", 0))
        if pos and cnt > 0:
            slots += [pos]*cnt

    pool_df = roster_df(sc, league_key)
    if pool_df.empty:
        st.info("No roster found (Yahoo may still be finalizing teams).")
    else:
        # Optional signals.csv (player_id keyed) via local path OR upload
        signals = {}

        st.markdown("**Optional signals for optimizer**")
        sig_col1, sig_col2 = st.columns([1,1])
        with sig_col1:
            sig_path = st.text_input("Signals CSV path (optional)", value="signals.csv")
        with sig_col2:
            sig_file = st.file_uploader("...or upload signals CSV", type=["csv"], key="signals_upload_startsit")

        try:
            if sig_file is not None:
                s = pd.read_csv(sig_file)
            elif sig_path and Path(sig_path).exists():
                s = pd.read_csv(sig_path)
            else:
                s = None

            if s is not None:
                cols_lower = {c.lower(): c for c in s.columns}
                pid_col = cols_lower.get("player_id")
                if pid_col:
                    for _, r in s.iterrows():
                        pid = str(r[pid_col])
                        if pid and pid != "nan":
                            signals[pid] = {k: r[k] for k in s.columns if k != pid_col}
                    st.success(f"Loaded {len(signals)} signals" + (" from upload" if sig_file is not None else f" from {sig_path}"))
                else:
                    st.warning("Signals CSV must include a 'player_id' column.")
        except Exception as e:
            st.warning(f"Could not read signals CSV: {e}")

        # Use your utils: apply_enrichment + optimize_lineup
        enriched_pool = apply_enrichment(pool_df.copy(), signals or {})


     # ---------- BASELINE (run BEFORE optimize_lineup) ----------
    def _all_zero(series) -> bool:
        try:
            return pd.Series(series).fillna(0).abs().sum() == 0
        except Exception:
            return True

    # If enrichment/points are all zero and no signals were loaded, set naïve baselines per position.
    if _all_zero(enriched_pool.get("points")) and not signals:
        baseline = {
            "QB": 16, "RB": 12, "WR": 12, "TE": 8, "K": 7, "DEF": 7,
            "DL": 6, "LB": 6, "DB": 6, "D": 6,  # IDP
        }
        def _guess_pos(row):
            elig = (row.get("eligible_positions") or "").upper().split(",")
            if not elig or elig == [""]:
                return (row.get("selected_position") or "").upper()
            # prefer non-bench elig
            for p in elig:
                if p not in ("BN","IR","NA","W/R/T","WR/RB/TE","Q/W/R/T","FLEX"):
                    return p
            return elig[0]
        enriched_pool["score"] = enriched_pool.apply(lambda r: float(baseline.get(_guess_pos(r), 6)), axis=1)
    else:
        # if your apply_enrichment sets 'score' already, keep it; otherwise fall back to 'points'
        if "score" not in enriched_pool.columns:
            enriched_pool["score"] = enriched_pool["points"].fillna(0).astype(float)
    # -----------------------------------------------------------

    # Now optimize AFTER score exists
    starters, bench_df = optimize_lineup(enriched_pool.copy(), slots)

    st.write("### Starters")
    if not starters:
        st.info("No starter picks could be computed.")
    else:
        for slot, p in starters:
            st.write(f"- **{slot}** → {p['name']} (eligible: {p.get('eligible_positions')})")

    st.write("### Bench (top 10 by score if provided)")
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        st.dataframe(bench_df.head(10), use_container_width=True)

    gained, benched = lineup_change_deltas(pool_df, starters)
    with st.expander("Change deltas vs current lineup"):
        st.write({"promoted": gained, "benched": benched})

    if st.button("Log this recommendation"):
        try:
            starters_payload = [{"slot": slot, **{k: (v.item() if hasattr(v,"item") else v) for k,v in p.to_dict().items()}} for slot, p in starters]
            bench_payload = []
            if isinstance(bench_df, pd.DataFrame):
                bench_payload = [{k: (v.item() if hasattr(v,"item") else v) for k,v in r.items()} for _, r in bench_df.head(10).iterrows()]
            log_event("lineup.reco",
                        league_key=league_key,
                        week=league_current_week(settings_full or {}),
                        data={"starters": starters_payload, "bench_top10": bench_payload})
            st.success("Recommendation logged")
        except Exception as _e:
            st.caption(f"Logging (lineup.reco) failed: {_e}")
# ───────────────── Waivers (v2) ─────────────────
with tab_waivers:
    st.subheader("Waiver targets & Approvals")
    df_ranked = pd.DataFrame()  # ← add this line
    try:
        L = yfa.League(sc, league_key)
        cands = free_agents(L)
    except Exception as e:
        st.error(f"Failed to get free agents: {e}")
        cands = []

    team_df = roster_df(sc, league_key)
    faab_budget = settings.get("faab_budget") or 100
    current_week = league_current_week(settings or {})

    if not cands or team_df.empty:
        st.info("Need both roster and free agents to rank waivers.")
    else:
        # Use your utils. rank_waivers should return a DataFrame.
        df_ranked = rank_waivers(cands, team_df, current_week, settings)
        st.write("### Ranked Waiver Candidates")
        st.dataframe(df_ranked, use_container_width=True)

        queue = multi_claim_queue(df_ranked, max_claims=5)
        st.write("### Proposed Claims (top 5)")
        st.json(queue)
    st.divider()
    st.subheader("Free Agent Watch (from latest snapshot)")

    snap = st.session_state.get("_last_snapshot_meta")
    if not snap:
        st.caption("No snapshot yet. Click **📸 Snapshot league data now** in the AI tab to cache free agents.")
    else:
        try:
            fa_path = Path(snap["dir"]) / "free_agents.json"
            if fa_path.exists():
                fa_list = json.loads(fa_path.read_text())
            else:
                # Fallback to live pull if file missing
                L = yfa.League(sc, league_key)
                fa_list = free_agents(L, positions=None, limit=80)
            if not fa_list:
                st.caption("No free agents found in snapshot.")
            else:
                df_fa = pd.DataFrame(fa_list)
                # quick filters
                cols = st.columns(3)
                with cols[0]:
                    pos_pick = st.multiselect("Positions", sorted(df_fa["position"].dropna().unique().tolist()))
                with cols[1]:
                    name_q = st.text_input("Name contains", "")
                with cols[2]:
                    top_n = st.number_input("Show top N", 10, 200, value=50)

                view = df_fa.copy()
                if pos_pick:
                    view = view[view["position"].isin(pos_pick)]
                if name_q.strip():
                    view = view[view["name"].str.contains(name_q.strip(), case=False, na=False)]
                view = view.sort_values("points", ascending=False).head(int(top_n))
                st.dataframe(view, use_container_width=True)
        except Exception as e:
            st.caption(f"Free Agent Watch unavailable: {e}")

        st.divider()
        st.write("### Approve & Execute (single claim)")

        if df_ranked.empty:
            st.caption("No ranked waivers available yet. Make sure your roster and free agents are populated.")
        else:
            max_idx = max(0, len(df_ranked) - 1)
            idx = st.number_input("Row to add (0-based from table above)",
                                min_value=0, max_value=max_idx, value=0)
            pick = df_ranked.iloc[int(idx)]
            drop_pid = st.text_input("Player ID to drop (optional)")
            default_bid = int(pick.get("suggested_faab", 0) or 0)
            faab_bid = st.number_input("FAAB bid (optional)", min_value=0, max_value=300, value=default_bid)

            if st.button("Approve transaction"):
                try:
                    result = execute_add_drop(sc, league_key,
                                            add_pid=str(pick["player_id"]),
                                            drop_pid=(drop_pid or None),
                                            faab_bid=int(faab_bid) if faab_bid else None)
                    if result.get("status") == "ok":
                        st.success("Transaction submitted ✅")
                        st.json(result.get("details"))

                        # log
                        try:
                            pick_dict = {k: (v.item() if hasattr(v, "item") else v) for k, v in pick.to_dict().items()}
                            log_event("waiver.approved",
                                    league_key=league_key,
                                    week=league_current_week(settings or {}),
                                    data={
                                        "add_player": pick_dict,
                                        "drop_player_id": (drop_pid or None),
                                        "faab_bid": int(faab_bid) if faab_bid else None,
                                        "result": _safe_dict(result),
                                    })
                        except Exception as _e:
                            st.caption(f"Logging (waiver.approved) failed: {_e}")
                    else:
                        st.error("Transaction failed"); st.json(result)
                except Exception as e:
                    st.error(f"Execution error: {e}")


# ───────────────── Opponent (overview) ─────────────────
with tab_opponent:
    st.subheader("Opponent (Overview)")
    def _get_current_opponent(sc, league_key: str, settings: dict):
        you_tk = resolve_team_key(sc, league_key)
        opp_tk = your_name = opp_name = None
        sb = get_scoreboard(sc, league_key, league_current_week(settings or {})) or {}
        try:
            fc = (sb or {}).get("fantasy_content", {})
            league = fc.get("league")
            scoreboard = league.get("scoreboard") if isinstance(league, dict) else None
            if not scoreboard and isinstance(league, list):
                for el in league:
                    if isinstance(el, dict) and "scoreboard" in el:
                        scoreboard = el["scoreboard"]; break
            if scoreboard:
                ms = scoreboard.get("matchups", {})
                for _, wrap in ms.items():
                    if isinstance(wrap, dict) and "matchup" in wrap:
                        m = wrap["matchup"]
                        teams_c = m.get("teams") if isinstance(m, dict) else None
                        if not teams_c and isinstance(m, list):
                            for x in m:
                                if isinstance(x, dict) and "teams" in x:
                                    teams_c = x["teams"]; break
                        if teams_c:
                            tkeys, tnames = [], []
                            for _, tw in teams_c.items():
                                if isinstance(tw, dict) and "team" in tw:
                                    team = tw["team"]
                                    tk = nm = None
                                    if isinstance(team, dict):
                                        tk = team.get("team_key"); nm = team.get("name")
                                    elif isinstance(team, list):
                                        for kv in team:
                                            if isinstance(kv, dict):
                                                if "team_key" in kv and not tk: tk = kv["team_key"]
                                                if "name" in kv and not nm: nm = kv["name"]
                                    if tk:
                                        tkeys.append(tk); tnames.append(nm or tk)
                            you_tk_local = you_tk
                            if you_tk_local and you_tk_local in tkeys and len(tkeys) == 2:
                                idx = tkeys.index(you_tk_local)
                                return you_tk_local, tnames[idx], tkeys[1-idx], tnames[1-idx]
        except Exception:
            pass
        return you_tk, None, None, None

    you_tk, your_name, opp_tk, opp_name = _get_current_opponent(sc, league_key, settings)
    # after trying _get_current_opponent(...)
    if not opp_tk:
        # fallback: manual pick
        try:
            league = yfa.League(sc, league_key)
            teams = []
            for t in league.teams():
                tk = nm = None
                if isinstance(t, dict):
                    tk = t.get("team_key"); nm = t.get("name") or tk
                elif isinstance(t, list):
                    for kv in t:
                        if isinstance(kv, dict):
                            if "team_key" in kv: tk = kv["team_key"]
                            if "name" in kv and not nm: nm = kv["name"]
                if tk: teams.append({"team_key": tk, "name": nm or tk})
            name_map = { t["name"]: t["team_key"] for t in teams if t["team_key"] != you_tk }
            pick = st.selectbox("Pick your Week-1 opponent (scoreboard not published yet)", list(name_map.keys()))
            opp_tk = name_map.get(pick)
            opp_name = pick
        except Exception:
            st.info("Couldn’t list teams to pick opponent.")

    if not opp_tk:
        st.info("Couldn’t identify your current opponent yet.")
    else:
        st.markdown(f"**You:** {your_name or you_tk}  |  **Opponent:** {opp_name or opp_tk}")
        opp_roster = team_roster_raw(sc, opp_tk) or []
        def _list_to_df(roster_list):
            rows = []
            for p in roster_list or []:
                elig = p.get("eligible_positions") or []
                if isinstance(elig, list): elig = ",".join(elig)
                rows.append({
                    "player_id": p.get("player_id"),
                    "name": p.get("name"),
                    "status": p.get("status"),
                    "eligible_positions": elig or "",
                    "selected_position": p.get("selected_position"),
                    "points": (p.get("points") or p.get("proj_points") or 0)
                })
            return pd.DataFrame(rows)
        df_opp = _list_to_df(opp_roster)
        if df_opp.empty:
            st.info("Opponent roster not available yet.")
        else:
            st.dataframe(df_opp, use_container_width=True)

# ───────────────── Trades (Evaluator) ─────────────────
with tab_trades:
    st.subheader("Trade Evaluator")
    st.caption("Pick players to offer/request. Uses current points/projections. For better accuracy, upload projections in Draft Assistant.")
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

# ───────────────── Opponent Scouting (tab) ─────────────────
with tab_opp_scout:
    st.subheader("Opponent Scouting")

    # --- helper: find current opponent (this week) ---
    def _get_current_opponent(sc, league_key: str, settings: dict):
        """Return (you_team_key, you_name, opp_team_key, opp_name) or (None,..) if not found."""
        you_tk = resolve_team_key(sc, league_key)
        opp_tk = your_name = opp_name = None
        sb = get_scoreboard(sc, league_key, league_current_week(settings or {})) or {}
        try:
            fc = (sb or {}).get("fantasy_content", {})
            league = fc.get("league")
            scoreboard = None
            if isinstance(league, list):
                for el in league:
                    if isinstance(el, dict) and "scoreboard" in el:
                        scoreboard = el["scoreboard"]; break
            elif isinstance(league, dict):
                scoreboard = league.get("scoreboard")
            if scoreboard:
                ms = scoreboard.get("matchups", {})
                for _, wrap in ms.items():
                    if isinstance(wrap, dict) and "matchup" in wrap:
                        m = wrap["matchup"]
                        teams_c = None
                        if isinstance(m, list):
                            for x in m:
                                if isinstance(x, dict) and "teams" in x:
                                    teams_c = x["teams"]; break
                        elif isinstance(m, dict):
                            teams_c = m.get("teams")
                        if teams_c:
                            tkeys, tnames = [], []
                            for _, tw in teams_c.items():
                                if isinstance(tw, dict) and "team" in tw:
                                    team = tw["team"]
                                    tk = nm = None
                                    if isinstance(team, list):
                                        for kv in team:
                                            if isinstance(kv, dict):
                                                if "team_key" in kv and not tk: tk = kv["team_key"]
                                                if "name" in kv and not nm: nm = kv["name"]
                                    elif isinstance(team, dict):
                                        tk = team.get("team_key")
                                        nm = team.get("name")
                                    if tk:
                                        tkeys.append(tk); tnames.append(nm or tk)
                            if you_tk and you_tk in tkeys and len(tkeys) == 2:
                                idx = tkeys.index(you_tk)
                                return you_tk, tnames[idx], tkeys[1-idx], tnames[1-idx]
        except Exception:
            pass
        return you_tk, None, None, None

    # --- helpers for optimizer/deltas on opponent roster ---
    def _list_to_df(roster_list):
        rows = []
        for p in roster_list or []:
            elig = p.get("eligible_positions") or []
            if isinstance(elig, list): elig = ",".join(elig)
            rows.append({
                "player_id": p.get("player_id"),
                "name": p.get("name"),
                "status": p.get("status"),
                "eligible_positions": elig or "",
                "selected_position": p.get("selected_position"),
                "points": (p.get("points") or p.get("proj_points") or 0)
            })
        return pd.DataFrame(rows)

    def _slots_from_settings(settings_dict: dict):
        slots = []
        for rp in (settings_dict or {}).get("roster_positions", []):
            pos, cnt = rp.get("position"), int(rp.get("count", 0))
            if pos and cnt > 0: slots += [pos]*cnt
        return slots

    # identify opponent
    you_tk, your_name, opp_tk, opp_name = _get_current_opponent(sc, league_key, settings)
    if not opp_tk:
        st.info("Couldn’t identify your current opponent yet (common pre-draft or before schedule is posted).")
        st.stop()

    # pull live data
    your_roster = team_roster_raw(sc, you_tk) if you_tk else []
    opp_roster  = team_roster_raw(sc, opp_tk) if opp_tk else []

    try:
        L = yfa.League(sc, league_key)
        fa_pool = free_agents(L, positions=None, limit=120)
    except Exception:
        fa_pool = []

    # opponent weak spots + block moves (utils)
    rep = scout_weak_spots(opp_roster)              # expects {"weak": [...], "counts": {...}, ...}
    blocks = recommend_blocks(rep.get("weak", []), fa_pool)

    # Optional signals.csv just for opponent optimizer (independent of Start/Sit tab)
    st.caption("Optional: upload a signals CSV to enrich opponent projections (must include player_id).")
    sig_file_opp = st.file_uploader("Upload opponent signals CSV", type=["csv"], key="signals_upload_opp")
    signals_opp = {}
    if sig_file_opp is not None:
        try:
            sdf = pd.read_csv(sig_file_opp)
            cols_lower = {c.lower(): c for c in sdf.columns}
            if "player_id" in cols_lower:
                pid_col = cols_lower["player_id"]
                for _, r in sdf.iterrows():
                    pid = str(r[pid_col])
                    if pid and pid != "nan":
                        signals_opp[pid] = {k: r[k] for k in sdf.columns if k != pid_col}
                st.success(f"Loaded {len(signals_opp)} signal rows for opponent enrichment.")
            else:
                st.warning("Signals CSV must include a 'player_id' column.")
        except Exception as e:
            st.warning(f"Could not read uploaded signals CSV: {e}")

    # opponent optimizer: what *they* should start
    st.markdown(f"**You:** {your_name or you_tk}  |  **Opponent:** {opp_name or opp_tk}")
    opp_df = _list_to_df(opp_roster)
    slots = _slots_from_settings(settings or league_settings(sc, league_key))

    if opp_df.empty or not slots:
        st.info("Opponent roster or league slots not available yet.")
    else:
        # apply enrichment & optimize (reuse the same optimizer API you use)
        opp_pool_enriched = apply_enrichment(opp_df.copy(), signals_opp or {})
        picks, opp_bench = optimize_lineup(opp_pool_enriched.copy(), slots)

        # compute deltas vs current opponent starters
        gained, benched = lineup_change_deltas(opp_df, picks)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Opponent optimal starters (our estimate)**")
            rows = []
            for slot, p in picks:
                rows.append({
                    "slot": slot,
                    "player_id": p["player_id"],
                    "name": p["name"],
                    "status": p.get("status"),
                    "eligible_positions": p.get("eligible_positions"),
                    "est_score": float(p.get("score") if "score" in p else p.get("points", 0) or 0)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with colB:
            st.markdown("**Opponent bench (best → worst)**")
            st.dataframe(opp_bench[["player_id","name","eligible_positions","status","score"]].head(12), use_container_width=True)

        st.markdown("### Change deltas (if they optimize)")
        colC, colD = st.columns(2)
        with colC:
            st.write("**Players they would add to starting lineup**")
            if gained:
                st.dataframe(opp_df[opp_df["player_id"].isin(gained)][["player_id","name","eligible_positions","status","points"]], use_container_width=True)
            else:
                st.caption("No upgrades vs current starters detected.")
        with colD:
            st.write("**Players they would bench**")
            if benched:
                st.dataframe(opp_df[opp_df["player_id"].isin(benched)][["player_id","name","eligible_positions","status","points"]], use_container_width=True)
            else:
                st.caption("No benches detected.")

    st.divider()
    st.markdown("### Weak spots & Block Moves")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Opponent weak spots**")
        st.json(rep)
    with col2:
        st.markdown("**Recommended blocks (top)**")
        if blocks:
            df_blocks = pd.DataFrame(blocks)
            st.dataframe(df_blocks, use_container_width=True)
        else:
            st.info("No obvious block opportunities right now.")

    st.divider()
    st.markdown("### Quick Block Waiver")
    if not fa_pool:
        st.caption("Free agents list empty or unavailable.")
    else:
        pick_opts = []
        if blocks:
            for b in blocks[:15]:
                pick_opts.append(f"{b.get('player_id')} — {b.get('name')} ({b.get('pos') or b.get('position')})")
        add_choice = st.selectbox("Pick a block target", pick_opts) if pick_opts else ""
        add_pid_default = (add_choice.split(" — ")[0] if add_choice else "")
        add_pid = st.text_input("Player ID to add", value=add_pid_default)

        drop_pid = st.text_input("Player ID to drop (optional)")
        faab_budget = settings.get("faab_budget") or 100
        faab_bid = st.number_input("FAAB bid", min_value=0, max_value=300, value=min(7, int(faab_budget*0.07)))

        if st.button("Place block as waiver"):
            if not add_pid.strip():
                st.warning("Provide a player_id to add.")
            else:
                try:
                    result = execute_add_drop(sc, league_key, add_pid=str(add_pid).strip(),
                                              drop_pid=(drop_pid.strip() or None),
                                              faab_bid=int(faab_bid))
                    if result.get("status") == "ok":
                        st.success("Block waiver submitted ✅")
                        st.json(result.get("details"))
                        try:
                            log_event("waiver.approved",
                                      league_key=league_key,
                                      week=league_current_week(settings or {}),
                                      data={
                                          "add_player": {"player_id": add_pid},
                                          "drop_player_id": (drop_pid or None),
                                          "faab_bid": int(faab_bid),
                                          "result": _safe_dict(result),
                                          "reason": "opponent_block"
                                      })
                        except Exception as _e:
                            st.caption(f"Logging (waiver.approved) failed: {_e}")
                    else:
                        st.error("Transaction failed"); st.json(result)
                except Exception as e:
                    st.error(f"Execution error: {e}")

# ───────────────── AI Assistant (Snapshot → Summarize → Retrieve → Answer) ─────────────────
with tab_ai:
    st.subheader("AI Assistant (snapshot → summarize → retrieve → answer)")

    # Week chooser (defaults to current week from league settings)
    wk_guess = league_current_week(settings or {})
    wk = st.number_input("Week to snapshot", min_value=1, max_value=25, value=int(wk_guess or 1), step=1)

    col_a, col_b = st.columns([1, 2], gap="large")

    # Always show the Snapshot button prominently
    with col_a:
        st.markdown("#### Step 1 — Take a Snapshot")
        if st.button("📸 Snapshot league data now", key="snap_now"):
            try:
                meta = snapshot_now(sc, league_key, settings, week_override=int(wk))
                st.session_state["_last_snapshot_meta"] = meta
                st.success(f"Snapshot saved under: {meta['dir']}")
                st.json(meta)
                try:
                    log_event("snapshot", league_key=league_key, week=int(meta.get("week") or 0), data=_safe_dict(meta))
                except Exception as _e:
                    st.caption(f"Snapshot logging failed: {_e}")
            except Exception as e:
                st.error(f"Snapshot failed: {e}")

        # Optional: quick reminder if no snapshot yet
        if "_last_snapshot_meta" not in st.session_state:
            st.info("No snapshot found yet — click **📸 Snapshot league data now** to build AI context.")

    # Helpers (same signatures as your previous version)
    def summarize_roster(roster_list, title):
        if not roster_list:
            return f"{title}: (no players)"
        def _name(p): return p.get("name") if isinstance(p, dict) else None
        def _pts(p):
            if isinstance(p, dict):
                return float(p.get("points") or p.get("proj_points") or 0.0)
            return 0.0
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
            return "Recent transactions (sample): " + ", ".join(uniq[:10])
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
        opp  = read_json("opp_roster.json")
        tx   = read_json("transactions.json")
        fas  = read_json("free_agents.json")
        you_sum = summarize_roster(your, "Your roster")
        opp_sum = summarize_roster(opp, f"Opponent roster ({snapshot_meta.get('opp_team_name') or 'unknown'})")
        tx_sum  = summarize_transactions(tx)
        fa_sum  = "Top free agents snapshot: " + "; ".join([f"{p.get('name')} {float(p.get('points') or 0):.1f}" for p in (fas or [])[:10]])
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

    def answer_with_ai(question: str, summaries: list[str], structured: dict,
                       league_key: Optional[str] = None, mode: str = "assistant") -> str:
        if not OPENAI_OK:
            ans = "OpenAI key not configured; cannot run the AI answer. (Set OPENAI_API_KEY and rerun.)"
            log_ai_qa(league_key, question, summaries, ans, structured, model="(missing)", mode=mode)
            return ans
        sys = "You are a concise fantasy football assistant. Use the provided CONTEXT to ground your answer. When uncertain, say what additional data you need."
        ctx = "\n".join(f"- {s}" for s in summaries)
        struct_txt = json.dumps(structured, ensure_ascii=False)
        prompt = f"""CONTEXT_SUMMARIES:
{ctx}

STRUCTURED_SNAPSHOT (JSON):
{struct_txt}

QUESTION:
{question}

STYLE:
- Bullet points (1–4), then 1–2 sentence recommendation."""
        try:
            resp = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.2
            )
            ans = resp.choices[0].message.content.strip()
            log_ai_qa(league_key, question, summaries, ans, structured, model="gpt-4o-mini", mode=mode)
            return ans
        except Exception as e:
            ans = f"OpenAI call failed: {e}"
            log_ai_qa(league_key, question, summaries, ans, structured, model="gpt-4o-mini", mode=mode)
            return ans

    # After a snapshot exists, show summaries + Q&A
    with col_b:
        st.markdown("#### Step 2 — Ask Questions")
        meta = st.session_state.get("_last_snapshot_meta")
        if not meta:
            st.caption("Once a snapshot exists, you’ll see summaries here and can ask questions.")
        else:
            st.caption(f"Using snapshot dir: {meta['dir']}")
            try:
                summaries = build_context_summaries(meta)
            except Exception as e:
                summaries = []
                st.warning(f"Could not build context summaries: {e}")

            if summaries:
                st.write("**Context summaries (top of mind):**")
                for s in summaries:
                    st.write("- " + str(s))

            q = st.text_area(
                "Ask a question (e.g., 'Who should I flex in PPR this week?' or 'What are my opponent’s weak spots?')",
                height=80,
                key="ai_question",
            )
            k_ctx = st.slider("How much context to feed the model", 1, 5, 3)

            if st.button("🤖 Ask AI", key="ask_ai"):
                if not summaries:
                    st.warning("Take a snapshot first.")
                else:
                    try:
                        chosen = select_context(q or "", summaries, k=int(k_ctx))
                        structured = {
                            "league_key": league_key,
                            "week": meta.get("week"),
                            "you_team_key": meta.get("you_team_key"),
                            "opp_team_key": meta.get("opp_team_key"),
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                        ans = answer_with_ai(q or "Give me a weekly plan.", chosen, structured,
                                             league_key=league_key, mode="assistant")
                        st.markdown("### Answer")
                        st.write(ans)
                        with st.expander("Context used"):
                            st.write(chosen)
                    except Exception as e:
                        st.error(f"AI answer failed: {e}")

    # Tiny helper: if OpenAI isn’t configured, warn (doesn’t block snapshots)
    if not OPENAI_OK:
        st.caption("⚠️ OpenAI key not configured — you can still take snapshots; Q&A will show a helpful notice.")


# ───────────────── Scheduler / Automation ─────────────────
with tab_sched:
    st.subheader("Scheduler / Automation")
    st.caption("These preferences can be read by a cron/Lambda/Cloud Run task to automate Morning Brief, lineup checks, and waivers.")
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
    if st.button("Run best-lineup now (no set)"):
        try:
            pool_df = roster_df(sc, league_key)
            settings_full = settings or league_settings(sc, league_key)
            slots = []
            for rp in settings_full.get("roster_positions", []):
                pos, cnt = rp.get("position"), int(rp.get("count", 0))
                if pos and cnt > 0: slots += [pos]*cnt
            enriched_pool = apply_enrichment(pool_df.copy(), {})  # no signals => baselines kick in
            starters, bench_df = optimize_lineup(enriched_pool.copy(), slots)
            st.success("Best-lineup computed")
            for slot, p in starters:
                st.write(f"- **{slot}** → {p['name']}")
        except Exception as e:
            st.error(f"Lineup run failed: {e}")

    if st.button("Simulate waivers now (no submit)"):
        try:
            L = yfa.League(sc, league_key)
            cands = free_agents(L)  # now IDP aware
            team_df = roster_df(sc, league_key)
            df_ranked = rank_waivers(cands, team_df, current_week=league_current_week(settings or {}), settings=settings or {})
            st.dataframe(df_ranked, use_container_width=True)
        except Exception as e:
            st.error(f"Waiver sim failed: {e}")
  

def _iter_recent_log_files(days: int = 30) -> List[Path]:
    out = []
    today = datetime.utcnow()
    for d in range(days):
        day = (today - pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        p = _local_log_path_for_day(day)
        if p.exists():
            out.append(p)
    return out

def read_logs_local(days: int = 30, limit: int = 2000) -> List[Dict[str, Any]]:
    files = _iter_recent_log_files(days)
    rows = []
    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("ts",""), reverse=True)
    return rows[:limit]   
  # ───────────────── Logs ─────────────────
with tab_logs:
    st.subheader("Logs")

    kinds_all = ["ai.qa", "snapshot", "waiver.approved", "lineup.reco"]
    colf1, colf2, colf3, colf4 = st.columns([1.3, 1, 1, 1])
    with colf1:
        kind_filter = st.multiselect("Kinds", kinds_all, default=kinds_all)
    with colf2:
        league_opts = ["(All)"] + [league_key]
        league_pick = st.selectbox("League", league_opts, index=(0 if len(league_opts) > 1 else 0))
    with colf3:
        week_pick = st.text_input("Week (optional)", "")
    with colf4:
        days_back = st.number_input("Days back", min_value=1, max_value=120, value=30)

    logs = read_logs_local(days=int(days_back), limit=4000)
    if kind_filter:
        logs = [r for r in logs if r.get("kind") in kind_filter]
    if league_pick and league_pick != "(All)":
        logs = [r for r in logs if r.get("league_key") == league_pick]
    if week_pick.strip():
        try:
            wv = int(week_pick.strip())
            logs = [r for r in logs if (r.get("week") == wv)]
        except Exception:
            st.caption("Week filter ignored (not an int)")

    ai_logs = [r for r in logs if r.get("kind") == "ai.qa"]
    evt_logs = [r for r in logs if r.get("kind") != "ai.qa"]

    st.markdown("### AI Q&A feed")
    if not ai_logs:
        st.caption("No AI logs yet.")
    else:
        for r in ai_logs[:200]:
            ai = r.get("ai") or {}
            q = (ai.get("question") or "")[:120]
            with st.expander(f"{r.get('ts')} • {r.get('league_key') or '(no league)'} • Q: {q}…"):
                st.write("**Question:**", ai.get("question"))
                st.write("**Context used:**")
                for c in ai.get("context") or []:
                    st.write("- " + str(c))
                st.write("**Answer:**")
                st.write(ai.get("answer"))
                with st.expander("Structured snapshot"):
                    st.json(ai.get("structured") or {})
                st.caption(f"model={ai.get('model')}  |  mode={ai.get('mode')}  |  week={r.get('week')}")

    st.markdown("### Events feed")
    if not evt_logs:
        st.caption("No events yet.")
    else:
        rows = []
        for r in evt_logs[:500]:
            kind = r.get("kind")
            summary = ""
            if kind == "snapshot":
                meta = r.get("data") or {}
                summary = f"dir={meta.get('dir')} files={len(meta.get('files', []))} opp={meta.get('opp_team_name')}"
            elif kind == "waiver.approved":
                dd = r.get("data") or {}
                addn = ((dd.get("add_player") or {}).get("name")) or "<unknown>"
                bid = dd.get("faab_bid")
                drp = dd.get("drop_player_id")
                summary = f"ADD {addn}  BID {bid}  DROP {drp}"
            elif kind == "lineup.reco":
                dd = r.get("data") or {}
                starters = dd.get("starters") or []
                names = ", ".join([s.get("name") for s in starters[:6]])
                summary = f"{len(starters)} starters → {names}"
            rows.append({
                "ts": r.get("ts"),
                "kind": kind,
                "league": r.get("league_key"),
                "week": r.get("week"),
                "summary": summary
            })
        if rows:
            df_logs = pd.DataFrame(rows)
            st.dataframe(df_logs, use_container_width=True)
         
