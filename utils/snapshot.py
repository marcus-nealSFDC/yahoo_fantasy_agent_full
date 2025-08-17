
import os, json, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

from .logger import DATA_DIR, log_event

# --- Yahoo helpers ---
def authorize_session_from_file(oauth_path: str = "oauth2.json") -> OAuth2:
    sc = OAuth2(None, None, from_file=oauth_path, browser_callback=False)
    if not sc.token_is_valid():
        sc.refresh_access_token()
    return sc

def yfs_get(sc: OAuth2, path: str) -> dict:
    url = "https://fantasysports.yahooapis.com/fantasy/v2" + path
    r = sc.session.get(url, headers={"Accept": "application/json"}, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized")
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} error: {r.text[:200]}")
    return r.json()

def parse_games_leagues_v2(raw: dict) -> Tuple[list, list]:
    fc = (raw or {}).get("fantasy_content", {})
    users = fc.get("users", {})
    user_arr = users.get("user") if "user" in users else None
    if not user_arr:
        for k, v in users.items():
            if str(k).isdigit() and isinstance(v, dict) and "user" in v:
                user_arr = v["user"]; break
    if not user_arr:
        return [], []

    games_container = None
    for entry in user_arr:
        if isinstance(entry, dict) and "games" in entry:
            games_container = entry["games"]; break
    if not games_container:
        return [], []

    games, leagues = [], []
    for _, game_wrapper in games_container.items():
        if not isinstance(game_wrapper, dict): continue
        glist = game_wrapper.get("game")
        if not isinstance(glist, list): continue

        gmeta = None
        for elem in glist:
            if isinstance(elem, dict) and ("code" in elem or "game_code" in elem) and "season" in elem:
                gmeta = elem; break
        if not gmeta:
            for elem in glist:
                if isinstance(elem, dict) and "game_key" in elem:
                    gmeta = elem; break
        if not gmeta: continue

        gk = gmeta.get("game_key")
        gc = gmeta.get("code") or gmeta.get("game_code")
        season = gmeta.get("season")
        gname = gmeta.get("name") or f"{gc} {season}"
        games.append({"game_key": gk, "game_code": gc, "season": season, "name": gname})

        for elem in glist:
            if isinstance(elem, dict) and "leagues" in elem:
                leagues_dict = elem["leagues"]
                for _, lwrap in leagues_dict.items():
                    if not isinstance(lwrap, dict): continue
                    lst = lwrap.get("league")
                    if isinstance(lst, list):
                        for lentry in lst:
                            lk = nm = None
                            if isinstance(lentry, dict):
                                lk = lentry.get("league_key"); nm = lentry.get("name") or lk
                            elif isinstance(lentry, list):
                                for kv in lentry:
                                    if "league_key" in kv: lk = kv["league_key"]
                                    if "name" in kv and not nm: nm = kv["name"]
                            if lk:
                                leagues.append({"game_key": gk, "league_key": lk, "name": nm})
    return games, leagues

def filter_latest_nfl_leagues(leagues: list, games: list) -> list:
    def to_int(s):
        try: return int(str(s))
        except: return -1
    nfl_games = [g for g in games if (g.get("game_code") == "nfl" and g.get("season"))]
    if not nfl_games: return []
    latest = max(nfl_games, key=lambda g: to_int(g["season"]))
    latest_gk = latest["game_key"]
    return [L for L in leagues if L.get("game_key") == latest_gk]

def resolve_team_key(sc: OAuth2, league_key: str) -> Optional[str]:
    L = yfa.League(sc, league_key)
    try:
        teams = L.teams()
        for t in teams:
            tk = owned = None
            name = None
            if isinstance(t, dict):
                tk = t.get("team_key"); owned = t.get("is_owned_by_current_login"); name = t.get("name")
            elif isinstance(t, list):
                for kv in t:
                    if isinstance(kv, dict):
                        if "team_key" in kv: tk = kv["team_key"]
                        if "is_owned_by_current_login" in kv: owned = kv["is_owned_by_current_login"]
            if (owned in (1, True, "1")) and tk:
                return tk
        for t in teams:
            tk = t.get("team_key") if isinstance(t, dict) else None
            if tk: return tk
    except Exception:
        pass
    # fallback via /users teams
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
                    tk = None
                    for el in tlist:
                        if isinstance(el, dict) and "team_key" in el:
                            tk = el["team_key"]
                    if tk and tk.startswith(league_key + ".t."):
                        return tk
    except Exception:
        pass
    return None

# --- Snapshot + summaries ---
def team_roster_raw(sc: OAuth2, team_key: str) -> list:
    try:
        T = yfa.Team(sc, team_key)
        return T.roster() or []
    except Exception:
        return []

def get_scoreboard(sc: OAuth2, league_key: str, week: int | None=None):
    try:
        wk_q = f";week={week}" if week else ""
        return yfs_get(sc, f"/league/{league_key}/scoreboard{wk_q}?format=json")
    except Exception:
        return None

def league_settings(sc: OAuth2, league_key: str) -> dict:
    try:
        return yfa.League(sc, league_key).settings() or {}
    except Exception:
        return {}

def league_current_week(settings_dict: dict) -> int:
    for k in ("current_week", "week", "standings_week"):
        v = settings_dict.get(k)
        if v:
            try: return int(v)
            except Exception: pass
    return 1

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def summarize_roster(roster_list, title):
    def _name(p): return p.get("name") if isinstance(p, dict) else None
    def _pts(p):
        if isinstance(p, dict):
            return float(p.get("points") or p.get("proj_points") or 0.0)
        return 0.0
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
                    txs = list(el["transactions"].values()); break
        elif isinstance(league, dict):
            txs = list(league.get("transactions", {}).values())
        names = []
        for w in txs:
            if not isinstance(w, dict): continue
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
                                if nm: names.append(nm)
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
    opp  = read_json("opp_roster.json")
    tx   = read_json("transactions.json")
    fas  = read_json("free_agents.json")

    you_sum = summarize_roster(your, "Your roster")
    opp_sum = summarize_roster(opp, f"Opponent roster ({snapshot_meta.get('opp_team_name') or 'unknown'})")
    tx_sum  = summarize_transactions(tx)
    fa_sum  = "Top free agents snapshot: " + "; ".join([
        f"{p.get('name')} {float(p.get('points') or 0):.1f}" for p in (fas[:10] if isinstance(fas, list) else [])
    ])
    return [you_sum, opp_sum, tx_sum, fa_sum]

def snapshot_now_cli(sc: OAuth2, league_key: str, week_override: int | None = None) -> dict:
    settings = league_settings(sc, league_key)
    wk = week_override or league_current_week(settings or {})
    league_dir = Path(DATA_DIR) / "league" / league_key / f"week_{wk}"
    ensure_dir(league_dir)

    try:
        (league_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    except Exception:
        pass

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
                                    tkeys.append(tk); tnames.append(nm or tk)
                        if you_tk and you_tk in tkeys and len(tkeys) == 2:
                            idx = tkeys.index(you_tk)
                            opp_tk  = tkeys[1-idx]
                            your_name = tnames[idx]
                            opp_name  = tnames[1-idx]
                            break
    except Exception:
        pass

    your_roster = team_roster_raw(sc, you_tk) if you_tk else []
    opp_roster  = team_roster_raw(sc, opp_tk) if opp_tk else []

    try:
        (league_dir / "your_roster.json").write_text(json.dumps(your_roster, indent=2))
        (league_dir / "opp_roster.json").write_text(json.dumps(opp_roster, indent=2))
    except Exception:
        pass

    # Transactions
    try:
        tx = yfs_get(sc, f"/league/{league_key}/transactions?format=json")
        (league_dir / "transactions.json").write_text(json.dumps(tx, indent=2))
    except Exception:
        tx = {}

    # Free agents (sample)
    fas = []
    try:
        L = yfa.League(sc, league_key)
        def _free_agents(L, positions=("WR","RB","TE","QB"), limit=50):
            out, seen = [], set()
            for pos in positions:
                try:
                    for p in L.free_agents(pos):
                        pid = p.get("player_id")
                        if pid in seen: 
                            continue
                        seen.add(pid)
                        pts = p.get("points") or p.get("proj_points") or 0
                        out.append({"player_id": pid, "name": p.get("name"), "position": p.get("position"), "points": pts})
                except Exception:
                    continue
            out.sort(key=lambda x: x.get("points") or 0, reverse=True)
            return out[:limit]
        fas = _free_agents(L, positions=("WR","RB","TE","QB"), limit=50)
        (league_dir / "free_agents.json").write_text(json.dumps(fas, indent=2))
    except Exception:
        pass

    meta = {
        "dir": str(league_dir),
        "week": wk,
        "you_team_key": you_tk,
        "opp_team_key": opp_tk,
        "you_team_name": your_name,
        "opp_team_name": opp_name,
        "files": [p.name for p in league_dir.glob("*.json")],
    }
    return meta

def write_morning_brief(md_path: Path, league_key: str, meta: dict):
    league_dir = Path(meta["dir"])
    def _read(name):
        p = league_dir / name
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: return {}
        return {}
    your = _read("your_roster.json")
    opp  = _read("opp_roster.json")
    fas  = _read("free_agents.json")

    def _name(p): return p.get("name") if isinstance(p, dict) else None
    def _pts(p): 
        if isinstance(p, dict): return float(p.get("points") or p.get("proj_points") or 0.0)
        return 0.0

    top_you = sorted([p for p in your if isinstance(p, dict)], key=lambda p: _pts(p), reverse=True)[:8]
    top_opp = sorted([p for p in opp if isinstance(p, dict)],  key=lambda p: _pts(p), reverse=True)[:8]
    top_fa  = sorted([p for p in fas if isinstance(p, dict)],  key=lambda p: p.get("points") or 0, reverse=True)[:8]

    lines = []
    lines.append(f"# Morning Brief — League `{league_key}` (Week {meta.get('week')})")
    lines.append(f"- **Your team:** {meta.get('you_team_name') or meta.get('you_team_key')}")
    lines.append(f"- **Opponent:** {meta.get('opp_team_name') or meta.get('opp_team_key')}")
    lines.append("")
    lines.append("## Opponent key players")
    for p in top_opp:
        lines.append(f"- {p.get('name')} — {','.join(p.get('eligible_positions', []))} — {_pts(p):.1f}")
    lines.append("")
    lines.append("## Your lineup holes (heuristic)")
    # simple hole heuristic: positions with < 2 players scoring > threshold
    pos_counts = {}
    for p in your:
        for pos in (p.get("eligible_positions") or []):
            pos_counts[pos] = pos_counts.get(pos, 0) + (1 if _pts(p) >= 8 else 0)
    for pos, cnt in sorted(pos_counts.items(), key=lambda kv: kv[0]):
        if cnt < 2:
            lines.append(f"- {pos}: only {cnt} viable (≥8 pts) — consider waiver/rearrange")
    if not any(cnt < 2 for cnt in pos_counts.values()):
        lines.append("- No obvious holes by this heuristic.")
    lines.append("")
    lines.append("## Top 5 waivers vs gaps")
    for p in top_fa[:5]:
        lines.append(f"- {p.get('name')} ({p.get('position')}) — {_pts(p):.1f} projected")
    md_path.write_text("\n".join(lines), encoding="utf-8")

def run_daily_snapshots(oauth_path: str = "oauth2.json", output_root: Path | None = None) -> dict:
    sc = authorize_session_from_file(oauth_path)
    raw = yfs_get(sc, "/users;use_login=1/games/leagues?format=json")
    games, leagues_all = parse_games_leagues_v2(raw)
    nfl_leagues = filter_latest_nfl_leagues(leagues_all, games)
    if not nfl_leagues:
        raise RuntimeError("No latest NFL leagues found")

    day = datetime.utcnow().strftime("%Y-%m-%d")
    output_root = output_root or (Path(DATA_DIR) / "daily" / day)
    output_root.mkdir(parents=True, exist_ok=True)

    index = {"day": day, "leagues": []}
    for lg in nfl_leagues:
        league_key = lg["league_key"]
        meta = snapshot_now_cli(sc, league_key, week_override=None)
        # write morning brief markdown
        md_file = output_root / f"{league_key.replace('.', '_')}_brief.md"
        write_morning_brief(md_file, league_key, meta)
        # log
        try:
            log_event("snapshot", league_key=league_key, week=meta.get("week"), data={
                "dir": meta.get("dir"),
                "you": meta.get("you_team_name"),
                "opp": meta.get("opp_team_name"),
                "files": meta.get("files"),
                "brief_md": str(md_file),
            })
        except Exception:
            pass
        index["leagues"].append({"league_key": league_key, "brief_md": str(md_file)})
    (output_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    # Convenience top-level daily index
    (Path(DATA_DIR) / "daily" / "latest.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
