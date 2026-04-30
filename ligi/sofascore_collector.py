"""
sofascore_collector.py
Pobiera mecze + statystyki + sklady + zdarzenia z Sofascore API
dla 22 lig + 11 puchar krajowych + 3 europejskich (UCL/UEL/UECL)
za ostatnie 15 sezonow.

Zapisuje do osobnej bazy SQLite per kraj: ./data/{COUNTRY}.db
- ENG, GER, ESP, ITA, FRA, SCO, BEL, NED, POR, GRE, TUR, EUR

Wymagania: pip install requests tqdm
Uruchomienie:
    python sofascore_collector.py
    python sofascore_collector.py --country GRE --seasons 2 --no-stats   # test
    python sofascore_collector.py --country ENG ESP                       # tylko wybrane
    python sofascore_collector.py --throttle 1.0                          # wolniej
"""
import argparse, json, sqlite3, time, sys
from pathlib import Path
from datetime import datetime, timezone
import requests
from tqdm import tqdm

API = "https://api.sofascore.com/api/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin":  "https://www.sofascore.com",
}

# Mapa: kraj -> ligi/puchary -> kod -> tournament_id (Sofascore unique-tournament)
COUNTRIES = {
    "ENG": {"leagues": {"E0":17,"E1":18,"E2":24,"E3":25,"EC":173}, "cups": {"FAC":19,"EFL":21}},
    "GER": {"leagues": {"D1":35,"D2":44},                          "cups": {"DFB":217}},
    "ESP": {"leagues": {"SP1":8,"SP2":54},                          "cups": {"CDR":329}},
    "ITA": {"leagues": {"I1":23,"I2":53},                           "cups": {"CIT":328}},
    "FRA": {"leagues": {"F1":34,"F2":182},                          "cups": {"CDF":335}},
    "SCO": {"leagues": {"SC0":36,"SC1":206,"SC2":207,"SC3":209},    "cups": {"SCC":347,"SLC":332}},
    "BEL": {"leagues": {"B1":38},                                   "cups": {"BCC":326}},
    "NED": {"leagues": {"N1":37},                                   "cups": {"KNV":330}},
    "POR": {"leagues": {"P1":238},                                  "cups": {"TDP":336}},
    "GRE": {"leagues": {"G1":185},                                  "cups": {"GRC":375}},
    "TUR": {"leagues": {"T1":52},                                   "cups": {"TRC":96}},
    "EUR": {"leagues": {},                                          "cups": {"UCL":7,"UEL":679,"UECL":17015}},
}

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    event_id INTEGER PRIMARY KEY, tournament TEXT, season_year TEXT, round INTEGER,
    date_utc TEXT, timestamp INTEGER,
    home_id INTEGER, home_name TEXT, away_id INTEGER, away_name TEXT,
    home_score INTEGER, away_score INTEGER, home_ht INTEGER, away_ht INTEGER,
    status TEXT, winner_code INTEGER, venue TEXT,
    home_manager TEXT, away_manager TEXT, fetched_full INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_m_tour ON matches(tournament, season_year);
CREATE TABLE IF NOT EXISTS statistics (
    event_id INTEGER, period TEXT, group_name TEXT, stat_name TEXT,
    home_val TEXT, away_val TEXT,
    PRIMARY KEY (event_id, period, group_name, stat_name));
CREATE TABLE IF NOT EXISTS incidents (
    event_id INTEGER, seq INTEGER, type TEXT, sub_type TEXT, minute INTEGER,
    is_home INTEGER, player_name TEXT, assist_name TEXT, json_blob TEXT,
    PRIMARY KEY (event_id, seq));
CREATE TABLE IF NOT EXISTS lineups (
    event_id INTEGER, is_home INTEGER, player_id INTEGER, player_name TEXT,
    position TEXT, jersey INTEGER, is_starter INTEGER, rating REAL,
    PRIMARY KEY (event_id, player_id));
CREATE TABLE IF NOT EXISTS pregame_form (
    event_id INTEGER PRIMARY KEY,
    home_avg REAL, home_pos INTEGER, home_pts TEXT, home_form TEXT,
    away_avg REAL, away_pos INTEGER, away_pts TEXT, away_form TEXT);
"""

def open_db(country):
    db = sqlite3.connect(DATA_DIR / f"{country}.db")
    db.executescript(SCHEMA); db.execute("PRAGMA journal_mode=WAL;"); return db

def get_json(url, retries=5, base_delay=0.5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200: return r.json()
            if r.status_code == 404: return None
            if r.status_code in (429, 502, 503):
                wait = base_delay * (2**attempt) + 1
                print(f"  ! HTTP {r.status_code}, czekam {wait:.1f}s", file=sys.stderr)
                time.sleep(wait); continue
            print(f"  ! HTTP {r.status_code}: {url}", file=sys.stderr); return None
        except Exception as e:
            wait = base_delay * (2**attempt) + 1
            print(f"  ! {type(e).__name__}: {e}, czekam {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    return None

def get_seasons(tid, n):
    j = get_json(f"{API}/unique-tournament/{tid}/seasons")
    if not j: return []
    return [(s["id"], s["year"]) for s in (j.get("seasons") or [])[:n]]

def fetch_season_events(tid, sid, kind):
    out = []
    for page in range(50):
        d = get_json(f"{API}/unique-tournament/{tid}/season/{sid}/events/{kind}/{page}")
        if not d: break
        out.extend(d.get("events", []))
        if not d.get("hasNextPage"): break
        time.sleep(0.4)
    return out

def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def upsert_match(db, ev, code, year):
    hs, as_ = ev.get("homeScore",{}), ev.get("awayScore",{})
    db.execute("""INSERT OR REPLACE INTO matches
        (event_id,tournament,season_year,round,date_utc,timestamp,home_id,home_name,
         away_id,away_name,home_score,away_score,home_ht,away_ht,status,winner_code,
         venue,home_manager,away_manager,fetched_full)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,COALESCE(
            (SELECT fetched_full FROM matches WHERE event_id=?),0))""",
        (ev["id"], code, year, (ev.get("roundInfo") or {}).get("round"),
         datetime.fromtimestamp(ev["startTimestamp"], tz=timezone.utc).isoformat(),
         ev["startTimestamp"], ev["homeTeam"]["id"], ev["homeTeam"]["name"],
         ev["awayTeam"]["id"], ev["awayTeam"]["name"],
         hs.get("current"), as_.get("current"), hs.get("period1"), as_.get("period1"),
         (ev.get("status") or {}).get("type"), ev.get("winnerCode"),
         (ev.get("venue") or {}).get("name"), None, None, ev["id"]))

def fetch_full(db, eid, throttle):
    m = get_json(f"{API}/event/{eid}/managers"); time.sleep(throttle)
    if m:
        db.execute("UPDATE matches SET home_manager=?,away_manager=? WHERE event_id=?",
            ((m.get("homeManager") or {}).get("name"), (m.get("awayManager") or {}).get("name"), eid))
    s = get_json(f"{API}/event/{eid}/statistics"); time.sleep(throttle)
    if s:
        for p in s.get("statistics", []):
            for g in p.get("groups", []):
                for it in g.get("statisticsItems", []):
                    db.execute("INSERT OR REPLACE INTO statistics VALUES (?,?,?,?,?,?)",
                        (eid, p.get("period","ALL"), g.get("groupName",""), it.get("name"),
                         str(it.get("home")), str(it.get("away"))))
    i = get_json(f"{API}/event/{eid}/incidents"); time.sleep(throttle)
    if i:
        for seq, inc in enumerate(i.get("incidents", [])):
            db.execute("INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, seq, inc.get("incidentType"), inc.get("incidentClass"), inc.get("time"),
                 1 if inc.get("isHome") else 0,
                 (inc.get("player") or {}).get("name"),
                 (inc.get("assist1") or {}).get("name"),
                 json.dumps(inc, ensure_ascii=False)))
    l = get_json(f"{API}/event/{eid}/lineups"); time.sleep(throttle)
    if l:
        for side, ih in (("home",1),("away",0)):
            for p in (l.get(side) or {}).get("players", []):
                pl = p.get("player") or {}
                db.execute("INSERT OR REPLACE INTO lineups VALUES (?,?,?,?,?,?,?,?)",
                    (eid, ih, pl.get("id"), pl.get("name"), pl.get("position"),
                     int(pl.get("jerseyNumber") or 0),
                     0 if p.get("substitute") else 1,
                     (p.get("statistics") or {}).get("rating")))
    pf = get_json(f"{API}/event/{eid}/pregame-form"); time.sleep(throttle)
    if pf:
        h, a = pf.get("homeTeam") or {}, pf.get("awayTeam") or {}
        db.execute("INSERT OR REPLACE INTO pregame_form VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, _f(h.get("avgRating")), h.get("position"), h.get("value"),
             ",".join(h.get("form") or []),
             _f(a.get("avgRating")), a.get("position"), a.get("value"),
             ",".join(a.get("form") or [])))
    db.execute("UPDATE matches SET fetched_full=1 WHERE event_id=?", (eid,))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", nargs="*")
    ap.add_argument("--seasons", type=int, default=15)
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--throttle", type=float, default=0.5)
    a = ap.parse_args()
    countries = a.country or list(COUNTRIES.keys())
    for c in countries:
        if c not in COUNTRIES: print(f"Nieznany kraj: {c}"); continue
        print(f"\n=== {c} ==="); db = open_db(c)
        try:
            for kind in ("leagues", "cups"):
                for code, tid in COUNTRIES[c][kind].items():
                    seasons = get_seasons(tid, a.seasons)
                    print(f"\n  -> {code} (tid={tid}), {len(seasons)} sezonow")
                    for sid, year in seasons:
                        for k in ("last", "next"):
                            for ev in fetch_season_events(tid, sid, k):
                                upsert_match(db, ev, code, year)
                            db.commit()
                        if a.no_stats: continue
                        rows = db.execute("""SELECT event_id FROM matches
                            WHERE tournament=? AND season_year=? AND status='finished'
                            AND (fetched_full=0 OR fetched_full IS NULL)""",
                            (code, year)).fetchall()
                        if not rows:
                            print(f"     {year}: nic nowego"); continue
                        for (eid,) in tqdm(rows, desc=f"     {code} {year}", unit="m"):
                            try: fetch_full(db, eid, a.throttle); db.commit()
                            except KeyboardInterrupt: db.commit(); raise
                            except Exception as e: print(f"\n   ! {eid}: {e}")
        finally: db.commit(); db.close()
    print("\nGotowe. Bazy w ./data/")

if __name__ == "__main__":
    main()
