"""
Sprawdza co jest dostepne na the-odds-api dla naszego klucza:
1. Quota (ile zapytan zostalo)
2. Lista lig pilkarskich
3. Dostepne rynki dla kazdej ligi (czy h2h/totals/shots OU)
"""
import sys, io, json
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API_KEY = "d6b24d43c9514d42b980f202edcb6cb5"
BASE = "https://api.the-odds-api.com/v4"

# 22 nasze ligi -> kody the-odds-api
LEAGUES = {
    "Belgium/First_Division_A": "soccer_belgium_first_div",
    "England/Premier_League":   "soccer_epl",
    "England/Championship":     "soccer_efl_champ",
    "England/League_One":       "soccer_england_league1",
    "England/League_Two":       "soccer_england_league2",
    "England/Conference":       "soccer_england_efl_cup",  # placeholder, sprawdzimy
    "France/Ligue_1":           "soccer_france_ligue_one",
    "France/Ligue_2":           "soccer_france_ligue_two",
    "Germany/Bundesliga_1":     "soccer_germany_bundesliga",
    "Germany/Bundesliga_2":     "soccer_germany_bundesliga2",
    "Greece/Super_League":      "soccer_greece_super_league",
    "Italy/Serie_A":            "soccer_italy_serie_a",
    "Italy/Serie_B":            "soccer_italy_serie_b",
    "Netherlands/Eredivisie":   "soccer_netherlands_eredivisie",
    "Portugal/Primeira_Liga":   "soccer_portugal_primeira_liga",
    "Scotland/Premiership":     "soccer_spl",
    "Spain/La_Liga":            "soccer_spain_la_liga",
    "Spain/Segunda_Division":   "soccer_spain_segunda_division",
    "Turkey/Super_Lig":         "soccer_turkey_super_league",
}

MARKETS_TO_TRY = [
    "h2h", "spreads", "totals", "btts",
    "alternate_totals", "team_totals",
    "alternate_team_totals", "h2h_3_way",
    "shots_on_target", "shots", "team_shots_on_target",
    "alternate_totals_shots_on_target",
]


def get_sports():
    r = requests.get(f"{BASE}/sports", params={"apiKey": API_KEY, "all": "true"}, timeout=20)
    print(f"GET /sports -> {r.status_code}")
    print(f"Quota: used={r.headers.get('x-requests-used')} remaining={r.headers.get('x-requests-remaining')}")
    return r.json() if r.status_code == 200 else None


def list_soccer_leagues(sports):
    soccer = [s for s in sports if s.get("group", "").lower() == "soccer"]
    print(f"\nLigi pilkarskie na API: {len(soccer)}")
    for s in soccer:
        active = "ACTIVE" if s.get("active") else "off"
        print(f"  {s['key']:<45} {active:<7} {s.get('title','')}")
    return soccer


def check_league_markets(sport_key, our_name):
    print(f"\n--- {our_name} ({sport_key}) ---")
    # 1 prosty request: h2h
    r = requests.get(f"{BASE}/sports/{sport_key}/odds",
                     params={"apiKey": API_KEY, "regions": "eu", "markets": "h2h"},
                     timeout=20)
    print(f"  h2h -> {r.status_code}  remaining={r.headers.get('x-requests-remaining')}")
    if r.status_code != 200:
        print(f"  ERR: {r.text[:200]}")
        return
    data = r.json()
    print(f"  Mecze: {len(data)}")
    if not data:
        return
    # zapytaj rozne markety jednoczesnie zeby zaoszczedzic requesty
    markets_str = ",".join(["h2h", "totals", "btts", "spreads"])
    r2 = requests.get(f"{BASE}/sports/{sport_key}/odds",
                      params={"apiKey": API_KEY, "regions": "eu",
                              "markets": markets_str, "oddsFormat": "decimal"},
                      timeout=20)
    print(f"  multi({markets_str}) -> {r2.status_code}  remaining={r2.headers.get('x-requests-remaining')}")
    if r2.status_code == 200:
        d2 = r2.json()
        if d2:
            avail_markets = set()
            for m in d2[:5]:
                for bk in m.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        avail_markets.add(mk.get("key"))
            print(f"  Dostepne rynki: {sorted(avail_markets)}")

    # sprob shots OU
    r3 = requests.get(f"{BASE}/sports/{sport_key}/odds",
                      params={"apiKey": API_KEY, "regions": "eu",
                              "markets": "alternate_totals_shots_on_target", "oddsFormat": "decimal"},
                      timeout=20)
    print(f"  shots_OU -> {r3.status_code}  remaining={r3.headers.get('x-requests-remaining')}")
    if r3.status_code == 200:
        d3 = r3.json()
        avail = set()
        for m in d3[:3]:
            for bk in m.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    avail.add(mk.get("key"))
        print(f"  shots rynki: {sorted(avail) if avail else 'BRAK'}")
    elif r3.status_code == 422:
        print(f"  shots: rynek niedostepny (422 - nie istnieje lub brak w planie)")


def main():
    print("=" * 70)
    print("  THE-ODDS-API — sprawdzenie dostepnosci dla naszego klucza")
    print("=" * 70)

    sports = get_sports()
    if sports is None:
        print("BLAD: nie udalo sie pobrac listy sportow.")
        return

    soccer = list_soccer_leagues(sports)

    # mapuj nasze ligi do kluczy API
    api_keys = {s["key"]: s for s in soccer}
    print("\n" + "=" * 70)
    print("  MAPOWANIE NASZYCH LIG")
    print("=" * 70)
    available = []
    for our, key in LEAGUES.items():
        ok = key in api_keys
        title = api_keys[key]["title"] if ok else "—"
        active = api_keys[key].get("active") if ok else False
        flag = "OK" if (ok and active) else ("INACTIVE" if ok else "BRAK")
        print(f"  {our:<32} -> {key:<40} [{flag}] {title}")
        if ok and active:
            available.append((our, key))

    print(f"\n  Aktywnych: {len(available)}/{len(LEAGUES)}")

    # sprawdz markety dla 2-3 lig (oszczednosc requestow)
    print("\n" + "=" * 70)
    print(f"  TEST RYNKOW (na 3 ligach, koszt ~9 requestow)")
    print("=" * 70)
    for our, key in available[:3]:
        check_league_markets(key, our)

    # final quota
    r = requests.get(f"{BASE}/sports", params={"apiKey": API_KEY}, timeout=20)
    print(f"\nKoncowa quota: used={r.headers.get('x-requests-used')} remaining={r.headers.get('x-requests-remaining')}")


if __name__ == "__main__":
    main()
