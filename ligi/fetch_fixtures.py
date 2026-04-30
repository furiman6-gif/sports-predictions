"""
Pobiera nadchodzace mecze z dwoch zrodel:
  1. Understat API  -> 5 lig (EPL, Bundesliga, La Liga, Ligue 1, Serie A)
  2. football-data.org API -> pozostale ligi (Championship, Eredivisie, Primeira Liga, ...)

Dopisuje przyszle mecze (bez FTHG/FTAG) na koniec CSV-ow kazdej ligi.
Dzieki temu bars5.py wyliczy cechy rankingowe pre-match.

Uruchom: python fetch_fixtures.py
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE = Path(__file__).parent

# Filtruj mecze tylko do przodu o tyle dni
DAYS_AHEAD = 7

# ============================================================
# Klucz API football-data.org
# ============================================================
FD_TOKEN = os.environ.get("FD_TOKEN", "17ce171c477445d28e3143db5ca2c7cb")
FD_BASE  = "https://api.football-data.org/v4"

# ============================================================
# Mapowania nazw: rozne zrodla -> football-data.co.uk
# ============================================================

# Understat -> FBD
TEAM_MAP_UNDERSTAT = {
    "Manchester City":        "Man City",
    "Manchester United":      "Man United",
    "Newcastle United":       "Newcastle",
    "Nottingham Forest":      "Nott'm Forest",
    "Queens Park Rangers":    "QPR",
    "West Bromwich Albion":   "West Brom",
    "Wolverhampton Wanderers":"Wolves",
    "Arminia Bielefeld":      "Bielefeld",
    "Bayer Leverkusen":       "Leverkusen",
    "Borussia Dortmund":      "Dortmund",
    "Borussia M.Gladbach":    "M'gladbach",
    "Eintracht Frankfurt":    "Ein Frankfurt",
    "FC Cologne":             "FC Koln",
    "FC Heidenheim":          "Heidenheim",
    "Fortuna Duesseldorf":    "Fortuna Dusseldorf",
    "Greuther Fuerth":        "Greuther Furth",
    "Hamburger SV":           "Hamburg",
    "Hannover 96":            "Hannover",
    "Hertha Berlin":          "Hertha",
    "Mainz 05":               "Mainz",
    "Nuernberg":              "Nurnberg",
    "RasenBallsport Leipzig": "RB Leipzig",
    "St. Pauli":              "St Pauli",
    "VfB Stuttgart":          "Stuttgart",
    "Athletic Club":          "Ath Bilbao",
    "Atletico Madrid":        "Ath Madrid",
    "Celta Vigo":             "Celta",
    "Deportivo La Coruna":    "La Coruna",
    "Espanyol":               "Espanol",
    "Rayo Vallecano":         "Vallecano",
    "Real Betis":             "Betis",
    "Real Oviedo":            "Oviedo",
    "Real Sociedad":          "Sociedad",
    "Real Valladolid":        "Valladolid",
    "SD Huesca":              "Huesca",
    "Sporting Gijon":         "Sp Gijon",
    "Clermont Foot":          "Clermont",
    "GFC Ajaccio":            "Ajaccio GFCO",
    "Paris Saint Germain":    "Paris SG",
    "SC Bastia":              "Bastia",
    "Saint-Etienne":          "St Etienne",
    "AC Milan":               "Milan",
    "Parma Calcio 1913":      "Parma",
    "SPAL 2013":              "Spal",
}

# football-data.org -> football-data.co.uk
TEAM_MAP_FDORG = {
    # Championship
    "Leeds United":             "Leeds",
    "Sheffield Wednesday":      "Sheffield Weds",
    "Sunderland AFC":           "Sunderland",
    "Queens Park Rangers":      "QPR",
    "Wolverhampton Wanderers":  "Wolves",
    "West Bromwich Albion":     "West Brom",
    "Coventry City":            "Coventry",
    "Birmingham City":          "Birmingham",
    "Watford FC":               "Watford",
    "Stoke City":               "Stoke",
    "Hull City":                "Hull",
    "Cardiff City":             "Cardiff",
    "Bristol City":             "Bristol City",
    "Norwich City":             "Norwich",
    "Millwall FC":              "Millwall",
    "Swansea City":             "Swansea",
    "Blackburn Rovers":         "Blackburn",
    "Derby County":             "Derby",
    "Preston North End":        "Preston",
    "Middlesbrough FC":         "Middlesbrough",
    "Luton Town":               "Luton",
    "Plymouth Argyle":          "Plymouth",
    "Oxford United":            "Oxford",
    # Eredivisie
    "AFC Ajax":                 "Ajax",
    "PSV":                      "PSV",
    "Feyenoord Rotterdam":      "Feyenoord",
    "AZ Alkmaar":               "AZ",
    "FC Twente":                "Twente",
    "Vitesse":                  "Vitesse",
    "SC Heerenveen":            "Heerenveen",
    "Sparta Rotterdam":         "Sparta Rotterdam",
    "FC Utrecht":               "Utrecht",
    "RKC Waalwijk":             "RKC",
    "NEC Nijmegen":             "NEC",
    "NAC Breda":                "NAC",
    "Almere City FC":           "Almere",
    "Willem II":                "Willem II",
    "PEC Zwolle":               "Zwolle",
    "Fortuna Sittard":          "Fortuna Sittard",
    "Go Ahead Eagles":          "Go Ahead",
    "FC Groningen":             "Groningen",
    # Primeira Liga
    "Sport Lisboa e Benfica":   "Benfica",
    "FC Porto":                 "Porto",
    "Sporting CP":              "Sporting",
    "SC Braga":                 "Braga",
    "Vitoria SC":               "Vitoria",
    "FC Vizela":                "Vizela",
    "GD Estoril Praia":         "Estoril",
    "Boavista FC":              "Boavista",
    "CD Nacional":              "Nacional",
    "CF Estrela Amadora":       "Estrela Amadora",
    "FC Famalicao":             "Famalicao",
    "SC Farense":               "Farense",
    "Moreirense FC":            "Moreirense",
    "AVS Futebol SAD":          "AVS",
    "Casa Pia AC":              "Casa Pia",
    "FC Arouca":                "Arouca",
    "GD Chaves":                "Chaves",
    "Gil Vicente FC":           "Gil Vicente",
    "Rio Ave FC":               "Rio Ave",
}

# ============================================================
# Konfiguracja lig
# source: "understat" lub "fdorg"
# ============================================================
LEAGUES_UNDERSTAT = [
    {
        "und_name": "EPL",
        "div":      "E0",
        "sezon":    2627,
        "csv_path": BASE / "England/Premier_League/wszystkie_sezony.csv",
    },
    {
        "und_name": "Bundesliga",
        "div":      "D1",
        "sezon":    2627,
        "csv_path": BASE / "Germany/Bundesliga_1/merged_data.csv",
    },
    {
        "und_name": "La_liga",
        "div":      "SP1",
        "sezon":    2627,
        "csv_path": BASE / "Spain/La_Liga/merged_data.csv",
    },
    {
        "und_name": "Ligue_1",
        "div":      "F1",
        "sezon":    2627,
        "csv_path": BASE / "France/Ligue_1/merged_data.csv",
    },
    {
        "und_name": "Serie_A",
        "div":      "I1",
        "sezon":    2627,
        "csv_path": BASE / "Italy/Serie_A/wszystkie_sezony.csv",
    },
]

LEAGUES_FDORG = [
    {
        "fd_code":  "ELC",
        "div":      "E1",
        "sezon":    2627,
        "csv_path": BASE / "England/Championship/wszystkie_sezony.csv",
    },
    {
        "fd_code":  "DED",
        "div":      "N1",
        "sezon":    2627,
        "csv_path": BASE / "Netherlands/Eredivisie/wszystkie_sezony.csv",
    },
    {
        "fd_code":  "PPL",
        "div":      "P1",
        "sezon":    2627,
        "csv_path": BASE / "Portugal/Primeira_Liga/merged_data.csv",
    },
]

UNDERSTAT_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}

FD_HEADERS = {
    "X-Auth-Token": FD_TOKEN,
}


# ============================================================
# Understat: pobierz przyszle mecze
# ============================================================
def fetch_understat_fixtures(league: str, season: str = "2025") -> pd.DataFrame:
    url = f"https://understat.com/getLeagueData/{league}/{season}"
    r = requests.get(url, headers=UNDERSTAT_HEADERS, timeout=20)
    r.raise_for_status()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today + timedelta(days=DAYS_AHEAD)
    rows = []
    for m in r.json()["dates"]:
        if m["goals"]["h"] is not None and m["goals"]["a"] is not None:
            continue  # rozegrany
        try:
            dt = datetime.strptime(m["datetime"][:10], "%Y-%m-%d")
        except Exception:
            continue
        if dt < today or dt > cutoff:
            continue  # poza oknem tygodnia
        ht = TEAM_MAP_UNDERSTAT.get(m["h"]["title"], m["h"]["title"])
        at = TEAM_MAP_UNDERSTAT.get(m["a"]["title"], m["a"]["title"])
        rows.append({"date_obj": dt, "date_str": dt.strftime("%d/%m/%Y"),
                     "HomeTeam": ht, "AwayTeam": at})
    return pd.DataFrame(rows)


# ============================================================
# football-data.org: pobierz przyszle mecze
# ============================================================
def fetch_fdorg_fixtures(fd_code: str) -> pd.DataFrame:
    url = f"{FD_BASE}/competitions/{fd_code}/matches?status=SCHEDULED"
    r = requests.get(url, headers=FD_HEADERS, timeout=20)
    r.raise_for_status()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today + timedelta(days=DAYS_AHEAD)
    rows = []
    for m in r.json().get("matches", []):
        raw_date = m.get("utcDate", "")[:10]
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d")
        except Exception:
            continue
        if dt < today or dt > cutoff:
            continue  # poza oknem tygodnia
        ht_raw = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "")
        at_raw = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "")
        ht = TEAM_MAP_FDORG.get(ht_raw, TEAM_MAP_FDORG.get(
            m["homeTeam"].get("name", ""), ht_raw))
        at = TEAM_MAP_FDORG.get(at_raw, TEAM_MAP_FDORG.get(
            m["awayTeam"].get("name", ""), at_raw))
        rows.append({"date_obj": dt, "date_str": dt.strftime("%d/%m/%Y"),
                     "HomeTeam": ht, "AwayTeam": at})
    return pd.DataFrame(rows)


# ============================================================
# Wspolna funkcja dopisywania
# ============================================================
def append_to_csv(csv_path: Path, fixtures: pd.DataFrame,
                  div: str, sezon: int, league_label: str) -> int:
    if not csv_path.exists():
        print(f"  BRAK PLIKU: {csv_path}")
        return 0
    if fixtures.empty:
        print(f"  {league_label}: brak nadchodzacych meczow")
        return 0

    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="warn")
    df_parsed = df.copy()
    df_parsed["_date"] = pd.to_datetime(df_parsed["Date"], dayfirst=True, errors="coerce")
    existing = set(
        zip(df_parsed["_date"].dt.strftime("%Y-%m-%d").fillna(""),
            df_parsed["HomeTeam"].astype(str).str.strip(),
            df_parsed["AwayTeam"].astype(str).str.strip())
    )

    new_rows = []
    for _, fix in fixtures.iterrows():
        key = (fix["date_obj"].strftime("%Y-%m-%d"),
               str(fix["HomeTeam"]).strip(),
               str(fix["AwayTeam"]).strip())
        if key in existing:
            continue
        row = {col: "" for col in df.columns}
        row["Sezon"]    = sezon
        row["Div"]      = div
        row["Date"]     = fix["date_str"]
        row["HomeTeam"] = fix["HomeTeam"]
        row["AwayTeam"] = fix["AwayTeam"]
        new_rows.append(row)

    if not new_rows:
        print(f"  {league_label}: {len(fixtures)} meczow - wszystkie juz sa")
        return 0

    df_out = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"  {league_label}: +{len(new_rows)} nowych meczow")
    return len(new_rows)


# ============================================================
# MAIN
# ============================================================
def main():
    today = datetime.now()
    cutoff = today + timedelta(days=DAYS_AHEAD)
    print("=" * 60)
    print(f"  FETCH FIXTURES  [{today.strftime('%d/%m/%Y')} - {cutoff.strftime('%d/%m/%Y')}]")
    print("=" * 60)
    total = 0

    print("\n[Understat - 5 lig]")
    for cfg in LEAGUES_UNDERSTAT:
        label = cfg["und_name"]
        print(f"  {label} ... ", end="", flush=True)
        try:
            fixtures = fetch_understat_fixtures(cfg["und_name"])
        except Exception as e:
            print(f"BLAD: {e}")
            continue
        total += append_to_csv(cfg["csv_path"], fixtures,
                                cfg["div"], cfg["sezon"], label)

    print("\n[football-data.org - pozostale ligi]")
    for cfg in LEAGUES_FDORG:
        label = cfg["fd_code"]
        print(f"  {label} ... ", end="", flush=True)
        try:
            fixtures = fetch_fdorg_fixtures(cfg["fd_code"])
        except Exception as e:
            print(f"BLAD: {e}")
            continue
        total += append_to_csv(cfg["csv_path"], fixtures,
                                cfg["div"], cfg["sezon"], label)

    print(f"\nGotowe! Lacznie dopisano: {total} meczow")
    print("Teraz mozesz uruchomic: run_bars5_all.bat")


if __name__ == "__main__":
    main()
