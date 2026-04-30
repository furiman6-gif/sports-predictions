"""
Skrypt wykonuje dwa kroki dla 5 lig (EPL, Bundesliga, La_liga, Ligue_1, Serie_A):

1. Pobiera Excel z football-data.co.uk (wszystkie 5 lig w jednym pliku) i dopisuje
   do każdego CSV nowe mecze, których jeszcze tam nie ma.

2. Dodaje kolumny xGH / xGA z plików xg_matches.xlsx (Understat).
"""

import requests
import pandas as pd
import os
from pathlib import Path
from io import BytesIO

BASE = Path(__file__).parent.parent.parent  # Desktop/Understat-scraping-2026-main

# ── Mapowania nazw: Understat → FBD ──────────────────────────────────────────
TEAM_MAP = {
    # EPL
    "Manchester City":        "Man City",
    "Manchester United":      "Man United",
    "Newcastle United":       "Newcastle",
    "Nottingham Forest":      "Nott'm Forest",
    "Queens Park Rangers":    "QPR",
    "West Bromwich Albion":   "West Brom",
    "Wolverhampton Wanderers":"Wolves",
    # Bundesliga
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
    # La_liga
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
    # Ligue_1
    "Clermont Foot":          "Clermont",
    "GFC Ajaccio":            "Ajaccio GFCO",
    "Paris Saint Germain":    "Paris SG",
    "SC Bastia":              "Bastia",
    "Saint-Etienne":          "St Etienne",
    # Serie_A
    "AC Milan":               "Milan",
    "Parma Calcio 1913":      "Parma",
    "SPAL 2013":              "Spal",
}

# URL do Excela z bieżącym sezonem (wszystkie 5 lig razem)
FBD_EXCEL_URL = "https://www.football-data.co.uk/mmz4281/2526/all-euro-data-2025-2026.xlsx"
FBD_SEZON = 2526

UPCOMING_DAYS_AHEAD = 7

LIGI = Path(__file__).parent  # Desktop/Understat-scraping-2026-main/engfoot/ligi

LEAGUES = [
    # ── Understat (mają xg_matches.xlsx) ────────────────────────────────────
    {
        "und_folder": "EPL",
        "div":        "E0",
        "csv_path":   LIGI / "England/Premier_League/wszystkie_sezony.csv",
    },
    {
        "und_folder": "Bundesliga",
        "div":        "D1",
        "csv_path":   LIGI / "Germany/Bundesliga_1/merged_data.csv",
    },
    {
        "und_folder": "La_liga",
        "div":        "SP1",
        "csv_path":   LIGI / "Spain/La_Liga/merged_data.csv",
    },
    {
        "und_folder": "Ligue_1",
        "div":        "F1",
        "csv_path":   LIGI / "France/Ligue_1/merged_data.csv",
    },
    {
        "und_folder": "Serie_A",
        "div":        "I1",
        "csv_path":   LIGI / "Italy/Serie_A/wszystkie_sezony.csv",
    },
    # ── Pozostałe ligi (tylko mecze z FBD, bez xG) ──────────────────────────
    {
        "und_folder": None,
        "div":        "E1",
        "csv_path":   LIGI / "England/Championship/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "E2",
        "csv_path":   LIGI / "England/League_One/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "E3",
        "csv_path":   LIGI / "England/League_Two/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "EC",
        "csv_path":   LIGI / "England/Conference/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "D2",
        "csv_path":   LIGI / "Germany/Bundesliga_2/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "SP2",
        "csv_path":   LIGI / "Spain/Segunda_Division/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "F2",
        "csv_path":   LIGI / "France/Ligue_2/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "I2",
        "csv_path":   LIGI / "Italy/Serie_B/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "N1",
        "csv_path":   LIGI / "Netherlands/Eredivisie/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "P1",
        "csv_path":   LIGI / "Portugal/Primeira_Liga/merged_data.csv",
    },
    {
        "und_folder": None,
        "div":        "G1",
        "csv_path":   LIGI / "Greece/Super_League/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "B1",
        "csv_path":   LIGI / "Belgium/First_Division_A/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "T1",
        "csv_path":   LIGI / "Turkey/Super_Lig/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "SC0",
        "csv_path":   LIGI / "Scotland/Premiership/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "SC1",
        "csv_path":   LIGI / "Scotland/Championship/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "SC2",
        "csv_path":   LIGI / "Scotland/League_One/wszystkie_sezony.csv",
    },
    {
        "und_folder": None,
        "div":        "SC3",
        "csv_path":   LIGI / "Scotland/League_Two/wszystkie_sezony.csv",
    },
]


def load_xg_all(und_folder: str) -> pd.DataFrame:
    """Ładuje wszystkie xg_matches.xlsx dla danej ligi (sezony + root)."""
    dfs = []
    folder = BASE / und_folder
    # Sezony
    for year in range(2014, 2027):
        path = folder / str(year) / "xg_matches.xlsx"
        if path.exists():
            dfs.append(pd.read_excel(path))
    # Root (bieżący sezon)
    root = folder / "xg_matches.xlsx"
    if root.exists():
        dfs.append(pd.read_excel(root))
    if not dfs:
        raise FileNotFoundError(f"Brak xg_matches.xlsx dla {und_folder}")
    xg = pd.concat(dfs, ignore_index=True)
    return xg


def build_lookup(xg: pd.DataFrame) -> dict:
    """
    Buduje słownik: (date_str 'YYYY-MM-DD', home_team_mapped) -> (home_xG, away_xG).
    """
    lookup = {}
    for _, row in xg.iterrows():
        dt = pd.to_datetime(row["datetime"])
        date_key = dt.strftime("%Y-%m-%d")
        home = row["home_team"]
        home_mapped = TEAM_MAP.get(home, home)
        away = row["away_team"]
        away_mapped = TEAM_MAP.get(away, away)
        hxg = round(float(row["home_xG"]), 2)
        axg = round(float(row["away_xG"]), 2)
        lookup[(date_key, home_mapped)] = (hxg, axg)
        # też po away (zapasowo)
        lookup[(date_key, away_mapped, "away")] = (hxg, axg)
    return lookup


def parse_date_fbd(date_str) -> str:
    """Konwertuje DD/MM/YYYY → YYYY-MM-DD."""
    try:
        dt = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def add_xg_to_csv(csv_path: Path, lookup: dict, und_folder: str):
    df = pd.read_csv(csv_path, low_memory=False)

    if "xGH" not in df.columns:
        df["xGH"] = None
    if "xGA" not in df.columns:
        df["xGA"] = None

    matched = 0
    unmatched = 0

    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, "xGH"]):
            matched += 1
            continue

        date_str = parse_date_fbd(row.get("Date", ""))
        if not date_str:
            continue

        home = str(row.get("HomeTeam", "")).strip()
        away = str(row.get("AwayTeam", "")).strip()

        result = lookup.get((date_str, home))
        if result is None:
            # Próba odwrotna – może nazwy są zamienione
            result_away = lookup.get((date_str, away, "away"))
            if result_away:
                result = result_away
        if result:
            df.at[idx, "xGH"] = result[0]
            df.at[idx, "xGA"] = result[1]
            matched += 1
        else:
            unmatched += 1

    df.to_csv(csv_path, index=False)
    total = matched + unmatched
    print(f"  {und_folder}: {matched}/{total} meczów dopasowanych ({unmatched} bez xG)")


def normalize_date_fbd(date_val) -> str:
    """Normalizuje datę z Excela FBD do formatu DD/MM/YYYY."""
    if pd.isna(date_val) or date_val == "":
        return ""
    if isinstance(date_val, str):
        s = date_val.strip()
        if not s:
            return ""
        parts = s.split("/")
        if len(parts) == 3:
            dd, mm, yy = parts
            if len(yy) == 2:
                y = int(yy)
                full_year = 1900 + y if y >= 93 else 2000 + y
                return f"{dd}/{mm}/{full_year}"
            # już DD/MM/YYYY z 4-cyfrowym rokiem
            return s
        # ISO / inne formaty (np. "2026-03-04 00:00:00" z dtype=str)
        try:
            dt = pd.to_datetime(s, dayfirst=False)
            return dt.strftime("%d/%m/%Y")
        except Exception:
            return s
    # datetime / Timestamp
    try:
        dt = pd.to_datetime(date_val)
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return str(date_val)


def download_fbd_excel() -> pd.DataFrame:
    """Pobiera Excel z football-data.co.uk i zwraca jako DataFrame.
    Plik ma wiele arkuszy (jeden na ligę) – wczytujemy wszystkie i łączymy.
    Nazwa arkusza odpowiada kodowi ligi (E0, D1, SP1, F1, I1 ...).
    """
    print(f"  Pobieranie: {FBD_EXCEL_URL}")
    r = requests.get(FBD_EXCEL_URL, timeout=30)
    r.raise_for_status()
    xl = pd.ExcelFile(BytesIO(r.content))
    dfs = []
    for sheet in xl.sheet_names:
        df_sheet = xl.parse(sheet, dtype=str)
        df_sheet.columns = [c.strip() for c in df_sheet.columns]
        df_sheet = df_sheet.dropna(how="all").reset_index(drop=True)
        # Kolumna Div może być pusta w niektórych arkuszach – wypełnij nazwą arkusza
        if "Div" not in df_sheet.columns or df_sheet["Div"].isna().all():
            df_sheet["Div"] = sheet
        else:
            df_sheet["Div"] = df_sheet["Div"].fillna(sheet)
        dfs.append(df_sheet)
    df = pd.concat(dfs, ignore_index=True)
    # Usuń całkowicie puste wiersze
    df = df.dropna(how="all").reset_index(drop=True)
    print(f"  Pobrано {len(df)} wierszy z {len(df.columns)} kolumnami ({len(xl.sheet_names)} arkuszy: {', '.join(xl.sheet_names)})")
    return df


def append_fbd_to_csv(csv_path: Path, fbd_df: pd.DataFrame, div: str, label: str) -> int:
    """
    Filtruje fbd_df po kolumnie Div == div, normalizuje daty,
    i dopisuje do csv_path wiersze których jeszcze tam nie ma.
    Dopasowanie po (Date_YYYY-MM-DD, HomeTeam, AwayTeam).
    """
    # Filtruj tylko tę ligę
    if "Div" not in fbd_df.columns:
        print(f"  {label}: brak kolumny 'Div' w Excelu")
        return 0
    league_df = fbd_df[fbd_df["Div"].astype(str).str.strip() == div].copy()
    if league_df.empty:
        print(f"  {label}: brak wierszy z Div={div}")
        return 0

    # Normalizuj daty w nowych danych
    league_df["Date"] = league_df["Date"].apply(normalize_date_fbd)

    date_dt = pd.to_datetime(league_df["Date"], dayfirst=True, errors="coerce")
    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=UPCOMING_DAYS_AHEAD)

    if "FTHG" in league_df.columns:
        fthg = league_df["FTHG"]
        fthg_str = fthg.astype(str).str.strip().str.lower()
        is_blank_fthg = fthg.isna() | (fthg_str == "") | (fthg_str == "nan")
        played_mask = ~is_blank_fthg
    else:
        played_mask = pd.Series(False, index=league_df.index)

    keep_mask = (date_dt.notna()) & ((date_dt <= today) | ((date_dt > today) & (date_dt <= cutoff) & (~played_mask)))
    league_df = league_df[keep_mask].copy()

    if not csv_path.exists():
        print(f"  {label}: BRAK pliku CSV: {csv_path}")
        return 0

    existing = pd.read_csv(csv_path, low_memory=False)

    # Napraw złe formaty dat w istniejącym CSV (np. "2026-03-04 00:00:00" → "04/03/2026")
    existing["Date"] = existing["Date"].apply(normalize_date_fbd)

    # Zbuduj zbiór istniejących kluczy (YYYY-MM-DD, HomeTeam, AwayTeam)
    ex_dates = pd.to_datetime(existing["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    existing_keys = set(
        zip(ex_dates,
            existing["HomeTeam"].astype(str).str.strip(),
            existing["AwayTeam"].astype(str).str.strip())
    )

    # Kolumny CSV (bez Sezon – zostanie dodany)
    csv_cols = list(existing.columns)

    new_rows = []
    for _, erow in league_df.iterrows():
        date_norm = parse_date_fbd(erow.get("Date", ""))
        if not date_norm:
            continue
        ht = str(erow.get("HomeTeam", "")).strip()
        at = str(erow.get("AwayTeam", "")).strip()
        if not ht or not at:
            continue
        key = (date_norm, ht, at)
        if key in existing_keys:
            continue

        # Zbuduj wiersz z kolumnami CSV
        row = {col: "" for col in csv_cols}
        row["Sezon"] = FBD_SEZON
        for col in csv_cols:
            if col == "Sezon":
                continue
            if col in erow.index:
                val = erow[col]
                row[col] = "" if pd.isna(val) else val
        row["Date"] = erow.get("Date", "")
        new_rows.append(row)

    if not new_rows:
        # Zapisz mimo to – żeby utrwalić naprawione formaty dat
        existing.to_csv(csv_path, index=False)
        print(f"  {label}: wszystkie mecze z Excela już są w CSV (daty znormalizowane)")
        return 0

    df_out = pd.concat([existing, pd.DataFrame(new_rows, columns=csv_cols)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"  {label}: +{len(new_rows)} nowych wierszy dopisanych do CSV")
    return len(new_rows)


def main():
    # ── KROK 1: Pobierz Excel FBD i dopisz nowe mecze ────────────────────────
    print("=" * 60)
    print("  KROK 1 – Aktualizacja CSV z football-data.co.uk")
    print("=" * 60)
    try:
        fbd_df = download_fbd_excel()
        total_new = 0
        for cfg in LEAGUES:
            total_new += append_fbd_to_csv(cfg["csv_path"], fbd_df, cfg["div"], cfg["und_folder"])
        print(f"\nLącznie dopisano: {total_new} nowych wierszy")
    except Exception as e:
        print(f"  BŁĄD pobierania Excela: {e}")
        print("  Kontynuuję z dodawaniem xG na istniejących danych...")

    # ── KROK 2: Dodaj xGH / xGA (tylko ligi z Understatu) ───────────────────
    print("\n" + "=" * 60)
    print("  KROK 2 – Dodawanie xGH / xGA")
    print("=" * 60)
    for cfg in LEAGUES:
        und_folder = cfg["und_folder"]
        if und_folder is None:
            continue  # brak danych xG dla tej ligi
        csv_path = cfg["csv_path"]
        print(f"\n=== {und_folder} ===")
        if not csv_path.exists():
            print(f"  BRAK pliku: {csv_path}")
            continue
        try:
            xg = load_xg_all(und_folder)
        except FileNotFoundError as e:
            print(f"  Brak xg_matches.xlsx: {e}")
            continue
        print(f"  Załadowano {len(xg)} meczów xG (lata 2014–2026)")
        lookup = build_lookup(xg)
        add_xg_to_csv(csv_path, lookup, und_folder)
    print("\nGotowe!")


if __name__ == "__main__":
    main()
