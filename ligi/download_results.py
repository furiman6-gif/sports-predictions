"""
Pobiera aktualne wyniki z football-data.co.uk dla biezacego sezonu,
DOPISUJE nowe mecze do wszystkie_sezony.csv (nie przebudowuje od zera),
dzieki czemu bars5.py wznawia obliczenia tylko dla nowych wierszy (pkl state).

Uruchom: python download_results.py
Lub:     python download_results.py --run-bars5
"""

import csv
import io
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent

LEAGUE_MAP = {
    "E0":  ("England",     "Premier_League"),
    "E1":  ("England",     "Championship"),
    "E2":  ("England",     "League_One"),
    "E3":  ("England",     "League_Two"),
    "EC":  ("England",     "Conference"),
    "D1":  ("Germany",     "Bundesliga_1"),
    "D2":  ("Germany",     "Bundesliga_2"),
    "SP1": ("Spain",       "La_Liga"),
    "SP2": ("Spain",       "Segunda_Division"),
    "I1":  ("Italy",       "Serie_A"),
    "I2":  ("Italy",       "Serie_B"),
    "F1":  ("France",      "Ligue_1"),
    "F2":  ("France",      "Ligue_2"),
    "SC0": ("Scotland",    "Premiership"),
    "SC1": ("Scotland",    "Championship"),
    "SC2": ("Scotland",    "League_One"),
    "SC3": ("Scotland",    "League_Two"),
    "B1":  ("Belgium",     "First_Division_A"),
    "N1":  ("Netherlands", "Eredivisie"),
    "P1":  ("Portugal",    "Primeira_Liga"),
    "G1":  ("Greece",      "Super_League"),
    "T1":  ("Turkey",      "Super_Lig"),
}

FD_BASE = "https://www.football-data.co.uk/mmz4281"


def current_season_url_path() -> str:
    """Zwraca sciezke URL dla biezacego sezonu, np. '2526'."""
    now = datetime.now()
    start_yr = now.year if now.month >= 7 else now.year - 1
    end_yr = start_yr + 1
    return f"{start_yr % 100:02d}{end_yr % 100:02d}"


def season_file_label(url_path: str) -> str:
    """
    Konwertuje sciezke URL sezonu na etykiete pliku uzywana przez organize.py.
    '2526' -> end_yr_short=26 -> start=2026, end=2027 -> '2627'
    """
    end_yr_short = int(url_path[2:])
    start = 2000 + end_yr_short
    end = start + 1
    return f"{str(start)[-2:]}{str(end)[-2:]}"


def normalize_date(date_str: str) -> str:
    if not date_str or not date_str.strip():
        return date_str
    s = date_str.strip()
    parts = s.split("/")
    if len(parts) != 3:
        return s
    dd, mm, yy = parts
    if len(yy) == 2:
        y = int(yy)
        full_year = 1900 + y if y >= 93 else 2000 + y
        return f"{dd}/{mm}/{full_year}"
    return s


def download_season_csv(code: str, url_season: str) -> tuple[list[str], list[list[str]]] | None:
    """Pobiera CSV ligi z football-data.co.uk. Zwraca (header, rows) lub None."""
    url = f"{FD_BASE}/{url_season}/{code}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    BLAD pobierania {url}: {e}")
        return None

    content = resp.content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    header = None
    rows = []
    for row in reader:
        if any(cell.strip() for cell in row):
            if header is None:
                header = [c.strip().lstrip("\ufeff") for c in row]
            else:
                if any(cell.strip() for cell in row):
                    rows.append(row)

    if not header or not rows:
        print(f"    OSTRZEZENIE: pusty CSV dla {code}")
        return None

    if "Date" in header:
        date_idx = header.index("Date")
        for row in rows:
            if date_idx < len(row):
                row[date_idx] = normalize_date(row[date_idx])

    return header, rows


def append_new_rows_to_merged(
    league_dir: Path,
    season_label: str,
    new_header: list[str],
    new_rows: list[list[str]],
    old_row_count: int,
) -> int:
    """
    Dopisuje tylko nowe wiersze do wszystkie_sezony.csv.
    Jezeli plik nie istnieje - tworzy go od podstaw.
    Zwraca laczna liczbe wierszy danych (bez naglowka).
    """
    merged_file = league_dir / "wszystkie_sezony.csv"
    truly_new = new_rows[old_row_count:]  # tylko nowe (po starym liczeniu)

    if not merged_file.exists():
        # Tworz od podstaw z samych nowych wierszy
        with open(merged_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sezon"] + new_header)
            for row in truly_new:
                writer.writerow([season_label] + row)
        return len(truly_new)

    # Odczytaj istniejacy naglowek
    with open(merged_file, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])
        existing_count = sum(1 for _ in reader)

    if not existing_header:
        # Plik pusty - zapis od podstaw
        with open(merged_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sezon"] + new_header)
            for row in truly_new:
                writer.writerow([season_label] + row)
        return len(truly_new)

    # Istniejace kolumny (bez "Sezon" na poczatku)
    global_cols = existing_header[1:] if existing_header[0] == "Sezon" else existing_header

    # Dopisz nowe wiersze (append) - mapujac na istniejace kolumny
    with open(merged_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in truly_new:
            row_dict = {col: (row[i] if i < len(row) else "") for i, col in enumerate(new_header)}
            out = [row_dict.get(col, "") for col in global_cols]
            writer.writerow([season_label] + out)

    return existing_count + len(truly_new)


def cleanup_stale_pkl(league_dir: Path, csv_file: str = "wszystkie_sezony.csv") -> None:
    """Usuwa pkl jezeli source_rows > current file rows (nieważny checkpoint)."""
    pkl_path = league_dir / f"{csv_file.replace('.csv', '')}_bars5_state.pkl"
    csv_path = league_dir / csv_file
    if not pkl_path.exists() or not csv_path.exists():
        return
    try:
        import re, struct
        with open(pkl_path, "rb") as f:
            data = f.read(600)
        m = re.search(rb"source_rows.{1,3}(M.{2}|J.{1}|K.{4})", data)
        if not m:
            return
        raw = m.group(1)
        if raw[0:1] == b"M":
            pkl_rows = struct.unpack("<H", raw[1:3])[0]
        elif raw[0:1] == b"J":
            pkl_rows = raw[1]
        elif raw[0:1] == b"K":
            pkl_rows = struct.unpack("<I", raw[1:5])[0]
        else:
            return
        current_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        if pkl_rows > current_rows:
            pkl_path.unlink()
            print(f"    Usunięto stary pkl (pkl={pkl_rows} > current={current_rows}) -> fresh run")
    except Exception:
        pass


def run_bars5(league_dir: Path) -> bool:
    bars5_script = BASE_DIR / "bars5.py"
    csv_file = "wszystkie_sezony.csv"
    if not (league_dir / csv_file).exists():
        print(f"    Brak {csv_file} - pomijam")
        return False
    cleanup_stale_pkl(league_dir, csv_file)
    print(f"    Uruchamiam bars5.py (wznowi od checkpointu jezeli istnieje)...")
    proc = subprocess.run(
        [sys.executable, str(bars5_script), csv_file],
        cwd=str(league_dir),
        text=True,
    )
    return proc.returncode == 0


def main():
    run_bars5_flag = "--run-bars5" in sys.argv

    url_season = current_season_url_path()
    file_label = season_file_label(url_season)
    season_file_name = f"sezon_{file_label}.csv"

    print(f"Biezacy sezon: URL={url_season}, plik={season_file_name}")
    print(f"Pobieranie z: {FD_BASE}/{url_season}/")
    print()

    updated_leagues: list[tuple[Path, int]] = []  # (league_dir, nowe_mecze)

    for code, (country, league) in LEAGUE_MAP.items():
        league_dir = BASE_DIR / country / league
        if not league_dir.exists():
            continue

        print(f"  [{code}] {country}/{league} ...")
        result = download_season_csv(code, url_season)
        if result is None:
            continue

        header, rows = result
        seasons_dir = league_dir / "sezony"
        seasons_dir.mkdir(parents=True, exist_ok=True)
        season_file = seasons_dir / season_file_name

        # Sprawdz stara liczbe meczow w pliku sezonu
        old_count = 0
        if season_file.exists():
            with open(season_file, encoding="utf-8", newline="") as f:
                old_count = sum(1 for _ in csv.reader(f)) - 1  # minus header

        if old_count == len(rows):
            print(f"    Brak nowych meczow ({len(rows)} meczow), pomijam.")
            continue

        # Zapisz zaktualizowany plik sezonu
        with open(season_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        new_count = len(rows) - old_count
        print(f"    Zapisano {len(rows)} meczow (+{new_count} nowych) -> {season_file.name}")

        # Dopisz tylko nowe wiersze do wszystkie_sezony.csv
        total = append_new_rows_to_merged(league_dir, file_label, header, rows, old_count)
        print(f"    wszystkie_sezony.csv: {total} wierszy lacznie (+{new_count} dopisano)")

        updated_leagues.append((league_dir, new_count))
        time.sleep(0.3)

    print()
    if not updated_leagues:
        print("Brak aktualizacji - wszystkie ligi sa aktualne.")
        return

    print(f"Zaktualizowano {len(updated_leagues)} lig.")
    print()

    if run_bars5_flag:
        print(f"Uruchamianie bars5.py (tylko nowe mecze dzieki pkl checkpoint)...")
        for league_dir, new_count in updated_leagues:
            print(f"\n  {league_dir.relative_to(BASE_DIR)} (+{new_count} meczow)")
            ok = run_bars5(league_dir)
            print(f"    {'OK' if ok else 'BLAD'}")
    else:
        print("Aby przeliczyc rankingi (tylko nowe mecze), uruchom:")
        print("  python download_results.py --run-bars5")
        print()
        print("Lub recznie dla konkretnej ligi (np. Premier League):")
        print("  cd England\\Premier_League && python ..\\..\\bars5.py wszystkie_sezony.csv")
        print()
        print("Podsumowanie zmian:")
        for league_dir, new_count in updated_leagues:
            print(f"  {league_dir.relative_to(BASE_DIR)}: +{new_count} nowych meczow")

    print("\nGotowe!")


if __name__ == "__main__":
    main()
