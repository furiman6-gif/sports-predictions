"""
aktualizuj_przyrostowo.py
=========================
Szybka aktualizacja przyrostowa — dopisuje tylko NOWE mecze do istniejących CSV.
Zamiast przebudowywać 151MB od zera, dodaje tylko nowe wiersze.

Kroki:
  1. Pobiera najnowszy 2026.xlsx z tennis-data.co.uk
  2. Dociąga live mecze przez scraper
  3. Porównuje z istniejącym final_2_projekt.csv
  4. Dopisuje tylko nowe mecze (Date+Winner+Loser jako klucz)
  5. Przebudowuje 5stats_projekt.csv (16MB, szybkie)
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

STATS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = STATS_DIR.parent
for _p in [str(ROOT_DIR), str(STATS_DIR)]:
    if _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(STATS_DIR))

from tennis_scraper2 import update_excel_from_current_tournaments

LATEST_URL    = "http://www.tennis-data.co.uk/2026/2026.xlsx"
FINAL2_PATH   = STATS_DIR / "csv" / "final_2_projekt.csv"
STATS5_PATH   = STATS_DIR / "csv" / "5stats_projekt.csv"
XLSX_2026     = STATS_DIR / "2026.xlsx"
KEY_COLS      = ["Date", "Winner", "Loser"]


def download_2026() -> None:
    print("Pobieranie 2026.xlsx...")
    req = urllib.request.Request(LATEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        XLSX_2026.write_bytes(r.read())
    print(f"  OK ({XLSX_2026.stat().st_size // 1024} KB)")


def update_live() -> None:
    print("Aktualizacja live (scraper)...")
    update_excel_from_current_tournaments(XLSX_2026, include_challengers=False)
    print("  OK")


def load_new_rows() -> pd.DataFrame:
    """Wczytaj 2026.xlsx i zwróć tylko wiersze których nie ma w final_2."""
    season_name = XLSX_2026.stem
    new_df = pd.read_excel(XLSX_2026, sheet_name=season_name)
    new_df["Date"] = pd.to_datetime(new_df["Date"], format="mixed", errors="coerce")

    if not FINAL2_PATH.exists():
        print(f"  Brak {FINAL2_PATH.name} — zostanie zbudowany od zera.")
        return new_df

    existing = pd.read_csv(
        FINAL2_PATH, usecols=KEY_COLS, low_memory=False
    )
    existing["Date"] = pd.to_datetime(existing["Date"], format="mixed", errors="coerce")

    # Klucz jako string do szybkiego porównania
    existing_keys = set(
        existing["Date"].astype(str) + "|" +
        existing["Winner"].fillna("") + "|" +
        existing["Loser"].fillna("")
    )
    new_df["_key"] = (
        new_df["Date"].astype(str) + "|" +
        new_df["Winner"].fillna("") + "|" +
        new_df["Loser"].fillna("")
    )
    only_new = new_df[~new_df["_key"].isin(existing_keys)].drop(columns=["_key"])
    return only_new


def append_to_final2(new_rows: pd.DataFrame) -> int:
    """Dopisuje nowe wiersze do final_2_projekt.csv."""
    if len(new_rows) == 0:
        return 0

    if not FINAL2_PATH.exists():
        # Buduj od zera — czytaj wszystkie xlsx
        print("  Budowanie final_2 od zera (pierwsze uruchomienie)...")
        frames = []
        for f in sorted(STATS_DIR.glob("20*.xls")) + sorted(STATS_DIR.glob("20*.xlsx")):
            try:
                frames.append(pd.read_excel(f, sheet_name=f.stem))
            except Exception as e:
                print(f"  Pominięto {f.name}: {e}")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(FINAL2_PATH, index=False, encoding="utf-8-sig")
        return len(df)

    # Dopisz do istniejącego
    new_rows.to_csv(
        FINAL2_PATH, mode="a", header=False,
        index=False, encoding="utf-8-sig"
    )
    return len(new_rows)


def rebuild_5stats() -> None:
    """Przebudowuje 5stats_projekt.csv (szybkie — 16MB)."""
    print("Budowanie 5stats_projekt.csv...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_5stats", STATS_DIR / "build_5stats.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  OK")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-live",     action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_2026()
    if not args.skip_live:
        update_live()

    print("Szukanie nowych meczów...")
    new_rows = load_new_rows()
    print(f"  Znaleziono {len(new_rows)} nowych wierszy.")

    added = append_to_final2(new_rows)
    if added == 0:
        print("  Brak nowych meczów — final_2 bez zmian.")
    else:
        size_mb = FINAL2_PATH.stat().st_size / 1024 / 1024
        print(f"  Dopisano {added} wierszy → final_2_projekt.csv ({size_mb:.0f} MB)")

    rebuild_5stats()
    print("\nGotowe.")


if __name__ == "__main__":
    main()
