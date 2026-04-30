#!/usr/bin/env python3
"""
pipeline_dzienny.py
===================
Jeden skrypt, cały proces:
  1. Pobiera najnowszy 2026.xlsx z tennis-data.co.uk
  2. Dopisuje live mecze przez scraper2
  3. Łączy wszystkie 26 sezonów w jeden DataFrame
  4. Pobiera Jeff Sackmann CSV (do XII 2025) + scrapeuje TA 2026
  5. Wylicza rolling stats per gracz
  6. Dołącza rolling stats do meczów (Winner + Loser)
  7. Zapisuje wszystko do jednego pliku: csv/final_z_statami.csv
"""
from __future__ import annotations

import sys
import unicodedata
import urllib.request
import bisect
import re
from pathlib import Path

import pandas as pd

# ── ścieżki ────────────────────────────────────────────────────────────────────
STATS_DIR  = Path(__file__).resolve().parent
ROOT_DIR   = STATS_DIR.parent
CSV_DIR    = STATS_DIR / "csv"
OUTPUT_CSV = CSV_DIR / "final_z_statami.csv"

for _p in [str(ROOT_DIR), str(STATS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── konfiguracja ───────────────────────────────────────────────────────────────
TENNIS_DATA_URL = "http://www.tennis-data.co.uk/2026/2026.xlsx"
XLSX_2026       = STATS_DIR / "2026.xlsx"

ROLL_COLS = [
    "roll_1st_serve_won_pct",
    "roll_2nd_serve_won_pct",
    "roll_bp_saved_pct",
    "roll_bp_conv_pct",
    "roll_return_pts_won_pct",
]

# Reczne aliasy dla nazw z xlsx, ktore moga nie zgadzac sie 1:1 z charting.
# Klucze i wartosci sa trzymane w postaci _norm(...).
MANUAL_NAME_ALIASES = {
    "sachko": "sachko v",
    "sachko v.": "sachko v",
    "huesler": "huesler ma",
    "huesler m.a.": "huesler ma",
    "huesler m a": "huesler ma",
}


# ══════════════════════════════════════════════════════════════════════════════
# KROK 1 – pobierz 2026.xlsx
# ══════════════════════════════════════════════════════════════════════════════
def krok1_pobierz_xlsx() -> None:
    print("\n[1/5] Pobieranie 2026.xlsx z tennis-data.co.uk ...")
    req = urllib.request.Request(
        TENNIS_DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        XLSX_2026.write_bytes(r.read())
    size_kb = XLSX_2026.stat().st_size // 1024
    print(f"  OK ({size_kb} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# KROK 2 – dopisz live mecze przez scraper2
# ══════════════════════════════════════════════════════════════════════════════
def krok2_scraper2() -> None:
    print("\n[2/5] Dopisywanie live meczów (scraper2) ...")
    try:
        from tennis_scraper2 import update_excel_from_current_tournaments
        update_excel_from_current_tournaments(XLSX_2026, include_challengers=False)
        print("  OK")
    except Exception as e:
        print(f"  OSTRZEZENIE: scraper2 blad: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# KROK 3 – merge 26 sezonów
# ══════════════════════════════════════════════════════════════════════════════
def krok3_merge_sezony() -> pd.DataFrame:
    print("\n[3/5] Laczenie 26 sezonow ...")
    frames = []
    files = sorted(
        {p.resolve() for p in
         list(STATS_DIR.glob("20*.xlsx")) + list(STATS_DIR.glob("20*.xls"))},
        key=lambda p: int(p.stem)
    )
    for path in files:
        try:
            df = pd.read_excel(path, sheet_name=path.stem)
            df["_season"] = path.stem
            frames.append(df)
        except Exception as e:
            print(f"  Pominieto {path.name}: {e}")

    merged = pd.concat(frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], format="mixed", errors="coerce")
    print(f"  {len(merged):,} meczow z {len(files)} sezonow")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# KROK 4 – rolling stats (Jeff + TA 2026)
# ══════════════════════════════════════════════════════════════════════════════
def krok4_rolling_stats(skip_ta: bool = False) -> pd.DataFrame:
    print("\n[4/5] Rolling stats (Jeff Sackmann + TennisAbstract 2026) ...")
    from charting_match_level import main as charting_main
    rolling = charting_main(skip_ta=skip_ta)
    return rolling


# ══════════════════════════════════════════════════════════════════════════════
# KROK 5 – join rolling stats do meczów
# ══════════════════════════════════════════════════════════════════════════════
def _norm(name: str) -> str:
    ascii_val = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", ascii_val.lower())).strip()


def krok5_dolacz_staty(matches: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/5] Dolaczanie rolling stats do meczow ...")

    avail_cols = [c for c in ROLL_COLS if c in rolling.columns]
    if not avail_cols:
        print("  OSTRZEZENIE: brak kolumn roll_* w rolling stats")
        return matches

    # Zbuduj lookup: (player_norm, date_int) -> {roll_col: val}
    rolling["_pnorm"] = rolling["player"].apply(_norm)
    rolling["date_int"] = pd.to_numeric(rolling["date_int"], errors="coerce").fillna(0).astype(int)
    rolling["surface_norm"] = rolling.get("surface", "").astype(str).str.strip().str.lower()

    # Fallback na wypadek braku dopasowania gracza (np. brak charting dla nazwiska):
    # mediany po nawierzchni + globalne mediany.
    fallback_global = {
        c: pd.to_numeric(rolling[c], errors="coerce").median()
        for c in avail_cols
    }
    fallback_by_surface: dict[str, dict] = {}
    for surf, grp in rolling.groupby("surface_norm"):
        fallback_by_surface[surf] = {
            c: pd.to_numeric(grp[c], errors="coerce").median()
            for c in avail_cols
        }

    lookup: dict[tuple[str, int], dict] = {}
    date_idx: dict[str, list[int]] = {}
    for _, row in rolling.iterrows():
        key = (row["_pnorm"], int(row["date_int"]))
        lookup[key] = {c: row[c] for c in avail_cols}
        date_idx.setdefault(row["_pnorm"], []).append(int(row["date_int"]))
    for pn in date_idx:
        date_idx[pn].sort()

    # Cache dopasowania nazw
    name_cache: dict[str, str] = {}

    def resolve(name: str) -> str:
        if name in name_cache:
            return name_cache[name]
        pn = _norm(name)
        pn = MANUAL_NAME_ALIASES.get(pn, pn)
        if pn not in date_idx:
            # xlsx: "Djokovic N." → first token = surname
            # Jeff: "Novak Djokovic" → last token = surname
            tokens = pn.split()
            first = tokens[0] if tokens else ""
            last  = tokens[-1] if tokens else ""
            for kp in date_idx:
                kp_tokens = kp.split()
                kp_last = kp_tokens[-1] if kp_tokens else ""
                if first and first == kp_last:
                    pn = kp
                    break
                if last and last == kp_last and last != first:
                    pn = kp
        name_cache[name] = pn
        return pn

    def get_stats(name: str, date_int: int, surface: str) -> dict:
        surf_key = str(surface).strip().lower()
        empty = fallback_by_surface.get(surf_key, fallback_global)
        pn = resolve(name)
        if (pn, date_int) in lookup:
            return lookup[(pn, date_int)]
        dates = date_idx.get(pn)
        if not dates:
            return empty
        idx2 = bisect.bisect_right(dates, date_int) - 1
        if idx2 < 0:
            return empty
        return lookup.get((pn, dates[idx2]), empty)

    # date_int dla każdego meczu
    date_ints = (
        matches["Date"]
        .dt.strftime("%Y%m%d")
        .fillna("0")
        .astype(int)
        .tolist()
    )
    winners = matches["Winner"].fillna("").tolist()
    losers  = matches["Loser"].fillna("").tolist()
    surfaces = matches.get("Surface", pd.Series([""] * len(matches))).fillna("").tolist()

    w_rows, l_rows = [], []
    for w, l, d, s in zip(winners, losers, date_ints, surfaces):
        w_rows.append(get_stats(w, d, s))
        l_rows.append(get_stats(l, d, s))

    w_df = pd.DataFrame(w_rows).rename(columns=lambda c: "W_" + c)
    l_df = pd.DataFrame(l_rows).rename(columns=lambda c: "L_" + c)

    result = pd.concat(
        [matches.reset_index(drop=True),
         w_df.reset_index(drop=True),
         l_df.reset_index(drop=True)],
        axis=1,
    )

    # Statystyki dopasowania
    w_matched = w_df.iloc[:, 0].notna().sum()
    l_matched = l_df.iloc[:, 0].notna().sum()
    total = len(result)
    print(f"  Winner dopasowany: {w_matched:,}/{total:,} ({w_matched/total*100:.1f}%)")
    print(f"  Loser  dopasowany: {l_matched:,}/{total:,} ({l_matched/total*100:.1f}%)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(
    skip_download: bool = False,
    skip_scraper:  bool = False,
    skip_ta:       bool = False,
) -> None:
    print("=" * 60)
    print("  PIPELINE DZIENNY")
    print("=" * 60)

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        krok1_pobierz_xlsx()
    else:
        print("\n[1/5] Pomijam pobieranie xlsx (--skip-download)")

    if not skip_scraper:
        krok2_scraper2()
    else:
        print("\n[2/5] Pomijam scraper2 (--skip-scraper)")

    matches = krok3_merge_sezony()
    rolling = krok4_rolling_stats(skip_ta=skip_ta)
    final   = krok5_dolacz_staty(matches, rolling)

    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    size_mb = OUTPUT_CSV.stat().st_size / 1024 / 1024
    print(f"\n=> Zapisano: {OUTPUT_CSV}")
    print(f"   Wiersze: {len(final):,}  |  Kolumny: {len(final.columns)}  |  Rozmiar: {size_mb:.1f} MB")

    # KROK 6 – odbuduj final_z_statami_extended.csv (gbm4_tenis go preferuje)
    print("\n[6/6] Odbudowa final_z_statami_extended.csv (Jeff 1968-1999 + nowe + upcoming) ...")
    try:
        import subprocess
        ext_script = ROOT_DIR / "build_extended_from_jeff.py"
        if ext_script.exists():
            r = subprocess.run([sys.executable, str(ext_script)], cwd=str(ROOT_DIR),
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                ext_path = CSV_DIR / "final_z_statami_extended.csv"
                if ext_path.exists():
                    print(f"  OK ({ext_path.stat().st_size // 1024 // 1024} MB)")
                else:
                    print("  OSTRZEZENIE: skrypt OK ale plik nie powstal")
            else:
                print(f"  BLAD build_extended:\n{r.stderr[-500:]}")
        else:
            print(f"  Brak skryptu: {ext_script}")
    except Exception as e:
        print(f"  OSTRZEZENIE: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dzienny pipeline tenisowy")
    p.add_argument("--skip-download", action="store_true", help="Nie pobieraj 2026.xlsx")
    p.add_argument("--skip-scraper",  action="store_true", help="Nie uruchamiaj scraper2")
    p.add_argument("--skip-ta",       action="store_true", help="Nie scrapeuj TA 2026 (tylko Jeff CSV)")
    args = p.parse_args()
    main(
        skip_download=args.skip_download,
        skip_scraper=args.skip_scraper,
        skip_ta=args.skip_ta,
    )
