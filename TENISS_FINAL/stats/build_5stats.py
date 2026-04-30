"""
Tworzy plik CSV z 5 kluczowymi statystykami tenisowymi w układzie match-level
(jak final_2_projekt.csv), dla obu graczy (Winner i Loser).

Statystyki:
  1. 1st serve points won %
  2. BP saved %
  3. BP converted %
  4. 2nd serve points won %
  5. Return points won %
  + Forma ostatnich 10 meczów na tej samej nawierzchni (win%, avg_sets, avg_games)

Źródła statystyk:
  - Kolumny career_* : zagregowane statystyki z całej kariery (TennisAbstract)
  - Kolumny roll_*   : rolling average per-mecz (Jeff Sackmann + TA scraping)
                       z charting_match_rolling.csv (generowany przez charting_match_level.py)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATS_DIR = Path(__file__).resolve().parent
CSV_DIR = STATS_DIR / "csv"

PROJEKT_CSV   = CSV_DIR / "final_2_projekt.csv"
FULL_WIDE_CSV = CSV_DIR / "all_seasons_charting_full_wide.csv"
ROLLING_CSV   = CSV_DIR / "charting_match_rolling.csv"   # per-mecz rolling stats
OUTPUT_CSV    = CSV_DIR / "5stats_projekt.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASE_COLS = [
    "ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface",
    "Round", "Best of", "Winner", "Loser", "WRank", "LRank",
    "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5",
    "Wsets", "Lsets", "Comment",
]


def extract_pct(value: object) -> float | None:
    """'83 (66%)' → 66.0  |  '0.66' → 66.0  |  NaN → None"""
    if pd.isna(value):
        return None
    s = str(value).strip()
    # format "N (P%)"
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", s)
    if m:
        return float(m.group(1))
    # plain float/int
    try:
        v = float(s)
        return v * 100 if v <= 1 else v
    except ValueError:
        return None


def build_player_match_frame(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wektoryzowane obliczanie formy rolling dla każdego gracza na nawierzchni.
    Zwraca DataFrame z kolumnami form_* z prefiksem W_ i L_ dla każdego meczu.
    """
    set_cols_w = ["W1", "W2", "W3", "W4", "W5"]
    set_cols_l = ["L1", "L2", "L3", "L4", "L5"]

    for c in set_cols_w + set_cols_l:
        matches[c] = pd.to_numeric(matches[c], errors="coerce")

    # Sety i gemy wygrane przez gracza w każdym meczu — z perspektywy winnera i losera
    matches["_w_sets"] = matches["Wsets"].fillna(0)
    matches["_l_sets"] = matches["Lsets"].fillna(0)
    matches["_w_games"] = matches[set_cols_w].sum(axis=1)
    matches["_l_games"] = matches[set_cols_l].sum(axis=1)

    # Rozwiń mecze do formatu long: jeden wiersz na gracza
    w_rows = matches[["Date", "Surface", "Winner", "_w_sets", "_w_games"]].copy()
    w_rows.columns = ["Date", "Surface", "player", "sets_won", "games_won"]
    w_rows["won"] = 1

    l_rows = matches[["Date", "Surface", "Loser", "_l_sets", "_l_games"]].copy()
    l_rows.columns = ["Date", "Surface", "player", "sets_won", "games_won"]
    l_rows["won"] = 0

    long = pd.concat([w_rows, l_rows], ignore_index=True).sort_values("Date")

    # Rolling per (player, surface) — shift(1) żeby nie liczyć bieżącego meczu
    def rolling_stats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date")
        g["form_win_pct"] = g["won"].shift(1).rolling(window, min_periods=1).mean() * 100
        g["form_avg_sets"] = g["sets_won"].shift(1).rolling(window, min_periods=1).mean()
        g["form_avg_games"] = g["games_won"].shift(1).rolling(window, min_periods=1).mean()
        return g

    long = long.groupby(["player", "Surface"], group_keys=False).apply(rolling_stats)

    # Przywróć oryginalne indeksy dla winnera i losera
    long_w = long[long["won"] == 1].copy()
    long_l = long[long["won"] == 0].copy()

    # Dopasuj z powrotem do oryginalnych wierszy matches
    matches = matches.copy()
    matches["_orig_idx"] = range(len(matches))

    merged_w = matches.merge(
        long_w[["Date", "Surface", "player", "form_win_pct", "form_avg_sets", "form_avg_games"]],
        left_on=["Date", "Surface", "Winner"], right_on=["Date", "Surface", "player"],
        how="left"
    ).rename(columns={
        "form_win_pct": "W_surf_form_win_pct",
        "form_avg_sets": "W_surf_form_avg_sets",
        "form_avg_games": "W_surf_form_avg_games",
    })

    merged_both = merged_w.merge(
        long_l[["Date", "Surface", "player", "form_win_pct", "form_avg_sets", "form_avg_games"]],
        left_on=["Date", "Surface", "Loser"], right_on=["Date", "Surface", "player"],
        how="left"
    ).rename(columns={
        "form_win_pct": "L_surf_form_win_pct",
        "form_avg_sets": "L_surf_form_avg_sets",
        "form_avg_games": "L_surf_form_avg_games",
    })

    form_cols = [
        "W_surf_form_win_pct", "W_surf_form_avg_sets", "W_surf_form_avg_games",
        "L_surf_form_win_pct", "L_surf_form_avg_sets", "L_surf_form_avg_games",
    ]
    # drop_duplicates na wypadek duplikatów po merge
    merged_both = merged_both.drop_duplicates(subset=["_orig_idx"])
    return merged_both[form_cols].round(2)


# ---------------------------------------------------------------------------
# Rolling stats helpers
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalizuje nazwę gracza: usuwa akcenty, małe litery, tylko alfanum+spacja."""
    ascii_val = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", ascii_val.lower())).strip()


def _build_rolling_lookup(
    rolling: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, float]]:
    """
    Buduje słownik (player_norm, date_int) → {roll_col: value, ...}

    Jeśli dla tego samego (gracz, data) jest kilka wierszy (np. dwa mecze w ten sam dzień)
    bierze ostatni.
    """
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    roll_cols = [c for c in rolling.columns if c.startswith("roll_")]
    for _, row in rolling.iterrows():
        key = (_norm(str(row["player"])), int(row["date_int"]))
        lookup[key] = {c: row[c] for c in roll_cols}
    return lookup


def _get_rolling_stats(
    player_name: str,
    date_int: int,
    lookup: dict[tuple[str, int], dict[str, float]],
    roll_cols: list[str],
    player_norm_cache: dict[str, str],
    date_idx: dict[str, list[int]],  # player_norm → sorted list of date_ints in lookup
) -> dict[str, float | None]:
    """
    Pobiera rolling stats dla gracza na dzień date_int.
    Jeśli brak dokładnego trafienia – bierze najbliższy wcześniejszy mecz.

    Obsługuje różne formaty nazwisk:
      - tennisexplorer: "Djokovic N."  → first token to nazwisko
      - Jeff's CSV:     "Novak Djokovic" → last token to nazwisko
    """
    pnorm = player_norm_cache.get(player_name)
    if pnorm is None:
        pnorm = _norm(player_name)

        if pnorm not in date_idx:
            tokens = pnorm.split()
            # tennisexplorer używa formatu "Lastname Firstname" więc surname = tokens[0]
            # Jeff używa "Firstname Lastname" więc surname = tokens[-1]
            # Próbuj dopasować surname z xlsx (tokens[0]) do klucza w Jeff (last token)
            first_token = tokens[0] if tokens else ""
            last_token  = tokens[-1] if tokens else ""
            best = None
            for key_player in date_idx:
                kp_tokens = key_player.split()
                kp_last   = kp_tokens[-1] if kp_tokens else ""
                kp_first  = kp_tokens[0]  if kp_tokens else ""
                # first_token (xlsx surname) == kp_last (Jeff surname)
                if first_token and first_token == kp_last:
                    best = key_player
                    break
                # fallback: last_token (xlsx first?) == kp_last (Jeff surname)
                if last_token and last_token == kp_last and last_token != first_token:
                    best = best or key_player
            if best:
                pnorm = best

        player_norm_cache[player_name] = pnorm

    empty = {c: None for c in roll_cols}

    # Szukaj dokładnego dnia
    if (pnorm, date_int) in lookup:
        return lookup[(pnorm, date_int)]

    # Fallback: ostatni mecz przed date_int
    dates_for_player = date_idx.get(pnorm)
    if not dates_for_player:
        return empty
    import bisect
    idx = bisect.bisect_right(dates_for_player, date_int) - 1
    if idx < 0:
        return empty
    return lookup.get((pnorm, dates_for_player[idx]), empty)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Wczytywanie final_2_projekt.csv ...")
projekt = pd.read_csv(PROJEKT_CSV, low_memory=False)
projekt["Date"] = pd.to_datetime(projekt["Date"], format="mixed", errors="coerce")

print("Wczytywanie all_seasons_charting_full_wide.csv ...")
full_wide = pd.read_csv(FULL_WIDE_CSV, low_memory=False)

# Wczytaj per-mecz rolling stats (opcjonalnie – generowane przez charting_match_level.py)
rolling_df: pd.DataFrame | None = None
rolling_lookup: dict = {}
rolling_date_idx: dict[str, list[int]] = {}
ROLL_COLS: list[str] = []
if ROLLING_CSV.exists():
    print("Wczytywanie charting_match_rolling.csv ...")
    rolling_df = pd.read_csv(ROLLING_CSV, low_memory=False)
    rolling_df["date_int"] = pd.to_numeric(rolling_df["date_int"], errors="coerce").fillna(0).astype(int)
    ROLL_COLS = [c for c in rolling_df.columns if c.startswith("roll_")]
    rolling_lookup = _build_rolling_lookup(rolling_df)
    # Zbuduj indeks dat per gracz (do fallback wyszukiwania)
    for (pnorm, d_int) in rolling_lookup:
        rolling_date_idx.setdefault(pnorm, []).append(d_int)
    for pnorm in rolling_date_idx:
        rolling_date_idx[pnorm].sort()
    print(f"  → {len(rolling_lookup):,} rekordów, {len(rolling_date_idx):,} graczy, kolumny: {ROLL_COLS}")
else:
    print(f"  Pominięto – brak pliku {ROLLING_CSV.name} (uruchom charting_match_level.py)")

# Build BP lookup: player_slug → (bp_saved_pct, bp_converted_pct)
# Kolumna BP ma kod gracza w nazwie (rf, nd, na...), więc szukamy dynamicznie
def extract_bp_col(df: pd.DataFrame, pattern: str) -> pd.Series:
    """Dla każdego wiersza znajdź pierwszą non-null kolumnę pasującą do wzorca."""
    matching_cols = [c for c in df.columns if pattern in c]
    if not matching_cols:
        return pd.Series([None] * len(df), index=df.index)
    subset = df[matching_cols].apply(lambda col: col.apply(extract_pct))
    return subset.bfill(axis=1).iloc[:, 0]

full_wide_dedup = full_wide.dropna(subset=["player_slug"]).drop_duplicates(subset="player_slug")
bp_saved = extract_bp_col(full_wide_dedup, "_bp_faced").values
bp_conv_vals = extract_bp_col(full_wide_dedup, "_bp_opps").values
bp_saved = pd.Series(bp_saved, index=full_wide_dedup["player_slug"].values)
bp_conv = pd.Series(bp_conv_vals, index=full_wide_dedup["player_slug"].values)

# ---------------------------------------------------------------------------
# Charting stats already in projekt (career-level, per-player)
# ---------------------------------------------------------------------------
CHARTING_SERVE1 = "winner_charting_serve_overview_won_pct_first_serves"
CHARTING_SERVE2 = "winner_charting_serve_overview_won_pct_second_serves"
CHARTING_RETURN = "winner_charting_return_breakdown_ptsw_pct_total"
CHARTING_SERVE1_L = "loser_charting_serve_overview_won_pct_first_serves"
CHARTING_SERVE2_L = "loser_charting_serve_overview_won_pct_second_serves"
CHARTING_RETURN_L = "loser_charting_return_breakdown_ptsw_pct_total"

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------
print("Budowanie statystyk ...")

out = projekt[[c for c in BASE_COLS if c in projekt.columns]].copy()

# ---- Winner charting stats ----
out["W_1st_serve_won_pct"] = projekt[CHARTING_SERVE1].apply(extract_pct)
out["W_2nd_serve_won_pct"] = projekt[CHARTING_SERVE2].apply(extract_pct)
out["W_return_pts_won_pct"] = projekt[CHARTING_RETURN].apply(extract_pct)

# Winner BP from full_wide via player name lookup
winner_slug_col = "winner_charting_player_slug"
if winner_slug_col in projekt.columns:
    out["W_bp_saved_pct"] = projekt[winner_slug_col].map(bp_saved)
    out["W_bp_converted_pct"] = projekt[winner_slug_col].map(bp_conv)
else:
    out["W_bp_saved_pct"] = None
    out["W_bp_converted_pct"] = None

# ---- Loser charting stats ----
out["L_1st_serve_won_pct"] = projekt[CHARTING_SERVE1_L].apply(extract_pct)
out["L_2nd_serve_won_pct"] = projekt[CHARTING_SERVE2_L].apply(extract_pct)
out["L_return_pts_won_pct"] = projekt[CHARTING_RETURN_L].apply(extract_pct)

loser_slug_col = "loser_charting_player_slug"
if loser_slug_col in projekt.columns:
    out["L_bp_saved_pct"] = projekt[loser_slug_col].map(bp_saved)
    out["L_bp_converted_pct"] = projekt[loser_slug_col].map(bp_conv)
else:
    out["L_bp_saved_pct"] = None
    out["L_bp_converted_pct"] = None

# ---- Rolling surface form ----
print("Liczenie formy na nawierzchni (ostatnie 10 meczów) ...")
projekt_sorted = projekt.sort_values("Date", na_position="last").reset_index(drop=True)
out_sorted = out.loc[projekt_sorted.index].reset_index(drop=True)
out_sorted["Date"] = projekt_sorted["Date"].values

form = build_player_match_frame(projekt_sorted)
out_sorted = pd.concat([out_sorted.reset_index(drop=True), form.reset_index(drop=True)], axis=1)
out_sorted["Date"] = projekt_sorted["Date"].values

# ---- Per-mecz rolling stats z charting_match_rolling.csv ----
if rolling_df is not None and ROLL_COLS:
    print("Dołączanie per-mecz rolling stats (Jeff Sackmann + TA) ...")
    norm_cache: dict[str, str] = {}

    # date_int dla każdego wiersza projektu
    date_ints = (
        pd.to_datetime(projekt_sorted["Date"], errors="coerce")
        .dt.strftime("%Y%m%d")
        .fillna("0")
        .astype(int)
        .tolist()
    )
    winners = projekt_sorted["Winner"].fillna("").tolist()
    losers  = projekt_sorted["Loser"].fillna("").tolist()

    w_rows = []
    l_rows = []
    for winner, loser, d_int in zip(winners, losers, date_ints):
        w_stats = _get_rolling_stats(winner, d_int, rolling_lookup, ROLL_COLS, norm_cache, rolling_date_idx)
        l_stats = _get_rolling_stats(loser,  d_int, rolling_lookup, ROLL_COLS, norm_cache, rolling_date_idx)
        w_rows.append(w_stats)
        l_rows.append(l_stats)

    # Dodaj prefiks W_roll_ i L_roll_
    w_roll_df = pd.DataFrame(w_rows).rename(columns=lambda c: "W_" + c)
    l_roll_df = pd.DataFrame(l_rows).rename(columns=lambda c: "L_" + c)
    out_sorted = pd.concat(
        [out_sorted.reset_index(drop=True),
         w_roll_df.reset_index(drop=True),
         l_roll_df.reset_index(drop=True)],
        axis=1,
    )
    matched_w = w_roll_df[w_roll_df.iloc[:, 0].notna()].shape[0]
    matched_l = l_roll_df[l_roll_df.iloc[:, 0].notna()].shape[0]
    print(f"  Dopasowano Winner: {matched_w:,}/{len(w_rows):,} ({matched_w/len(w_rows)*100:.1f}%)")
    print(f"  Dopasowano Loser:  {matched_l:,}/{len(l_rows):,} ({matched_l/len(l_rows)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_sorted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nGotowe! Zapisano: {OUTPUT_CSV}")
print(f"Wiersze: {len(out_sorted):,}  |  Kolumny: {len(out_sorted.columns)}")
print("\nKolumny wyjściowe:")
for c in out_sorted.columns:
    non_null = out_sorted[c].notna().sum()
    print(f"  {c:40s} {non_null:>7,} non-null ({non_null/len(out_sorted)*100:.1f}%)")
