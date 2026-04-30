#!/usr/bin/env python3
"""
charting_match_level.py

Buduje per-mecz rolling stats dla 5 kluczowych metryk tenisowych:
  1. 1st serve points won %
  2. 2nd serve points won %
  3. BP saved %
  4. BP converted %
  5. Return points won %

Źródła danych:
  - Jeff Sackmann GitHub CSV  → mecze do 31 XII 2025
  - TennisAbstract /charting/ → mecze od 1 I 2026
    Strona główna /charting/ zawiera WSZYSTKIE ~17 500 meczów jako linki
    w formacie: YYYYMMDD-M-Tournament-Round-Player1-Player2.html
    Scraper: pobiera index raz/dzień → filtruje 2026 → pobiera strony meczów

Wyjście: stats/csv/charting_match_rolling.csv
"""
from __future__ import annotations

import re
import unicodedata
import urllib.request
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

# ─── konfiguracja ─────────────────────────────────────────────────────────────
GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "JeffSackmann/tennis_MatchChartingProject/master"
)
TA_INDEX_URL = "https://www.tennisabstract.com/charting/"
TA_BASE      = "https://www.tennisabstract.com/charting"

STATS_DIR  = Path(__file__).resolve().parent
CACHE_DIR  = STATS_DIR / "charting_cache"
OUTPUT_CSV = STATS_DIR / "csv" / "charting_match_rolling.csv"

JEFF_CUTOFF = 20260101   # Jeff's CSV tylko do końca 2025
ROLL_WINDOW = 20
MIN_PERIODS = 1
USER_AGENT  = "Mozilla/5.0"
TIMEOUT     = 30

STAT_COLS = [
    "1st_serve_won_pct",
    "2nd_serve_won_pct",
    "bp_saved_pct",
    "bp_conv_pct",
    "return_pts_won_pct",
]


# ─── helpers ──────────────────────────────────────────────────────────────────
def _fetch_url(url: str, dest: Path) -> None:
    """Pobiera URL do pliku (pomija jeśli już w cache)."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Pobieranie {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        dest.write_bytes(resp.read())


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _norm(name: str) -> str:
    ascii_val = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", ascii_val.lower())).strip()


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den * 100).where(den > 0)


def _infer_surface(match_id: str) -> str:
    mid = match_id.lower()
    clay = ["clay", "roland", "monte_carlo", "montecarlo", "madrid", "rome",
            "barcelona", "hamburg", "estoril", "bucharest", "munich", "geneva",
            "lyon", "nice", "marrakech", "casablanca", "houston", "bogota", "kitzbuhel"]
    grass = ["wimbledon", "grass", "halle", "queens", "eastbourne",
             "hertogenbosch", "newport", "mallorca", "eastbourne", "bad_homburg"]
    if any(k in mid for k in clay):
        return "Clay"
    if any(k in mid for k in grass):
        return "Grass"
    return "Hard"


# ─── Jeff Sackmann CSV ────────────────────────────────────────────────────────
def _load_jeff_matches() -> pd.DataFrame:
    path = CACHE_DIR / "charting-m-matches.csv"
    _fetch_url(f"{GITHUB_BASE}/charting-m-matches.csv", path)
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={
        "Player 1": "player1", "Player 2": "player2",
        "Date": "date_int", "Surface": "surface",
    })
    df["date_int"] = pd.to_numeric(df["date_int"], errors="coerce").fillna(0).astype(int)
    return df[df["date_int"] < JEFF_CUTOFF].copy()


def _load_jeff_stats() -> pd.DataFrame:
    path = CACHE_DIR / "charting-m-stats-Overview.csv"
    _fetch_url(f"{GITHUB_BASE}/charting-m-stats-Overview.csv", path)
    df = pd.read_csv(path, low_memory=False)
    if "set" in df.columns:
        df = df[df["set"].astype(str).str.lower() == "total"].copy()
    return df


def build_jeff_records() -> pd.DataFrame:
    """
    Łączy mecze + statystyki z Jeff's CSV.
    Zwraca: player, date_int, surface + STAT_COLS
    """
    matches = _load_jeff_matches()
    stats   = _load_jeff_stats()

    num_cols = ["first_in", "first_won", "second_in", "second_won",
                "bk_pts", "bp_saved", "return_pts", "return_pts_won"]
    for col in num_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)

    merged = stats.merge(
        matches[["match_id", "date_int", "surface", "player1", "player2"]],
        on="match_id", how="inner",
    )

    # BP converted: potrzebujemy danych przeciwnika
    opp = merged[["match_id", "player", "bk_pts", "bp_saved"]].rename(columns={
        "player": "opp_player", "bk_pts": "opp_bk_pts", "bp_saved": "opp_bp_saved",
    })
    merged = merged.merge(opp, on="match_id", how="left")
    merged = merged[merged["player"] != merged["opp_player"]].copy()

    merged["1st_serve_won_pct"]  = _safe_pct(merged["first_won"],  merged["first_in"])
    merged["2nd_serve_won_pct"]  = _safe_pct(merged["second_won"], merged["second_in"])
    merged["bp_saved_pct"]       = _safe_pct(merged["bp_saved"],   merged["bk_pts"])
    merged["bp_conv_pct"]        = _safe_pct(
        merged["opp_bk_pts"] - merged["opp_bp_saved"], merged["opp_bk_pts"]
    )
    merged["return_pts_won_pct"] = _safe_pct(merged["return_pts_won"], merged["return_pts"])

    return merged[["player", "date_int", "surface"] + STAT_COLS].copy()


# ─── TennisAbstract charting index ────────────────────────────────────────────
def load_ta_index(min_date_int: int = JEFF_CUTOFF) -> list[dict]:
    """
    Pobiera stronę https://www.tennisabstract.com/charting/
    Zwraca listę meczów ATP od min_date_int:
      {href, date_int, surface, player1, player2}

    Strona cachowana z datą dzisiejszą (odświeżana raz na dobę).
    """
    today = date.today().strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"ta_index_{today}.html"

    # Usuń stare pliki cache (inne daty)
    for old in CACHE_DIR.glob("ta_index_*.html"):
        if old.name != cache_path.name:
            old.unlink(missing_ok=True)

    _fetch_url(TA_INDEX_URL, cache_path)
    html = cache_path.read_text(encoding="utf-8", errors="ignore")

    # Każdy mecz to <a href="YYYYMMDD-M-...-Player1-Player2.html">...</a>
    # Filtrujemy tylko ATP (M), data >= min_date_int
    pattern = re.compile(
        r'href="(\d{8}-M-([^"]+)\.html)"',
        re.IGNORECASE,
    )

    records = []
    for m in pattern.finditer(html):
        href    = m.group(1)
        body    = m.group(2)   # np. Monte_Carlo_Masters-R32-Matteo_Berrettini-Daniil_Medvedev

        # Parsuj: data z początku href
        try:
            date_int = int(href[:8])
        except ValueError:
            continue

        if date_int < min_date_int:
            continue

        # Rozbij na: Tournament-Round-Player1_Firstname_Lastname-Player2_Firstname_Lastname
        # Format: TOURNAMENT-ROUND-PLAYER1-PLAYER2
        # Ale nazwy graczy mogą mieć podkreślniki (np. Del_Potro)
        # Pewny separator: po round (R\d+|QF|SF|F) idą dwa nazwiska
        round_match = re.search(r"-(R\d+|QF|SF|F|RR)-(.+)", body)
        if not round_match:
            continue

        round_code  = round_match.group(1)
        players_raw = round_match.group(2)   # "Matteo_Berrettini-Daniil_Medvedev"

        # Podziel graczy po myślniku – nazwisko zawiera podkreślniki, ale gracze są oddzieleni myślnikiem
        # Założenie: każdy gracz to jedno lub dwa słowa z podkreślnikami
        # Szukamy miejsca podziału: dwie grupy [A-Z][a-z_]+ oddzielone '-'
        # Najprościej: split("-") i łączymy tokeny, które zaczynają się od WIELKIEJ litery
        player_parts = players_raw.split("-")

        # Znajdź indeks podziału między graczem1 a graczem2
        # Każdy gracz zaczyna się od wielkiej litery; szukamy drugiego tokenu z wielkiej
        split_idx = None
        for i in range(1, len(player_parts)):
            if player_parts[i] and player_parts[i][0].isupper():
                split_idx = i
                break

        if split_idx is None:
            continue

        player1 = " ".join(player_parts[:split_idx]).replace("_", " ")
        player2 = " ".join(player_parts[split_idx:]).replace("_", " ")

        # Wyciągnij nawierzchnię z nazwy turnieju
        tournament_part = body[: round_match.start()]
        surface = _infer_surface(tournament_part)

        records.append({
            "href":     href,
            "date_int": date_int,
            "surface":  surface,
            "player1":  player1,
            "player2":  player2,
        })

    return records


# ─── parsowanie strony meczu ──────────────────────────────────────────────────
def _extract_js_var(html: str, varname: str) -> str | None:
    """
    Wyciąga wartość zmiennej JS: var NAME = '...';
    Używa string.find zamiast regex żeby uniknąć problemów z backslash w Python 3.14.
    Zwraca zdekodowany HTML (zamienia backslash-n → newline, \\/ → /).
    """
    marker = f"var {varname} = '"
    idx = html.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = start
    backslash = chr(92)
    while end < len(html):
        if html[end] == "'" and html[end - 1] != backslash:
            break
        end += 1
    raw = html[start:end]
    if "<table" not in raw and "<span" not in raw:
        return None
    # Zamień escaped znaki (tylko \n i \/ są potrzebne, reszta zostawiamy)
    return raw.replace("\\n", "\n").replace("\\/", "/")


def _extract_pct(text: str) -> float | None:
    """
    Parsuje wartość procentową z różnych formatów:
    '72.7%' → 72.7 | '83 (66%)' → 66.0 | '0.66' → 66.0
    """
    s = str(text).strip()
    # format "N (P%)"
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", s)
    if m:
        return float(m.group(1))
    # format "P%"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    # plain float 0–1
    try:
        v = float(s.replace(",", "."))
        if 0.0 <= v <= 1.0:
            return round(v * 100, 2)
        if 1.0 < v <= 100.0:
            return round(v, 2)
    except (ValueError, TypeError):
        pass
    return None


def _parse_bp_fraction(text: str) -> float | None:
    """
    Parsuje ułamek break pointów: '2/8' → 25.0, '2/2' → 100.0, '0/0' → None
    """
    s = str(text).strip()
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return None
    return round(num / den * 100, 2)


def parse_match_page(html: str, player1: str, player2: str) -> dict[str, dict]:
    """
    Parsuje stronę meczu TennisAbstract używając var overview.

    var overview zawiera tabelę STATS OVERVIEW z kolumnami:
      STATS OVERVIEW | A% | DF% | 1stIn | 1st% | 2nd% | BPSaved | RPW% | ...
    Wiersze: pełne nazwiska graczy (np. "Matteo Berrettini", "Daniil Medvedev")
    powtórzone dla każdego seta; pierwsze dwa wiersze to statystyki meczu (Total).
    BPSaved: format ułamkowy "2/8" (saved/faced).

    Zwraca: {player1: {stat_col: val, ...}, player2: {stat_col: val, ...}}
    """
    empty = {c: None for c in STAT_COLS}
    results = {player1: dict(empty), player2: dict(empty)}

    # Wyciągnij var overview
    overview_html = _extract_js_var(html, "overview")
    if not overview_html:
        return results

    try:
        tables = pd.read_html(StringIO(overview_html))
    except Exception:
        return results

    if not tables:
        return results

    tbl = tables[0]

    # Kolumny: STATS OVERVIEW, A%, DF%, 1stIn, 1st%, 2nd%, BPSaved, RPW%, ...
    cols = list(tbl.columns)
    label_col = cols[0]   # "STATS OVERVIEW"

    def find_col(name: str) -> int | None:
        for i, c in enumerate(cols):
            if str(c).strip() == name:
                return i
        return None

    idx_1st    = find_col("1st%")
    idx_2nd    = find_col("2nd%")
    idx_bp     = find_col("BPSaved")
    idx_rpw    = find_col("RPW%")

    p1_norm = _norm(player1)
    p2_norm = _norm(player2)
    p1_surname = p1_norm.split()[-1]
    p2_surname = p2_norm.split()[-1]

    # Przejdź przez wiersze – zbierz tylko Total (pierwsze trafienie dla każdego gracza)
    for _, row in tbl.iterrows():
        cell0 = _norm(str(row.iloc[0]))
        if not cell0 or cell0 in ("nan", "set 1", "set 2", "set 3", "set 4", "set 5"):
            continue

        if p1_surname in cell0 or set(p1_norm.split()) & set(cell0.split()):
            target = player1
        elif p2_surname in cell0 or set(p2_norm.split()) & set(cell0.split()):
            target = player2
        else:
            continue

        r = results[target]
        # Pobierz wartości tylko jeśli jeszcze nie ustawione (pierwszy wiersz = Total)
        if r["1st_serve_won_pct"] is None and idx_1st is not None:
            r["1st_serve_won_pct"] = _extract_pct(row.iloc[idx_1st])
        if r["2nd_serve_won_pct"] is None and idx_2nd is not None:
            r["2nd_serve_won_pct"] = _extract_pct(row.iloc[idx_2nd])
        if r["bp_saved_pct"] is None and idx_bp is not None:
            r["bp_saved_pct"] = _parse_bp_fraction(row.iloc[idx_bp])
        if r["return_pts_won_pct"] is None and idx_rpw is not None:
            r["return_pts_won_pct"] = _extract_pct(row.iloc[idx_rpw])

    # BP converted: gracz1 konwertuje = 100% - bp_saved% gracza2 (i odwrotnie)
    bp1 = results[player1]["bp_saved_pct"]
    bp2 = results[player2]["bp_saved_pct"]
    # Ale potrzebujemy ile BP twardy (faced), nie tylko %) – mamy tylko %
    # Użyjemy prostego: bp_conv = 100 - opponent_bp_saved
    if bp2 is not None:
        results[player1]["bp_conv_pct"] = round(100.0 - bp2, 2)
    if bp1 is not None:
        results[player2]["bp_conv_pct"] = round(100.0 - bp1, 2)

    return results


# ─── pobierz mecze 2026 z TA ─────────────────────────────────────────────────
def fetch_ta_2026_records() -> pd.DataFrame:
    """
    1. Pobiera index /charting/ (cache dzienny)
    2. Filtruje mecze ATP od 1 I 2026
    3. Dla każdego meczu pobiera stronę i parsuje statystyki
    Zwraca DataFrame z kolumnami: player, date_int, surface + STAT_COLS
    """
    print("  Wczytywanie TA charting index ...")
    matches = load_ta_index(min_date_int=JEFF_CUTOFF)
    print(f"  Znaleziono {len(matches)} meczów ATP od 2026-01-01")

    rows = []
    for i, m in enumerate(matches, 1):
        print(f"  [{i}/{len(matches)}] {m['date_int']} {m['player1']} vs {m['player2']} ...", end=" ", flush=True)
        url = f"{TA_BASE}/{m['href']}"
        try:
            html = _fetch_html(url)
        except Exception as e:
            print(f"BLAD: {e}")
            continue

        stats = parse_match_page(html, m["player1"], m["player2"])
        parsed_any = False
        for player, s in stats.items():
            if any(v is not None for v in s.values()):
                rows.append({
                    "player":   player,
                    "date_int": m["date_int"],
                    "surface":  m["surface"],
                    **s,
                })
                parsed_any = True
        print("OK" if parsed_any else "brak danych")

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["player", "date_int", "surface"] + STAT_COLS)


# ─── rolling averages ─────────────────────────────────────────────────────────
def build_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling average per gracz – shift(1) żeby nie używać bieżącego meczu.
    Gdy rolling zwraca NaN (za mało historii), ffill ostatnią znana wartość."""
    df = df.sort_values(["player", "date_int"]).copy()
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = None
        rolled = df.groupby("player")[col].transform(
            lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
        )
        rolled = rolled.groupby(df["player"]).transform(lambda x: x.ffill())
        df[f"roll_{col}"] = rolled
    return df


# ─── main ─────────────────────────────────────────────────────────────────────
def main(skip_ta: bool = False) -> pd.DataFrame:
    """
    Buduje pełną bazę rolling stats i zapisuje do CSV.

    Args:
        skip_ta: pomiń scraping TennisAbstract (przydatne do testów)
    """
    print("=" * 60)
    print("  charting_match_level.py – per-mecz rolling stats")
    print("=" * 60)

    # 1. Jeff Sackmann CSV (do XII 2025)
    print("\n[1/2] Jeff Sackmann CSV (do 31 XII 2025) ...")
    jeff_df = build_jeff_records()
    print(f"  -> {len(jeff_df):,} rekordow gracz x mecz, {jeff_df['player'].nunique():,} graczy")

    # 2. TennisAbstract (od I 2026)
    if skip_ta:
        ta_df = pd.DataFrame(columns=["player", "date_int", "surface"] + STAT_COLS)
        print("\n[2/2] Pomijam TennisAbstract (skip_ta=True)")
    else:
        print("\n[2/2] TennisAbstract 2026 ...")
        ta_df = fetch_ta_2026_records()
        print(f"  -> {len(ta_df):,} rekordow gracz x mecz")

    # Zlacz i oblicz rolling
    all_frames = [jeff_df]
    if not ta_df.empty:
        for col in STAT_COLS:
            if col not in ta_df.columns:
                ta_df[col] = None
        all_frames.append(ta_df[["player", "date_int", "surface"] + STAT_COLS])

    full_df = pd.concat(all_frames, ignore_index=True)
    print(f"\nLacznie: {len(full_df):,} rekordow, obliczam rolling ...")
    result = build_rolling(full_df)

    keep = ["player", "date_int", "surface"] + [f"roll_{c}" for c in STAT_COLS]
    out = result[[c for c in keep if c in result.columns]].round(3)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n=> Zapisano: {OUTPUT_CSV}")
    print(f"   Wiersze: {len(out):,}  |  Gracze: {out['player'].nunique():,}")
    for col in [c for c in out.columns if c.startswith("roll_")]:
        cnt = out[col].notna().sum()
        print(f"   {col:40s}  {cnt:>8,}  ({cnt/max(len(out),1)*100:.1f}%)")

    return out


if __name__ == "__main__":
    main()
