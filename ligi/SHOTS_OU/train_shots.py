"""
SHOTS O/U — regresja LightGBM dla strzalow celnych (HST/AST/total)
Per liga: 3 modele (total, home, away), kazdy z 3 progami O/U na bazie sredniej z ostatnich 2 sezonow.
Cechy: rolling HST/AST (form 5, 10), venue-specific (home u siebie, away na wyjezdzie),
       Elo (z FTR), SIAM_8D/16D z wszystkie_sezony.csv.
Output: SHOTS_OU/predykcje_<data>.csv
"""
from __future__ import annotations
import sys, io, os, math, time
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import norm
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
TODAY = pd.Timestamp.today().normalize()

LEAGUES = [
    "Belgium/First_Division_A",
    "England/Premier_League", "England/Championship", "England/League_One",
    "England/League_Two", "England/Conference",
    "France/Ligue_1", "France/Ligue_2",
    "Germany/Bundesliga_1", "Germany/Bundesliga_2",
    "Greece/Super_League",
    "Italy/Serie_A", "Italy/Serie_B",
    "Netherlands/Eredivisie",
    "Portugal/Primeira_Liga",
    "Scotland/Premiership", "Scotland/Championship", "Scotland/League_One", "Scotland/League_Two",
    "Spain/La_Liga", "Spain/Segunda_Division",
    "Turkey/Super_Lig",
]

LAST_2_SEASONS = ["2526", "2627"]
ROLL_WINDOWS = [5, 10]


def parse_date(s):
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def infer_season(d):
    if pd.isna(d):
        return None
    y = d.year
    m = d.month
    if m >= 7:
        return f"{str(y)[-2:]}{str(y+1)[-2:]}"
    return f"{str(y-1)[-2:]}{str(y)[-2:]}"


def build_team_rolling(df: pd.DataFrame, stat_h: str, stat_a: str, suffix: str) -> pd.DataFrame:
    """Rolling form per team (overall) — bierze stat_h gdy team=Home, stat_a gdy Away."""
    df = df.sort_values("Date").reset_index(drop=True)
    long_rows = []
    for i, r in df.iterrows():
        long_rows.append({"team": r["HomeTeam"], "date": r["Date"], "idx": i, "venue": "H", "val": r.get(stat_h)})
        long_rows.append({"team": r["AwayTeam"], "date": r["Date"], "idx": i, "venue": "A", "val": r.get(stat_a)})
    long = pd.DataFrame(long_rows).sort_values(["team", "date", "idx"]).reset_index(drop=True)
    long["val"] = pd.to_numeric(long["val"], errors="coerce")
    for w in ROLL_WINDOWS:
        long[f"roll_{w}_{suffix}"] = long.groupby("team")["val"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    # venue-specific
    for w in ROLL_WINDOWS:
        long[f"vroll_{w}_{suffix}"] = long.groupby(["team", "venue"])["val"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    out_cols = [f"roll_{w}_{suffix}" for w in ROLL_WINDOWS] + [f"vroll_{w}_{suffix}" for w in ROLL_WINDOWS]
    h_long = long[long["venue"] == "H"].set_index("idx")[out_cols].rename(columns=lambda c: f"H_{c}")
    a_long = long[long["venue"] == "A"].set_index("idx")[out_cols].rename(columns=lambda c: f"A_{c}")
    return df.join(h_long).join(a_long)


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    K = 24.0
    initial = 1500.0
    ratings = {}
    elo_h, elo_a, elo_diff = [], [], []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        ra = ratings.get(h, initial)
        rb = ratings.get(a, initial)
        elo_h.append(ra)
        elo_a.append(rb)
        elo_diff.append(ra - rb)
        ftr = r.get("FTR")
        if isinstance(ftr, str) and ftr in ("H", "D", "A"):
            outcome = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
            exp_a = 1 / (1 + 10 ** ((rb - ra) / 400))
            ratings[h] = ra + K * (outcome - exp_a)
            ratings[a] = rb + K * ((1 - outcome) - (1 - exp_a))
    df["elo_h"] = elo_h
    df["elo_a"] = elo_a
    df["elo_diff"] = elo_diff
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)
    df["Sezon"] = df["Date"].apply(infer_season)
    # rollingi
    df = build_team_rolling(df, "HST", "AST", "shots")
    df = build_team_rolling(df, "FTHG", "FTAG", "goals")
    df = add_elo(df)
    # SIAM - jesli sa
    siam = [c for c in df.columns if c.startswith("SIAM_")]
    # encoding miesiaca / dnia tygodnia
    df["month"] = df["Date"].dt.month
    df["dow"] = df["Date"].dt.dayofweek
    return df


FEATURE_COLS = (
    [f"H_roll_{w}_shots" for w in ROLL_WINDOWS] +
    [f"H_vroll_{w}_shots" for w in ROLL_WINDOWS] +
    [f"A_roll_{w}_shots" for w in ROLL_WINDOWS] +
    [f"A_vroll_{w}_shots" for w in ROLL_WINDOWS] +
    [f"H_roll_{w}_goals" for w in ROLL_WINDOWS] +
    [f"H_vroll_{w}_goals" for w in ROLL_WINDOWS] +
    [f"A_roll_{w}_goals" for w in ROLL_WINDOWS] +
    [f"A_vroll_{w}_goals" for w in ROLL_WINDOWS] +
    ["elo_h", "elo_a", "elo_diff", "month", "dow"]
)


def thresholds_around(avg: float) -> list[float]:
    """3 linie waska: avg-1, avg, avg+1 zaokraglone do .5"""
    base = round(avg * 2) / 2
    if base == round(base):
        base += 0.5
    return [base - 1.0, base, base + 1.0]


def train_one_clf(df_full: pd.DataFrame, target_col: str, league: str, lines: list[float]):
    """Klasyfikacja: osobny LGBM per (target x linia). Zwroc DF predykcji."""
    feat_cols = [c for c in FEATURE_COLS if c in df_full.columns]
    siam_cols = [c for c in df_full.columns if c.startswith("SIAM_")]
    feat_cols = feat_cols + siam_cols

    df_full = df_full.copy()
    df_full[target_col] = pd.to_numeric(df_full[target_col], errors="coerce")
    known = df_full[df_full[target_col].notna()].copy()
    future = df_full[df_full[target_col].isna() & (df_full["Date"] >= TODAY)].copy()
    if len(known) < 200 or len(future) == 0:
        return None

    seasons = sorted([s for s in known["Sezon"].dropna().unique()])
    if len(seasons) < 2:
        return None
    valid_seasons = seasons[-1:]
    train_df = known[~known["Sezon"].isin(valid_seasons)]
    valid_df = known[known["Sezon"].isin(valid_seasons)]
    if len(train_df) < 100 or len(valid_df) < 30:
        train_df = known.iloc[:int(len(known) * 0.85)]
        valid_df = known.iloc[int(len(known) * 0.85):]

    X_tr = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_va = valid_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_fu = future[feat_cols].apply(pd.to_numeric, errors="coerce")

    out = future[["Date", "HomeTeam", "AwayTeam"]].copy()
    out["liga"] = league
    out[f"{target_col}_avg2y"] = round(float(known[known["Sezon"].isin(LAST_2_SEASONS)][target_col].mean()
                                              if known["Sezon"].isin(LAST_2_SEASONS).any() else known[target_col].mean()), 2)

    # rownolegle: regresja zeby miec pred + sigma
    y_tr_reg = train_df[target_col]
    y_va_reg = valid_df[target_col]
    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                             min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
                             random_state=42, verbose=-1)
    reg.fit(X_tr, y_tr_reg, eval_set=[(X_va, y_va_reg)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    pred_va = reg.predict(X_va)
    sigma = float(np.sqrt(np.mean((y_va_reg - pred_va) ** 2)))
    pred_fu = reg.predict(X_fu)
    out[f"{target_col}_pred"] = np.round(pred_fu, 2)
    out[f"{target_col}_sigma"] = round(sigma, 3)

    for L in lines:
        y_tr = (train_df[target_col] > L).astype(int)
        y_va = (valid_df[target_col] > L).astype(int)
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        prob = model.predict_proba(X_fu)[:, 1]
        # accuracy walidacyjna
        acc = float((model.predict(X_va) == y_va).mean())
        out[f"{target_col}_P_over_{L}"] = np.round(prob, 3)
        out[f"{target_col}_acc_{L}"] = round(acc, 3)
    return out


def train_one(df_full: pd.DataFrame, target_col: str, league: str):
    """Zwroc: model, sigma, threshold_list, pred_future_df."""
    feat_cols = [c for c in FEATURE_COLS if c in df_full.columns]
    siam_cols = [c for c in df_full.columns if c.startswith("SIAM_")]
    feat_cols = feat_cols + siam_cols

    df_full = df_full.copy()
    df_full[target_col] = pd.to_numeric(df_full[target_col], errors="coerce")
    known = df_full[df_full[target_col].notna()].copy()
    future = df_full[df_full[target_col].isna() & (df_full["Date"] >= TODAY)].copy()

    if len(known) < 200 or len(future) == 0:
        return None

    # split: ostatni sezon = valid, reszta = train
    seasons = sorted([s for s in known["Sezon"].dropna().unique()])
    if len(seasons) < 2:
        return None
    valid_seasons = seasons[-1:]
    train_df = known[~known["Sezon"].isin(valid_seasons)]
    valid_df = known[known["Sezon"].isin(valid_seasons)]
    if len(train_df) < 100 or len(valid_df) < 30:
        train_df = known.iloc[:int(len(known) * 0.85)]
        valid_df = known.iloc[int(len(known) * 0.85):]

    X_tr = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_tr = train_df[target_col]
    X_va = valid_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_va = valid_df[target_col]
    X_fu = future[feat_cols].apply(pd.to_numeric, errors="coerce")

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31,
        min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    # sigma z residuow walidacyjnych
    pred_va = model.predict(X_va)
    sigma = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))

    # progi z 2 ostatnich sezonow
    last2 = known[known["Sezon"].isin(LAST_2_SEASONS)]
    if len(last2) < 30:
        last2 = known.tail(min(len(known), 600))
    avg = float(last2[target_col].mean())
    lines = thresholds_around(avg)

    # predykcje przyszlosci
    pred_fu = model.predict(X_fu)
    out = future[["Date", "HomeTeam", "AwayTeam"]].copy()
    out["liga"] = league
    out[f"{target_col}_pred"] = np.round(pred_fu, 2)
    out[f"{target_col}_sigma"] = round(sigma, 3)
    out[f"{target_col}_avg2y"] = round(avg, 2)
    for L in lines:
        prob_over = 1 - norm.cdf((L - pred_fu) / sigma)
        out[f"{target_col}_P_over_{L}"] = np.round(prob_over, 3)
    return out


def process_league(league: str, mode: str = "reg"):
    csv_path = ROOT / league / "wszystkie_sezony.csv"
    if not csv_path.exists():
        print(f"  [skip] brak {csv_path}")
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    if "HST" not in df.columns or "AST" not in df.columns:
        print(f"  [skip] brak HST/AST: {league}")
        return None
    df = build_features(df)
    df["TOT_SHOTS"] = pd.to_numeric(df.get("HST"), errors="coerce") + pd.to_numeric(df.get("AST"), errors="coerce")
    df["H_SHOTS"] = pd.to_numeric(df.get("HST"), errors="coerce")
    df["A_SHOTS"] = pd.to_numeric(df.get("AST"), errors="coerce")

    parts = []
    for tcol in ["TOT_SHOTS", "H_SHOTS", "A_SHOTS"]:
        if mode == "reg":
            r = train_one(df, tcol, league)
        else:  # clf
            known = df[df[tcol].notna()]
            last2 = known[known["Sezon"].isin(LAST_2_SEASONS)]
            if len(last2) < 30:
                last2 = known.tail(min(len(known), 600))
            avg = float(last2[tcol].mean())
            lines = thresholds_around(avg)
            r = train_one_clf(df, tcol, league, lines)
        if r is not None:
            parts.append(r.set_index(["Date", "HomeTeam", "AwayTeam", "liga"]))
    if not parts:
        return None
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how="outer")
    merged = merged.reset_index()
    return merged


def main():
    print("=" * 70)
    print(f"  SHOTS O/U — trening i predykcje  ({TODAY.date()})")
    print("=" * 70)
    print()
    print("Wybierz system liczenia:")
    print("  [1] WSZYSTKIE TARGETY  — auto_future_optimizer (FTR/O25/BTTS/kartki/")
    print("                           faule/strzaly/rozne/gole H/A; ~kilkanascie min)")
    print("  [2] STRZALY SZYBKO     — regresja LGBM (3 linie z normal CDF; ~2 min)")
    print("  [3] STRZALY DOKLADNIE  — klasyfikacja LGBM (osobny model na kazda linie;")
    print("                           9 modeli/liga = 189 modeli; ~10-15 min)")
    print()
    choice = input("Twoj wybor [1/2/3, Enter=2]: ").strip() or "2"
    if choice == "1":
        import subprocess
        print("\n>>> Uruchamiam WSZYSTKIE TARGETY...\n")
        subprocess.run([sys.executable, str(ROOT / "auto_future_optimizer.py")], cwd=str(ROOT))
        return
    if choice not in ("2", "3"):
        print("Niepoprawny wybor. Koncze.")
        return
    mode = "reg" if choice == "2" else "clf"
    label = "STRZALY SZYBKO (regresja)" if mode == "reg" else "STRZALY DOKLADNIE (klasyfikacja)"
    print(f"\n>>> Uruchamiam {label}...\n")
    all_parts = []
    t0 = time.time()
    for i, lg in enumerate(LEAGUES, 1):
        t = time.time()
        print(f"[{i}/{len(LEAGUES)}] {lg} ...", end=" ", flush=True)
        try:
            r = process_league(lg, mode=mode)
            if r is not None and len(r) > 0:
                all_parts.append(r)
                print(f"OK ({len(r)} meczy, {time.time()-t:.0f}s)")
            else:
                print("brak danych")
        except Exception as e:
            print(f"BLAD: {e}")
    if not all_parts:
        print("Brak wynikow.")
        return
    full = pd.concat(all_parts, ignore_index=True)
    full = full.sort_values(["Date", "liga", "HomeTeam"])
    suffix = "_reg" if mode == "reg" else "_clf"
    out = OUT_DIR / f"predykcje_{TODAY.date()}{suffix}.csv"
    full.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nZapisano: {out}")
    print(f"Lacznie: {len(full)} przyszlych meczow z {len(all_parts)} lig")
    print(f"Czas: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
