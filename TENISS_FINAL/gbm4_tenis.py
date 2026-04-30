import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.isotonic import IsotonicRegression
import joblib

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
_csv_ext  = SCRIPT_DIR / "stats" / "csv" / "final_z_statami_extended.csv"
_csv_new  = SCRIPT_DIR / "stats" / "csv" / "final_z_statami.csv"
_csv5     = SCRIPT_DIR / "stats" / "csv" / "5stats_projekt.csv"
_csv2     = SCRIPT_DIR / "stats" / "csv" / "final_2_projekt.csv"
CSV_PATH  = str(
    _csv_ext if _csv_ext.exists() else (_csv_new if _csv_new.exists() else (_csv5 if _csv5.exists() else _csv2))
)

DATE_COL = "Date"
TARGET_COL = "target"      # 1 = P1 wygrywa, 0 = P2 wygrywa

OUTPUT_BASE_DIR = str(SCRIPT_DIR / "outputs_tenis")

LGBM_REGRESSOR_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",   # gbdt = deterministyczny (goss był losowy przy n_jobs=-1)
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 31,
    "min_child_samples": 120,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.95,
    "reg_alpha": 0.4,
    "reg_lambda": 8.0,
    "random_state": 42,
    "n_jobs": 1,               # n_jobs=1 gwarantuje pełny determinizm
    "verbose": -1,
}

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

BEST_ELO_K = 40
BEST_ELO_INITIAL = 1500
BEST_ELO_INACTIVITY_UNCERTAINTY = 0.0
BEST_N_LAST_SEASONS = 25
BEST_CLASSIFICATION_THRESHOLD = 0.44
USE_ODDS_FEATURES = True
VALUE_MIN_EDGE = 0.04
VALUE_MIN_EV = 0.03
CALIB_BLEND_WITH_NO_ODDS = True
CALIB_BLEND_WEIGHT_ODDS = 0.50
ODDS_FEATURE_MARKERS = ("odds", "implied_prob", "norm_prob")
BANKROLL_INITIAL = 1000.0
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.03

# Profile stawek zapisywane automatycznie po treningu.
PROFILE_AGGRESSIVE = {
    "profile_name": "aggressive",
    "model_conf_threshold": 0.55,
    "kelly_fraction": 0.55,
    "max_stake_pct": 0.05,
}
PROFILE_RECOMMENDED = {
    "profile_name": "recommended",
    "model_conf_threshold": 0.72,
    "min_edge_prob": 0.03,
    "min_odds": 1.50,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.025,
}
BEST_FEATURE_SET = {
    "glicko_surf_r_diff",
    "glicko_surf_rd_diff",
    "diff_days_rest",
    "diff_surface_change",
    "h2h_win_pct_3",
    "h2h_win_pct_5",
    "h2h_surf_win_pct_3",
    "h2h_surf_win_pct_5",
    "diff_roll_1st_serve_won_pct_form4",
    "diff_roll_2nd_serve_won_pct_form4",
    "diff_roll_bp_saved_pct_form4",
    "diff_roll_bp_conv_pct_form4",
    "diff_roll_return_pts_won_pct_form4",
    "diff_roll_1st_serve_won_pct_form10",
    "diff_roll_2nd_serve_won_pct_form10",
    "diff_roll_bp_saved_pct_form10",
    "diff_roll_bp_conv_pct_form10",
    "diff_roll_return_pts_won_pct_form10",
    "diff_roll_1st_serve_won_pct_form20",
    "diff_roll_2nd_serve_won_pct_form20",
    "diff_roll_bp_saved_pct_form20",
    "diff_roll_bp_conv_pct_form20",
    "diff_roll_return_pts_won_pct_form20",
    "diff_roll_1st_serve_won_pct_trend_4_20",
    "diff_roll_2nd_serve_won_pct_trend_4_20",
    "diff_roll_bp_saved_pct_trend_4_20",
    "diff_roll_bp_conv_pct_trend_4_20",
    "diff_roll_return_pts_won_pct_trend_4_20",
    "diff_roll_1st_serve_won_pct_trend_10_20",
    "diff_roll_2nd_serve_won_pct_trend_10_20",
    "diff_roll_bp_saved_pct_trend_10_20",
    "diff_roll_bp_conv_pct_trend_10_20",
    "diff_roll_return_pts_won_pct_trend_10_20",
    "diff_odds_mean",
    "diff_implied_prob_mean",
    "diff_norm_prob_mean",
    "diff_odds_ps",
    "diff_implied_prob_ps",
    "diff_odds_b365",
    "diff_implied_prob_b365",
}

CSV5_ROLL_BASE = [
    "roll_1st_serve_won_pct",
    "roll_2nd_serve_won_pct",
    "roll_bp_saved_pct",
    "roll_bp_conv_pct",
    "roll_return_pts_won_pct",
]

PLAYER_NAME_ALIASES = {
    "huesler": "huesler ma",
    "sachko": "sachko v",
    "ofner": "ofner s",
    "topo": "topo m",
    "engel": "engel j",
    "dedura-palomero": "dedura-palomero d",
}

ODDS_BOOK_COLUMNS = {
    "ps": ("PSW", "PSL"),
    "b365": ("B365W", "B365L"),
    "max": ("MaxW", "MaxL"),
    "avg": ("AvgW", "AvgL"),
    "bfe": ("BFEW", "BFEL"),
    "ex": ("EXW", "EXL"),
    "lb": ("LBW", "LBL"),
    "sj": ("SJW", "SJL"),
    "cb": ("CBW", "CBL"),
    "gb": ("GBW", "GBL"),
    "iw": ("IWW", "IWL"),
    "sb": ("SBW", "SBL"),
    "ub": ("UBW", "UBL"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_date(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


def extract_pct(value):
    """'83 (66%)' → 66.0  |  plain float → as-is  |  NaN → NaN"""
    if pd.isna(value):
        return np.nan
    import re
    s = str(value).strip()
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", s)
    if m:
        return float(m.group(1))
    try:
        v = float(s)
        return v * 100 if v <= 1 else v
    except ValueError:
        return np.nan


def assign_players(df):
    """
    Tworzy kolumny P1/P2: P1 = zawodnik z niższym (lepszym) WRank.
    Target = 1 jeśli P1 wygrywa (P1 == Winner).
    Dla meczów bez wyniku target = NaN.
    """
    df = df.copy()

    w_rank = pd.to_numeric(df["WRank"], errors="coerce")
    l_rank = pd.to_numeric(df["LRank"], errors="coerce")

    # P1 = niższy rank (lepszy), P2 = wyższy rank (gorszy)
    p1_is_winner = (w_rank <= l_rank) | l_rank.isna()

    df["P1"] = np.where(p1_is_winner, df["Winner"], df["Loser"])
    df["P2"] = np.where(p1_is_winner, df["Loser"], df["Winner"])
    df["P1Rank"] = np.where(p1_is_winner, w_rank, l_rank)
    df["P2Rank"] = np.where(p1_is_winner, l_rank, w_rank)

    # ATP ranking points (re-mapped do P1/P2, żeby nie było leakage z prefiksem W/L)
    if "WPts" in df.columns and "LPts" in df.columns:
        w_pts = pd.to_numeric(df["WPts"], errors="coerce")
        l_pts = pd.to_numeric(df["LPts"], errors="coerce")
        df["P1Pts"] = np.where(p1_is_winner, w_pts, l_pts)
        df["P2Pts"] = np.where(p1_is_winner, l_pts, w_pts)
        df["diff_Pts"] = df["P1Pts"] - df["P2Pts"]

    # Rolling charting stats z final_z_statami.csv (Jeff CSV + TA 2026, shift(1) – brak leakage)
    charting_pairs = [
        ("roll_1st_serve_won_pct",  "W_roll_1st_serve_won_pct",  "L_roll_1st_serve_won_pct"),
        ("roll_2nd_serve_won_pct",  "W_roll_2nd_serve_won_pct",  "L_roll_2nd_serve_won_pct"),
        ("roll_bp_saved_pct",       "W_roll_bp_saved_pct",       "L_roll_bp_saved_pct"),
        ("roll_bp_conv_pct",        "W_roll_bp_conv_pct",        "L_roll_bp_conv_pct"),
        ("roll_return_pts_won_pct", "W_roll_return_pts_won_pct", "L_roll_return_pts_won_pct"),
    ]
    for feat, w_col, l_col in charting_pairs:
        if w_col in df.columns and l_col in df.columns:
            df[f"P1_{feat}"] = np.where(p1_is_winner, df[w_col], df[l_col])
            df[f"P2_{feat}"] = np.where(p1_is_winner, df[l_col], df[w_col])

    # Fallback rankingu/pts dla upcoming: ostatnia znana wartosc gracza
    # (po posortowaniu po dacie, bez leakage na historycznych meczach predykcyjnych).
    def _name_key(v):
        s = str(v).lower().replace(".", "").replace(",", "").strip()
        s = " ".join(s.split())
        return PLAYER_NAME_ALIASES.get(s, s)

    rank_long = pd.concat(
        [
            df[[DATE_COL, "Winner", "WRank"]].rename(columns={"Winner": "player", "WRank": "rank"}),
            df[[DATE_COL, "Loser", "LRank"]].rename(columns={"Loser": "player", "LRank": "rank"}),
        ],
        ignore_index=True,
    )
    rank_long["pkey"] = rank_long["player"].apply(_name_key)
    rank_long["rank"] = pd.to_numeric(rank_long["rank"], errors="coerce")
    rank_map = (
        rank_long.dropna(subset=["pkey", "rank"])
        .sort_values([DATE_COL, "pkey"])
        .groupby("pkey", as_index=False)
        .last()
        .set_index("pkey")["rank"]
        .to_dict()
    )
    df["P1Rank"] = df["P1Rank"].fillna(df["P1"].apply(_name_key).map(rank_map))
    df["P2Rank"] = df["P2Rank"].fillna(df["P2"].apply(_name_key).map(rank_map))

    if "P1Pts" in df.columns and "P2Pts" in df.columns:
        pts_long = pd.concat(
            [
                df[[DATE_COL, "Winner", "WPts"]].rename(columns={"Winner": "player", "WPts": "pts"}),
                df[[DATE_COL, "Loser", "LPts"]].rename(columns={"Loser": "player", "LPts": "pts"}),
            ],
            ignore_index=True,
        )
        pts_long["pkey"] = pts_long["player"].apply(_name_key)
        pts_long["pts"] = pd.to_numeric(pts_long["pts"], errors="coerce")
        pts_map = (
            pts_long.dropna(subset=["pkey", "pts"])
            .sort_values([DATE_COL, "pkey"])
            .groupby("pkey", as_index=False)
            .last()
            .set_index("pkey")["pts"]
            .to_dict()
        )
        df["P1Pts"] = df["P1Pts"].fillna(df["P1"].apply(_name_key).map(pts_map))
        df["P2Pts"] = df["P2Pts"].fillna(df["P2"].apply(_name_key).map(pts_map))
        df["diff_Pts"] = df["P1Pts"] - df["P2Pts"]

    # Target: 1 = P1 wygrywa, 0 = P2 wygrywa
    # Brak wyniku = Upcoming lub Sched; wszystko inne traktujemy jako rozegrany mecz
    no_result = (
        df["Comment"].str.strip().str.lower().isin(["upcoming", "sched"])
        if "Comment" in df.columns
        else pd.Series(False, index=df.index)
    )
    has_result = ~no_result
    df[TARGET_COL] = np.where(
        has_result,
        np.where(p1_is_winner, 1, 0),
        np.nan
    )
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje cechy kursowe po stronie P1/P2 (bez leakage):
    mapowanie W/L -> P1/P2 przez relację P1==Winner.
    """
    df = df.copy()
    if "P1" not in df.columns or "Winner" not in df.columns:
        return df

    p1_is_winner_side = df["P1"].eq(df["Winner"])
    p1_odds_cols = []
    p2_odds_cols = []
    p1_imp_cols = []
    p2_imp_cols = []

    for bk, (w_col, l_col) in ODDS_BOOK_COLUMNS.items():
        if w_col not in df.columns or l_col not in df.columns:
            continue
        w = pd.to_numeric(df[w_col], errors="coerce")
        l = pd.to_numeric(df[l_col], errors="coerce")
        p1_odds = pd.Series(np.where(p1_is_winner_side, w, l), index=df.index)
        p2_odds = pd.Series(np.where(p1_is_winner_side, l, w), index=df.index)

        # Kursy <= 1 traktujemy jako brak danych.
        p1_odds = p1_odds.where(p1_odds > 1.0, np.nan)
        p2_odds = p2_odds.where(p2_odds > 1.0, np.nan)

        p1_col = f"P1_odds_{bk}"
        p2_col = f"P2_odds_{bk}"
        imp1_col = f"P1_implied_prob_{bk}"
        imp2_col = f"P2_implied_prob_{bk}"
        diff_odds_col = f"diff_odds_{bk}"
        diff_imp_col = f"diff_implied_prob_{bk}"

        df[p1_col] = p1_odds
        df[p2_col] = p2_odds
        df[imp1_col] = 1.0 / p1_odds
        df[imp2_col] = 1.0 / p2_odds
        df[diff_odds_col] = df[p1_col] - df[p2_col]
        df[diff_imp_col] = df[imp1_col] - df[imp2_col]

        p1_odds_cols.append(p1_col)
        p2_odds_cols.append(p2_col)
        p1_imp_cols.append(imp1_col)
        p2_imp_cols.append(imp2_col)

    if p1_odds_cols and p2_odds_cols:
        df["P1_odds_mean"] = df[p1_odds_cols].mean(axis=1)
        df["P2_odds_mean"] = df[p2_odds_cols].mean(axis=1)
        df["diff_odds_mean"] = df["P1_odds_mean"] - df["P2_odds_mean"]

    if p1_imp_cols and p2_imp_cols:
        df["P1_implied_prob_mean"] = df[p1_imp_cols].mean(axis=1)
        df["P2_implied_prob_mean"] = df[p2_imp_cols].mean(axis=1)
        df["diff_implied_prob_mean"] = df["P1_implied_prob_mean"] - df["P2_implied_prob_mean"]
        margin = df["P1_implied_prob_mean"] + df["P2_implied_prob_mean"]
        df["P1_norm_prob_mean"] = df["P1_implied_prob_mean"] / margin.replace(0, np.nan)
        df["P2_norm_prob_mean"] = df["P2_implied_prob_mean"] / margin.replace(0, np.nan)
        df["diff_norm_prob_mean"] = df["P1_norm_prob_mean"] - df["P2_norm_prob_mean"]

    return df


# ---------------------------------------------------------------------------
# Rolling features per player
# ---------------------------------------------------------------------------

def build_player_long(df):
    """Rozwiń do formatu long: jeden wiersz per gracz per mecz."""
    set_cols_w = ["W1", "W2", "W3", "W4", "W5"]
    set_cols_l = ["L1", "L2", "L3", "L4", "L5"]
    for c in set_cols_w + set_cols_l:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    total_games = df[set_cols_w].sum(axis=1) + df[set_cols_l].sum(axis=1)

    w_rows = pd.DataFrame({
        "match_id": df.index,
        "Date": df[DATE_COL],
        "Surface": df["Surface"],
        "player": df["Winner"],
        "won": 1,
        "sets_won": df["Wsets"].fillna(0),
        "sets_lost": df["Lsets"].fillna(0),
        "games_won": df[set_cols_w].sum(axis=1),
        "games_lost": df[set_cols_l].sum(axis=1),
        "games_total": total_games,
        "rank": pd.to_numeric(df["WRank"], errors="coerce"),
    })
    l_rows = pd.DataFrame({
        "match_id": df.index,
        "Date": df[DATE_COL],
        "Surface": df["Surface"],
        "player": df["Loser"],
        "won": 0,
        "sets_won": df["Lsets"].fillna(0),
        "sets_lost": df["Wsets"].fillna(0),
        "games_won": df[set_cols_l].sum(axis=1),
        "games_lost": df[set_cols_w].sum(axis=1),
        "games_total": total_games,
        "rank": pd.to_numeric(df["LRank"], errors="coerce"),
    })
    long = pd.concat([w_rows, l_rows], ignore_index=True).sort_values(["player", "Date", "match_id"])
    return long


def add_rolling_features(long):
    long = long.copy()
    windows = [3, 5, 10, 20]
    metrics = ["won", "sets_won", "sets_lost", "games_won", "games_lost"]
    surf_key = long["Surface"].fillna("Unknown").astype(str).str.strip()
    surf_key = surf_key.where(surf_key != "", "Unknown")

    for metric in metrics:
        shifted = long.groupby("player")[metric].shift(1)
        for w in windows:
            rolled = shifted.groupby(long["player"]).rolling(w, min_periods=1).mean()
            long[f"{metric}_avg{w}"] = rolled.reset_index(level=0, drop=True).values

    for metric in metrics:
        long[f"momentum_{metric}"] = long[f"{metric}_avg5"] - long[f"{metric}_avg20"]

    # Na nawierzchni
    for metric in ["won", "sets_won", "games_won"]:
        shifted = long.groupby([long["player"], surf_key])[metric].shift(1)
        for w in [5, 10]:
            rolled = shifted.groupby([long["player"], surf_key]).rolling(w, min_periods=1).mean()
            long[f"{metric}_surf_avg{w}"] = rolled.reset_index(level=[0, 1], drop=True).reindex(long.index).values

    return long


def merge_player_features(df, long):
    df = df.copy()
    df["match_id"] = df.index

    roll_cols = [c for c in long.columns if ("avg" in c or "momentum_" in c)]

    p1_feats = long.merge(
        df[["match_id", "P1"]].rename(columns={"P1": "player"}),
        on=["match_id", "player"], how="inner"
    )[["match_id"] + roll_cols].rename(columns={c: f"P1_{c}" for c in roll_cols})

    p2_feats = long.merge(
        df[["match_id", "P2"]].rename(columns={"P2": "player"}),
        on=["match_id", "player"], how="inner"
    )[["match_id"] + roll_cols].rename(columns={c: f"P2_{c}" for c in roll_cols})

    df = df.merge(p1_feats, on="match_id", how="left")
    df = df.merge(p2_feats, on="match_id", how="left")

    # Różnice P1 - P2
    for col in roll_cols:
        p1c = f"P1_{col}"
        p2c = f"P2_{col}"
        if p1c in df.columns and p2c in df.columns:
            df[f"diff_{col}"] = df[p1c] - df[p2c]

    return df


def add_csv5_form4_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = df.index
    required_pairs = [
        ("roll_1st_serve_won_pct", "W_roll_1st_serve_won_pct", "L_roll_1st_serve_won_pct"),
        ("roll_2nd_serve_won_pct", "W_roll_2nd_serve_won_pct", "L_roll_2nd_serve_won_pct"),
        ("roll_bp_saved_pct", "W_roll_bp_saved_pct", "L_roll_bp_saved_pct"),
        ("roll_bp_conv_pct", "W_roll_bp_conv_pct", "L_roll_bp_conv_pct"),
        ("roll_return_pts_won_pct", "W_roll_return_pts_won_pct", "L_roll_return_pts_won_pct"),
    ]
    available = [(base, w_col, l_col) for base, w_col, l_col in required_pairs if w_col in df.columns and l_col in df.columns]
    if len(available) == 0:
        return df
    w_data = {"match_id": df["match_id"], "Date": df[DATE_COL], "player": df["Winner"]}
    l_data = {"match_id": df["match_id"], "Date": df[DATE_COL], "player": df["Loser"]}
    for base, w_col, l_col in available:
        w_data[base] = pd.to_numeric(df[w_col], errors="coerce")
        l_data[base] = pd.to_numeric(df[l_col], errors="coerce")
    long = pd.concat([pd.DataFrame(w_data), pd.DataFrame(l_data)], ignore_index=True)
    long = long.sort_values(["player", "Date", "match_id"]).reset_index(drop=True)
    form_windows = [4, 10, 20]
    form_cols = []
    for base, _, _ in available:
        shifted = long.groupby("player")[base].shift(1)
        for w in form_windows:
            rolled = shifted.groupby(long["player"]).rolling(w, min_periods=1).mean()
            col = f"{base}_form{w}"
            long[col] = rolled.reset_index(level=0, drop=True).values
            form_cols.append(col)
    p1 = long.merge(
        df[["match_id", "P1"]].rename(columns={"P1": "player"}),
        on=["match_id", "player"],
        how="inner",
    )[["match_id"] + form_cols].rename(columns={c: f"P1_{c}" for c in form_cols})
    p2 = long.merge(
        df[["match_id", "P2"]].rename(columns={"P2": "player"}),
        on=["match_id", "player"],
        how="inner",
    )[["match_id"] + form_cols].rename(columns={c: f"P2_{c}" for c in form_cols})
    df = df.merge(p1, on="match_id", how="left")
    df = df.merge(p2, on="match_id", how="left")
    for col in form_cols:
        p1c = f"P1_{col}"
        p2c = f"P2_{col}"
        if p1c in df.columns and p2c in df.columns:
            df[f"diff_{col}"] = df[p1c] - df[p2c]
    # Trend formy: krotkie okno minus dlugie okno
    trend_pairs = [(4, 20), (10, 20)]
    for base, _, _ in available:
        for short_w, long_w in trend_pairs:
            p1_short = f"P1_{base}_form{short_w}"
            p1_long = f"P1_{base}_form{long_w}"
            p2_short = f"P2_{base}_form{short_w}"
            p2_long = f"P2_{base}_form{long_w}"
            p1_trend = f"P1_{base}_trend_{short_w}_{long_w}"
            p2_trend = f"P2_{base}_trend_{short_w}_{long_w}"
            diff_trend = f"diff_{base}_trend_{short_w}_{long_w}"
            if p1_short in df.columns and p1_long in df.columns:
                df[p1_trend] = df[p1_short] - df[p1_long]
            if p2_short in df.columns and p2_long in df.columns:
                df[p2_trend] = df[p2_short] - df[p2_long]
            if p1_trend in df.columns and p2_trend in df.columns:
                df[diff_trend] = df[p1_trend] - df[p2_trend]
    return df


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def compute_elo(
    df,
    k=32,
    initial=1500,
    surface_specific=False,
    inactivity_uncertainty=0.0,
    inactivity_cap_days=120,
):
    ratings = {}
    last_played = {}
    p1_elos, p2_elos = [], []
    suffix = "_surf" if surface_specific else ""

    known = df[TARGET_COL].notna()

    for idx, row in df.iterrows():
        p1, p2 = row["P1"], row["P2"]
        surf = row["Surface"] if surface_specific else "all"
        key1 = (p1, surf)
        key2 = (p2, surf)

        r1 = ratings.get(key1, initial)
        r2 = ratings.get(key2, initial)
        p1_elos.append(r1)
        p2_elos.append(r2)

        if not known[idx] or pd.isna(row[TARGET_COL]):
            continue

        match_date = row[DATE_COL]
        last_date1 = last_played.get(key1)
        last_date2 = last_played.get(key2)
        days1 = 0 if last_date1 is None else max((match_date - last_date1).days, 0)
        days2 = 0 if last_date2 is None else max((match_date - last_date2).days, 0)
        inactive1 = min(days1, inactivity_cap_days)
        inactive2 = min(days2, inactivity_cap_days)
        k1 = k * (1.0 + inactivity_uncertainty * inactive1)
        k2 = k * (1.0 + inactivity_uncertainty * inactive2)

        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        s1 = float(row[TARGET_COL])

        ratings[key1] = r1 + k1 * (s1 - e1)
        ratings[key2] = r2 + k2 * ((1 - s1) - (1 - e1))
        last_played[key1] = match_date
        last_played[key2] = match_date

    df = df.copy()
    df[f"elo{suffix}_P1"] = p1_elos
    df[f"elo{suffix}_P2"] = p2_elos
    df[f"elo{suffix}_diff"] = df[f"elo{suffix}_P1"] - df[f"elo{suffix}_P2"]
    return df


def compute_glicko(df, initial_r=1500, initial_rd=350, min_rd=30, surface_specific=False):
    q = math.log(10) / 400
    ratings = {}
    p1_r, p1_rd, p2_r, p2_rd = [], [], [], []
    suffix = "_surf" if surface_specific else ""

    def _g(rd):
        return 1 / math.sqrt(1 + 3 * q**2 * rd**2 / math.pi**2)

    def _e(r, ro, rdo):
        return 1 / (1 + 10 ** (-_g(rdo) * (r - ro) / 400))

    for idx, row in df.iterrows():
        p1, p2 = row["P1"], row["P2"]
        surf = row["Surface"] if surface_specific else "all"
        k1 = (p1, surf)
        k2 = (p2, surf)

        r1, rd1 = ratings.get(k1, (initial_r, initial_rd))
        r2, rd2 = ratings.get(k2, (initial_r, initial_rd))
        p1_r.append(r1); p1_rd.append(rd1)
        p2_r.append(r2); p2_rd.append(rd2)

        if pd.isna(row[TARGET_COL]):
            continue

        s1 = float(row[TARGET_COL])
        s2 = 1 - s1

        g2 = _g(rd2)
        e1 = _e(r1, r2, rd2)
        d2_1 = 1 / (q**2 * g2**2 * e1 * (1 - e1))
        new_r1 = r1 + q / (1/rd1**2 + 1/d2_1) * g2 * (s1 - e1)
        new_rd1 = max(math.sqrt(1 / (1/rd1**2 + 1/d2_1)), min_rd)

        g1 = _g(rd1)
        e2 = _e(r2, r1, rd1)
        d2_2 = 1 / (q**2 * g1**2 * e2 * (1 - e2))
        new_r2 = r2 + q / (1/rd2**2 + 1/d2_2) * g1 * (s2 - e2)
        new_rd2 = max(math.sqrt(1 / (1/rd2**2 + 1/d2_2)), min_rd)

        ratings[k1] = (new_r1, new_rd1)
        ratings[k2] = (new_r2, new_rd2)

    df = df.copy()
    df[f"glicko{suffix}_P1_r"] = p1_r
    df[f"glicko{suffix}_P1_rd"] = p1_rd
    df[f"glicko{suffix}_P2_r"] = p2_r
    df[f"glicko{suffix}_P2_rd"] = p2_rd
    df[f"glicko{suffix}_r_diff"] = df[f"glicko{suffix}_P1_r"] - df[f"glicko{suffix}_P2_r"]
    df[f"glicko{suffix}_rd_diff"] = df[f"glicko{suffix}_P1_rd"] - df[f"glicko{suffix}_P2_rd"]
    return df


# ---------------------------------------------------------------------------
# H2H
# ---------------------------------------------------------------------------

def add_h2h_features(df):
    df = df.copy()
    df["match_id"] = df.index

    records = []
    for idx, row in df.iterrows():
        won = np.nan if pd.isna(row[TARGET_COL]) else float(row[TARGET_COL])
        records.append({
            "match_id": idx,
            "P1": row["P1"],
            "P2": row["P2"],
            "Surface": row["Surface"],
            "h2h_won": won,
        })

    h2h = pd.DataFrame(records).sort_values("match_id")

    # H2H ogólny
    for w in [3, 5, 10]:
        h2h[f"h2h_win_pct_{w}"] = (
            h2h.groupby(["P1", "P2"])["h2h_won"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    h2h["h2h_momentum"] = h2h["h2h_win_pct_5"] - h2h["h2h_win_pct_10"]

    # H2H na nawierzchni
    for w in [3, 5]:
        h2h[f"h2h_surf_win_pct_{w}"] = (
            h2h.groupby(["P1", "P2", "Surface"])["h2h_won"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    h2h["h2h_surf_momentum"] = h2h["h2h_surf_win_pct_5"] - h2h["h2h_win_pct_5"]

    keep = [
        "match_id",
        "h2h_win_pct_3", "h2h_win_pct_5", "h2h_win_pct_10", "h2h_momentum",
        "h2h_surf_win_pct_3", "h2h_surf_win_pct_5", "h2h_surf_momentum",
    ]
    df = df.merge(h2h[keep], on="match_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def seasonal_split(unique_seasons, mode="auto"):
    seasons = sorted(unique_seasons)
    n = len(seasons)
    if n < 4:
        raise ValueError("Za mało sezonów do sensownego splitu")

    if mode == "80_10_10":
        n_test = max(1, round(n * 0.10))
        n_valid = max(1, round(n * 0.10))
    elif mode == "75_12_5_12_5":
        n_test = max(1, round(n * 0.125))
        n_valid = max(1, round(n * 0.125))
    else:
        if n <= 8:
            n_valid, n_test = 1, 1
        elif n <= 12:
            n_valid, n_test = 1, 1
        elif n <= 19:
            n_valid, n_test = 1, 2
        else:
            n_test = max(1, round(n * 0.10))
            n_valid = max(1, round(n * 0.10))

    n_train = n - n_valid - n_test
    if n_train < 2:
        raise ValueError("Za mało sezonów treningowych po splicie")
    return seasons[:n_train], seasons[n_train:n_train + n_valid], seasons[n_train + n_valid:]


def get_feature_columns(df):
    skip = {
        DATE_COL, TARGET_COL, "Season", "P1", "P2", "Winner", "Loser",
        "match_id", "Comment", "Location", "Tournament", "ATP",
        "Series", "Court", "Round", "Surface",
        "WRank", "LRank", "WPts", "LPts",
        "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5",
        "Wsets", "Lsets", "Best of",
        "total_games",  # target regresji — nie może być featurem klasyfikatora
    }
    odds_prefixes = (
        "CB", "GB", "IW", "LB", "SB", "WH", "B365", "BW", "SJ", "PS",
        "Max", "Avg", "BFE", "BF", "UB", "EX", "LB",
    )
    # Wyklucz surowe kolumny źródłowe (W_roll_* i L_roll_* → używamy P1_/P2_/diff_ pochodnych)
    charting_raw = (
        "winner_charting_", "loser_charting_",   # stare career stats z TA
        "W_roll_", "L_roll_",                    # surowe rolling — używamy P1_/P2_ zamiast
        "_season",                               # wewnętrzna kolumna pipeline
    )

    allowed = BEST_FEATURE_SET

    feature_cols = []
    for col in df.columns:
        if col in skip:
            continue
        if any(col.startswith(p) for p in odds_prefixes):
            continue
        if any(col.startswith(p) for p in charting_raw):
            continue
        if col not in allowed:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def remove_odds_features(feature_names):
    return [c for c in feature_names if not any(marker in c for marker in ODDS_FEATURE_MARKERS)]


def filter_features(train_df, valid_df, test_df, future_df, feature_cols):
    dropped = []
    keep = []

    for col in feature_cols:
        miss = train_df[col].isna().mean()
        uniq = train_df[col].nunique(dropna=False)
        if miss > 0.85:
            dropped.append((col, "too_many_missing"))
            continue
        if uniq <= 1:
            dropped.append((col, "constant"))
            continue
        keep.append(col)

    X_train = train_df[keep].replace([np.inf, -np.inf], np.nan).copy()
    X_valid = valid_df[keep].replace([np.inf, -np.inf], np.nan).copy()
    X_test = test_df[keep].replace([np.inf, -np.inf], np.nan).copy()
    X_future = future_df[keep].replace([np.inf, -np.inf], np.nan).copy() if len(future_df) > 0 else pd.DataFrame(columns=keep)

    if len(keep) > 1:
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Trendy chcemy zachowac jako sygnal kierunku zmian formy.
        protected = {c for c in keep if "_trend_" in c}
        to_drop = [c for c in upper.columns if c not in protected and any(upper[c] > 0.97)]
        for c in to_drop:
            dropped.append((c, "high_corr"))
        keep = [c for c in keep if c not in to_drop]
        X_train = X_train[keep]
        X_valid = X_valid[keep]
        X_test = X_test[keep]
        X_future = X_future[keep] if len(X_future) > 0 else X_future

    return X_train, X_valid, X_test, X_future, keep, dropped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_brier_score(y_true, proba_pos):
    return np.mean((proba_pos - y_true) ** 2)


def calibration_table_binary(y_true, pred_prob, step=0.05):
    bins = np.arange(0.0, 1.0 + step, step)
    rows = []
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (pred_prob >= left) & (pred_prob < right) if i < len(bins) - 2 else (pred_prob >= left) & (pred_prob <= right)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_left": left, "bin_right": right, "count": 0, "avg_pred": np.nan, "actual_rate": np.nan, "diff": np.nan})
            continue
        avg_pred = float(pred_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        rows.append({"bin_left": left, "bin_right": right, "count": n, "avg_pred": avg_pred, "actual_rate": actual_rate, "diff": actual_rate - avg_pred})
    return pd.DataFrame(rows)


def expected_calibration_error(calib_df):
    df = calib_df.dropna().copy()
    if df.empty:
        return np.nan
    total = df["count"].sum()
    return np.sum((df["count"] / total) * np.abs(df["actual_rate"] - df["avg_pred"]))


def build_threshold_report(y_true, proba_p1, thresholds=None):
    """
    Raport progow dla symetrycznej decyzji:
      kwalifikuje mecz gdy max(P1, P2) >= threshold
      typ = strona o wyzszym prawdopodobienstwie.
    """
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    p1 = pd.Series(proba_p1).astype(float).reset_index(drop=True)
    p2 = 1.0 - p1
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 0.76, 0.01), 2)
    rows = []
    for t in thresholds:
        qualified = (p1 >= t) | (p2 >= t)
        n_q = int(qualified.sum())
        coverage = float(qualified.mean() * 100.0)
        if n_q == 0:
            acc = np.nan
            p1_share = np.nan
        else:
            pred = (p1 >= p2).astype(int)
            acc = float((pred[qualified] == y[qualified]).mean() * 100.0)
            p1_share = float((pred[qualified] == 1).mean() * 100.0)
        rows.append({
            "threshold": float(t),
            "qualified_matches": n_q,
            "coverage_pct": round(coverage, 2),
            "accuracy_pct": round(acc, 2) if pd.notna(acc) else np.nan,
            "p1_pick_share_pct": round(p1_share, 2) if pd.notna(p1_share) else np.nan,
        })
    return pd.DataFrame(rows)


def add_value_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje kolumny value-bet:
      - wybrana strona (P1/P2),
      - kurs wybranej strony,
      - implied probability,
      - edge (p_model - p_implied),
      - EV,
      - flaga zakładu wg progów.
    """
    df = pred_df.copy()
    side = np.where(df["prob_P1_wins"] >= df["prob_P2_wins"], "P1", "P2")
    p_model = np.where(side == "P1", df["prob_P1_wins"], df["prob_P2_wins"])
    odds = np.where(side == "P1", df.get("P1_odds_mean", np.nan), df.get("P2_odds_mean", np.nan))
    odds = pd.to_numeric(pd.Series(odds, index=df.index), errors="coerce")
    p_model = pd.to_numeric(pd.Series(p_model, index=df.index), errors="coerce")
    implied = 1.0 / odds.replace(0, np.nan)
    edge = p_model - implied
    ev = p_model * (odds - 1.0) - (1.0 - p_model)
    df["pick_side"] = side
    df["pick_odds"] = odds
    df["pick_implied_prob"] = implied
    df["pick_edge_prob"] = edge
    df["pick_ev"] = ev
    df["bet_flag"] = (
        odds.notna()
        & (odds > 1.01)
        & edge.notna()
        & ev.notna()
        & (edge >= VALUE_MIN_EDGE)
        & (ev >= VALUE_MIN_EV)
    ).astype(int)
    return df


def build_value_report(test_pred_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """
    Raport value na teście:
      - próg pewności modelu (max prob),
      - liczba zakładów,
      - hit-rate,
      - ROI (flat stake 1u).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.55, 0.76, 0.01), 2)
    rows = []
    base = test_pred_df.copy()
    base["model_conf"] = base[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)
    base["pick_correct"] = np.where(
        (base["pick_side"] == "P1") & (base[TARGET_COL] == 1), 1,
        np.where((base["pick_side"] == "P2") & (base[TARGET_COL] == 0), 1, 0),
    )
    for t in thresholds:
        sel = base[(base["bet_flag"] == 1) & (base["model_conf"] >= t)].copy()
        n = len(sel)
        if n == 0:
            rows.append({
                "threshold": float(t),
                "bets": 0,
                "hit_rate_pct": np.nan,
                "roi_pct": np.nan,
                "avg_edge_pct": np.nan,
                "avg_ev_pct": np.nan,
            })
            continue
        profit = np.where(sel["pick_correct"] == 1, sel["pick_odds"] - 1.0, -1.0)
        roi = float(np.nanmean(profit) * 100.0)
        hit = float(sel["pick_correct"].mean() * 100.0)
        rows.append({
            "threshold": float(t),
            "bets": int(n),
            "hit_rate_pct": round(hit, 2),
            "roi_pct": round(roi, 2),
            "avg_edge_pct": round(float(sel["pick_edge_prob"].mean() * 100.0), 2),
            "avg_ev_pct": round(float(sel["pick_ev"].mean() * 100.0), 2),
        })
    return pd.DataFrame(rows)


def simulate_bankroll_kelly(
    bets_df: pd.DataFrame,
    initial_bankroll=BANKROLL_INITIAL,
    kelly_fraction=KELLY_FRACTION,
    max_stake_pct=MAX_STAKE_PCT,
):
    """
    Symulacja bankrollu dla listy zakładów:
      - Kelly fraction (fractional Kelly),
      - cap procentowy na stawkę.
    """
    if len(bets_df) == 0:
        return pd.DataFrame(), {
            "bets": 0,
            "final_bankroll": float(initial_bankroll),
            "profit": 0.0,
            "roi_on_bankroll_pct": 0.0,
            "yield_on_staked_pct": np.nan,
            "max_drawdown_pct": 0.0,
            "avg_stake_pct": 0.0,
            "total_staked": 0.0,
        }

    work = bets_df.copy()
    if DATE_COL in work.columns:
        work[DATE_COL] = pd.to_datetime(work[DATE_COL], format="mixed", errors="coerce")
        work = work.sort_values([DATE_COL], na_position="last").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    bankroll = float(initial_bankroll)
    peak = bankroll
    max_dd = 0.0
    total_staked = 0.0
    rows = []

    for i, r in work.iterrows():
        odds = float(r.get("pick_odds", np.nan))
        p = float(max(0.0, min(1.0, r.get("model_conf", np.nan))))
        correct = int(r.get("pick_correct", 0))
        if np.isnan(odds) or odds <= 1.01 or np.isnan(p):
            continue
        b = odds - 1.0
        k_full = (p * odds - 1.0) / b
        stake_frac = max(0.0, float(kelly_fraction) * k_full)
        stake_frac = min(stake_frac, float(max_stake_pct))
        stake = bankroll * stake_frac
        pnl = stake * b if correct == 1 else -stake

        bankroll_before = bankroll
        bankroll = bankroll + pnl
        peak = max(peak, bankroll)
        dd = 0.0 if peak <= 0 else (peak - bankroll) / peak
        max_dd = max(max_dd, dd)
        total_staked += stake

        rows.append({
            "bet_no": i + 1,
            DATE_COL: r.get(DATE_COL, pd.NaT),
            "P1": r.get("P1", np.nan),
            "P2": r.get("P2", np.nan),
            "pick_side": r.get("pick_side", np.nan),
            "pick_odds": odds,
            "model_conf": p,
            "pick_ev": r.get("pick_ev", np.nan),
            "pick_correct": correct,
            "stake_frac": stake_frac,
            "stake_units": stake,
            "pnl_units": pnl,
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll,
            "drawdown_pct": dd * 100.0,
        })

    curve_df = pd.DataFrame(rows)
    profit = bankroll - float(initial_bankroll)
    roi_bankroll_pct = 0.0 if initial_bankroll == 0 else (profit / float(initial_bankroll)) * 100.0
    yield_pct = np.nan if total_staked <= 0 else (profit / total_staked) * 100.0
    avg_stake_pct = 0.0 if len(curve_df) == 0 else float(curve_df["stake_frac"].mean() * 100.0)

    stats = {
        "bets": int(len(curve_df)),
        "final_bankroll": float(bankroll),
        "profit": float(profit),
        "roi_on_bankroll_pct": float(roi_bankroll_pct),
        "yield_on_staked_pct": float(yield_pct) if pd.notna(yield_pct) else np.nan,
        "max_drawdown_pct": float(max_dd * 100.0),
        "avg_stake_pct": float(avg_stake_pct),
        "total_staked": float(total_staked),
    }
    return curve_df, stats


def build_bankroll_report(test_pred_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """
    Raport bankrollu po progach confidence:
      - przyrost bankrollu,
      - max drawdown,
      - yield na stawkowanym kapitale.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.55, 0.76, 0.01), 2)
    base = test_pred_df.copy()
    base["model_conf"] = base[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)
    base["pick_correct"] = np.where(
        (base["pick_side"] == "P1") & (base[TARGET_COL] == 1), 1,
        np.where((base["pick_side"] == "P2") & (base[TARGET_COL] == 0), 1, 0),
    )
    rows = []
    for t in thresholds:
        sel = base[(base["bet_flag"] == 1) & (base["model_conf"] >= t)].copy()
        _, stats = simulate_bankroll_kelly(sel)
        rows.append({
            "threshold": float(t),
            "bets": int(stats["bets"]),
            "final_bankroll": round(float(stats["final_bankroll"]), 2),
            "profit_units": round(float(stats["profit"]), 2),
            "roi_on_bankroll_pct": round(float(stats["roi_on_bankroll_pct"]), 2),
            "yield_on_staked_pct": round(float(stats["yield_on_staked_pct"]), 2) if pd.notna(stats["yield_on_staked_pct"]) else np.nan,
            "max_drawdown_pct": round(float(stats["max_drawdown_pct"]), 2),
            "avg_stake_pct": round(float(stats["avg_stake_pct"]), 2),
            "total_staked": round(float(stats["total_staked"]), 2),
        })
    return pd.DataFrame(rows)


def save_staking_profiles_csv(output_dir: str) -> None:
    """
    Zapisuje dwa osobne profile stawek do CSV:
      - agresywny,
      - rekomendowany (stabilniejszy).
    """
    aggressive_df = pd.DataFrame([PROFILE_AGGRESSIVE])
    recommended_df = pd.DataFrame([PROFILE_RECOMMENDED])
    aggressive_df.to_csv(os.path.join(output_dir, "staking_profile_aggressive.csv"), index=False)
    recommended_df.to_csv(os.path.join(output_dir, "staking_profile_recommended.csv"), index=False)


def save_future_profile_predictions(fut_pred_df: pd.DataFrame, output_dir: str) -> None:
    """
    Zapisuje osobne selekcje przyszłych predykcji dla 2 profili:
      - aggressive
      - recommended
    """
    df = fut_pred_df.copy()
    df["model_conf"] = df[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)

    aggr_sel = df[
        (df["bet_flag"] == 1)
        & (df["model_conf"] >= PROFILE_AGGRESSIVE["model_conf_threshold"])
    ].copy()
    rec_sel = df[
        (df["bet_flag"] == 1)
        & (df["model_conf"] >= PROFILE_RECOMMENDED["model_conf_threshold"])
        & (pd.to_numeric(df["pick_edge_prob"], errors="coerce") >= PROFILE_RECOMMENDED["min_edge_prob"])
        & (pd.to_numeric(df["pick_odds"], errors="coerce") >= PROFILE_RECOMMENDED["min_odds"])
    ].copy()

    aggr_sel.to_csv(os.path.join(output_dir, "future_predictions_aggressive.csv"), index=False)
    rec_sel.to_csv(os.path.join(output_dir, "future_predictions_recommended.csv"), index=False)


def print_separator(title=None):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


# ---------------------------------------------------------------------------
# Fatigue
# ---------------------------------------------------------------------------

def _add_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """Liczba meczów rozegranych przez każdego gracza w ostatnich 7 i 14 dniach."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df[DATE_COL], format="mixed", errors="coerce")

    # Long format
    w = df[["_date", "Winner"]].rename(columns={"Winner": "player"})
    l = df[["_date", "Loser"]].rename(columns={"Loser": "player"})
    long = pd.concat([w, l], ignore_index=True).dropna(subset=["_date"]).sort_values("_date")

    # Per gracz: posortowane daty → searchsorted dla szybkiego zliczania okna
    player_dates: dict[str, np.ndarray] = {
        p: g["_date"].values.astype("datetime64[D]")
        for p, g in long.groupby("player")
    }

    def count_window(player, date, days):
        dates = player_dates.get(player)
        if dates is None or pd.isna(date):
            return 0
        d = np.datetime64(date, "D")
        cutoff = d - np.timedelta64(days, "D")
        hi = int(np.searchsorted(dates, d, side="left"))       # exclude current
        lo = int(np.searchsorted(dates, cutoff, side="left"))  # include cutoff
        return hi - lo

    for side in ["P1", "P2"]:
        df[f"{side}_matches_7d"]  = [count_window(r[side], r["_date"], 7)  for _, r in df.iterrows()]
        df[f"{side}_matches_14d"] = [count_window(r[side], r["_date"], 14) for _, r in df.iterrows()]

    df["diff_matches_7d"]  = df["P1_matches_7d"]  - df["P2_matches_7d"]
    df["diff_matches_14d"] = df["P1_matches_14d"] - df["P2_matches_14d"]
    df.drop(columns=["_date"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Surface transition + rest days
# ---------------------------------------------------------------------------

def _add_surface_transition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per gracz: nawierzchnia poprzedniego meczu + dni od ostatniego meczu.
    surface_change = 1 jeśli gracz przyszedł z innej nawierzchni.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[DATE_COL], format="mixed", errors="coerce")
    df["_mid"] = np.arange(len(df))

    w = df[["_mid", "_date", "Winner", "Surface"]].rename(columns={"Winner": "player"})
    l = df[["_mid", "_date", "Loser",  "Surface"]].rename(columns={"Loser": "player"})
    long = pd.concat([w, l], ignore_index=True).dropna(subset=["_date"]).sort_values(["player", "_date", "_mid"])

    surf_enc = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
    long["prev_surface"] = long.groupby("player")["Surface"].shift(1)
    long["prev_date"]    = long.groupby("player")["_date"].shift(1)
    long["surface_change"]    = (long["Surface"] != long["prev_surface"]).astype(float)
    long["prev_surface_enc"]  = long["prev_surface"].map(surf_enc)
    long["days_rest"] = (long["_date"] - long["prev_date"]).dt.days

    # Merge przez match_id + player (P1/P2 — nie Winner/Loser, bo P1 != zawsze Winner)
    for side in ["P1", "P2"]:
        merged = (
            df[["_mid", side]]
            .merge(
                long[["_mid", "player", "surface_change", "days_rest", "prev_surface_enc"]],
                left_on=["_mid", side],
                right_on=["_mid", "player"],
                how="left"
            )
            .drop_duplicates(subset=["_mid"])
        )
        df[f"{side}_surface_change"]   = merged["surface_change"].values
        df[f"{side}_days_rest"]        = merged["days_rest"].values
        df[f"{side}_prev_surface_enc"] = merged["prev_surface_enc"].values

    df["diff_surface_change"] = df["P1_surface_change"] - df["P2_surface_change"]
    df["diff_days_rest"]      = df["P1_days_rest"]      - df["P2_days_rest"]
    df.drop(columns=["_date", "_mid"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Rank filling
# ---------------------------------------------------------------------------

def _fill_missing_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnij brakujące WRank/LRank poprzednim znanym rankingiem gracza (forward-fill).

    Dane muszą być posortowane chronologicznie (zapewnia parse_date).
    Dla każdego brakującego ranku używamy ostatniego WCZEŚNIEJSZEGO meczu gracza,
    żeby nie wyciekał ranking z przyszłości do historycznych wierszy.
    """
    df = df.copy()
    df["WRank"] = pd.to_numeric(df["WRank"], errors="coerce")
    df["LRank"] = pd.to_numeric(df["LRank"], errors="coerce")

    # Budujemy long frame: gracz → (pozycja w df, rank) — tylko wiersze ze znanym rankiem
    records: list[dict] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        if pd.notna(row["WRank"]):
            records.append({"pos": pos, "player": row["Winner"], "rank": row["WRank"]})
        if pd.notna(row["LRank"]):
            records.append({"pos": pos, "player": row["Loser"],  "rank": row["LRank"]})

    if records:
        known = pd.DataFrame(records).sort_values("pos")
        # Dla każdego gracza forward-fill: rank z pozycji X dotyczy wszystkich kolejnych brakujących
        last_rank_at: dict[str, tuple[int, float]] = {}  # player → (pos, rank)
        for _, r in known.iterrows():
            last_rank_at[r["player"]] = (int(r["pos"]), float(r["rank"]))

        # Uzupełnij NaN tylko poprzednim rankingiem (nie przyszłym)
        # Iterujemy wiersz po wierszu żeby zachować porządek czasowy
        winner_rank_ffill: dict[str, float] = {}
        loser_rank_ffill: dict[str, float] = {}

        w_fill = {}
        l_fill = {}
        for pos, (idx, row) in enumerate(df.iterrows()):
            winner = row["Winner"]
            loser  = row["Loser"]
            # Najpierw uzupełniamy brakujące
            if pd.isna(row["WRank"]) and winner in winner_rank_ffill:
                w_fill[idx] = winner_rank_ffill[winner]
            if pd.isna(row["LRank"]) and loser in loser_rank_ffill:
                l_fill[idx] = loser_rank_ffill[loser]
            # Potem aktualizujemy słownik dopiero po zapisaniu (forward only)
            if pd.notna(row["WRank"]):
                winner_rank_ffill[winner] = float(row["WRank"])
            if pd.notna(row["LRank"]):
                loser_rank_ffill[loser] = float(row["LRank"])

        for idx, val in w_fill.items():
            df.at[idx, "WRank"] = val
        for idx, val in l_fill.items():
            df.at[idx, "LRank"] = val

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Nie znaleziono pliku: {CSV_PATH}")
        return

    print(f"Plik danych: {CSV_PATH}")

    n_last_seasons = int((input(f"Ile ostatnich sezonów użyć? (Enter={BEST_N_LAST_SEASONS}): ").strip() or str(BEST_N_LAST_SEASONS)))
    split_mode = input("Tryb splitu [auto / 80_10_10 / 75_12_5_12_5] (Enter=auto): ").strip() or "auto"
    n_features_input = input("Ile features użyć? (Enter=wszystkie): ").strip()
    n_features_limit = int(n_features_input) if n_features_input else None
    surface_filter = input("Nawierzchnia [Hard/Clay/Grass/Carpet/Enter=wszystkie]: ").strip() or None

    output_dir = os.path.join(OUTPUT_BASE_DIR, f"tenis_{surface_filter or 'ALL'}")
    ensure_dir(output_dir)

    df_all = pd.read_csv(CSV_PATH, low_memory=False)
    df_all = parse_date(df_all)
    df_all["Season"] = df_all[DATE_COL].dt.year

    if surface_filter:
        surface_filter = surface_filter.strip().capitalize()
        df_all = df_all[df_all["Surface"].str.strip().str.capitalize() == surface_filter].copy().reset_index(drop=True)
        if df_all.empty:
            print(f"Brak meczów na nawierzchni: {surface_filter}")
            return

    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        print(f"Dostępnych sezonów: {len(all_seasons)}, a podałeś {n_last_seasons}")
        return

    selected_seasons = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected_seasons)].copy().reset_index(drop=True)

    print(f"\nSezony: {selected_seasons[0]}–{selected_seasons[-1]} | Mecze: {len(df_all):,}")

    # Uzupełnij brakujące rankingi z ostatniego meczu gracza
    df_all = _fill_missing_ranks(df_all)

    # Assign P1/P2, target
    df_all = assign_players(df_all)
    if USE_ODDS_FEATURES:
        df_all = add_odds_features(df_all)
        print("Cechy kursowe: WLACZONE")
    else:
        print("Cechy kursowe: WYLACZONE")
    df_all = add_csv5_form4_features(df_all)

    # Total games (target regresji)
    set_cols = [("W1","L1"),("W2","L2"),("W3","L3"),("W4","L4"),("W5","L5")]
    df_all["total_games"] = sum(
        pd.to_numeric(df_all[w], errors="coerce").fillna(0) +
        pd.to_numeric(df_all[l], errors="coerce").fillna(0)
        for w, l in set_cols
    )
    # Tylko dla ukończonych meczów z sensowną wartością
    df_all.loc[df_all[TARGET_COL].isna() | (df_all["total_games"] < 12), "total_games"] = np.nan

    known_mask = df_all[TARGET_COL].notna()
    future_df = df_all[~known_mask].copy().reset_index(drop=True)

    # Oblicz wszystkie cechy na pełnym zbiorze (łącznie z future)
    print("Budowanie cech rolling...")
    long = build_player_long(df_all)
    long = add_rolling_features(long)

    df_all = merge_player_features(df_all, long)

    print("ELO / Glicko...")
    df_all = compute_elo(
        df_all,
        k=BEST_ELO_K,
        initial=BEST_ELO_INITIAL,
        surface_specific=False,
        inactivity_uncertainty=BEST_ELO_INACTIVITY_UNCERTAINTY,
    )
    df_all = compute_elo(
        df_all,
        k=BEST_ELO_K,
        initial=BEST_ELO_INITIAL,
        surface_specific=True,
        inactivity_uncertainty=BEST_ELO_INACTIVITY_UNCERTAINTY,
    )
    df_all = compute_glicko(df_all, surface_specific=False)
    df_all = compute_glicko(df_all, surface_specific=True)

    print("H2H...")
    df_all = add_h2h_features(df_all)

    # Zmiana nawierzchni + dni odpoczynku
    df_all = _add_surface_transition(df_all)

    # Rank features
    df_all["rank_diff"] = df_all["P1Rank"] - df_all["P2Rank"]
    df_all["rank_ratio"] = df_all["P1Rank"] / df_all["P2Rank"].replace(0, np.nan)
    df_all["log_rank_diff"] = np.log1p(df_all["P1Rank"].fillna(500)) - np.log1p(df_all["P2Rank"].fillna(500))

    # Best of encoding (kluczowe dla total games)
    df_all["best_of"] = pd.to_numeric(df_all["Best of"], errors="coerce").fillna(3)

    # Surface encoding
    df_all["surface_enc"] = pd.Categorical(df_all["Surface"]).codes

    # Round encoding
    round_order = {
        "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
        "Quarterfinals": 5, "Semifinals": 6, "The Final": 7, "Final": 7,
        "Round Robin": 3,
    }
    df_all["round_enc"] = df_all["Round"].map(round_order).fillna(2)

    # Series encoding
    series_order = {
        "Grand Slam": 7, "Masters Cup": 6, "Masters 1000": 5,
        "ATP500": 4, "ATP250": 3, "International Gold": 4,
        "International": 3, "Series": 2,
    }
    df_all["series_enc"] = df_all["Series"].map(series_order).fillna(2) if "Series" in df_all.columns else 2

    # Split known/future
    known_df = df_all[df_all[TARGET_COL].notna()].copy().reset_index(drop=True)
    if len(future_df) > 0:
        future_df = df_all[df_all[TARGET_COL].isna()].copy().reset_index(drop=True)

    train_s, valid_s, test_s = seasonal_split(known_df["Season"].unique(), split_mode)
    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()

    print(f"\nTrain: {train_s[0]}–{train_s[-1]} ({len(train_df):,})")
    print(f"Valid: {valid_s[0]}–{valid_s[-1]} ({len(valid_df):,})")
    print(f"Test:  {test_s[0]}–{test_s[-1]} ({len(test_df):,})")

    # Imputacja rolling charting stats (z train, bez leakage):
    #   1. ffill per gracz — już zrobione w charting_match_level (ostatnia znana wartość)
    #   2. fallback dla graczy bez żadnych charting danych:
    #      mediana per rank_group × nawierzchnia (top50 / 51-150 / 151-300 / 300+)
    roll_feats = [
        "roll_1st_serve_won_pct", "roll_2nd_serve_won_pct",
        "roll_bp_saved_pct", "roll_bp_conv_pct", "roll_return_pts_won_pct",
    ]
    roll_cols_all = [f"{side}_{feat}" for feat in roll_feats for side in ("P1", "P2")]
    roll_cols_present = [c for c in roll_cols_all if c in train_df.columns]

    def _rank_group(rank_series: pd.Series) -> pd.Series:
        r = pd.to_numeric(rank_series, errors="coerce")
        return pd.cut(
            r,
            bins=[0, 50, 150, 300, np.inf],
            labels=["top50", "51-150", "151-300", "300+"],
        ).astype(str).where(r.notna(), "300+")

    if roll_cols_present and "Surface" in train_df.columns:
        # Buduj lookup: (rank_group, surface, col) → mediana z train
        tmp = train_df.copy()
        tmp["_rg_P1"] = _rank_group(tmp["P1Rank"])
        tmp["_rg_P2"] = _rank_group(tmp["P2Rank"])

        rg_medians: dict[tuple, float] = {}
        surf_medians: dict[str, float] = {}
        global_medians: dict[str, float] = {}

        for col in roll_cols_present:
            side = "P1" if col.startswith("P1_") else "P2"
            rg_col = f"_rg_{side}"
            global_medians[col] = tmp[col].median()
            surf_medians[col] = tmp.groupby("Surface")[col].median().to_dict()
            for (rg, surf), grp in tmp.groupby([rg_col, "Surface"]):
                med = grp[col].median()
                if not pd.isna(med):
                    rg_medians[(col, rg, surf)] = med

        def _impute_roll(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["_rg_P1"] = _rank_group(df["P1Rank"])
            df["_rg_P2"] = _rank_group(df["P2Rank"])
            for col in roll_cols_present:
                mask = df[col].isna()
                if not mask.any():
                    continue
                side = "P1" if col.startswith("P1_") else "P2"
                rg_col = f"_rg_{side}"
                filled = df.loc[mask].apply(
                    lambda row: rg_medians.get(
                        (col, row[rg_col], row["Surface"]),
                        surf_medians[col].get(row["Surface"], global_medians[col])
                    ),
                    axis=1,
                )
                df.loc[mask, col] = filled
            df.drop(columns=["_rg_P1", "_rg_P2"], inplace=True)
            return df

        before = sum(train_df[c].isna().sum() for c in roll_cols_present)
        train_df  = _impute_roll(train_df)
        valid_df  = _impute_roll(valid_df)
        test_df   = _impute_roll(test_df)
        if len(future_df) > 0:
            future_df = _impute_roll(future_df)
        after = sum(train_df[c].isna().sum() for c in roll_cols_present)
        print(f"  Imputacja roll stats (rank_group x surface): {before:,} -> {after:,} NaN w train")

    # Rolling charting diff features (P1 - P2) — po imputacji
    for feat in roll_feats:
        p1c = f"P1_{feat}"
        p2c = f"P2_{feat}"
        for dfx in (train_df, valid_df, test_df, future_df):
            if p1c in dfx.columns and p2c in dfx.columns:
                dfx[f"diff_{feat}"] = dfx[p1c] - dfx[p2c]

    feature_cols = get_feature_columns(known_df)
    X_train, X_valid, X_test, X_future, final_feats, dropped = filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )

    y_train = train_df[TARGET_COL].values.astype(int)
    y_valid = valid_df[TARGET_COL].values.astype(int)
    y_test = test_df[TARGET_COL].values.astype(int)

    if n_features_limit and n_features_limit < len(final_feats):
        print(f"\nAuto-select: wybieram top {n_features_limit} z {len(final_feats)} features")
        selector = LGBMClassifier(
            objective="binary", learning_rate=0.05,
            n_estimators=300, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
        )
        selector.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                     eval_metric="binary_logloss",
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        fi = pd.Series(selector.feature_importances_, index=final_feats)
        top_feats = fi.nlargest(n_features_limit).index.tolist()
        X_train = X_train[top_feats]
        X_valid = X_valid[top_feats]
        X_test = X_test[top_feats]
        X_future = X_future[top_feats] if len(X_future) > 0 else X_future
        final_feats = top_feats

    pd.DataFrame(dropped, columns=["feature", "reason"]).to_csv(
        os.path.join(output_dir, "dropped_features.csv"), index=False
    )

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)]
    )
    best_iter = model.best_iteration_

    valid_proba = model.predict_proba(X_valid, num_iteration=best_iter)[:, 1]
    test_proba  = model.predict_proba(X_test,  num_iteration=best_iter)[:, 1]

    # ── Kalibracja izotoniczna (fit na valid, apply na test i future) ──
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_proba, y_valid)
    joblib.dump(calibrator, os.path.join(output_dir, "calibrator.pkl"))

    valid_proba_cal = calibrator.predict(valid_proba)
    test_proba_cal  = calibrator.predict(test_proba)

    # Metryki RAW
    valid_logloss_raw = log_loss(y_valid, valid_proba)
    test_logloss_raw  = log_loss(y_test,  test_proba)

    calib_df_raw = calibration_table_binary(y_test, test_proba)
    calib_df     = calibration_table_binary(y_test, test_proba_cal)
    ece_raw = expected_calibration_error(calib_df_raw)
    ece     = expected_calibration_error(calib_df)

    # Automatyczny wybór: raw jeśli lepiej skalibrowany
    if ece_raw <= ece:
        print(f"  Kalibracja: używam RAW (ECE raw={ece_raw:.5f} <= ECE cal={ece:.5f})")
        test_proba_cal  = test_proba
        valid_proba_cal = valid_proba
        calib_used = "raw"
    else:
        print(f"  Kalibracja: używam ISOTONIC (ECE raw={ece_raw:.5f} > ECE cal={ece:.5f})")
        calib_used = "isotonic"

    blend_applied = False
    no_odds_calib_used = "n/a"
    if USE_ODDS_FEATURES and CALIB_BLEND_WITH_NO_ODDS:
        no_odds_feats = remove_odds_features(final_feats)
        if len(no_odds_feats) >= 3:
            print("\nTrenowanie modelu pomocniczego (bez kursów) do blendu kalibracji...")
            X_train_no = X_train[no_odds_feats]
            X_valid_no = X_valid[no_odds_feats]
            X_test_no = X_test[no_odds_feats]
            X_future_no = X_future[no_odds_feats] if len(X_future) > 0 else X_future

            model_no_odds = LGBMClassifier(**LGBM_PARAMS)
            model_no_odds.fit(
                X_train_no, y_train,
                eval_set=[(X_valid_no, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
            )
            best_iter_no_odds = model_no_odds.best_iteration_

            valid_proba_no = model_no_odds.predict_proba(X_valid_no, num_iteration=best_iter_no_odds)[:, 1]
            test_proba_no = model_no_odds.predict_proba(X_test_no, num_iteration=best_iter_no_odds)[:, 1]

            calibrator_no_odds = IsotonicRegression(out_of_bounds="clip")
            calibrator_no_odds.fit(valid_proba_no, y_valid)
            joblib.dump(calibrator_no_odds, os.path.join(output_dir, "calibrator_no_odds.pkl"))

            valid_proba_no_cal = calibrator_no_odds.predict(valid_proba_no)
            test_proba_no_cal = calibrator_no_odds.predict(test_proba_no)
            ece_no_raw = expected_calibration_error(calibration_table_binary(y_test, test_proba_no))
            ece_no_cal = expected_calibration_error(calibration_table_binary(y_test, test_proba_no_cal))
            if ece_no_raw <= ece_no_cal:
                valid_proba_no_cal = valid_proba_no
                test_proba_no_cal = test_proba_no
                no_odds_calib_used = "raw"
            else:
                no_odds_calib_used = "isotonic"

            blend_w = float(np.clip(CALIB_BLEND_WEIGHT_ODDS, 0.0, 1.0))
            valid_proba_cal = blend_w * valid_proba_cal + (1.0 - blend_w) * valid_proba_no_cal
            test_proba_cal = blend_w * test_proba_cal + (1.0 - blend_w) * test_proba_no_cal
            calib_used = f"blend_odds_noodds_w{blend_w:.2f}"
            blend_applied = True
        else:
            print("  Blend pominięty: za mało cech po usunięciu kursowych.")

    valid_pred = (valid_proba_cal >= BEST_CLASSIFICATION_THRESHOLD).astype(int)
    test_pred  = (test_proba_cal  >= BEST_CLASSIFICATION_THRESHOLD).astype(int)

    # Metryki po wybranej kalibracji
    valid_logloss = log_loss(y_valid, valid_proba_cal)
    test_logloss  = log_loss(y_test,  test_proba_cal)
    valid_brier   = binary_brier_score(y_valid, valid_proba_cal)
    test_brier    = binary_brier_score(y_test,  test_proba_cal)
    valid_acc     = accuracy_score(y_valid, valid_pred)
    test_acc      = accuracy_score(y_test,  test_pred)
    test_auc      = roc_auc_score(y_test,   test_proba_cal)

    # Przelicz calib_df dla wybranych prawdopodobieństw
    calib_df = calibration_table_binary(y_test, test_proba_cal)
    ece      = expected_calibration_error(calib_df)
    calib_df.to_csv(os.path.join(output_dir, "calibration_bins.csv"), index=False)

    fi_df = pd.DataFrame({
        "feature": final_feats,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # ── Model total games (regresja) ──
    print("\nTrenowanie modelu total games...")
    tg_train_mask = known_df[known_df["Season"].isin(train_s)]["total_games"].notna()
    tg_valid_mask = known_df[known_df["Season"].isin(valid_s)]["total_games"].notna()
    tg_test_mask  = known_df[known_df["Season"].isin(test_s)]["total_games"].notna()

    tg_mask_train = train_df["total_games"].notna().values
    tg_mask_valid = valid_df["total_games"].notna().values
    tg_mask_test  = test_df["total_games"].notna().values

    y_tg_train = train_df["total_games"].values[tg_mask_train]
    y_tg_valid = valid_df["total_games"].values[tg_mask_valid]
    y_tg_test  = test_df["total_games"].values[tg_mask_test]

    X_tg_train = X_train.iloc[tg_mask_train]
    X_tg_valid = X_valid.iloc[tg_mask_valid]
    X_tg_test  = X_test.iloc[tg_mask_test]

    tg_model = LGBMRegressor(**LGBM_REGRESSOR_PARAMS)
    tg_model.fit(
        X_tg_train, y_tg_train,
        eval_set=[(X_tg_valid, y_tg_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)]
    )
    tg_best_iter = tg_model.best_iteration_

    test_tg_pred = tg_model.predict(X_tg_test, num_iteration=tg_best_iter)
    tg_mae  = np.mean(np.abs(test_tg_pred - y_tg_test))
    tg_rmse = np.sqrt(np.mean((test_tg_pred - y_tg_test) ** 2))
    tg_naive_mae = np.mean(np.abs(y_tg_test - y_tg_test.mean()))
    bo_col = test_df["best_of"].reset_index(drop=True).values[tg_mask_test] if "best_of" in test_df.columns else None
    if bo_col is not None:
        bo3 = bo_col <= 3
        bo5 = bo_col >= 5
        mae_bo3 = np.mean(np.abs(test_tg_pred[bo3] - y_tg_test[bo3])) if bo3.sum() > 0 else np.nan
        mae_bo5 = np.mean(np.abs(test_tg_pred[bo5] - y_tg_test[bo5])) if bo5.sum() > 0 else np.nan
        print(f"Total games — MAE: {tg_mae:.2f} | RMSE: {tg_rmse:.2f} | Naiwne MAE: {tg_naive_mae:.2f}")
        print(f"             BO3 MAE: {mae_bo3:.2f} ({bo3.sum()} meczów) | BO5 MAE: {mae_bo5:.2f} ({bo5.sum()} meczów)")
    else:
        print(f"Total games — MAE: {tg_mae:.2f} | RMSE: {tg_rmse:.2f} | Naiwne MAE: {tg_naive_mae:.2f}")

    # Predykcje testowe (skalibrowane)
    base_cols = [
        c for c in [
            DATE_COL, "P1", "P2", "Surface", "Round", "Tournament", TARGET_COL, "Season",
            "P1_odds_mean", "P2_odds_mean"
        ] if c in test_df.columns
    ]
    pred_df = test_df[base_cols].copy()
    pred_df["prob_P1_wins"] = test_proba_cal
    pred_df["prob_P2_wins"]     = 1 - test_proba_cal
    pred_df["pred"]    = ["P1" if p >= BEST_CLASSIFICATION_THRESHOLD else "P2" for p in test_proba_cal]
    pred_df["correct"] = (test_pred == y_test).astype(int)
    pred_df = add_value_columns(pred_df)
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    threshold_points = [0.55, 0.60, 0.65, 0.69, 0.70]
    threshold_grid = np.round(np.arange(0.50, 0.76, 0.01), 2)
    thr_report = build_threshold_report(y_test, test_proba_cal, thresholds=threshold_grid)
    thr_report.to_csv(os.path.join(output_dir, "threshold_report.csv"), index=False)
    value_report = build_value_report(pred_df, thresholds=threshold_grid)
    value_report.to_csv(os.path.join(output_dir, "value_report.csv"), index=False)
    bankroll_report = build_bankroll_report(pred_df, thresholds=threshold_grid)
    bankroll_report.to_csv(os.path.join(output_dir, "bankroll_report.csv"), index=False)

    # Zapis szczegółów krzywej bankrollu dla najlepszego progu (max final bankroll)
    pred_df_bank = pred_df.copy()
    pred_df_bank["model_conf"] = pred_df_bank[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)
    pred_df_bank["pick_correct"] = np.where(
        (pred_df_bank["pick_side"] == "P1") & (pred_df_bank[TARGET_COL] == 1), 1,
        np.where((pred_df_bank["pick_side"] == "P2") & (pred_df_bank[TARGET_COL] == 0), 1, 0),
    )
    if len(bankroll_report.dropna(subset=["final_bankroll"])) > 0:
        best_row = bankroll_report.sort_values("final_bankroll", ascending=False).iloc[0]
        best_thr = float(best_row["threshold"])
        best_sel = pred_df_bank[(pred_df_bank["bet_flag"] == 1) & (pred_df_bank["model_conf"] >= best_thr)].copy()
        best_curve, _ = simulate_bankroll_kelly(best_sel)
        best_curve.to_csv(os.path.join(output_dir, "bankroll_curve_best.csv"), index=False)

    # Przyszłe mecze (skalibrowane)
    n_future_saved = 0
    if len(future_df) > 0 and len(X_future) > 0:
        future_proba     = model.predict_proba(X_future, num_iteration=best_iter)[:, 1]
        future_proba_cal = calibrator.predict(future_proba)
        if blend_applied:
            future_proba_no = model_no_odds.predict_proba(X_future_no, num_iteration=best_iter_no_odds)[:, 1]
            future_proba_no_cal = calibrator_no_odds.predict(future_proba_no)
            blend_w = float(np.clip(CALIB_BLEND_WEIGHT_ODDS, 0.0, 1.0))
            future_proba_cal = blend_w * future_proba_cal + (1.0 - blend_w) * future_proba_no_cal
        future_tg = tg_model.predict(X_future, num_iteration=tg_best_iter)
        fut_base = [
            c for c in [
                DATE_COL, "P1", "P2", "P1Rank", "P2Rank", "Surface", "Round", "Tournament",
                "P1_odds_mean", "P2_odds_mean"
            ] if c in future_df.columns
        ]
        fut_pred_df = future_df[fut_base].copy()
        fut_pred_df["prob_P1_wins"] = future_proba_cal
        fut_pred_df["prob_P2_wins"]     = 1 - future_proba_cal
        fut_pred_df["pred"] = ["P1" if p >= BEST_CLASSIFICATION_THRESHOLD else "P2" for p in future_proba_cal]
        fut_pred_df["pred_total_games"] = np.round(future_tg, 1)
        fut_pred_df = add_value_columns(fut_pred_df)
        # Zapisuj tylko realne przyszłe mecze (bez historycznych braków wyniku).
        fut_pred_df[DATE_COL] = pd.to_datetime(fut_pred_df[DATE_COL], format="mixed", errors="coerce")
        today = pd.Timestamp.today().normalize()
        fut_pred_df = fut_pred_df[
            fut_pred_df[DATE_COL].isna() | (fut_pred_df[DATE_COL] >= today)
        ].sort_values(DATE_COL)
        num_cols = fut_pred_df.select_dtypes(include=[np.number]).columns
        fut_pred_df[num_cols] = fut_pred_df[num_cols].round(2)
        n_future_saved = len(fut_pred_df)
        fut_pred_df.to_csv(os.path.join(output_dir, "future_predictions.csv"), index=False)
        save_future_profile_predictions(fut_pred_df, output_dir)

    summary = pd.DataFrame([{
        "file": Path(CSV_PATH).name,
        "surface": surface_filter or "ALL",
        "n_selected_seasons": n_last_seasons,
        "train_seasons": f"{train_s[0]}-{train_s[-1]}",
        "valid_seasons": f"{valid_s[0]}-{valid_s[-1]}",
        "test_seasons": f"{test_s[0]}-{test_s[-1]}",
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_future": int(n_future_saved),
        "n_features": len(final_feats),
        "pred_threshold": BEST_CLASSIFICATION_THRESHOLD,
        "value_min_edge": VALUE_MIN_EDGE,
        "value_min_ev": VALUE_MIN_EV,
        "bankroll_initial": BANKROLL_INITIAL,
        "kelly_fraction": KELLY_FRACTION,
        "max_stake_pct": MAX_STAKE_PCT,
        "elo_k": BEST_ELO_K,
        "elo_initial": BEST_ELO_INITIAL,
        "elo_inactivity_uncertainty": BEST_ELO_INACTIVITY_UNCERTAINTY,
        "best_iteration": best_iter,
        "valid_logloss_raw": round(valid_logloss_raw, 5),
        "test_logloss_raw":  round(test_logloss_raw,  5),
        "valid_logloss_cal": round(valid_logloss, 5),
        "test_logloss_cal":  round(test_logloss,  5),
        "valid_brier": round(valid_brier, 5),
        "test_brier":  round(test_brier,  5),
        "valid_accuracy": round(valid_acc, 4),
        "test_accuracy":  round(test_acc,  4),
        "test_auc":  round(test_auc, 4),
        "test_ece_raw": round(ece_raw, 5),
        "test_ece_cal": round(ece,     5),
        "calib_used":  calib_used,
        "blend_applied": int(blend_applied),
        "blend_weight_odds": round(float(CALIB_BLEND_WEIGHT_ODDS), 2),
        "no_odds_calib_used": no_odds_calib_used,
        "tg_mae": round(tg_mae, 3),
        "tg_rmse": round(tg_rmse, 3),
    }])
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    save_staking_profiles_csv(output_dir)

    print_separator("PODSUMOWANIE MODELU")
    print(summary.T.to_string(header=False))

    print_separator("KALIBRACJA — RAW vs SKALIBROWANA (P1 wygrywa)")
    print(f"ECE raw: {ece_raw:.5f}  ->  ECE cal: {ece:.5f}\n")
    # Oba DataFramy mają te same biny (obliczone na tych samych 20 binach 0..1),
    # więc wyrównujemy po bin_left.
    merged_calib = calib_df_raw[["bin_left","bin_right","count","actual_rate","avg_pred","diff"]].copy()
    merged_calib.columns = ["bin_left","bin_right","count","actual_rate","raw_pred","raw_diff"]
    cal_lookup = calib_df.set_index("bin_left")[["avg_pred","diff"]]
    merged_calib["cal_pred"] = merged_calib["bin_left"].map(cal_lookup["avg_pred"])
    merged_calib["cal_diff"] = merged_calib["bin_left"].map(cal_lookup["diff"])
    print(merged_calib.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_separator("TOP 30 FEATURE IMPORTANCE")
    print(fi_df.head(30).to_string(index=False))

    print_separator("RAPORT PROGOW (SYM: P1 lub P2 >= prog)")
    show_thr = thr_report[thr_report["threshold"].isin(threshold_points)]
    print(show_thr.to_string(index=False))

    print_separator("RAPORT VALUE (EV + EDGE)")
    show_value = value_report[value_report["threshold"].isin(threshold_points)]
    print(show_value.to_string(index=False))

    print_separator("RAPORT BANKROLL (KELLY + CAP)")
    show_bank = bankroll_report[bankroll_report["threshold"].isin(threshold_points)]
    print(show_bank.to_string(index=False))

    print_separator("PREDYKCJE PRZYSZŁYCH MECZÓW")
    if len(future_df) > 0:
        fut_pred_df_loaded = pd.read_csv(os.path.join(output_dir, "future_predictions.csv"))
        fut_pred_df_loaded[DATE_COL] = pd.to_datetime(fut_pred_df_loaded[DATE_COL], format="mixed", errors="coerce")
        today = pd.Timestamp.today().normalize()
        # Pokaż tylko mecze z datą >= dziś lub bez daty (prawdziwe przyszłe)
        fut_pred_df_loaded = fut_pred_df_loaded[
            fut_pred_df_loaded[DATE_COL].isna() | (fut_pred_df_loaded[DATE_COL] >= today)
        ].sort_values(DATE_COL)
        print(fut_pred_df_loaded.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("Brak meczów bez wyniku w CSV.")

    print_separator("ZAPIS")
    print(f"Zapisano do: {output_dir}")


if __name__ == "__main__":
    main()
