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
    precision_recall_fscore_support
)

import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
_csv1 = SCRIPT_DIR / "wszystkie_sezony.csv"
_csv2 = SCRIPT_DIR / "merged_data.csv"
CSV_PATH = str(_csv1 if _csv1.exists() else _csv2)
DATE_COL = "Date"
TARGET_COL = "FTR"

CLASS_ORDER = ["H", "D", "A"]
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_CLASS = {0: "H", 1: "D", 2: "A"}
CLASS_LABELS_PRINT = {"H": "win", "D": "draw", "A": "loss"}

OUTPUT_BASE_DIR = str(SCRIPT_DIR / "outputs_single")

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "goss",
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 63,
    "min_child_samples": 60,
    "subsample": 1.0,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_jobs": -1
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_date(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


def infer_season_from_date(date_series):
    years = date_series.dt.year
    months = date_series.dt.month
    return np.where(months >= 7, years, years - 1)


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


def split_known_future(df):
    df = df.copy()

    known_mask = df[TARGET_COL].isin(CLASS_TO_INT.keys())
    known_df = df[known_mask].copy().reset_index(drop=True)
    future_df = df[~known_mask].copy().reset_index(drop=True)

    known_df["target"] = known_df[TARGET_COL].map(CLASS_TO_INT)
    return known_df, future_df


def build_team_match_history(df, use_xg=True):
    xg_h = df["xGH"] if (use_xg and "xGH" in df.columns) else np.nan
    xg_a = df["xGA"] if (use_xg and "xGA" in df.columns) else np.nan

    home = pd.DataFrame({
        "match_id": df.index,
        "Date": df[DATE_COL],
        "Team": df["HomeTeam"],
        "is_home": 1,
        "goals_for": df["FTHG"],
        "goals_against": df["FTAG"],
        "shots_for": df["HS"] if "HS" in df.columns else np.nan,
        "shots_against": df["AS"] if "AS" in df.columns else np.nan,
        "sot_for": df["HST"] if "HST" in df.columns else np.nan,
        "sot_against": df["AST"] if "AST" in df.columns else np.nan,
        "corners_for": df["HC"] if "HC" in df.columns else np.nan,
        "corners_against": df["AC"] if "AC" in df.columns else np.nan,
        "fouls_for": df["HF"] if "HF" in df.columns else np.nan,
        "fouls_against": df["AF"] if "AF" in df.columns else np.nan,
        "yellows_for": df["HY"] if "HY" in df.columns else np.nan,
        "yellows_against": df["AY"] if "AY" in df.columns else np.nan,
        "reds_for": df["HR"] if "HR" in df.columns else np.nan,
        "reds_against": df["AR"] if "AR" in df.columns else np.nan,
        "xg_for": xg_h,
        "xg_against": xg_a,
        "points": np.where(df["FTHG"] > df["FTAG"], 3, np.where(df["FTHG"] == df["FTAG"], 1, 0))
    })

    away = pd.DataFrame({
        "match_id": df.index,
        "Date": df[DATE_COL],
        "Team": df["AwayTeam"],
        "is_home": 0,
        "goals_for": df["FTAG"],
        "goals_against": df["FTHG"],
        "shots_for": df["AS"] if "AS" in df.columns else np.nan,
        "shots_against": df["HS"] if "HS" in df.columns else np.nan,
        "sot_for": df["AST"] if "AST" in df.columns else np.nan,
        "sot_against": df["HST"] if "HST" in df.columns else np.nan,
        "corners_for": df["AC"] if "AC" in df.columns else np.nan,
        "corners_against": df["HC"] if "HC" in df.columns else np.nan,
        "fouls_for": df["AF"] if "AF" in df.columns else np.nan,
        "fouls_against": df["HF"] if "HF" in df.columns else np.nan,
        "yellows_for": df["AY"] if "AY" in df.columns else np.nan,
        "yellows_against": df["HY"] if "HY" in df.columns else np.nan,
        "reds_for": df["AR"] if "AR" in df.columns else np.nan,
        "reds_against": df["HR"] if "HR" in df.columns else np.nan,
        "xg_for": xg_a,
        "xg_against": xg_h,
        "points": np.where(df["FTAG"] > df["FTHG"], 3, np.where(df["FTAG"] == df["FTHG"], 1, 0))
    })

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["Team", "Date", "match_id"]).reset_index(drop=True)
    return long_df


def add_advanced_rolling_features(long_df):
    long_df = long_df.copy()

    metrics = [
        "goals_for", "goals_against",
        "shots_for", "shots_against",
        "sot_for", "sot_against",
        "corners_for", "corners_against",
        "fouls_for", "fouls_against",
        "yellows_for", "yellows_against",
        "reds_for", "reds_against",
        "xg_for", "xg_against",
        "points"
    ]
    windows = [3, 5, 10, 20, 30]

    for metric in metrics:
        shifted = long_df.groupby("Team")[metric].shift(1)
        for w in windows:
            rolled = shifted.groupby(long_df["Team"]).rolling(w, min_periods=1).mean()
            long_df[f"{metric}_avg_last{w}"] = rolled.reset_index(level=0, drop=True).values

    for metric in metrics:
        long_df[f"momentum_{metric}"] = long_df[f"{metric}_avg_last5"] - long_df[f"{metric}_avg_last20"]

    return long_df


def merge_form_features_to_match(df, long_df):
    df = df.copy()
    df["match_id"] = df.index

    cols = [c for c in long_df.columns if ("avg_last" in c or "momentum_" in c)]

    home = long_df[long_df["is_home"] == 1][["match_id"] + cols].copy()
    away = long_df[long_df["is_home"] == 0][["match_id"] + cols].copy()

    home = home.rename(columns={c: f"home_{c}" for c in cols})
    away = away.rename(columns={c: f"away_{c}" for c in cols})

    df = df.merge(home, on="match_id", how="left")
    df = df.merge(away, on="match_id", how="left")

    bases = [
        "goals_for", "goals_against",
        "shots_for", "shots_against",
        "sot_for", "sot_against",
        "corners_for", "corners_against",
        "fouls_for", "fouls_against",
        "yellows_for", "yellows_against",
        "reds_for", "reds_against",
        "xg_for", "xg_against",
        "points"
    ]

    for base in bases:
        for w in [3, 5, 10, 30]:
            h = f"home_{base}_avg_last{w}"
            a = f"away_{base}_avg_last{w}"
            if h in df.columns and a in df.columns:
                df[f"roll_{base}_{w}_diff"] = df[h] - df[a]

    for base in bases:
        h = f"home_momentum_{base}"
        a = f"away_momentum_{base}"
        if h in df.columns and a in df.columns:
            df[f"momentum_{base}_diff"] = df[h] - df[a]

    return df


def add_h2h_features(df):
    df = df.copy()
    df["match_id"] = df.index

    records = []
    for idx, row in df.iterrows():
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            gd = np.nan
            pts = np.nan
        else:
            gd = row["FTHG"] - row["FTAG"]
            pts = 3 if row["FTHG"] > row["FTAG"] else 1 if row["FTHG"] == row["FTAG"] else 0

        records.append({
            "match_id": idx,
            "TeamA": row["HomeTeam"],
            "TeamB": row["AwayTeam"],
            "h2h_gd": gd,
            "h2h_pts": pts
        })

    h2h = pd.DataFrame(records)
    h2h = h2h.sort_values(["TeamA", "TeamB", "match_id"]).reset_index(drop=True)

    for col in ["h2h_gd", "h2h_pts"]:
        for w in [3, 5, 10]:
            h2h[f"{col}_avg_last{w}"] = (
                h2h.groupby(["TeamA", "TeamB"])[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    h2h["h2h_momentum_gd"] = h2h["h2h_gd_avg_last5"] - h2h["h2h_gd_avg_last10"]
    h2h["h2h_momentum_pts"] = h2h["h2h_pts_avg_last5"] - h2h["h2h_pts_avg_last10"]

    keep_cols = [
        "match_id",
        "h2h_gd_avg_last3", "h2h_gd_avg_last5", "h2h_gd_avg_last10",
        "h2h_pts_avg_last3", "h2h_pts_avg_last5", "h2h_pts_avg_last10",
        "h2h_momentum_gd", "h2h_momentum_pts"
    ]

    h2h = h2h[keep_cols].copy()

    h2h = h2h.rename(columns={
        "h2h_gd_avg_last3": "h2h_gd_3",
        "h2h_gd_avg_last5": "h2h_gd_5",
        "h2h_gd_avg_last10": "h2h_gd_10",
        "h2h_pts_avg_last3": "h2h_pts_3",
        "h2h_pts_avg_last5": "h2h_pts_5",
        "h2h_pts_avg_last10": "h2h_pts_10"
    })

    df = df.merge(h2h, on="match_id", how="left")
    return df


def compute_elo(df, k=20, initial=1500):
    ratings = {}
    elo_h, elo_a = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = ratings.get(home, initial)
        ra = ratings.get(away, initial)
        elo_h.append(rh)
        elo_a.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        if row["FTHG"] > row["FTAG"]:
            sh = 1
        elif row["FTHG"] == row["FTAG"]:
            sh = 0.5
        else:
            sh = 0

        ratings[home] = rh + k * (sh - eh)
        ratings[away] = ra + k * ((1 - sh) - (1 - eh))

    df = df.copy()
    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def compute_xg_elo(df, k=25, initial=1500):
    ratings = {}
    elo_h, elo_a = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = ratings.get(home, initial)
        ra = ratings.get(away, initial)
        elo_h.append(rh)
        elo_a.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("xGH")) or pd.isna(row.get("xGA")):
            continue

        xg_diff = row["xGH"] - row["xGA"]
        sh = 1 / (1 + 10 ** (-xg_diff / 2))

        eh = 1 / (1 + 10 ** ((ra - rh) / 400))

        ratings[home] = rh + k * (sh - eh)
        ratings[away] = ra + k * ((1 - sh) - (1 - eh))

    df = df.copy()
    df["xg_elo_home"] = elo_h
    df["xg_elo_away"] = elo_a
    df["xg_elo_diff"] = df["xg_elo_home"] - df["xg_elo_away"]
    return df


def compute_goals_elo(df, k=20, initial=1500):
    ratings = {}
    elo_h, elo_a = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = ratings.get(home, initial)
        ra = ratings.get(away, initial)
        elo_h.append(rh)
        elo_a.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        gd = abs(row["FTHG"] - row["FTAG"])
        k_adj = k * np.log(max(gd, 1) + 1)

        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        if row["FTHG"] > row["FTAG"]:
            sh = 1
        elif row["FTHG"] == row["FTAG"]:
            sh = 0.5
        else:
            sh = 0

        ratings[home] = rh + k_adj * (sh - eh)
        ratings[away] = ra + k_adj * ((1 - sh) - (1 - eh))

    df = df.copy()
    df["goals_elo_home"] = elo_h
    df["goals_elo_away"] = elo_a
    df["goals_elo_diff"] = df["goals_elo_home"] - df["goals_elo_away"]
    return df


def compute_glicko(df, initial_r=1500, initial_rd=350, min_rd=30):
    q = math.log(10) / 400
    ratings = {}
    g_hr, g_hrd, g_ar, g_ard = [], [], [], []

    def _g(rd):
        return 1 / math.sqrt(1 + 3 * q ** 2 * rd ** 2 / math.pi ** 2)

    def _e(r, ro, rdo):
        return 1 / (1 + 10 ** (-_g(rdo) * (r - ro) / 400))

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh, rdh = ratings.get(home, (initial_r, initial_rd))
        ra, rda = ratings.get(away, (initial_r, initial_rd))

        g_hr.append(rh)
        g_hrd.append(rdh)
        g_ar.append(ra)
        g_ard.append(rda)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        if row["FTHG"] > row["FTAG"]:
            sh, sa = 1, 0
        elif row["FTHG"] == row["FTAG"]:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0, 1

        ga = _g(rda)
        e_h = _e(rh, ra, rda)
        d2_h = 1 / (q ** 2 * ga ** 2 * e_h * (1 - e_h))
        new_rh = rh + q / (1 / rdh ** 2 + 1 / d2_h) * ga * (sh - e_h)
        new_rdh = max(math.sqrt(1 / (1 / rdh ** 2 + 1 / d2_h)), min_rd)

        gh = _g(rdh)
        e_a = _e(ra, rh, rdh)
        d2_a = 1 / (q ** 2 * gh ** 2 * e_a * (1 - e_a))
        new_ra = ra + q / (1 / rda ** 2 + 1 / d2_a) * gh * (sa - e_a)
        new_rda = max(math.sqrt(1 / (1 / rda ** 2 + 1 / d2_a)), min_rd)

        ratings[home] = (new_rh, new_rdh)
        ratings[away] = (new_ra, new_rda)

    df = df.copy()
    df["glicko_home_r"] = g_hr
    df["glicko_home_rd"] = g_hrd
    df["glicko_away_r"] = g_ar
    df["glicko_away_rd"] = g_ard
    df["glicko_r_diff"] = df["glicko_home_r"] - df["glicko_away_r"]
    df["glicko_rd_diff"] = df["glicko_home_rd"] - df["glicko_away_rd"]
    return df


def compute_elo_home_away(df, k=20, initial=1500):
    h_ratings = {}
    a_ratings = {}
    eh_list, ea_list = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = h_ratings.get(home, initial)
        ra = a_ratings.get(away, initial)
        eh_list.append(rh)
        ea_list.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        sh = 1 if row["FTHG"] > row["FTAG"] else 0.5 if row["FTHG"] == row["FTAG"] else 0

        h_ratings[home] = rh + k * (sh - exp_h)
        a_ratings[away] = ra + k * ((1 - sh) - (1 - exp_h))

    df = df.copy()
    df["elo_H_home"] = eh_list
    df["elo_A_away"] = ea_list
    df["elo_HA_diff"] = df["elo_H_home"] - df["elo_A_away"]
    return df


def compute_xg_elo_home_away(df, k=25, initial=1500):
    h_ratings = {}
    a_ratings = {}
    eh_list, ea_list = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = h_ratings.get(home, initial)
        ra = a_ratings.get(away, initial)
        eh_list.append(rh)
        ea_list.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("xGH")) or pd.isna(row.get("xGA")):
            continue

        xg_diff = row["xGH"] - row["xGA"]
        sh = 1 / (1 + 10 ** (-xg_diff / 2))
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))

        h_ratings[home] = rh + k * (sh - exp_h)
        a_ratings[away] = ra + k * ((1 - sh) - (1 - exp_h))

    df = df.copy()
    df["xg_elo_H_home"] = eh_list
    df["xg_elo_A_away"] = ea_list
    df["xg_elo_HA_diff"] = df["xg_elo_H_home"] - df["xg_elo_A_away"]
    return df


def compute_goals_elo_home_away(df, k=20, initial=1500):
    h_ratings = {}
    a_ratings = {}
    eh_list, ea_list = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = h_ratings.get(home, initial)
        ra = a_ratings.get(away, initial)
        eh_list.append(rh)
        ea_list.append(ra)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        gd = abs(row["FTHG"] - row["FTAG"])
        k_adj = k * np.log(max(gd, 1) + 1)
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        sh = 1 if row["FTHG"] > row["FTAG"] else 0.5 if row["FTHG"] == row["FTAG"] else 0

        h_ratings[home] = rh + k_adj * (sh - exp_h)
        a_ratings[away] = ra + k_adj * ((1 - sh) - (1 - exp_h))

    df = df.copy()
    df["goals_elo_H_home"] = eh_list
    df["goals_elo_A_away"] = ea_list
    df["goals_elo_HA_diff"] = df["goals_elo_H_home"] - df["goals_elo_A_away"]
    return df


def compute_glicko_home_away(df, initial_r=1500, initial_rd=350, min_rd=30):
    q = math.log(10) / 400
    h_ratings = {}
    a_ratings = {}
    ghr, ghrd, gar, gard = [], [], [], []

    def _g(rd):
        return 1 / math.sqrt(1 + 3 * q ** 2 * rd ** 2 / math.pi ** 2)

    def _e(r, ro, rdo):
        return 1 / (1 + 10 ** (-_g(rdo) * (r - ro) / 400))

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh, rdh = h_ratings.get(home, (initial_r, initial_rd))
        ra, rda = a_ratings.get(away, (initial_r, initial_rd))

        ghr.append(rh)
        ghrd.append(rdh)
        gar.append(ra)
        gard.append(rda)

        if row[TARGET_COL] not in CLASS_TO_INT:
            continue
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            continue

        if row["FTHG"] > row["FTAG"]:
            sh, sa = 1, 0
        elif row["FTHG"] == row["FTAG"]:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0, 1

        ga = _g(rda)
        e_h = _e(rh, ra, rda)
        d2_h = 1 / (q ** 2 * ga ** 2 * e_h * (1 - e_h))
        new_rh = rh + q / (1 / rdh ** 2 + 1 / d2_h) * ga * (sh - e_h)
        new_rdh = max(math.sqrt(1 / (1 / rdh ** 2 + 1 / d2_h)), min_rd)

        gh = _g(rdh)
        e_a = _e(ra, rh, rdh)
        d2_a = 1 / (q ** 2 * gh ** 2 * e_a * (1 - e_a))
        new_ra = ra + q / (1 / rda ** 2 + 1 / d2_a) * gh * (sa - e_a)
        new_rda = max(math.sqrt(1 / (1 / rda ** 2 + 1 / d2_a)), min_rd)

        h_ratings[home] = (new_rh, new_rdh)
        a_ratings[away] = (new_ra, new_rda)

    df = df.copy()
    df["glicko_H_home_r"] = ghr
    df["glicko_H_home_rd"] = ghrd
    df["glicko_A_away_r"] = gar
    df["glicko_A_away_rd"] = gard
    df["glicko_HA_r_diff"] = df["glicko_H_home_r"] - df["glicko_A_away_r"]
    df["glicko_HA_rd_diff"] = df["glicko_H_home_rd"] - df["glicko_A_away_rd"]
    return df


def get_feature_columns(df):
    raw_stats = {
        "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST", "HC", "AC",
        "HF", "AF", "HO", "AO", "HY", "AY", "HR", "AR",
        "HBP", "ABP", "HFKC", "AFKC", "xGH", "xGA"
    }

    odds_prefixes = (
        "GB", "IW", "LB", "SB", "WH", "SY", "B365", "SO", "BW", "SJ", "VC",
        "Bb", "PS", "Max", "Avg", "PC", "AH", "BF", "1XB", "BMG", "CL", "LBC", "BFE", "BFD",
        "BS", "BV", "HH", "P"
    )

    feature_cols = []
    for col in df.columns:
        if col in {DATE_COL, "Season", TARGET_COL, "target", "HomeTeam", "AwayTeam", "match_id"}:
            continue
        if col in raw_stats:
            continue
        if any(col.startswith(p) for p in odds_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


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
        to_drop = [c for c in upper.columns if any(upper[c] > 0.995)]

        for c in to_drop:
            dropped.append((c, "high_corr"))

        keep = [c for c in keep if c not in to_drop]
        X_train = X_train[keep]
        X_valid = X_valid[keep]
        X_test = X_test[keep]
        X_future = X_future[keep] if len(X_future) > 0 else X_future

    return X_train, X_valid, X_test, X_future, keep, dropped


def multiclass_brier_score(y_true, proba, n_classes=3):
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))


def per_class_brier(y_true, proba, class_idx):
    y_bin = (y_true == class_idx).astype(int)
    p = proba[:, class_idx]
    return np.mean((p - y_bin) ** 2)


def calibration_table_binary(y_true_binary, pred_prob, step=0.05):
    bins = np.arange(0.0, 1.0 + step, step)
    rows = []

    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]

        if i == len(bins) - 2:
            mask = (pred_prob >= left) & (pred_prob <= right)
        else:
            mask = (pred_prob >= left) & (pred_prob < right)

        n = int(mask.sum())

        if n == 0:
            rows.append({
                "bin_left": left,
                "bin_right": right,
                "count": 0,
                "avg_pred": np.nan,
                "actual_rate": np.nan,
                "diff": np.nan
            })
            continue

        avg_pred = float(pred_prob[mask].mean())
        actual_rate = float(y_true_binary[mask].mean())

        rows.append({
            "bin_left": left,
            "bin_right": right,
            "count": n,
            "avg_pred": avg_pred,
            "actual_rate": actual_rate,
            "diff": actual_rate - avg_pred
        })

    return pd.DataFrame(rows)


def expected_calibration_error(calib_df):
    df = calib_df.dropna().copy()
    if df.empty:
        return np.nan
    total = df["count"].sum()
    return np.sum((df["count"] / total) * np.abs(df["actual_rate"] - df["avg_pred"]))


def print_separator(title=None):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Nie znaleziono pliku: {CSV_PATH}")
        return

    n_last_seasons = int(input("Ile ostatnich sezonów użyć? "))
    split_mode = input("Tryb splitu [auto / 80_10_10 / 75_12_5_12_5] (Enter=auto): ").strip()
    if split_mode == "":
        split_mode = "auto"
    n_features_input = input("Ile features użyć? (Enter=wszystkie): ").strip()
    n_features_limit = int(n_features_input) if n_features_input else None
    use_xg_input = input("Użyć statystyk xG? [t/n] (Enter=tak): ").strip().lower()
    use_xg = use_xg_input not in ("n", "nie", "no")

    output_dir = os.path.join(
        OUTPUT_BASE_DIR,
        Path(CSV_PATH).stem + "_FULL_FORM_SHOTS_SOT_XG_H2H"
    )
    ensure_dir(output_dir)

    df_all = pd.read_csv(CSV_PATH)
    df_all = parse_date(df_all)
    df_all["Season"] = infer_season_from_date(df_all[DATE_COL])

    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        print(f"Plik ma tylko {len(all_seasons)} sezonów, a podałeś {n_last_seasons}")
        return

    selected_seasons = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected_seasons)].copy().reset_index(drop=True)

    known_df, future_df = split_known_future(df_all)

    # Oblicz cechy raz na połączonym zbiorze (known + future), żeby uniknąć
    # konfliktów nazw kolumn (_x/_y) i NaN w cechach przyszłych meczów.
    all_df = pd.concat(
        [known_df.drop(columns=["target"], errors="ignore"), future_df],
        ignore_index=True
    ).sort_values(DATE_COL).reset_index(drop=True)
    all_df["Season"] = infer_season_from_date(all_df[DATE_COL])

    long_df_all = build_team_match_history(all_df, use_xg=use_xg)
    long_df_all = add_advanced_rolling_features(long_df_all)
    full_with_features = merge_form_features_to_match(all_df, long_df_all)
    full_with_features = add_h2h_features(full_with_features)
    full_with_features = compute_elo(full_with_features)
    full_with_features = compute_goals_elo(full_with_features)
    full_with_features = compute_glicko(full_with_features)
    full_with_features = compute_elo_home_away(full_with_features)
    full_with_features = compute_goals_elo_home_away(full_with_features)
    full_with_features = compute_glicko_home_away(full_with_features)
    if use_xg:
        full_with_features = compute_xg_elo(full_with_features)
        full_with_features = compute_xg_elo_home_away(full_with_features)

    known_df = full_with_features[full_with_features[TARGET_COL].isin(CLASS_TO_INT.keys())].copy().reset_index(drop=True)
    known_df["target"] = known_df[TARGET_COL].map(CLASS_TO_INT)

    if len(future_df) > 0:
        future_df = full_with_features[~full_with_features[TARGET_COL].isin(CLASS_TO_INT.keys())].copy().reset_index(drop=True)
    else:
        future_df = pd.DataFrame(columns=known_df.columns)

    train_s, valid_s, test_s = seasonal_split(known_df["Season"].unique(), split_mode)

    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()

    feature_cols = get_feature_columns(known_df)
    X_train, X_valid, X_test, X_future, final_feats, dropped = filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    if n_features_limit and n_features_limit < len(final_feats):
        print(f"\n--- Auto-select: wybieram top {n_features_limit} z {len(final_feats)} features ---")
        selector = LGBMClassifier(
            objective="multiclass", num_class=3, learning_rate=0.05,
            n_estimators=300, num_leaves=31, random_state=42, n_jobs=-1
        )
        selector.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      eval_metric="multi_logloss",
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        fi = pd.Series(selector.feature_importances_, index=final_feats)
        top_feats = fi.nlargest(n_features_limit).index.tolist()

        X_train = X_train[top_feats]
        X_valid = X_valid[top_feats]
        X_test = X_test[top_feats]
        X_future = X_future[top_feats] if len(X_future) > 0 else X_future
        final_feats = top_feats
        print(f"--- Wybrano: {final_feats[:10]}{'...' if len(final_feats) > 10 else ''} ---\n")

    dropped_df = pd.DataFrame(dropped, columns=["feature", "reason"])
    dropped_df.to_csv(os.path.join(output_dir, "dropped_features.csv"), index=False)

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(150),
            lgb.log_evaluation(50)
        ]
    )

    best_iter = model.best_iteration_

    valid_proba = model.predict_proba(X_valid, num_iteration=best_iter)
    test_proba = model.predict_proba(X_test, num_iteration=best_iter)

    valid_pred = np.argmax(valid_proba, axis=1)
    test_pred = np.argmax(test_proba, axis=1)

    valid_logloss = log_loss(y_valid, valid_proba, labels=[0, 1, 2])
    test_logloss = log_loss(y_test, test_proba, labels=[0, 1, 2])

    valid_brier = multiclass_brier_score(y_valid, valid_proba, n_classes=3)
    test_brier = multiclass_brier_score(y_test, test_proba, n_classes=3)

    valid_acc = accuracy_score(y_valid, valid_pred)
    test_acc = accuracy_score(y_test, test_pred)

    test_brier_h = per_class_brier(y_test, test_proba, 0)
    test_brier_d = per_class_brier(y_test, test_proba, 1)
    test_brier_a = per_class_brier(y_test, test_proba, 2)

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_test,
        test_pred,
        labels=[0, 1, 2],
        zero_division=0
    )

    class_metrics_df = pd.DataFrame({
        "class": ["win", "draw", "loss"],
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": supp
    })

    ece_map = {}
    calib_print_tables = {}

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_bin = (y_test == cls_idx).astype(int)
        p_cls = test_proba[:, cls_idx]
        calib_df = calibration_table_binary(y_bin, p_cls, step=0.05)
        calib_df.to_csv(os.path.join(output_dir, f"calibration_bins_{cls_name}.csv"), index=False)
        ece_map[cls_name] = expected_calibration_error(calib_df)
        calib_print_tables[cls_name] = calib_df

    fi_df = pd.DataFrame({
        "feature": final_feats,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", TARGET_COL, "Season"] if c in test_df.columns]
    pred_df = test_df[base_cols].copy()
    pred_df["pred_H"] = test_proba[:, 0]
    pred_df["pred_D"] = test_proba[:, 1]
    pred_df["pred_A"] = test_proba[:, 2]
    pred_df["pred_class"] = [INT_TO_CLASS[i] for i in test_pred]
    pred_df["max_prob"] = test_proba.max(axis=1)
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    future_pred_df = pd.DataFrame()
    if len(future_df) > 0:
        future_proba = model.predict_proba(X_future, num_iteration=best_iter)
        future_pred = np.argmax(future_proba, axis=1)

        future_base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in future_df.columns]
        future_pred_df = future_df[future_base_cols].copy()
        future_pred_df["pred_H"] = future_proba[:, 0]
        future_pred_df["pred_D"] = future_proba[:, 1]
        future_pred_df["pred_A"] = future_proba[:, 2]
        future_pred_df["pred_class"] = [INT_TO_CLASS[i] for i in future_pred]
        future_pred_df["max_prob"] = future_proba.max(axis=1)
        future_pred_df.to_csv(os.path.join(output_dir, "future_predictions.csv"), index=False)

    summary = pd.DataFrame([{
        "file": Path(CSV_PATH).name,
        "n_selected_seasons": n_last_seasons,
        "train_seasons": "|".join(map(str, train_s)),
        "valid_seasons": "|".join(map(str, valid_s)),
        "test_seasons": "|".join(map(str, test_s)),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_future": len(future_df),
        "n_features": len(final_feats),
        "best_iteration": best_iter,
        "valid_logloss": valid_logloss,
        "test_logloss": test_logloss,
        "valid_brier_multiclass": valid_brier,
        "test_brier_multiclass": test_brier,
        "valid_accuracy": valid_acc,
        "test_accuracy": test_acc,
        "test_brier_H": test_brier_h,
        "test_brier_D": test_brier_d,
        "test_brier_A": test_brier_a,
        "ece_H": ece_map["H"],
        "ece_D": ece_map["D"],
        "ece_A": ece_map["A"]
    }])
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    print_separator("PODSUMOWANIE MODELU")
    print(summary.T.to_string(header=False))

    print_separator("PRECISION / RECALL / F1")
    print(class_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_separator("KALIBRACJA - WIN (H)")
    print(calib_print_tables["H"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_separator("KALIBRACJA - DRAW (D)")
    print(calib_print_tables["D"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_separator("KALIBRACJA - LOSS (A)")
    print(calib_print_tables["A"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_separator("PREDYKCJE PRZYSZLYCH MECZOW (najblizszy tydzien)")
    if len(future_pred_df) > 0:
        today = pd.Timestamp.today().normalize()
        week_mask = (future_pred_df[DATE_COL] >= today) & (future_pred_df[DATE_COL] <= today + pd.Timedelta(days=7))
        week_df = future_pred_df[week_mask]
        if len(week_df) > 0:
            print(week_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("Brak meczow w ciagu najblizszych 7 dni.")
    else:
        print("Brak przyszlych meczow bez wyniku w CSV.")

    print_separator("TOP 30 FEATURE IMPORTANCE")
    print(fi_df.head(30).to_string(index=False))

    print_separator("ZAPIS")
    print(f"Zapisano do: {output_dir}")


if __name__ == "__main__":
    main()
