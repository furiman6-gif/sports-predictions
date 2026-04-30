import math
import os
import warnings
from functools import reduce
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
RANKING_CSV_PATH = str(SCRIPT_DIR / "merged_data_Tylko_Cechy_Rankingowe.csv")
DATE_COL = "Date"
TARGET_COL = "FTR"

CLASS_ORDER = ["H", "D", "A"]
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_CLASS = {0: "H", 1: "D", 2: "A"}
CLASS_LABELS_PRINT = {"H": "win", "D": "draw", "A": "loss"}

OUTPUT_BASE_DIR = str(SCRIPT_DIR / "outputs_single")
TODAY = pd.Timestamp.today().normalize()
FUTURE_DAYS_AHEAD = 7
OVERDUE_DAYS_BACK = 3
TARGET_MODES = ["FTR", "O25", "BTTS", "CRD25", "CORNOU", "SHTOU", "FOULSOU"]
KEEP_ORIGINAL_TRAINING = True
CONFIG_GRID = [
    {"name": "baseline", "params": {}},
    {"name": "lr_002", "params": {"learning_rate": 0.02, "n_estimators": 2200}},
    {"name": "leaves_31", "params": {"num_leaves": 31}},
]
STAT_OU_MODES = {
    "CRD25": {"h_cols": ["HY"], "a_cols": ["AY"], "fixed_threshold": 2.5},
    "CORNOU": {"h_cols": ["HC"], "a_cols": ["AC"], "fixed_threshold": None},
    "SHTOU": {"h_cols": ["HST"], "a_cols": ["AST"], "fixed_threshold": None},
    "FOULSOU": {"h_cols": ["HF"], "a_cols": ["AF"], "fixed_threshold": None},
}

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


def ensure_ftr_column(df):
    data = df.copy()
    if "FTR" not in data.columns and "result" in data.columns:
        data["FTR"] = data["result"].map({"W": "H", "D": "D", "L": "A", "H": "H", "A": "A"})
    return data


def deduplicate_matches(df):
    if len(df) == 0:
        return df
    needed = [DATE_COL, "HomeTeam", "AwayTeam", "Season"]
    for c in needed:
        if c not in df.columns:
            return df
    data = df.copy().reset_index(drop=True)
    if "FTR" in data.columns:
        ftr_known = data["FTR"].isin(["H", "D", "A"])
    else:
        ftr_known = pd.Series(False, index=data.index)
    if "FTHG" in data.columns and "FTAG" in data.columns:
        goals_known = pd.to_numeric(data["FTHG"], errors="coerce").notna() & pd.to_numeric(data["FTAG"], errors="coerce").notna()
    else:
        goals_known = pd.Series(False, index=data.index)
    data["_q"] = (ftr_known.astype(int) * 1000) + (goals_known.astype(int) * 500) + data.notna().sum(axis=1)
    data["_i"] = np.arange(len(data))
    data = data.sort_values(needed + ["_q", "_i"], ascending=[True, True, True, True, False, False])
    data = data.drop_duplicates(subset=needed, keep="first").drop(columns=["_q", "_i"]).reset_index(drop=True)
    return data


def split_known_future_by_target(df, target_mode):
    df = df.copy()
    if target_mode == "FTR":
        known_mask = df[TARGET_COL].isin(CLASS_TO_INT.keys())
        known_df = df[known_mask].copy().reset_index(drop=True)
        unresolved_df = df[~known_mask].copy().reset_index(drop=True)
        known_df["target"] = known_df[TARGET_COL].map(CLASS_TO_INT)
        return known_df, unresolved_df, "multiclass", None

    gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
    ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
    known_goals = gh.notna() & ga.notna()

    if target_mode == "O25":
        known_df = df[known_goals].copy().reset_index(drop=True)
        unresolved_df = df[~known_goals].copy().reset_index(drop=True)
        gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
        ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")
        known_df["target"] = ((gh_k + ga_k) > 2.5).astype(int)
        return known_df, unresolved_df, "binary", 2.5

    if target_mode == "BTTS":
        known_df = df[known_goals].copy().reset_index(drop=True)
        unresolved_df = df[~known_goals].copy().reset_index(drop=True)
        gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
        ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")
        known_df["target"] = ((gh_k > 0) & (ga_k > 0)).astype(int)
        return known_df, unresolved_df, "binary", None

    if target_mode in STAT_OU_MODES:
        cfg = STAT_OU_MODES[target_mode]
        missing = [c for c in cfg["h_cols"] + cfg["a_cols"] if c not in df.columns]
        if missing:
            raise ValueError(f"Brak kolumn dla {target_mode}: {missing}")
        stat_h0 = pd.to_numeric(df[cfg["h_cols"][0]], errors="coerce")
        known_mask = known_goals & stat_h0.notna()
        known_df = df[known_mask].copy().reset_index(drop=True)
        unresolved_df = df[~known_mask].copy().reset_index(drop=True)
        h_tot = sum(pd.to_numeric(known_df[c], errors="coerce").fillna(0) for c in cfg["h_cols"])
        a_tot = sum(pd.to_numeric(known_df[c], errors="coerce").fillna(0) for c in cfg["a_cols"])
        total = h_tot + a_tot
        if cfg["fixed_threshold"] is not None:
            threshold = cfg["fixed_threshold"]
        else:
            if len(total) == 0:
                raise ValueError(f"Brak danych do obliczenia progu dla {target_mode}")
            threshold = float(np.floor(float(total.mean()) * 2) / 2)
        known_df["target"] = (total > threshold).astype(int)
        return known_df, unresolved_df, "binary", threshold

    raise ValueError(f"Nieznany target_mode: {target_mode}")


def binary_labels(target_mode):
    if target_mode == "BTTS":
        return "NO", "YES"
    return "U", "O"


def filter_future_window(df):
    if len(df) == 0 or DATE_COL not in df.columns:
        return df
    dt = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    mask = dt >= TODAY
    if FUTURE_DAYS_AHEAD is not None:
        mask &= dt <= (TODAY + pd.Timedelta(days=FUTURE_DAYS_AHEAD))
    return df[mask].copy().reset_index(drop=True)


def filter_overdue_window(df):
    if len(df) == 0 or DATE_COL not in df.columns:
        return df
    dt = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    mask = dt < TODAY
    if OVERDUE_DAYS_BACK is not None:
        mask &= dt >= (TODAY - pd.Timedelta(days=OVERDUE_DAYS_BACK))
    return df[mask].copy().reset_index(drop=True)


def add_light_bars_features(df):
    if len(df) == 0:
        return df
    data = df.copy()
    if DATE_COL in data.columns:
        data = data.sort_values(DATE_COL).reset_index(drop=True)
    default_rating = 1500.0
    k_factor = 24.0
    nu = 0.35
    ratings = {}
    total_matches = 0
    total_draws = 0
    win_home, draw_prob_list, edge = [], [], []

    for _, row in data.iterrows():
        h = row.get("HomeTeam")
        a = row.get("AwayTeam")
        if h not in ratings:
            ratings[h] = default_rating
        if a not in ratings:
            ratings[a] = default_rating
        ra = ratings[h]
        rb = ratings[a]
        ga = 10.0 ** (ra / 400.0)
        gb = 10.0 ** (rb / 400.0)
        sg = np.sqrt(ga * gb)
        den = ga + gb + nu * sg
        if den <= 0:
            pw, p_draw, pl = 1 / 3, 1 / 3, 1 / 3
        else:
            pw = ga / den
            p_draw = (nu * sg) / den
            pl = gb / den
        win_home.append(float(pw))
        draw_prob_list.append(float(p_draw))
        edge.append(float(pw - pl))

        outcome = None
        ftr = row.get("FTR")
        if isinstance(ftr, str):
            if ftr == "H":
                outcome = 1.0
            elif ftr == "D":
                outcome = 0.5
            elif ftr == "A":
                outcome = 0.0
        if outcome is None:
            hg = pd.to_numeric(row.get("FTHG"), errors="coerce")
            ag = pd.to_numeric(row.get("FTAG"), errors="coerce")
            if pd.notna(hg) and pd.notna(ag):
                if hg > ag:
                    outcome = 1.0
                elif hg < ag:
                    outcome = 0.0
                else:
                    outcome = 0.5
        if outcome is not None:
            ratings[h] = ra + k_factor * (outcome - (pw + 0.5 * p_draw))
            ratings[a] = rb + k_factor * ((1.0 - outcome) - (pl + 0.5 * p_draw))
            total_matches += 1
            if abs(outcome - 0.5) < 0.01:
                total_draws += 1
            if total_matches >= 50:
                draw_rate = total_draws / total_matches
                if draw_rate < 0.95:
                    nu = float(max(0.05, min(2.0, nu * 0.995 + (2.0 * draw_rate / (1.0 - draw_rate)) * 0.005)))

    data["bars_ed_win_home"] = win_home
    data["bars_ed_draw"] = draw_prob_list
    data["bars_ed_edge"] = edge
    return data


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


def build_full_feature_frame(df_all, use_xg=True):
    data = df_all.copy().sort_values(DATE_COL).reset_index(drop=True)
    long_df = build_team_match_history(data, use_xg=use_xg)
    long_df = add_advanced_rolling_features(long_df)
    full = merge_form_features_to_match(data, long_df)
    full = add_h2h_features(full)
    full = compute_elo(full)
    full = compute_goals_elo(full)
    full = compute_glicko(full)
    full = compute_elo_home_away(full)
    full = compute_goals_elo_home_away(full)
    full = compute_glicko_home_away(full)
    if use_xg:
        full = compute_xg_elo(full)
        full = compute_xg_elo_home_away(full)
    return full


def make_bundle_for_target(full_df, target_mode, split_mode):
    known_df, unresolved_df, task_kind, ou_threshold = split_known_future_by_target(full_df, target_mode)
    future_df = filter_future_window(unresolved_df)
    overdue_df = filter_overdue_window(unresolved_df)
    train_s, valid_s, test_s = seasonal_split(known_df["Season"].unique(), split_mode)
    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()
    feature_cols = get_feature_columns(known_df)
    X_train, X_valid, X_test, X_future, final_feats, dropped = filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )
    if len(overdue_df) > 0:
        X_overdue = overdue_df.reindex(columns=final_feats).copy()
        X_overdue = X_overdue.apply(pd.to_numeric, errors="coerce")
        fill_values = X_train.median(numeric_only=True)
        X_overdue = X_overdue.fillna(fill_values).fillna(0.0)
    else:
        X_overdue = pd.DataFrame(columns=final_feats)
    return {
        "target_mode": target_mode,
        "task_kind": task_kind,
        "ou_threshold": ou_threshold,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "future_df": future_df,
        "overdue_df": overdue_df,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "X_future": X_future,
        "X_overdue": X_overdue,
        "y_train": train_df["target"].values,
        "y_valid": valid_df["target"].values,
        "y_test": test_df["target"].values,
        "features": final_feats,
        "dropped": dropped,
        "train_seasons": "|".join(map(str, train_s)),
        "valid_seasons": "|".join(map(str, valid_s)),
        "test_seasons": "|".join(map(str, test_s)),
    }


def evaluate_bundle(bundle, cfg_name, cfg_override):
    params = dict(LGBM_PARAMS)
    params.update(cfg_override)
    params.setdefault("verbose", -1)
    params.setdefault("force_col_wise", True)
    if bundle["task_kind"] == "binary":
        params["objective"] = "binary"
        params.pop("num_class", None)
    if params.get("boosting_type") == "goss" and "data_sample_strategy" not in params:
        params["data_sample_strategy"] = "goss"

    model = LGBMClassifier(**params)
    model.fit(
        bundle["X_train"],
        bundle["y_train"],
        eval_set=[(bundle["X_valid"], bundle["y_valid"])],
        eval_metric="multi_logloss" if bundle["task_kind"] == "multiclass" else "binary_logloss",
        callbacks=[
            lgb.early_stopping(150 if KEEP_ORIGINAL_TRAINING else 40),
            lgb.log_evaluation(50 if KEEP_ORIGINAL_TRAINING else 0)
        ]
    )

    best_iter = model.best_iteration_
    valid_proba = model.predict_proba(bundle["X_valid"], num_iteration=best_iter)
    test_proba = model.predict_proba(bundle["X_test"], num_iteration=best_iter)

    if bundle["task_kind"] == "multiclass":
        valid_pred = np.argmax(valid_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)
        valid_logloss = log_loss(bundle["y_valid"], valid_proba, labels=[0, 1, 2])
        test_logloss = log_loss(bundle["y_test"], test_proba, labels=[0, 1, 2])
        valid_acc = accuracy_score(bundle["y_valid"], valid_pred)
        test_acc = accuracy_score(bundle["y_test"], test_pred)
        valid_brier = float(np.mean(np.sum((valid_proba - np.eye(3)[bundle["y_valid"]]) ** 2, axis=1)))
        test_brier = float(np.mean(np.sum((test_proba - np.eye(3)[bundle["y_test"]]) ** 2, axis=1)))
    else:
        valid_p = valid_proba[:, 1]
        test_p = test_proba[:, 1]
        valid_pred = (valid_p >= 0.5).astype(int)
        test_pred = (test_p >= 0.5).astype(int)
        valid_logloss = log_loss(bundle["y_valid"], valid_proba, labels=[0, 1])
        test_logloss = log_loss(bundle["y_test"], test_proba, labels=[0, 1])
        valid_acc = accuracy_score(bundle["y_valid"], valid_pred)
        test_acc = accuracy_score(bundle["y_test"], test_pred)
        valid_brier = float(np.mean((valid_p - bundle["y_valid"]) ** 2))
        test_brier = float(np.mean((test_p - bundle["y_test"]) ** 2))

    return {
        "model": model,
        "cfg_name": cfg_name,
        "cfg_override": cfg_override,
        "best_iter": int(best_iter),
        "valid_logloss": float(valid_logloss),
        "test_logloss": float(test_logloss),
        "valid_acc": float(valid_acc),
        "test_acc": float(test_acc),
        "valid_brier": float(valid_brier),
        "test_brier": float(test_brier),
    }


def build_calibration_5pct(predictions_df):
    if len(predictions_df) == 0 or "prediction_split" not in predictions_df.columns:
        return pd.DataFrame()
    test_df = predictions_df[predictions_df["prediction_split"] == "test"].copy()
    if len(test_df) == 0 or "target_true" not in test_df.columns:
        return pd.DataFrame()

    def bin_and_agg(df_src, prob_col, y_true, target_mode, outcome):
        p = pd.to_numeric(df_src.get(prob_col), errors="coerce")
        y = pd.to_numeric(y_true, errors="coerce")
        ok = p.notna() & y.notna()
        if int(ok.sum()) == 0:
            return []
        d = pd.DataFrame({
            "p": p.loc[ok].astype(float).values,
            "y": y.loc[ok].astype(float).values,
        })
        d["bin_idx"] = np.floor(np.clip(d["p"], 0, 0.999999) / 0.05).astype(int)
        d["bin_start"] = d["bin_idx"] * 0.05
        d["bin_end"] = d["bin_start"] + 0.05
        grouped = d.groupby(["bin_idx", "bin_start", "bin_end"], as_index=False).agg(
            n=("p", "size"),
            avg_pred=("p", "mean"),
            actual_rate=("y", "mean"),
            hits=("y", "sum"),
        )
        grouped["target_mode"] = target_mode
        grouped["outcome"] = outcome
        grouped["calib_gap"] = grouped["avg_pred"] - grouped["actual_rate"]
        return grouped.to_dict("records")

    rows = []
    for mode in TARGET_MODES:
        m = test_df[test_df["target_mode"] == mode].copy()
        if len(m) == 0:
            continue
        y = pd.to_numeric(m["target_true"], errors="coerce")
        if mode == "FTR":
            mapping = [("H", "pred_H", 0), ("D", "pred_D", 1), ("A", "pred_A", 2)]
        elif mode == "BTTS":
            mapping = [("NO", "pred_NO", 0), ("YES", "pred_YES", 1)]
        else:
            mapping = [("U", "pred_U", 0), ("O", "pred_O", 1)]
        for outcome, col, cls in mapping:
            if col in m.columns:
                y_true = (y == cls).astype(float)
                rows.extend(bin_and_agg(m, col, y_true, mode, outcome))
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target_mode", "outcome", "bin_idx"]).reset_index(drop=True)


def build_future_predictions_1row(best_df, predictions_df):
    if len(predictions_df) == 0:
        return pd.DataFrame()
    fut = predictions_df[predictions_df.get("prediction_split", "") == "future"].copy()
    if len(fut) == 0:
        return pd.DataFrame()
    keys = [k for k in ["Date", "Season", "HomeTeam", "AwayTeam"] if k in fut.columns]
    chunks = []
    for mode in TARGET_MODES:
        m = fut[fut["target_mode"] == mode].copy()
        if len(m) == 0:
            continue
        if mode == "FTR":
            cols = [c for c in ["pred_H", "pred_D", "pred_A", "pred_class", "max_prob"] if c in m.columns]
        elif mode == "BTTS":
            cols = [c for c in ["pred_NO", "pred_YES", "pred_class", "max_prob"] if c in m.columns]
        else:
            cols = [c for c in ["pred_U", "pred_O", "pred_class", "max_prob", "ou_threshold"] if c in m.columns]
        rename = {c: f"{mode}_{c}" for c in cols}
        chunks.append(m[keys + cols].rename(columns=rename))
    if len(chunks) == 0:
        return pd.DataFrame()
    merged = reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), chunks)
    if len(best_df) > 0:
        metrics = best_df[["target_mode", "test_acc", "test_logloss", "valid_acc", "valid_logloss"]].copy()
        for mode in TARGET_MODES:
            mm = metrics[metrics["target_mode"] == mode].drop(columns=["target_mode"])
            if len(mm) == 0:
                continue
            mm = mm.rename(columns={
                "test_acc": f"{mode}_test_acc",
                "test_logloss": f"{mode}_test_logloss",
                "valid_acc": f"{mode}_valid_acc",
                "valid_logloss": f"{mode}_valid_logloss",
            })
            merged = pd.concat([merged, pd.DataFrame([mm.iloc[0].to_dict()] * len(merged))], axis=1)
    return merged


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
    if os.path.exists(RANKING_CSV_PATH):
        rank_df = pd.read_csv(RANKING_CSV_PATH, low_memory=False)
        rank_df[DATE_COL] = pd.to_datetime(rank_df[DATE_COL], errors="coerce", dayfirst=False)
        rank_df = rank_df.drop(columns=["result"], errors="ignore")
        rank_df = rank_df.dropna(subset=[DATE_COL])
        df_all = df_all.merge(rank_df, on=[DATE_COL, "HomeTeam", "AwayTeam"], how="left")
    else:
        print(f"[WARN] Nie znaleziono pliku rankingowego: {RANKING_CSV_PATH}")
    df_all["Season"] = infer_season_from_date(df_all[DATE_COL])

    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        print(f"Plik ma tylko {len(all_seasons)} sezonów, a podałeś {n_last_seasons}")
        return

    selected_seasons = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected_seasons)].copy().reset_index(drop=True)
    full_with_features = build_full_feature_frame(df_all, use_xg=use_xg)

    all_rows = []
    best_rows = []
    all_features = []
    all_predictions = []
    all_future = []
    all_overdue = []
    errors = []
    ftr_artifacts = {}

    print_separator("RUN - WSZYSTKIE TARGETY")

    for target_mode in TARGET_MODES:
        print(f"target={target_mode}")
        try:
            bundle = make_bundle_for_target(full_with_features, target_mode, split_mode)
            if len(bundle["features"]) == 0:
                raise ValueError("Brak cech po filtracji")

            if n_features_limit and n_features_limit < len(bundle["features"]):
                variances = bundle["X_train"].var(axis=0, skipna=True).fillna(0.0)
                keep = variances.sort_values(ascending=False).head(n_features_limit).index.tolist()
                bundle["X_train"] = bundle["X_train"][keep]
                bundle["X_valid"] = bundle["X_valid"][keep]
                bundle["X_test"] = bundle["X_test"][keep]
                bundle["X_future"] = bundle["X_future"][keep] if len(bundle["X_future"]) > 0 else bundle["X_future"]
                bundle["X_overdue"] = bundle["X_overdue"][keep] if len(bundle["X_overdue"]) > 0 else bundle["X_overdue"]
                bundle["features"] = keep

            best_eval = None
            cfg_list = [{"name": "original", "params": {}}] if KEEP_ORIGINAL_TRAINING else CONFIG_GRID
            for cfg in cfg_list:
                ev = evaluate_bundle(bundle, cfg["name"], cfg["params"])
                row = {
                    "target_mode": target_mode,
                    "stage": "param_search",
                    "n_seasons": n_last_seasons,
                    "cfg_name": cfg["name"],
                    "cfg_override": str(cfg["params"]),
                    "valid_logloss": ev["valid_logloss"],
                    "test_logloss": ev["test_logloss"],
                    "valid_acc": ev["valid_acc"],
                    "test_acc": ev["test_acc"],
                    "valid_brier": ev["valid_brier"],
                    "test_brier": ev["test_brier"],
                    "best_iter": ev["best_iter"],
                    "n_train": len(bundle["train_df"]),
                    "n_valid": len(bundle["valid_df"]),
                    "n_test": len(bundle["test_df"]),
                    "n_future": len(bundle["future_df"]),
                    "n_overdue": len(bundle["overdue_df"]),
                    "n_features": len(bundle["features"]),
                    "train_seasons": bundle["train_seasons"],
                    "valid_seasons": bundle["valid_seasons"],
                    "test_seasons": bundle["test_seasons"],
                }
                all_rows.append(row)
                if best_eval is None or (ev["valid_logloss"], ev["test_logloss"], -ev["test_acc"]) < (
                    best_eval["valid_logloss"], best_eval["test_logloss"], -best_eval["test_acc"]
                ):
                    best_eval = ev

            best_row = {
                "target_mode": target_mode,
                "csv_path": CSV_PATH,
                "best_n_seasons": n_last_seasons,
                "best_cfg_name": best_eval["cfg_name"],
                "best_cfg_override": str(best_eval["cfg_override"]),
                "valid_logloss": best_eval["valid_logloss"],
                "test_logloss": best_eval["test_logloss"],
                "valid_acc": best_eval["valid_acc"],
                "test_acc": best_eval["test_acc"],
                "valid_brier": best_eval["valid_brier"],
                "test_brier": best_eval["test_brier"],
                "best_iter": best_eval["best_iter"],
                "n_train": len(bundle["train_df"]),
                "n_valid": len(bundle["valid_df"]),
                "n_test": len(bundle["test_df"]),
                "n_future": len(bundle["future_df"]),
                "n_overdue": len(bundle["overdue_df"]),
                "n_features": len(bundle["features"]),
                "train_seasons": bundle["train_seasons"],
                "valid_seasons": bundle["valid_seasons"],
                "test_seasons": bundle["test_seasons"],
                "use_xg": use_xg,
                "task_kind": bundle["task_kind"],
                "ou_threshold": bundle["ou_threshold"],
            }
            best_rows.append(best_row)

            all_features.extend([
                {"target_mode": target_mode, "feature_order": i + 1, "feature_name": f}
                for i, f in enumerate(bundle["features"])
            ])

            model = best_eval["model"]
            best_iter = best_eval["best_iter"]
            test_proba = model.predict_proba(bundle["X_test"], num_iteration=best_iter)
            test_base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in bundle["test_df"].columns]
            test_pred_df = bundle["test_df"][test_base_cols].copy()
            test_pred_df["target_mode"] = target_mode
            test_pred_df["prediction_split"] = "test"
            test_pred_df["target_true"] = bundle["test_df"]["target"].values
            if bundle["ou_threshold"] is not None:
                test_pred_df["ou_threshold"] = bundle["ou_threshold"]

            if bundle["task_kind"] == "multiclass":
                test_pred = np.argmax(test_proba, axis=1)
                test_pred_df["pred_H"] = test_proba[:, 0]
                test_pred_df["pred_D"] = test_proba[:, 1]
                test_pred_df["pred_A"] = test_proba[:, 2]
                test_pred_df["pred_class"] = [INT_TO_CLASS[i] for i in test_pred]
                test_pred_df["max_prob"] = np.max(test_proba, axis=1)
            else:
                neg_label, pos_label = binary_labels(target_mode)
                test_pred = (test_proba[:, 1] >= 0.5).astype(int)
                test_pred_df[f"pred_{neg_label}"] = test_proba[:, 0]
                test_pred_df[f"pred_{pos_label}"] = test_proba[:, 1]
                test_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in test_pred]
                test_pred_df["max_prob"] = np.maximum(test_proba[:, 0], test_proba[:, 1])
            all_predictions.extend(test_pred_df.to_dict("records"))

            future_pred_df = pd.DataFrame()
            if len(bundle["future_df"]) > 0 and len(bundle["X_future"]) > 0:
                future_proba = model.predict_proba(bundle["X_future"], num_iteration=best_iter)
                future_base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in bundle["future_df"].columns]
                future_pred_df = bundle["future_df"][future_base_cols].copy()
                future_pred_df["target_mode"] = target_mode
                future_pred_df["prediction_split"] = "future"
                if bundle["ou_threshold"] is not None:
                    future_pred_df["ou_threshold"] = bundle["ou_threshold"]
                if bundle["task_kind"] == "multiclass":
                    future_pred = np.argmax(future_proba, axis=1)
                    future_pred_df["pred_H"] = future_proba[:, 0]
                    future_pred_df["pred_D"] = future_proba[:, 1]
                    future_pred_df["pred_A"] = future_proba[:, 2]
                    future_pred_df["pred_class"] = [INT_TO_CLASS[i] for i in future_pred]
                    future_pred_df["max_prob"] = np.max(future_proba, axis=1)
                else:
                    neg_label, pos_label = binary_labels(target_mode)
                    future_pred = (future_proba[:, 1] >= 0.5).astype(int)
                    future_pred_df[f"pred_{neg_label}"] = future_proba[:, 0]
                    future_pred_df[f"pred_{pos_label}"] = future_proba[:, 1]
                    future_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in future_pred]
                    future_pred_df["max_prob"] = np.maximum(future_proba[:, 0], future_proba[:, 1])
                all_predictions.extend(future_pred_df.to_dict("records"))
                all_future.append(future_pred_df)

            overdue_pred_df = pd.DataFrame()
            if len(bundle["overdue_df"]) > 0 and len(bundle["X_overdue"]) > 0:
                overdue_proba = model.predict_proba(bundle["X_overdue"], num_iteration=best_iter)
                overdue_base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in bundle["overdue_df"].columns]
                overdue_pred_df = bundle["overdue_df"][overdue_base_cols].copy()
                overdue_pred_df["target_mode"] = target_mode
                overdue_pred_df["prediction_split"] = "overdue"
                if bundle["ou_threshold"] is not None:
                    overdue_pred_df["ou_threshold"] = bundle["ou_threshold"]
                if bundle["task_kind"] == "multiclass":
                    overdue_pred = np.argmax(overdue_proba, axis=1)
                    overdue_pred_df["pred_H"] = overdue_proba[:, 0]
                    overdue_pred_df["pred_D"] = overdue_proba[:, 1]
                    overdue_pred_df["pred_A"] = overdue_proba[:, 2]
                    overdue_pred_df["pred_class"] = [INT_TO_CLASS[i] for i in overdue_pred]
                    overdue_pred_df["max_prob"] = np.max(overdue_proba, axis=1)
                else:
                    neg_label, pos_label = binary_labels(target_mode)
                    overdue_pred = (overdue_proba[:, 1] >= 0.5).astype(int)
                    overdue_pred_df[f"pred_{neg_label}"] = overdue_proba[:, 0]
                    overdue_pred_df[f"pred_{pos_label}"] = overdue_proba[:, 1]
                    overdue_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in overdue_pred]
                    overdue_pred_df["max_prob"] = np.maximum(overdue_proba[:, 0], overdue_proba[:, 1])
                all_predictions.extend(overdue_pred_df.to_dict("records"))
                all_overdue.append(overdue_pred_df)

            if target_mode == "FTR":
                test_proba_ftr = test_proba
                y_test = bundle["y_test"]
                test_pred = np.argmax(test_proba_ftr, axis=1)
                test_brier_h = per_class_brier(y_test, test_proba_ftr, 0)
                test_brier_d = per_class_brier(y_test, test_proba_ftr, 1)
                test_brier_a = per_class_brier(y_test, test_proba_ftr, 2)
                prec, rec, f1, supp = precision_recall_fscore_support(
                    y_test, test_pred, labels=[0, 1, 2], zero_division=0
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
                    p_cls = test_proba_ftr[:, cls_idx]
                    calib_df = calibration_table_binary(y_bin, p_cls, step=0.05)
                    ece_map[cls_name] = expected_calibration_error(calib_df)
                    calib_print_tables[cls_name] = calib_df
                fi_df = pd.DataFrame({
                    "feature": bundle["features"],
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)
                dropped_df = pd.DataFrame(bundle["dropped"], columns=["feature", "reason"])
                summary = pd.DataFrame([{
                    "file": Path(CSV_PATH).name,
                    "n_selected_seasons": n_last_seasons,
                    "train_seasons": bundle["train_seasons"],
                    "valid_seasons": bundle["valid_seasons"],
                    "test_seasons": bundle["test_seasons"],
                    "n_train": len(bundle["train_df"]),
                    "n_valid": len(bundle["valid_df"]),
                    "n_test": len(bundle["test_df"]),
                    "n_future": len(bundle["future_df"]),
                    "n_features": len(bundle["features"]),
                    "best_iteration": best_iter,
                    "valid_logloss": best_eval["valid_logloss"],
                    "test_logloss": best_eval["test_logloss"],
                    "valid_brier_multiclass": best_eval["valid_brier"],
                    "test_brier_multiclass": best_eval["test_brier"],
                    "valid_accuracy": best_eval["valid_acc"],
                    "test_accuracy": best_eval["test_acc"],
                    "test_brier_H": test_brier_h,
                    "test_brier_D": test_brier_d,
                    "test_brier_A": test_brier_a,
                    "ece_H": ece_map["H"],
                    "ece_D": ece_map["D"],
                    "ece_A": ece_map["A"]
                }])
                ftr_artifacts = {
                    "summary": summary,
                    "class_metrics": class_metrics_df,
                    "calib": calib_print_tables,
                    "fi": fi_df,
                    "dropped": dropped_df,
                    "test_predictions": test_pred_df.copy(),
                    "future_predictions": future_pred_df.copy(),
                }

            print(
                f"  OK cfg={best_row['best_cfg_name']} "
                f"v_ll={best_row['valid_logloss']:.4f} "
                f"t_ll={best_row['test_logloss']:.4f} "
                f"t_acc={best_row['test_acc']:.3f}"
            )
        except Exception as e:
            errors.append({"target_mode": target_mode, "error": str(e)})
            print(f"  BLAD: {e}")

    tuning_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)
    feature_df = pd.DataFrame(all_features)
    predictions_df = pd.DataFrame(all_predictions)
    errors_df = pd.DataFrame(errors)
    calibration_df = build_calibration_5pct(predictions_df)
    future_1row_df = build_future_predictions_1row(best_df, predictions_df)

    tuning_df.to_csv(os.path.join(output_dir, "tuning_runs.csv"), index=False)
    best_df.to_csv(os.path.join(output_dir, "best_settings.csv"), index=False)
    feature_df.to_csv(os.path.join(output_dir, "feature_list_full.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions_full.csv"), index=False)
    calibration_df.to_csv(os.path.join(output_dir, "calibration_5pct_bins.csv"), index=False)
    future_1row_df.to_csv(os.path.join(output_dir, "future_predictions_1row_all_targets.csv"), index=False)
    errors_df.to_csv(os.path.join(output_dir, "errors.csv"), index=False)

    if len(all_future) > 0:
        pd.concat(all_future, ignore_index=True).to_csv(os.path.join(output_dir, "future_predictions_all.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(output_dir, "future_predictions_all.csv"), index=False)

    if len(all_overdue) > 0:
        pd.concat(all_overdue, ignore_index=True).to_csv(os.path.join(output_dir, "overdue_predictions_all.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(output_dir, "overdue_predictions_all.csv"), index=False)

    if len(ftr_artifacts) > 0:
        ftr_artifacts["dropped"].to_csv(os.path.join(output_dir, "dropped_features.csv"), index=False)
        ftr_artifacts["fi"].to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        ftr_artifacts["test_predictions"].to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        if len(ftr_artifacts["future_predictions"]) > 0:
            ftr_artifacts["future_predictions"].to_csv(os.path.join(output_dir, "future_predictions.csv"), index=False)
        for cls in CLASS_ORDER:
            ftr_artifacts["calib"][cls].to_csv(os.path.join(output_dir, f"calibration_bins_{cls}.csv"), index=False)
        ftr_artifacts["summary"].to_csv(os.path.join(output_dir, "summary.csv"), index=False)

        print_separator("PODSUMOWANIE FTR")
        print(ftr_artifacts["summary"].T.to_string(header=False))
        print_separator("PRECISION / RECALL / F1")
        print(ftr_artifacts["class_metrics"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print_separator("KALIBRACJA - WIN (H)")
        print(ftr_artifacts["calib"]["H"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print_separator("KALIBRACJA - DRAW (D)")
        print(ftr_artifacts["calib"]["D"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print_separator("KALIBRACJA - LOSS (A)")
        print(ftr_artifacts["calib"]["A"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print_separator("TOP 30 FEATURE IMPORTANCE")
        print(ftr_artifacts["fi"].head(30).to_string(index=False))

    print_separator("PODSUMOWANIE MULTI-TARGET")
    if len(best_df) > 0:
        print(best_df[["target_mode", "best_cfg_name", "valid_logloss", "test_logloss", "test_acc"]].to_string(index=False))
    else:
        print("Brak udanych targetów.")
    print_separator("ZAPIS")
    print(f"Zapisano do: {output_dir}")


if __name__ == "__main__":
    main()
