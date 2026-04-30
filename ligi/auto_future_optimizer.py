import importlib.util
import json
import traceback
from functools import reduce
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "auto_outputs_future"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric_columns(df):
    """Konwertuje string/object columns do numeric jesli wszystkie niepuste wartosci sa liczbami.
    Po naszym CSV fix, kolumny ktore mialy puste stringi (przyszle mecze) sa object/string dtype
    zamiast float. Trzeba je wymusic na numeric bo gbm4.py robi df['FTHG'] > df['FTAG']."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        # object lub StringDtype - sprobuj skonwertowac
        try:
            col_str = df[col].astype(str).str.strip()
        except Exception:
            continue
        non_empty_mask = (col_str != "") & (col_str.str.lower() != "nan")
        if not non_empty_mask.any():
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # jesli wszystkie niepuste wartosci sa liczbami, uzyj numeric
        if converted[non_empty_mask].notna().all():
            df[col] = converted
    return df


def _prefix_csv_columns(path):
    """Naprawia trailing comma / ragged rows w CSV przed pd.read_csv."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return
    lines = text.splitlines()
    if not lines:
        return
    header_fields = lines[0].rstrip("\r").split(",")
    while header_fields and header_fields[-1].strip() == "":
        header_fields.pop()
    n_cols = len(header_fields)
    if n_cols == 0:
        return
    needs_fix = False
    for line in lines[1:]:
        if not line.strip():
            continue
        if line.rstrip("\r").count(",") + 1 != n_cols:
            needs_fix = True
            break
    if not needs_fix and lines[0].rstrip("\r") == ",".join(header_fields):
        return
    fixed = [",".join(header_fields)]
    for line in lines[1:]:
        line = line.rstrip("\r")
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) > n_cols:
            fields = fields[:n_cols]
        elif len(fields) < n_cols:
            fields = fields + [""] * (n_cols - len(fields))
        fixed.append(",".join(fields))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(fixed) + "\n")
TODAY = pd.Timestamp.today().normalize()
FUTURE_DAYS_AHEAD = 7
OVERDUE_DAYS_BACK = 3
MAX_FEATURES = None
MAX_LEAGUES = None
_BASE_OU_TARGETS = [
    "O25",
    "CRD25", "SHTOU",
]
# dla kazdego OU dodajemy 2 dodatkowe linie: _LO (base-1) i _HI (base+1)
# ODCHUDZONE: tylko bazowe linie, bez _LO/_HI i bez wariantow per-druzyna
TARGET_MODES = ["FTR", "BTTS"] + _BASE_OU_TARGETS


def _strip_lohi(target_mode: str):
    """Zwroc (base_target, offset). Offset to -1.0/+1.0/0.0."""
    if target_mode.endswith("_LO"):
        return target_mode[:-3], -1.0
    if target_mode.endswith("_HI"):
        return target_mode[:-3], +1.0
    return target_mode, 0.0
FORCED_RUNS = []
TRAIN_ALL_YEARS = True
FAST_DEFAULT_PARAMS = {
    "n_estimators": 1800,
    "num_leaves": 47,
    "min_child_samples": 80,
    "learning_rate": 0.03,
}


CONFIG_GRID = [
    {"name": "baseline", "params": {}},
    {"name": "lr_002", "params": {"learning_rate": 0.02, "n_estimators": 2200}},
    {"name": "leaves_31", "params": {"num_leaves": 31}},
]


def find_main_csv(league_dir: Path) -> Path | None:
    p1 = league_dir / "wszystkie_sezony.csv"
    p2 = league_dir / "merged_data.csv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def detect_future_leagues() -> list[dict]:
    rows = []
    for gbm4_path in ROOT.glob("*/*/gbm4.py"):
        league_dir = gbm4_path.parent
        csv_path = find_main_csv(league_dir)
        if csv_path is None:
            continue
        try:
            _prefix_csv_columns(csv_path)
            df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="warn")
        except Exception:
            continue
        if "Date" not in df.columns:
            continue
        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        if "FTR" in df.columns:
            known_mask = df["FTR"].isin(["H", "D", "A"])
        else:
            gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
            ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
            known_mask = gh.notna() & ga.notna()
        future_mask = (dt >= TODAY) & (~known_mask)
        future_count = int(future_mask.sum())
        if future_count > 0:
            rows.append({
                "league_dir": league_dir,
                "league_name": str(league_dir.relative_to(ROOT)).replace("\\", "/"),
                "gbm4_path": gbm4_path,
                "csv_path": csv_path,
                "future_count": future_count,
            })
    rows.sort(key=lambda x: x["league_name"])
    return rows


def detect_forced_leagues() -> list[dict]:
    rows = []
    for rel_dir, csv_name in FORCED_RUNS:
        league_dir = ROOT / rel_dir
        gbm4_path = league_dir / "gbm4.py"
        csv_path = league_dir / csv_name
        if not gbm4_path.exists() or not csv_path.exists():
            continue
        try:
            _prefix_csv_columns(csv_path)
            df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="warn")
            df = ensure_ftr_column(df)
            if "Date" in df.columns:
                dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                if "FTR" in df.columns:
                    known_mask = df["FTR"].isin(["H", "D", "A"])
                else:
                    gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
                    ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
                    known_mask = gh.notna() & ga.notna()
                future_mask = (dt >= TODAY) & (~known_mask)
                future_count = int(future_mask.sum())
            else:
                future_count = 0
        except Exception:
            future_count = 0
        rows.append({
            "league_dir": league_dir,
            "league_name": f"{rel_dir}::{Path(csv_name).stem}",
            "gbm4_path": gbm4_path,
            "csv_path": csv_path,
            "future_count": future_count,
        })
    rows.sort(key=lambda x: x["league_name"])
    return rows


def ensure_ftr_column(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "FTR" not in data.columns and "result" in data.columns:
        data["FTR"] = data["result"].map({"W": "H", "D": "D", "L": "A", "H": "H", "A": "A"})
    return data


def is_ranking_only_csv(csv_path: str | Path) -> bool:
    return "tylko_cechy_rankingowe" in str(csv_path).lower()


def get_target_modes_for_csv(csv_path: str | Path) -> list[str]:
    try:
        probe = pd.read_csv(csv_path, nrows=1, low_memory=False)
    except Exception:
        return ["FTR"]
    base = ["FTR"]
    if "FTHG" in probe.columns and "FTAG" in probe.columns:
        base.extend(["O25", "BTTS", "HGOOU", "AGOOU"])
    if "HY" in probe.columns and "AY" in probe.columns:
        base.extend(["CRD25", "HCRDOU", "ACRDOU"])
    if "HC" in probe.columns and "AC" in probe.columns:
        base.extend(["CORNOU", "HCORNOU", "ACORNOU"])
    if "HST" in probe.columns and "AST" in probe.columns:
        base.extend(["SHTOU", "HSHTOU", "ASHTOU"])
    if "HF" in probe.columns and "AF" in probe.columns:
        base.extend(["FOULSOU", "HFOULOU", "AFOULOU"])
    # rozszerz o _LO/_HI dla kazdego OU
    out = []
    for b in base:
        out.append(b)
        if b in _BASE_OU_TARGETS:
            out.extend([f"{b}_LO", f"{b}_HI"])
    out_set = set(out)
    return [m for m in TARGET_MODES if m in out_set]


def load_gbm4_module(gbm4_path: Path):
    module_name = "gbm4_" + str(abs(hash(str(gbm4_path))))
    spec = importlib.util.spec_from_file_location(module_name, str(gbm4_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_balanced_threshold(total_values: pd.Series, step: float = 0.5) -> float:
    s = pd.to_numeric(total_values, errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError("Brak danych do wyznaczenia progu")
    lo = float(np.floor(s.min() / step) * step)
    hi = float(np.ceil(s.max() / step) * step)
    if hi <= lo:
        return float(lo)
    candidates = np.arange(lo, hi + step, step)
    best_any = None
    best_mid = None
    for th in candidates:
        over_rate = float((s > th).mean())
        score = abs(over_rate - 0.5)
        rec = (score, th)
        if best_any is None or rec < best_any:
            best_any = rec
        if 0.30 <= over_rate <= 0.70:
            if best_mid is None or rec < best_mid:
                best_mid = rec
    if best_mid is not None:
        return float(best_mid[1])
    return float(best_any[1])


LEAGUE_THRESHOLD_OVERRIDES = {
    # liga (substring matchowane do league_name): {target: prog}
    "England/Premier_League":   {"CRD25": 3.5, "FOULSOU": 21.5},
    "England/Championship":     {"CRD25": 3.5, "FOULSOU": 21.5},
    "England/League_One":       {"CRD25": 3.5, "FOULSOU": 21.5},
    "England/League_Two":       {"CRD25": 3.5, "FOULSOU": 21.5},
    "England/Conference":       {"CRD25": 3.5, "FOULSOU": 21.5},
    "Germany/Bundesliga_1":     {"CRD25": 2.5, "FOULSOU": 21.5},
    "Germany/Bundesliga_2":     {"CRD25": 2.5, "FOULSOU": 21.5},
    "Spain/La_Liga":            {"CRD25": 5.5, "FOULSOU": 23.5},
    "Spain/Segunda_Division":   {"CRD25": 5.5, "FOULSOU": 23.5},
    "Italy/Serie_A":            {"CRD25": 4.5, "FOULSOU": 24.5},
    "Italy/Serie_B":            {"CRD25": 4.5, "FOULSOU": 24.5},
    "France/Ligue_1":           {"CRD25": 3.5, "FOULSOU": 22.5},
    "France/Ligue_2":           {"CRD25": 3.5, "FOULSOU": 22.5},
    "Netherlands/Eredivisie":   {"CRD25": 3.5, "FOULSOU": 21.5},
    "Portugal/Primeira_Liga":   {"CRD25": 3.5, "FOULSOU": 21.5},
    "Belgium/First_Division_A": {"CRD25": 3.5, "FOULSOU": 21.5},
    "Scotland/Premiership":     {"CRD25": 3.5, "FOULSOU": 21.5},
    "Scotland/Championship":    {"CRD25": 3.5, "FOULSOU": 21.5},
    "Scotland/League_One":      {"CRD25": 3.5, "FOULSOU": 21.5},
    "Scotland/League_Two":      {"CRD25": 3.5, "FOULSOU": 21.5},
    "Turkey/Super_Lig":         {"CRD25": 4.5, "FOULSOU": 23.5},
    "Greece/Super_League":      {"CRD25": 4.5, "FOULSOU": 23.5},
}


def get_league_threshold(league_name: str | None, target_mode: str):
    if not league_name:
        return None
    for key, overrides in LEAGUE_THRESHOLD_OVERRIDES.items():
        if key in league_name and target_mode in overrides:
            return overrides[target_mode]
    return None


def split_known_future_by_target(module, df: pd.DataFrame, target_mode: str, league_name: str | None = None):
    df = df.copy()
    base_target, offset = _strip_lohi(target_mode)

    if target_mode == "FTR":
        known_mask = df[module.TARGET_COL].isin(module.CLASS_TO_INT.keys())
        known_df = df[known_mask].copy().reset_index(drop=True)
        future_df = df[~known_mask].copy().reset_index(drop=True)
        known_df["target"] = known_df[module.TARGET_COL].map(module.CLASS_TO_INT)
        return known_df, future_df, "multiclass", None

    gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
    ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
    known_mask = gh.notna() & ga.notna()
    known_df = df[known_mask].copy().reset_index(drop=True)
    future_df = df[~known_mask].copy().reset_index(drop=True)
    gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
    ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")
    if base_target == "O25":
        threshold_o25 = 2.5 + offset  # 1.5 / 2.5 / 3.5
        known_df["target"] = ((gh_k + ga_k) > threshold_o25).astype(int)
        return known_df, future_df, "binary", threshold_o25
    if target_mode == "BTTS":
        known_df["target"] = ((gh_k > 0) & (ga_k > 0)).astype(int)
        return known_df, future_df, "binary", None

    if base_target in STAT_OU_MODES:
        cfg = STAT_OU_MODES[base_target]
        missing = [c for c in cfg["h_cols"] + cfg["a_cols"] if c not in df.columns]
        if missing:
            raise ValueError(f"Brak kolumn dla {target_mode}: {missing}")
        side_mode = cfg.get("side_mode", "total")
        if side_mode == "home":
            stat_known = pd.to_numeric(df[cfg["h_cols"][0]], errors="coerce").notna()
        elif side_mode == "away":
            stat_known = pd.to_numeric(df[cfg["a_cols"][0]], errors="coerce").notna()
        else:
            stat_h0 = pd.to_numeric(df[cfg["h_cols"][0]], errors="coerce")
            stat_known = stat_h0.notna()
        known_mask_stat = known_mask & stat_known
        known_df = df[known_mask_stat].copy().reset_index(drop=True)
        future_df = df[~known_mask_stat].copy().reset_index(drop=True)
        stat_k = _stat_target_series(known_df, cfg)
        # Wyznacz BAZOWY próg (dla _LO/_HI dodajemy offset na koniec)
        league_override = get_league_threshold(league_name, base_target)
        if league_override is not None:
            base_threshold = league_override
        elif cfg["fixed_threshold"] is not None:
            base_threshold = cfg["fixed_threshold"]
        else:
            if len(stat_k) == 0:
                raise ValueError(f"Brak danych do obliczenia progu dla {target_mode}")
            if cfg.get("dynamic_threshold") == "balanced":
                base_threshold = choose_balanced_threshold(stat_k, step=0.5)
            else:
                mean_val = float(stat_k.mean())
                base_threshold = float(np.floor(mean_val * 2) / 2)
        threshold = base_threshold + offset
        known_df["target"] = (stat_k > threshold).astype(int)
        return known_df, future_df, "binary", threshold

    raise ValueError(f"Nieznany target_mode: {target_mode}")


STAT_OU_MODES = {
    # legacy: suma home+away
    "CRD25":   {"h_cols": ["HY"],  "a_cols": ["AY"],  "fixed_threshold": 2.5, "side_mode": "total"},
    "CORNOU":  {"h_cols": ["HC"],  "a_cols": ["AC"],  "fixed_threshold": None, "side_mode": "total"},
    "SHTOU":   {"h_cols": ["HST"], "a_cols": ["AST"], "fixed_threshold": None, "side_mode": "total"},
    "FOULSOU": {"h_cols": ["HF"],  "a_cols": ["AF"],  "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "total"},
    # nowe: over/under per gospodarz/gosc
    "HGOOU":   {"h_cols": ["FTHG"], "a_cols": ["FTAG"], "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "home"},
    "AGOOU":   {"h_cols": ["FTHG"], "a_cols": ["FTAG"], "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "away"},
    "HCRDOU":  {"h_cols": ["HY"],   "a_cols": ["AY"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "home"},
    "ACRDOU":  {"h_cols": ["HY"],   "a_cols": ["AY"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "away"},
    "HFOULOU": {"h_cols": ["HF"],   "a_cols": ["AF"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "home"},
    "AFOULOU": {"h_cols": ["HF"],   "a_cols": ["AF"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "away"},
    "HCORNOU": {"h_cols": ["HC"],   "a_cols": ["AC"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "home"},
    "ACORNOU": {"h_cols": ["HC"],   "a_cols": ["AC"],   "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "away"},
    "HSHTOU":  {"h_cols": ["HST"],  "a_cols": ["AST"],  "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "home"},
    "ASHTOU":  {"h_cols": ["HST"],  "a_cols": ["AST"],  "fixed_threshold": None, "dynamic_threshold": "balanced", "side_mode": "away"},
}


def _stat_target_series(df_in: pd.DataFrame, cfg: dict) -> pd.Series:
    h_tot = sum(pd.to_numeric(df_in[c], errors="coerce").fillna(0) for c in cfg["h_cols"])
    a_tot = sum(pd.to_numeric(df_in[c], errors="coerce").fillna(0) for c in cfg["a_cols"])
    side_mode = cfg.get("side_mode", "total")
    if side_mode == "home":
        return h_tot
    if side_mode == "away":
        return a_tot
    return h_tot + a_tot


def binary_labels(target_mode: str):
    base, _ = _strip_lohi(target_mode)
    if base == "O25":
        return "U", "O"
    if base == "BTTS":
        return "NO", "YES"
    if base in STAT_OU_MODES:
        return "U", "O"
    return "0", "1"


def is_la_liga(league_name: str) -> bool:
    return league_name == "Spain/La_Liga"


def detect_xg_start_season(df_all: pd.DataFrame):
    if "Season" not in df_all.columns or "xGH" not in df_all.columns or "xGA" not in df_all.columns:
        return None
    xg_mask = pd.to_numeric(df_all["xGH"], errors="coerce").notna() & pd.to_numeric(df_all["xGA"], errors="coerce").notna()
    if int(xg_mask.sum()) == 0:
        return None
    season_counts = df_all.loc[xg_mask, "Season"].value_counts()
    valid_seasons = sorted(season_counts[season_counts >= 50].index.tolist())
    if len(valid_seasons) == 0:
        valid_seasons = sorted(df_all.loc[xg_mask, "Season"].unique().tolist())
    if len(valid_seasons) == 0:
        return None
    return valid_seasons[0]


def get_config_grid(league_name: str, target_mode: str):
    if is_la_liga(league_name):
        if target_mode == "FTR":
            return [{"name": "laliga_ftr_child120", "params": {"min_child_samples": 120}}]
        if target_mode in {"O25", "BTTS"}:
            return [{"name": "laliga_binary_leaves15", "params": {"num_leaves": 15}}]
    return CONFIG_GRID


def filter_future_window(module, df: pd.DataFrame):
    if len(df) == 0:
        return df
    if module.DATE_COL not in df.columns:
        return df
    dt = pd.to_datetime(df[module.DATE_COL], dayfirst=True, errors="coerce")
    mask = dt >= TODAY
    if FUTURE_DAYS_AHEAD is not None:
        mask &= dt <= (TODAY + pd.Timedelta(days=FUTURE_DAYS_AHEAD))
    return df[mask].copy().reset_index(drop=True)


def filter_overdue_window(module, df: pd.DataFrame):
    if len(df) == 0:
        return df
    if module.DATE_COL not in df.columns:
        return df
    dt = pd.to_datetime(df[module.DATE_COL], dayfirst=True, errors="coerce")
    mask = dt < TODAY
    if OVERDUE_DAYS_BACK is not None:
        mask &= dt >= (TODAY - pd.Timedelta(days=OVERDUE_DAYS_BACK))
    return df[mask].copy().reset_index(drop=True)


def deduplicate_matches(module, df: pd.DataFrame):
    if len(df) == 0:
        return df
    needed = [module.DATE_COL, "HomeTeam", "AwayTeam", "Season"]
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


def add_light_bars_features(module, df: pd.DataFrame):
    if len(df) == 0:
        return df
    data = df.copy()
    if module.DATE_COL in data.columns:
        data = data.sort_values(module.DATE_COL).reset_index(drop=True)
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


def add_home_away_context_features(module, df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodatkowe cechy kontekstowe:
    - forma gospodarza liczona tylko z jego meczow u siebie,
    - forma goscia liczona tylko z jego meczow na wyjezdzie.
    Bez leakage: wszystkie rollingi sa po shift(1).
    """
    if len(df) == 0:
        return df
    data = df.copy().reset_index(drop=True)
    if module.DATE_COL in data.columns:
        data = data.sort_values(module.DATE_COL).reset_index(drop=True)
    data["_match_id_ctx"] = np.arange(len(data))

    def _to_num(col: str) -> pd.Series:
        return pd.to_numeric(data.get(col), errors="coerce")

    # HOME-only: statystyki gospodarza z jego meczow domowych
    home_long = pd.DataFrame(
        {
            "_match_id_ctx": data["_match_id_ctx"],
            "Date": data[module.DATE_COL],
            "Team": data["HomeTeam"],
            "goals_for": _to_num("FTHG"),
            "goals_against": _to_num("FTAG"),
            "yellows_for": _to_num("HY"),
            "yellows_against": _to_num("AY"),
            "fouls_for": _to_num("HF"),
            "fouls_against": _to_num("AF"),
        }
    ).sort_values(["Team", "Date", "_match_id_ctx"])

    # AWAY-only: statystyki goscia z jego meczow wyjazdowych
    away_long = pd.DataFrame(
        {
            "_match_id_ctx": data["_match_id_ctx"],
            "Date": data[module.DATE_COL],
            "Team": data["AwayTeam"],
            "goals_for": _to_num("FTAG"),
            "goals_against": _to_num("FTHG"),
            "yellows_for": _to_num("AY"),
            "yellows_against": _to_num("HY"),
            "fouls_for": _to_num("AF"),
            "fouls_against": _to_num("HF"),
        }
    ).sort_values(["Team", "Date", "_match_id_ctx"])

    metrics = ["goals_for", "goals_against", "yellows_for", "yellows_against", "fouls_for", "fouls_against"]
    windows = [3, 5, 10]

    for metric in metrics:
        shifted_h = home_long.groupby("Team")[metric].shift(1)
        shifted_a = away_long.groupby("Team")[metric].shift(1)
        for w in windows:
            roll_h = shifted_h.groupby(home_long["Team"]).rolling(w, min_periods=1).mean()
            roll_a = shifted_a.groupby(away_long["Team"]).rolling(w, min_periods=1).mean()
            home_long[f"home_ctx_{metric}_avg_last{w}"] = roll_h.reset_index(level=0, drop=True).values
            away_long[f"away_ctx_{metric}_avg_last{w}"] = roll_a.reset_index(level=0, drop=True).values

    home_cols = [c for c in home_long.columns if c.startswith("home_ctx_")]
    away_cols = [c for c in away_long.columns if c.startswith("away_ctx_")]
    home_map = home_long[["_match_id_ctx"] + home_cols].copy()
    away_map = away_long[["_match_id_ctx"] + away_cols].copy()
    data = data.merge(home_map, on="_match_id_ctx", how="left")
    data = data.merge(away_map, on="_match_id_ctx", how="left")

    # Różnice kontekstowe HOME vs AWAY dla kluczowych metryk.
    for metric in ["goals_for", "goals_against", "yellows_for", "fouls_for"]:
        for w in windows:
            h_col = f"home_ctx_{metric}_avg_last{w}"
            a_col = f"away_ctx_{metric}_avg_last{w}"
            if h_col in data.columns and a_col in data.columns:
                data[f"ctx_{metric}_diff_last{w}"] = data[h_col] - data[a_col]

    return data.drop(columns=["_match_id_ctx"], errors="ignore")


def prepare_bundle(module, n_last_seasons: int, target_mode: str, split_mode: str = "auto", use_xg: bool = True, league_name: str | None = None):
    _prefix_csv_columns(module.CSV_PATH)
    df_all = pd.read_csv(module.CSV_PATH, low_memory=False, on_bad_lines="warn")
    df_all = _coerce_numeric_columns(df_all)
    df_all = ensure_ftr_column(df_all)
    df_all = module.parse_date(df_all)
    df_all["Season"] = module.infer_season_from_date(df_all[module.DATE_COL])
    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        return None
    selected = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected)].copy().reset_index(drop=True)
    df_all = deduplicate_matches(module, df_all)

    known_df, future_df, task_kind, threshold = split_known_future_by_target(module, df_all, target_mode, league_name=league_name)
    ranking_only = is_ranking_only_csv(module.CSV_PATH)
    if ranking_only and target_mode != "FTR":
        return None

    if ranking_only:
        full = df_all.copy().sort_values(module.DATE_COL).reset_index(drop=True)
    else:
        all_df = pd.concat(
            [known_df.drop(columns=["target"], errors="ignore"), future_df],
            ignore_index=True
        ).sort_values(module.DATE_COL).reset_index(drop=True)
        all_df["Season"] = module.infer_season_from_date(all_df[module.DATE_COL])
        all_df = add_light_bars_features(module, all_df)
        all_df = add_home_away_context_features(module, all_df)

        long_df = module.build_team_match_history(all_df, use_xg=use_xg)
        long_df = module.add_advanced_rolling_features(long_df)
        full = module.merge_form_features_to_match(all_df, long_df)
        full = module.add_h2h_features(full)
        full = module.compute_elo(full)
        full = module.compute_goals_elo(full)
        full = module.compute_glicko(full)
        full = module.compute_elo_home_away(full)
        full = module.compute_goals_elo_home_away(full)
        full = module.compute_glicko_home_away(full)
        if use_xg:
            full = module.compute_xg_elo(full)
            full = module.compute_xg_elo_home_away(full)

    if task_kind == "multiclass":
        known_mask = full[module.TARGET_COL].isin(module.CLASS_TO_INT.keys())
        known_df = full[known_mask].copy().reset_index(drop=True)
        known_df["target"] = known_df[module.TARGET_COL].map(module.CLASS_TO_INT)
        if len(future_df) > 0:
            unresolved_df = full[~known_mask].copy().reset_index(drop=True)
        else:
            unresolved_df = pd.DataFrame(columns=known_df.columns)
    elif _strip_lohi(target_mode)[0] in STAT_OU_MODES:
        cfg = STAT_OU_MODES[_strip_lohi(target_mode)[0]]
        gh = pd.to_numeric(full.get("FTHG"), errors="coerce")
        ga = pd.to_numeric(full.get("FTAG"), errors="coerce")
        side_mode = cfg.get("side_mode", "total")
        if side_mode == "home":
            stat_known = pd.to_numeric(full.get(cfg["h_cols"][0]), errors="coerce").notna()
        elif side_mode == "away":
            stat_known = pd.to_numeric(full.get(cfg["a_cols"][0]), errors="coerce").notna()
        else:
            stat_known = pd.to_numeric(full.get(cfg["h_cols"][0]), errors="coerce").notna()
        known_mask = gh.notna() & ga.notna() & stat_known
        known_df = full[known_mask].copy().reset_index(drop=True)
        stat_k = _stat_target_series(known_df, cfg)
        known_df["target"] = (stat_k > threshold).astype(int)
        if len(future_df) > 0:
            unresolved_df = full[~known_mask].copy().reset_index(drop=True)
        else:
            unresolved_df = pd.DataFrame(columns=known_df.columns)
    else:
        gh = pd.to_numeric(full.get("FTHG"), errors="coerce")
        ga = pd.to_numeric(full.get("FTAG"), errors="coerce")
        known_mask = gh.notna() & ga.notna()
        known_df = full[known_mask].copy().reset_index(drop=True)
        gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
        ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")
        base_t, _ = _strip_lohi(target_mode)
        if base_t == "O25":
            known_df["target"] = ((gh_k + ga_k) > threshold).astype(int)
        else:
            known_df["target"] = ((gh_k > 0) & (ga_k > 0)).astype(int)
        if len(future_df) > 0:
            unresolved_df = full[~known_mask].copy().reset_index(drop=True)
        else:
            unresolved_df = pd.DataFrame(columns=known_df.columns)
    future_df = filter_future_window(module, unresolved_df)
    overdue_df = filter_overdue_window(module, unresolved_df)

    try:
        train_s, valid_s, test_s = module.seasonal_split(known_df["Season"].unique(), split_mode)
    except Exception:
        return None

    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()
    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        return None

    feature_cols = module.get_feature_columns(known_df)
    if ranking_only:
        ranking_tokens = ("rank", "elo", "glicko", "mu", "sigma", "ordinal", "certainty", "wl_", "ts_", "pl_", "lnet", "lm", "melo")
        feature_cols = [c for c in feature_cols if any(tok in c.lower() for tok in ranking_tokens)]
    X_train, X_valid, X_test, X_future, final_feats, dropped = module.filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )
    if len(final_feats) == 0:
        return None
    if MAX_FEATURES is not None and len(final_feats) > MAX_FEATURES:
        variances = X_train.var(axis=0, skipna=True).fillna(0.0)
        keep = variances.sort_values(ascending=False).head(MAX_FEATURES).index.tolist()
        X_train = X_train[keep]
        X_valid = X_valid[keep]
        X_test = X_test[keep]
        X_future = X_future[keep] if len(X_future) > 0 else X_future
        final_feats = keep
    if len(overdue_df) > 0:
        X_overdue = overdue_df.reindex(columns=final_feats).copy()
        X_overdue = X_overdue.apply(pd.to_numeric, errors="coerce")
        fill_values = X_train.median(numeric_only=True)
        X_overdue = X_overdue.fillna(fill_values).fillna(0.0)
    else:
        X_overdue = pd.DataFrame(columns=final_feats)

    if league_name is not None and is_la_liga(league_name):
        xg_elo_features = [c for c in final_feats if c == "xg_elo_diff"]
        if len(xg_elo_features) == 0:
            return None
        X_train = X_train[xg_elo_features]
        X_valid = X_valid[xg_elo_features]
        X_test = X_test[xg_elo_features]
        X_future = X_future[xg_elo_features] if len(X_future) > 0 else X_future
        X_overdue = X_overdue[xg_elo_features] if len(X_overdue) > 0 else X_overdue
        final_feats = xg_elo_features

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "X_future": X_future,
        "X_overdue": X_overdue,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "test_df": test_df,
        "future_df": future_df,
        "overdue_df": overdue_df,
        "features": final_feats,
        "train_seasons": "|".join(map(str, train_s)),
        "valid_seasons": "|".join(map(str, valid_s)),
        "test_seasons": "|".join(map(str, test_s)),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_future": len(future_df),
        "n_overdue": len(overdue_df),
        "n_features": len(final_feats),
        "dropped_count": len(dropped),
        "task_kind": task_kind,
        "target_mode": target_mode,
        "ou_threshold": threshold,
    }


def evaluate_model(module, bundle, cfg_name: str, cfg_override: dict):
    params = dict(module.LGBM_PARAMS)
    params.update(FAST_DEFAULT_PARAMS)
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
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
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
        "best_iter": int(best_iter),
        "valid_logloss": float(valid_logloss),
        "test_logloss": float(test_logloss),
        "valid_acc": float(valid_acc),
        "test_acc": float(test_acc),
        "valid_brier": float(valid_brier),
        "test_brier": float(test_brier),
        "cfg_name": cfg_name,
        "cfg_override": cfg_override,
    }


def choose_best(rows: list[dict]) -> dict:
    rows_sorted = sorted(rows, key=lambda r: (r["valid_logloss"], r["test_logloss"], -r["test_acc"]))
    return rows_sorted[0]


def build_calibration_5pct(predictions_df: pd.DataFrame):
    if len(predictions_df) == 0:
        return pd.DataFrame()
    if "prediction_split" not in predictions_df.columns:
        return pd.DataFrame()
    test_df = predictions_df[predictions_df["prediction_split"] == "test"].copy()
    if len(test_df) == 0:
        return pd.DataFrame()
    if "target_true" not in test_df.columns:
        return pd.DataFrame()

    def bin_and_agg(df_src: pd.DataFrame, prob_col: str, y_true: pd.Series, target_mode: str, outcome: str):
        p = pd.to_numeric(df_src.get(prob_col), errors="coerce")
        y = pd.to_numeric(y_true, errors="coerce")
        ok = p.notna() & y.notna()
        if int(ok.sum()) == 0:
            return []
        d = pd.DataFrame({
            "league": df_src.loc[ok, "league"].astype(str).values if "league" in df_src.columns else "ALL",
            "p": p.loc[ok].astype(float).values,
            "y": y.loc[ok].astype(float).values,
        })
        d["bin_idx"] = np.floor(np.clip(d["p"], 0, 0.999999) / 0.05).astype(int)
        d["bin_start"] = d["bin_idx"] * 0.05
        d["bin_end"] = d["bin_start"] + 0.05
        out_rows = []
        grouped = d.groupby(["league", "bin_idx", "bin_start", "bin_end"], as_index=False).agg(
            n=("p", "size"),
            avg_pred=("p", "mean"),
            actual_rate=("y", "mean"),
            hits=("y", "sum"),
        )
        grouped["target_mode"] = target_mode
        grouped["outcome"] = outcome
        grouped["calib_gap"] = grouped["avg_pred"] - grouped["actual_rate"]
        out_rows.extend(grouped.to_dict("records"))
        grouped_all = d.groupby(["bin_idx", "bin_start", "bin_end"], as_index=False).agg(
            n=("p", "size"),
            avg_pred=("p", "mean"),
            actual_rate=("y", "mean"),
            hits=("y", "sum"),
        )
        grouped_all["league"] = "ALL"
        grouped_all["target_mode"] = target_mode
        grouped_all["outcome"] = outcome
        grouped_all["calib_gap"] = grouped_all["avg_pred"] - grouped_all["actual_rate"]
        out_rows.extend(grouped_all.to_dict("records"))
        return out_rows

    rows = []
    all_modes = TARGET_MODES[:]
    for mode in all_modes:
        m = test_df[test_df["target_mode"] == mode].copy()
        if len(m) == 0:
            continue
        y = pd.to_numeric(m["target_true"], errors="coerce")
        base_m, _ = _strip_lohi(mode)
        if base_m == "FTR":
            mapping = [("H", "pred_H", 0), ("D", "pred_D", 1), ("A", "pred_A", 2)]
        elif base_m == "BTTS":
            mapping = [("NO", "pred_NO", 0), ("YES", "pred_YES", 1)]
        else:
            mapping = [("U", "pred_U", 0), ("O", "pred_O", 1)]
        for outcome, col, cls in mapping:
            if col not in m.columns:
                continue
            y_true = (y == cls).astype(float)
            rows.extend(bin_and_agg(m, col, y_true, mode, outcome))
    if len(rows) == 0:
        return pd.DataFrame()
    calib_df = pd.DataFrame(rows)
    calib_df = calib_df.sort_values(["target_mode", "outcome", "league", "bin_idx"]).reset_index(drop=True)
    return calib_df


def run_league(league_info: dict, target_mode: str):
    league_name = league_info["league_name"]
    gbm4_path = league_info["gbm4_path"]
    module = load_gbm4_module(gbm4_path)
    module.CSV_PATH = str(league_info["csv_path"])

    _prefix_csv_columns(module.CSV_PATH)
    df_all = pd.read_csv(module.CSV_PATH, low_memory=False, on_bad_lines="warn")
    df_all = _coerce_numeric_columns(df_all)
    df_all = ensure_ftr_column(df_all)
    df_all = module.parse_date(df_all)
    df_all["Season"] = module.infer_season_from_date(df_all[module.DATE_COL])
    seasons = sorted(df_all["Season"].unique())
    max_s = len(seasons)
    if max_s < 4:
        raise RuntimeError(f"Za mało sezonów: {max_s}")

    if TRAIN_ALL_YEARS:
        season_candidates = [max_s]
    else:
        start_n = max(4, max_s - 2)
        season_candidates = list(range(start_n, max_s + 1))
    use_xg = ("xGH" in df_all.columns and "xGA" in df_all.columns)
    if is_la_liga(league_name):
        xg_start = detect_xg_start_season(df_all)
        if xg_start is not None:
            seasons_from_xg = [s for s in seasons if s >= xg_start]
            if len(seasons_from_xg) >= 4:
                season_candidates = [len(seasons_from_xg)]
        use_xg = True

    cache = {}
    season_stage_rows = []
    for n in season_candidates:
        bundle = prepare_bundle(module, n_last_seasons=n, target_mode=target_mode, split_mode="auto", use_xg=use_xg, league_name=league_name)
        if bundle is None:
            continue
        cache[n] = bundle
        base_eval = evaluate_model(module, bundle, "baseline", {})
        season_stage_rows.append({
            "league": league_name,
            "target_mode": target_mode,
            "stage": "season_search",
            "n_seasons": n,
            "cfg_name": "baseline",
            "cfg_override": json.dumps({}, ensure_ascii=False),
            "valid_logloss": base_eval["valid_logloss"],
            "test_logloss": base_eval["test_logloss"],
            "valid_acc": base_eval["valid_acc"],
            "test_acc": base_eval["test_acc"],
            "valid_brier": base_eval["valid_brier"],
            "test_brier": base_eval["test_brier"],
            "best_iter": base_eval["best_iter"],
            "n_train": bundle["n_train"],
            "n_valid": bundle["n_valid"],
            "n_test": bundle["n_test"],
            "n_future": bundle["n_future"],
            "n_features": bundle["n_features"],
            "train_seasons": bundle["train_seasons"],
            "valid_seasons": bundle["valid_seasons"],
            "test_seasons": bundle["test_seasons"],
        })

    if len(season_stage_rows) == 0:
        raise RuntimeError("Brak udanych runów dla testu sezonów")

    season_sorted = sorted(season_stage_rows, key=lambda r: (r["valid_logloss"], r["test_logloss"]))
    top_n = [r["n_seasons"] for r in season_sorted[:1]]

    param_rows = []
    best_eval = None
    best_bundle = None
    best_n = None
    grid = get_config_grid(league_name, target_mode)
    for n in top_n:
        bundle = cache[n]
        for cfg in grid:
            ev = evaluate_model(module, bundle, cfg["name"], cfg["params"])
            row = {
                "league": league_name,
                "target_mode": target_mode,
                "stage": "param_search",
                "n_seasons": n,
                "cfg_name": cfg["name"],
                "cfg_override": json.dumps(cfg["params"], ensure_ascii=False),
                "valid_logloss": ev["valid_logloss"],
                "test_logloss": ev["test_logloss"],
                "valid_acc": ev["valid_acc"],
                "test_acc": ev["test_acc"],
                "valid_brier": ev["valid_brier"],
                "test_brier": ev["test_brier"],
                "best_iter": ev["best_iter"],
                "n_train": bundle["n_train"],
                "n_valid": bundle["n_valid"],
                "n_test": bundle["n_test"],
                "n_future": bundle["n_future"],
                "n_features": bundle["n_features"],
                "train_seasons": bundle["train_seasons"],
                "valid_seasons": bundle["valid_seasons"],
                "test_seasons": bundle["test_seasons"],
            }
            param_rows.append(row)
            if best_eval is None or (ev["valid_logloss"], ev["test_logloss"], -ev["test_acc"]) < (
                best_eval["valid_logloss"], best_eval["test_logloss"], -best_eval["test_acc"]
            ):
                best_eval = ev
                best_bundle = bundle
                best_n = n

    if best_eval is None:
        raise RuntimeError("Brak udanych runów dla testu parametrów")

    feature_list_df = pd.DataFrame({
        "league": league_name,
        "target_mode": target_mode,
        "feature_order": np.arange(1, len(best_bundle["features"]) + 1),
        "feature_name": best_bundle["features"],
    })

    test_proba = best_eval["model"].predict_proba(best_bundle["X_test"], num_iteration=best_eval["best_iter"])
    test_pred_df = best_bundle["test_df"][[c for c in [module.DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in best_bundle["test_df"].columns]].copy()
    test_pred_df["league"] = league_name
    test_pred_df["target_mode"] = target_mode
    test_pred_df["prediction_split"] = "test"
    if "target" in best_bundle["test_df"].columns:
        test_pred_df["target_true"] = best_bundle["test_df"]["target"].values
    if best_bundle["ou_threshold"] is not None:
        test_pred_df["ou_threshold"] = best_bundle["ou_threshold"]
    if best_bundle["task_kind"] == "multiclass":
        test_pred = np.argmax(test_proba, axis=1)
        test_pred_df["pred_H"] = test_proba[:, 0]
        test_pred_df["pred_D"] = test_proba[:, 1]
        test_pred_df["pred_A"] = test_proba[:, 2]
        test_pred_df["pred_class"] = [module.INT_TO_CLASS[i] for i in test_pred]
        test_pred_df["max_prob"] = np.max(test_proba, axis=1)
    else:
        neg_label, pos_label = binary_labels(target_mode)
        test_pred = (test_proba[:, 1] >= 0.5).astype(int)
        test_pred_df[f"pred_{neg_label}"] = test_proba[:, 0]
        test_pred_df[f"pred_{pos_label}"] = test_proba[:, 1]
        test_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in test_pred]
        test_pred_df["max_prob"] = np.maximum(test_proba[:, 0], test_proba[:, 1])

    future_pred_df = pd.DataFrame()
    if best_bundle["n_future"] > 0 and len(best_bundle["X_future"]) > 0:
        future_proba = best_eval["model"].predict_proba(best_bundle["X_future"], num_iteration=best_eval["best_iter"])
        if best_bundle["task_kind"] == "multiclass":
            future_pred = np.argmax(future_proba, axis=1)
        else:
            future_pred = (future_proba[:, 1] >= 0.5).astype(int)
        base_cols = [c for c in [module.DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in best_bundle["future_df"].columns]
        future_pred_df = best_bundle["future_df"][base_cols].copy()
        if best_bundle["task_kind"] == "multiclass":
            future_pred_df["pred_H"] = future_proba[:, 0]
            future_pred_df["pred_D"] = future_proba[:, 1]
            future_pred_df["pred_A"] = future_proba[:, 2]
            future_pred_df["pred_class"] = [module.INT_TO_CLASS[i] for i in future_pred]
            future_pred_df["max_prob"] = np.max(future_proba, axis=1)
        else:
            neg_label, pos_label = binary_labels(target_mode)
            future_pred_df[f"pred_{neg_label}"] = future_proba[:, 0]
            future_pred_df[f"pred_{pos_label}"] = future_proba[:, 1]
            future_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in future_pred]
            future_pred_df["max_prob"] = np.maximum(future_proba[:, 0], future_proba[:, 1])
        future_pred_df["league"] = league_name
        future_pred_df["target_mode"] = target_mode
        future_pred_df["prediction_split"] = "future"
        if best_bundle["ou_threshold"] is not None:
            future_pred_df["ou_threshold"] = best_bundle["ou_threshold"]

    overdue_pred_df = pd.DataFrame()
    if best_bundle["n_overdue"] > 0 and len(best_bundle["X_overdue"]) > 0:
        overdue_proba = best_eval["model"].predict_proba(best_bundle["X_overdue"], num_iteration=best_eval["best_iter"])
        if best_bundle["task_kind"] == "multiclass":
            overdue_pred = np.argmax(overdue_proba, axis=1)
        else:
            overdue_pred = (overdue_proba[:, 1] >= 0.5).astype(int)
        base_cols = [c for c in [module.DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in best_bundle["overdue_df"].columns]
        overdue_pred_df = best_bundle["overdue_df"][base_cols].copy()
        if best_bundle["task_kind"] == "multiclass":
            overdue_pred_df["pred_H"] = overdue_proba[:, 0]
            overdue_pred_df["pred_D"] = overdue_proba[:, 1]
            overdue_pred_df["pred_A"] = overdue_proba[:, 2]
            overdue_pred_df["pred_class"] = [module.INT_TO_CLASS[i] for i in overdue_pred]
            overdue_pred_df["max_prob"] = np.max(overdue_proba, axis=1)
        else:
            neg_label, pos_label = binary_labels(target_mode)
            overdue_pred_df[f"pred_{neg_label}"] = overdue_proba[:, 0]
            overdue_pred_df[f"pred_{pos_label}"] = overdue_proba[:, 1]
            overdue_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in overdue_pred]
            overdue_pred_df["max_prob"] = np.maximum(overdue_proba[:, 0], overdue_proba[:, 1])
        overdue_pred_df["league"] = league_name
        overdue_pred_df["target_mode"] = target_mode
        overdue_pred_df["prediction_split"] = "overdue"
        if best_bundle["ou_threshold"] is not None:
            overdue_pred_df["ou_threshold"] = best_bundle["ou_threshold"]

    best_row = {
        "league": league_name,
        "target_mode": target_mode,
        "csv_path": str(league_info["csv_path"]),
        "future_matches_detected": league_info["future_count"],
        "best_n_seasons": best_n,
        "best_cfg_name": best_eval["cfg_name"],
        "best_cfg_override": json.dumps(best_eval["cfg_override"], ensure_ascii=False),
        "valid_logloss": best_eval["valid_logloss"],
        "test_logloss": best_eval["test_logloss"],
        "valid_acc": best_eval["valid_acc"],
        "test_acc": best_eval["test_acc"],
        "valid_brier": best_eval["valid_brier"],
        "test_brier": best_eval["test_brier"],
        "best_iter": best_eval["best_iter"],
        "n_train": best_bundle["n_train"],
        "n_valid": best_bundle["n_valid"],
        "n_test": best_bundle["n_test"],
        "n_future": best_bundle["n_future"],
        "n_overdue": best_bundle["n_overdue"],
        "n_features": best_bundle["n_features"],
        "features_json": json.dumps(best_bundle["features"], ensure_ascii=False),
        "train_seasons": best_bundle["train_seasons"],
        "valid_seasons": best_bundle["valid_seasons"],
        "test_seasons": best_bundle["test_seasons"],
        "use_xg": use_xg,
        "task_kind": best_bundle["task_kind"],
        "ou_threshold": best_bundle["ou_threshold"],
    }

    return season_stage_rows, param_rows, best_row, feature_list_df, test_pred_df, future_pred_df, overdue_pred_df


def save_outputs(leagues, all_rows, best_rows, all_features, all_predictions, all_future, all_overdue, errors):
    tuning_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)
    feature_df = pd.DataFrame(all_features)
    predictions_df = pd.DataFrame(all_predictions)
    err_df = pd.DataFrame(errors)

    tuning_path = OUT_DIR / "tuning_runs.csv"
    best_path = OUT_DIR / "best_settings.csv"
    feature_path = OUT_DIR / "feature_list_full.csv"
    predictions_full_path = OUT_DIR / "predictions_full.csv"
    predictions_1row_path = OUT_DIR / "future_predictions_1row_all_targets.csv"
    overdue_predictions_path = OUT_DIR / "overdue_predictions_all.csv"
    calibration_path = OUT_DIR / "calibration_5pct_bins.csv"
    err_path = OUT_DIR / "errors.csv"
    tuning_df.to_csv(tuning_path, index=False)
    best_df.to_csv(best_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    predictions_df.to_csv(predictions_full_path, index=False)
    calibration_df = build_calibration_5pct(predictions_df)
    calibration_df.to_csv(calibration_path, index=False)
    err_df.to_csv(err_path, index=False)

    if len(all_future) > 0:
        future_all_df = pd.concat(all_future, ignore_index=True)
        if "Date" in future_all_df.columns:
            future_all_df["Date"] = pd.to_datetime(future_all_df["Date"], dayfirst=True, errors="coerce")
            future_all_df = future_all_df[
                (future_all_df["Date"] >= TODAY)
                & (future_all_df["Date"] <= (TODAY + pd.Timedelta(days=FUTURE_DAYS_AHEAD)))
            ].copy()
            future_all_df = future_all_df.sort_values(["Date", "league", "target_mode"]).reset_index(drop=True)
            future_all_df["Date"] = future_all_df["Date"].dt.strftime("%d/%m/%Y")
        future_all_df.to_csv(OUT_DIR / "future_predictions_all.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "future_predictions_all.csv", index=False)

    if len(all_overdue) > 0:
        overdue_all_df = pd.concat(all_overdue, ignore_index=True)
        if "Date" in overdue_all_df.columns:
            overdue_all_df["Date"] = pd.to_datetime(overdue_all_df["Date"], dayfirst=True, errors="coerce")
            overdue_all_df = overdue_all_df.sort_values(["Date", "league", "target_mode"]).reset_index(drop=True)
            overdue_all_df["Date"] = overdue_all_df["Date"].dt.strftime("%d/%m/%Y")
        overdue_all_df.to_csv(overdue_predictions_path, index=False)
    else:
        pd.DataFrame().to_csv(overdue_predictions_path, index=False)

    if len(predictions_df) > 0:
        fut = predictions_df[predictions_df.get("prediction_split", "") == "future"].copy()
        if len(fut) > 0:
            if "Date" in fut.columns:
                fut["Date"] = pd.to_datetime(fut["Date"], dayfirst=True, errors="coerce")
                fut = fut[
                    (fut["Date"] >= TODAY)
                    & (fut["Date"] <= (TODAY + pd.Timedelta(days=FUTURE_DAYS_AHEAD)))
                ].copy()
            keys = [k for k in ["Date", "league", "Season", "HomeTeam", "AwayTeam"] if k in fut.columns]
            chunks = []
            all_output_modes = TARGET_MODES[:]
            for mode in all_output_modes:
                m = fut[fut["target_mode"] == mode].copy()
                if len(m) == 0:
                    continue
                base_m, _ = _strip_lohi(mode)
                cols = []
                if base_m == "FTR":
                    cols = [c for c in ["pred_H", "pred_D", "pred_A", "pred_class", "max_prob"] if c in m.columns]
                    rename = {c: f"FTR_{c}" for c in cols}
                elif base_m == "BTTS":
                    cols = [c for c in ["pred_NO", "pred_YES", "pred_class", "max_prob"] if c in m.columns]
                    rename = {c: f"BTTS_{c}" for c in cols}
                else:
                    cols = [c for c in ["pred_U", "pred_O", "pred_class", "max_prob"] if c in m.columns]
                    extra_cols = ["ou_threshold"] if "ou_threshold" in m.columns else []
                    cols = cols + extra_cols
                    rename = {c: f"{mode}_{c}" for c in cols}
                m = m[keys + cols].rename(columns=rename)
                chunks.append(m)
            if len(chunks) > 0:
                merged = reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), chunks)
                if len(best_df) > 0 and "league" in best_df.columns and "target_mode" in best_df.columns:
                    metrics = best_df[["league", "target_mode", "test_acc", "test_logloss", "valid_acc", "valid_logloss"]].copy()
                    for mode in all_output_modes:
                        mm = metrics[metrics["target_mode"] == mode].copy()
                        if len(mm) == 0:
                            continue
                        mm = mm.drop(columns=["target_mode"]).rename(columns={
                            "test_acc": f"{mode}_test_acc",
                            "test_logloss": f"{mode}_test_logloss",
                            "valid_acc": f"{mode}_valid_acc",
                            "valid_logloss": f"{mode}_valid_logloss",
                        })
                        merged = merged.merge(mm, on="league", how="left")
                if "Date" in merged.columns:
                    merged = merged.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
                    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
                merged.to_csv(predictions_1row_path, index=False)
            else:
                pd.DataFrame().to_csv(predictions_1row_path, index=False)
        else:
            pd.DataFrame().to_csv(predictions_1row_path, index=False)
    else:
        pd.DataFrame().to_csv(predictions_1row_path, index=False)

    summary = {
        "n_leagues_with_future": len(leagues),
        "n_success": len(best_rows),
        "n_failed": len(errors),
        "tuning_runs_path": str(tuning_path),
        "best_settings_path": str(best_path),
        "feature_list_full_path": str(feature_path),
        "predictions_full_path": str(predictions_full_path),
        "predictions_1row_all_targets_path": str(predictions_1row_path),
        "calibration_5pct_bins_path": str(calibration_path),
        "future_predictions_path": str(OUT_DIR / "future_predictions_all.csv"),
        "overdue_predictions_path": str(overdue_predictions_path),
        "errors_path": str(err_path),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    leagues = detect_forced_leagues() if len(FORCED_RUNS) > 0 else detect_future_leagues()
    if MAX_LEAGUES is not None:
        leagues = leagues[:MAX_LEAGUES]
    print(f"Znalezione ligi z przyszlymi meczami: {len(leagues)}")
    for l in leagues:
        print(f" - {l['league_name']} (future={l['future_count']})")

    all_rows = []
    best_rows = []
    all_features = []
    all_predictions = []
    all_future = []
    all_overdue = []
    errors = []

    for i, league in enumerate(leagues, 1):
        name = league["league_name"]
        print(f"\n[{i}/{len(leagues)}] {name}")
        target_modes = get_target_modes_for_csv(league["csv_path"])
        for target_mode in target_modes:
            print(f"  target={target_mode}")
            try:
                season_rows, param_rows, best_row, feature_df, test_pred_df, future_df, overdue_df = run_league(league, target_mode)
                all_rows.extend(season_rows)
                all_rows.extend(param_rows)
                best_rows.append(best_row)
                if len(feature_df) > 0:
                    all_features.extend(feature_df.to_dict("records"))
                if len(test_pred_df) > 0:
                    all_predictions.extend(test_pred_df.to_dict("records"))
                if len(future_df) > 0:
                    all_predictions.extend(future_df.to_dict("records"))
                    all_future.append(future_df)
                if len(overdue_df) > 0:
                    all_predictions.extend(overdue_df.to_dict("records"))
                    all_overdue.append(overdue_df)
                print(
                    f"    OK: n={best_row['best_n_seasons']} cfg={best_row['best_cfg_name']} "
                    f"v_ll={best_row['valid_logloss']:.4f} t_ll={best_row['test_logloss']:.4f} "
                    f"t_acc={best_row['test_acc']:.3f}"
                )
            except Exception as e:
                err = {
                    "league": name,
                    "target_mode": target_mode,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(f"    BLAD: {e}")
                if not getattr(detect_future_leagues, "_traceback_printed", False):
                    traceback.print_exc()
                    detect_future_leagues._traceback_printed = True
            save_outputs(leagues, all_rows, best_rows, all_features, all_predictions, all_future, all_overdue, errors)
    summary = save_outputs(leagues, all_rows, best_rows, all_features, all_predictions, all_future, all_overdue, errors)

    print("\nPodsumowanie:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
