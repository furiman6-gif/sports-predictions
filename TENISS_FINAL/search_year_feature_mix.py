import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

import gbm4_tenis as g


OUT_DIR = Path(g.OUTPUT_BASE_DIR) / "tenis_ALL"
RESULTS_PATH = OUT_DIR / "year_feature_mix_search.csv"
BEST_PATH = OUT_DIR / "year_feature_mix_best.csv"

RANDOM_SEED = 42
N_TRIES = 100
THRESHOLDS = np.round(np.arange(0.55, 0.76, 0.01), 2)


def profile_features() -> dict[str, set[str]]:
    base = set(g.BEST_FEATURE_SET)
    core = {c for c in base if not c.startswith("diff_roll_") and "odds" not in c and "implied_prob" not in c and "norm_prob" not in c}
    rolling = {c for c in base if c.startswith("diff_roll_")}
    odds = {c for c in base if ("odds" in c) or ("implied_prob" in c) or ("norm_prob" in c)}
    return {
        "full": base,
        "core_only": core,
        "rolling_only": rolling,
        "core_plus_rolling": core | rolling,
        "core_plus_odds": core | odds,
        "rolling_plus_odds": rolling | odds,
        "no_odds": base - odds,
    }


def build_feature_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = g.parse_date(df_raw)
    df["Season"] = df[g.DATE_COL].dt.year
    # Stare rekordy miewaja puste Surface; rolling per (player, Surface) wymaga pelnej osi.
    df["Surface"] = df.get("Surface").fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df = g._fill_missing_ranks(df)
    df = g.assign_players(df)
    if g.USE_ODDS_FEATURES:
        df = g.add_odds_features(df)
    df = g.add_csv5_form4_features(df)

    print("[search] building heavy features once...")
    long = g.build_player_long(df)
    long = g.add_rolling_features(long)
    df = g.merge_player_features(df, long)
    df = g.compute_elo(
        df,
        k=g.BEST_ELO_K,
        initial=g.BEST_ELO_INITIAL,
        surface_specific=False,
        inactivity_uncertainty=g.BEST_ELO_INACTIVITY_UNCERTAINTY,
    )
    df = g.compute_elo(
        df,
        k=g.BEST_ELO_K,
        initial=g.BEST_ELO_INITIAL,
        surface_specific=True,
        inactivity_uncertainty=g.BEST_ELO_INACTIVITY_UNCERTAINTY,
    )
    df = g.compute_glicko(df, surface_specific=False)
    df = g.compute_glicko(df, surface_specific=True)
    df = g.add_h2h_features(df)
    df = g._add_surface_transition(df)
    return df


def sample_configs(rng: random.Random, profiles: list[str], seasons_all: list[int]) -> list[dict]:
    cfgs = []
    for _ in range(N_TRIES):
        n_last = rng.choice([16, 20, 24, 27, 30, 35, 40, 50, 60])
        min_year = rng.choice([1968, 1975, 1980, 1985, 1990, 1995, 2000])
        cfgs.append(
            {
                "profile": rng.choice(profiles),
                "n_last": n_last,
                "min_year": min_year,
                "split_mode": "auto",
            }
        )
    return cfgs


def evaluate_one(
    df_feat: pd.DataFrame,
    allowed_features: set[str],
    n_last: int,
    min_year: int,
    split_mode: str,
) -> dict | None:
    df = df_feat.copy()
    df = df[df["Season"] >= min_year].copy()
    known = df[df[g.TARGET_COL].notna()].copy()
    if known.empty:
        return None

    seasons = sorted(known["Season"].unique())
    if len(seasons) < 6:
        return None
    if n_last > len(seasons):
        n_last = len(seasons)
    selected = seasons[-n_last:]
    known = known[known["Season"].isin(selected)].copy()
    if known[g.TARGET_COL].nunique() < 2:
        return None

    train_s, valid_s, test_s = g.seasonal_split(known["Season"].unique(), split_mode)
    train_df = known[known["Season"].isin(train_s)].copy()
    valid_df = known[known["Season"].isin(valid_s)].copy()
    test_df = known[known["Season"].isin(test_s)].copy()
    if min(len(train_df), len(valid_df), len(test_df)) < 200:
        return None

    # Feature list from numeric cols + profile allowlist.
    feats = []
    base_feats = g.get_feature_columns(known)
    for c in base_feats:
        if c in allowed_features:
            feats.append(c)
    if len(feats) < 5:
        return None

    dummy_future = pd.DataFrame(columns=train_df.columns)
    X_train, X_valid, X_test, _, final_feats, _ = g.filter_features(train_df, valid_df, test_df, dummy_future, feats)
    if len(final_feats) < 5:
        return None

    y_train = train_df[g.TARGET_COL].astype(int).values
    y_valid = valid_df[g.TARGET_COL].astype(int).values
    y_test = test_df[g.TARGET_COL].astype(int).values

    # Fast train params for search.
    p = dict(g.LGBM_PARAMS)
    p["n_estimators"] = 700
    p["learning_rate"] = 0.03
    model = g.LGBMClassifier(**p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    bi = model.best_iteration_
    valid_p = model.predict_proba(X_valid, num_iteration=bi)[:, 1]
    test_p = model.predict_proba(X_test, num_iteration=bi)[:, 1]

    # Calibrate (same rule as main file).
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(valid_p, y_valid)
    test_p_cal = cal.predict(test_p)
    ece_raw = g.expected_calibration_error(g.calibration_table_binary(y_test, test_p))
    ece_cal = g.expected_calibration_error(g.calibration_table_binary(y_test, test_p_cal))
    if ece_raw <= ece_cal:
        test_use = test_p
        calib = "raw"
    else:
        test_use = test_p_cal
        calib = "isotonic"

    # Build pred_df for value / bankroll reports.
    pred_df = test_df[[g.DATE_COL, "P1", "P2", g.TARGET_COL, "P1_odds_mean", "P2_odds_mean"]].copy()
    pred_df["prob_P1_wins"] = test_use
    pred_df["prob_P2_wins"] = 1.0 - test_use
    pred_df = g.add_value_columns(pred_df)
    val_rep = g.build_value_report(pred_df, thresholds=THRESHOLDS).dropna(subset=["roi_pct"])
    bank_rep = g.build_bankroll_report(pred_df, thresholds=THRESHOLDS).dropna(subset=["roi_on_bankroll_pct"])
    if len(val_rep) == 0 or len(bank_rep) == 0:
        return None

    best_val = val_rep.sort_values("roi_pct", ascending=False).iloc[0]
    best_bank = bank_rep.sort_values("roi_on_bankroll_pct", ascending=False).iloc[0]
    return {
        "profile": "",
        "n_last": n_last,
        "min_year": min_year,
        "train_seasons": f"{train_s[0]}-{train_s[-1]}",
        "valid_seasons": f"{valid_s[0]}-{valid_s[-1]}",
        "test_seasons": f"{test_s[0]}-{test_s[-1]}",
        "n_features": len(final_feats),
        "calib_used": calib,
        "test_logloss": round(float(log_loss(y_test, test_use)), 5),
        "test_ece": round(float(g.expected_calibration_error(g.calibration_table_binary(y_test, test_use))), 5),
        "best_value_threshold": float(best_val["threshold"]),
        "best_value_roi_pct": float(best_val["roi_pct"]),
        "best_value_bets": int(best_val["bets"]),
        "best_bank_threshold": float(best_bank["threshold"]),
        "best_bank_roi_pct": float(best_bank["roi_on_bankroll_pct"]),
        "best_bank_dd_pct": float(best_bank["max_drawdown_pct"]),
        "best_bank_final": float(best_bank["final_bankroll"]),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    profiles = profile_features()

    df_raw = pd.read_csv(g.CSV_PATH, low_memory=False)
    df_feat = build_feature_table(df_raw)
    seasons_all = sorted(df_feat[g.DATE_COL].dt.year.dropna().astype(int).unique().tolist())
    cfgs = sample_configs(rng, list(profiles.keys()), seasons_all)
    print(f"[search] sampled configs: {len(cfgs)}")

    rows = []
    for i, c in enumerate(cfgs, start=1):
        print(f"[search] {i}/{len(cfgs)} profile={c['profile']} n_last={c['n_last']} min_year={c['min_year']}")
        res = evaluate_one(
            df_feat=df_feat,
            allowed_features=profiles[c["profile"]],
            n_last=c["n_last"],
            min_year=c["min_year"],
            split_mode=c["split_mode"],
        )
        if res is None:
            continue
        res["profile"] = c["profile"]
        rows.append(res)

    rep = pd.DataFrame(rows)
    if len(rep) == 0:
        raise RuntimeError("Brak wynikow w searchu.")
    rep = rep.sort_values(["best_bank_roi_pct", "best_value_roi_pct"], ascending=[False, False]).reset_index(drop=True)
    rep.to_csv(RESULTS_PATH, index=False)

    best_bank = rep.iloc[0].copy()
    best_value = rep.sort_values("best_value_roi_pct", ascending=False).iloc[0].copy()
    best = pd.DataFrame(
        [
            {"objective": "bankroll", **best_bank.to_dict()},
            {"objective": "flat_value", **best_value.to_dict()},
        ]
    )
    best.to_csv(BEST_PATH, index=False)

    print("\n[search] saved:")
    print(RESULTS_PATH)
    print(BEST_PATH)
    print("\n[search] top 5:")
    print(rep.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
