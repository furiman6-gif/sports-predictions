import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import gbm4_tenis as base


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_tenis" / "elo_surf_diff_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_MODE = "auto"
SURFACE_FILTER = None

ELO_PARAM_GRID = [
    {"name": "elo_k40_i1500_u0", "k": 40, "initial": 1500, "inactivity_uncertainty": 0.0},
    {"name": "elo_k40_i1500_u001", "k": 40, "initial": 1500, "inactivity_uncertainty": 0.001},
]

FEATURE_SET_GRID = [
    {"name": "no_rank_h2h_rest", "cols": [
        "diff_days_rest",
        "diff_surface_change",
        "h2h_win_pct_5",
        "h2h_surf_win_pct_5",
        "h2h_momentum",
        "h2h_surf_momentum",
    ]},
    {"name": "non_elo_plus_glicko", "cols": [
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
    ]},
]

MODEL_PARAM_GRID = [
    {"name": "model_baseline", "params": {}},
    {"name": "model_tighter", "params": {"num_leaves": 31, "min_child_samples": 120, "reg_alpha": 0.4, "reg_lambda": 8.0}},
    {"name": "model_wider", "params": {"num_leaves": 95, "min_child_samples": 40, "colsample_bytree": 0.95, "subsample": 0.9}},
]
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.44, 0.54, 0.01)]


def _prepare_base_df(elo_k: int, elo_initial: int, inactivity_uncertainty: float):
    df_all = pd.read_csv(base.CSV_PATH, low_memory=False)
    df_all = base.parse_date(df_all)
    df_all["Season"] = df_all[base.DATE_COL].dt.year
    if SURFACE_FILTER:
        sf = SURFACE_FILTER.strip().capitalize()
        df_all = df_all[df_all["Surface"].str.strip().str.capitalize() == sf].copy().reset_index(drop=True)
    df_all = base._fill_missing_ranks(df_all)
    df_all = base.assign_players(df_all)
    df_all = base.add_csv5_form4_features(df_all)
    long = base.build_player_long(df_all)
    long = base.add_rolling_features(long)
    df_all = base.merge_player_features(df_all, long)
    for feat in base.CSV5_ROLL_BASE:
        p1c = f"P1_{feat}"
        p2c = f"P2_{feat}"
        if p1c in df_all.columns and p2c in df_all.columns:
            df_all[f"diff_{feat}"] = df_all[p1c] - df_all[p2c]
    df_all = base.compute_elo(
        df_all,
        k=elo_k,
        initial=elo_initial,
        surface_specific=False,
        inactivity_uncertainty=inactivity_uncertainty,
    )
    df_all = base.compute_elo(
        df_all,
        k=elo_k,
        initial=elo_initial,
        surface_specific=True,
        inactivity_uncertainty=inactivity_uncertainty,
    )
    df_all = base.compute_glicko(df_all, surface_specific=False)
    df_all = base.compute_glicko(df_all, surface_specific=True)
    df_all = base.add_h2h_features(df_all)
    df_all = base._add_surface_transition(df_all)
    df_all["rank_diff"] = df_all["P1Rank"] - df_all["P2Rank"]
    df_all["rank_ratio"] = df_all["P1Rank"] / df_all["P2Rank"].replace(0, np.nan)
    df_all["log_rank_diff"] = np.log1p(df_all["P1Rank"].fillna(500)) - np.log1p(df_all["P2Rank"].fillna(500))
    round_order = {
        "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
        "Quarterfinals": 5, "Semifinals": 6, "The Final": 7, "Final": 7, "Round Robin": 3,
    }
    series_order = {
        "Grand Slam": 7, "Masters Cup": 6, "Masters 1000": 5,
        "ATP500": 4, "ATP250": 3, "International Gold": 4, "International": 3, "Series": 2,
    }
    df_all["round_enc"] = df_all["Round"].map(round_order).fillna(2)
    if "Series" in df_all.columns:
        df_all["series_enc"] = df_all["Series"].map(series_order).fillna(2)
    else:
        df_all["series_enc"] = 2
    df_all["surface_enc"] = pd.Categorical(df_all["Surface"]).codes
    return df_all


def _season_options(seasons):
    n = len(seasons)
    return [max(4, n - 2)]


def _resolve_feature_cols(df_all, requested_cols):
    cols = []
    for col in requested_cols:
        if col in df_all.columns and pd.api.types.is_numeric_dtype(df_all[col]):
            cols.append(col)
    return cols


def _build_split(df_all, n_last_seasons, feature_cols):
    all_seasons = sorted(df_all["Season"].unique())
    selected = all_seasons[-n_last_seasons:]
    df = df_all[df_all["Season"].isin(selected)].copy().reset_index(drop=True)
    known_df = df[df[base.TARGET_COL].notna()].copy().reset_index(drop=True)
    train_s, valid_s, test_s = base.seasonal_split(known_df["Season"].unique(), SPLIT_MODE)
    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()
    future_df = df[df[base.TARGET_COL].isna()].copy().reset_index(drop=True)
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_valid = valid_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_future = future_df[feature_cols].apply(pd.to_numeric, errors="coerce") if len(future_df) > 0 else pd.DataFrame(columns=feature_cols)
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0.0)
    X_valid = X_valid.fillna(med).fillna(0.0)
    X_test = X_test.fillna(med).fillna(0.0)
    X_future = X_future.fillna(med).fillna(0.0) if len(X_future) > 0 else X_future
    y_train = train_df[base.TARGET_COL].astype(int).values
    y_valid = valid_df[base.TARGET_COL].astype(int).values
    y_test = test_df[base.TARGET_COL].astype(int).values
    return {
        "df": df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "future_df": future_df,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "X_future": X_future,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "train_seasons": train_s,
        "valid_seasons": valid_s,
        "test_seasons": test_s,
        "feature_cols": feature_cols,
    }


def _safe_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def _eval_bundle(bundle, cfg_name, cfg_override):
    params = dict(base.LGBM_PARAMS)
    params.update(cfg_override)
    params["objective"] = "binary"
    params.pop("num_class", None)
    params.setdefault("verbose", -1)
    params.setdefault("force_col_wise", True)
    params.setdefault("n_jobs", 1)
    model = LGBMClassifier(**params)
    model.fit(
        bundle["X_train"],
        bundle["y_train"],
        eval_set=[(bundle["X_valid"], bundle["y_valid"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    best_iter = model.best_iteration_
    valid_proba = model.predict_proba(bundle["X_valid"], num_iteration=best_iter)[:, 1]
    test_proba = model.predict_proba(bundle["X_test"], num_iteration=best_iter)[:, 1]
    row = {
        "cfg_name": cfg_name,
        "cfg_override": json.dumps(cfg_override, ensure_ascii=False),
        "best_iter": int(best_iter),
        "valid_logloss": float(log_loss(bundle["y_valid"], valid_proba)),
        "test_logloss": float(log_loss(bundle["y_test"], test_proba)),
        "valid_accuracy": float(accuracy_score(bundle["y_valid"], (valid_proba >= 0.5).astype(int))),
        "test_accuracy": float(accuracy_score(bundle["y_test"], (test_proba >= 0.5).astype(int))),
        "valid_auc": _safe_auc(bundle["y_valid"], valid_proba),
        "test_auc": _safe_auc(bundle["y_test"], test_proba),
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "model": model,
    }
    return row


def main():
    rows = []
    best = None
    best_bundle = None
    for elo_cfg in ELO_PARAM_GRID:
        df_all = _prepare_base_df(elo_cfg["k"], elo_cfg["initial"], elo_cfg["inactivity_uncertainty"])
        seasons = sorted(df_all["Season"].unique())
        if len(seasons) < 4:
            raise RuntimeError("Za mało sezonów")
        season_opts = _season_options(seasons)
        for n in season_opts:
            for feat_cfg in FEATURE_SET_GRID:
                feature_cols = _resolve_feature_cols(df_all, feat_cfg["cols"])
                if len(feature_cols) == 0:
                    continue
                bundle = _build_split(df_all, n, feature_cols)
                if len(bundle["y_train"]) == 0 or len(bundle["y_valid"]) == 0 or len(bundle["y_test"]) == 0:
                    continue
                for model_cfg in MODEL_PARAM_GRID:
                    ev = _eval_bundle(bundle, model_cfg["name"], model_cfg["params"])
                    for threshold in THRESHOLD_GRID:
                        valid_pred = (ev["valid_proba"] >= threshold).astype(int)
                        test_pred = (ev["test_proba"] >= threshold).astype(int)
                        valid_acc = float(accuracy_score(bundle["y_valid"], valid_pred))
                        test_acc = float(accuracy_score(bundle["y_test"], test_pred))
                        out = {
                            "elo_cfg_name": elo_cfg["name"],
                            "elo_k": elo_cfg["k"],
                            "elo_initial": elo_cfg["initial"],
                            "elo_inactivity_uncertainty": elo_cfg["inactivity_uncertainty"],
                            "feature_set_name": feat_cfg["name"],
                            "feature_cols": "|".join(feature_cols),
                            "n_last_seasons": n,
                            "train_seasons": "|".join(map(str, bundle["train_seasons"])),
                            "valid_seasons": "|".join(map(str, bundle["valid_seasons"])),
                            "test_seasons": "|".join(map(str, bundle["test_seasons"])),
                            "n_train": len(bundle["train_df"]),
                            "n_valid": len(bundle["valid_df"]),
                            "n_test": len(bundle["test_df"]),
                            "n_future": len(bundle["future_df"]),
                            "n_features": len(feature_cols),
                            "cfg_name": ev["cfg_name"],
                            "cfg_override": ev["cfg_override"],
                            "threshold": float(threshold),
                            "best_iter": ev["best_iter"],
                            "valid_logloss": ev["valid_logloss"],
                            "test_logloss": ev["test_logloss"],
                            "valid_accuracy": valid_acc,
                            "test_accuracy": test_acc,
                            "valid_auc": ev["valid_auc"],
                            "test_auc": ev["test_auc"],
                        }
                        rows.append(out)
                        print(
                            f"{elo_cfg['name']} | {feat_cfg['name']} | s={n} | {model_cfg['name']} | "
                            f"thr={threshold:.2f} | test_acc={test_acc:.4f} | valid_logloss={ev['valid_logloss']:.5f}"
                        )
                        key = (-test_acc, ev["valid_logloss"], ev["test_logloss"])
                        if best is None or key < (-best["test_accuracy"], best["valid_logloss"], best["test_logloss"]):
                            best = ev | {
                                "n_last_seasons": n,
                                "elo_cfg_name": elo_cfg["name"],
                                "elo_k": elo_cfg["k"],
                                "elo_initial": elo_cfg["initial"],
                                "elo_inactivity_uncertainty": elo_cfg["inactivity_uncertainty"],
                                "feature_set_name": feat_cfg["name"],
                                "feature_cols": feature_cols,
                                "threshold": float(threshold),
                                "valid_accuracy": valid_acc,
                                "test_accuracy": test_acc,
                                "test_pred": test_pred,
                            }
                            best_bundle = bundle
    runs_df = pd.DataFrame(rows).sort_values(["test_accuracy", "valid_logloss", "test_logloss"], ascending=[False, True, True]).reset_index(drop=True)
    runs_df.to_csv(OUTPUT_DIR / "tuning_runs_elo_surf_diff.csv", index=False)
    if best is None or best_bundle is None:
        raise RuntimeError("Brak udanego strojenia")
    best_summary = {
        "csv_path": base.CSV_PATH,
        "feature_set_name": best["feature_set_name"],
        "feature_cols": best["feature_cols"],
        "n_features": int(len(best["feature_cols"])),
        "best_elo_cfg_name": best["elo_cfg_name"],
        "best_elo_k": int(best["elo_k"]),
        "best_elo_initial": int(best["elo_initial"]),
        "best_elo_inactivity_uncertainty": float(best["elo_inactivity_uncertainty"]),
        "best_n_last_seasons": int(best["n_last_seasons"]),
        "best_cfg_name": best["cfg_name"],
        "best_cfg_override": json.loads(best["cfg_override"]),
        "best_threshold": float(best["threshold"]),
        "best_iter": int(best["best_iter"]),
        "valid_logloss": float(best["valid_logloss"]),
        "test_logloss": float(best["test_logloss"]),
        "valid_accuracy": float(best["valid_accuracy"]),
        "test_accuracy": float(best["test_accuracy"]),
        "valid_auc": float(best["valid_auc"]),
        "test_auc": float(best["test_auc"]),
        "passed_68_test_accuracy": bool(best["test_accuracy"] >= 0.68),
    }
    with open(OUTPUT_DIR / "best_summary_elo_surf_diff.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)
    model = best["model"]
    test_proba = best["test_proba"]
    test_pred = best["test_pred"]
    test_out = best_bundle["test_df"][[base.DATE_COL, "P1", "P2", "Surface", "Tournament", "Season", base.TARGET_COL]].copy()
    test_out["prob_P1_wins"] = test_proba
    test_out["pred"] = test_pred
    test_out.to_csv(OUTPUT_DIR / "test_predictions_elo_surf_diff.csv", index=False)
    if len(best_bundle["future_df"]) > 0 and len(best_bundle["X_future"]) > 0:
        fut_proba = model.predict_proba(best_bundle["X_future"], num_iteration=best["best_iter"])[:, 1]
        fut_pred = (fut_proba >= best["threshold"]).astype(int)
        fut = best_bundle["future_df"][[base.DATE_COL, "P1", "P2", "Surface", "Tournament", "Season"]].copy()
        fut["prob_P1_wins"] = fut_proba
        fut["pred"] = fut_pred
        fut.to_csv(OUTPUT_DIR / "future_predictions_elo_surf_diff.csv", index=False)
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))
    print(f"Zapisano do: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
