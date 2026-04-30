from pathlib import Path

import pandas as pd

import gbm4_tenis as g
import search_year_feature_mix as s


OUT_DIR = Path(g.OUTPUT_BASE_DIR) / "tenis_ALL"
OUT_PATH = OUT_DIR / "benchmark_1to1.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = s.profile_features()

    # Identyczne ramy czasowe i split, różnimy tylko zestaw cech.
    n_last = 59
    min_year = 1968
    split_mode = "auto"

    df_raw = pd.read_csv(g.CSV_PATH, low_memory=False)
    df_feat = s.build_feature_table(df_raw)

    cfgs = [
        ("baseline_full", profiles["full"]),
        ("candidate_core_plus_odds", profiles["core_plus_odds"]),
    ]

    rows = []
    for name, feat_set in cfgs:
        r = s.evaluate_one(
            df_feat=df_feat,
            allowed_features=feat_set,
            n_last=n_last,
            min_year=min_year,
            split_mode=split_mode,
        )
        if r is None:
            continue
        r["config"] = name
        rows.append(r)

    if not rows:
        raise RuntimeError("Brak wynikow benchmark 1:1.")

    rep = pd.DataFrame(rows)
    base = rep[rep["config"] == "baseline_full"].iloc[0]
    cand = rep[rep["config"] == "candidate_core_plus_odds"].iloc[0]

    delta = {
        "config": "delta_candidate_minus_baseline",
        "test_period": f"{base['test_seasons']}",
        "delta_bank_roi_pct": cand["best_bank_roi_pct"] - base["best_bank_roi_pct"],
        "delta_value_roi_pct": cand["best_value_roi_pct"] - base["best_value_roi_pct"],
        "delta_bank_dd_pct": cand["best_bank_dd_pct"] - base["best_bank_dd_pct"],
        "delta_test_logloss": cand["test_logloss"] - base["test_logloss"],
        "delta_test_ece": cand["test_ece"] - base["test_ece"],
        "baseline_bank_roi_pct": base["best_bank_roi_pct"],
        "candidate_bank_roi_pct": cand["best_bank_roi_pct"],
        "baseline_value_roi_pct": base["best_value_roi_pct"],
        "candidate_value_roi_pct": cand["best_value_roi_pct"],
    }

    rep["test_period"] = rep["test_seasons"]
    out = pd.concat([rep, pd.DataFrame([delta])], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
