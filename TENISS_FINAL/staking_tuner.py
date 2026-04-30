from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs_tenis" / "tenis_ALL"
TEST_PRED_PATH = OUT_DIR / "test_predictions.csv"

THRESHOLDS = np.round(np.arange(0.55, 0.76, 0.01), 2)
KELLYS = np.round(np.arange(0.10, 0.61, 0.05), 2)
MAX_CAPS = np.round(np.arange(0.01, 0.051, 0.005), 3)
INITIAL_BANKROLL = 1000.0


def simulate_bankroll(sel: pd.DataFrame, kelly_fraction: float, max_stake_pct: float):
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_dd = 0.0
    total_staked = 0.0
    bets = 0

    for _, r in sel.iterrows():
        odds = float(r["pick_odds"])
        p = float(r["model_conf"])
        if not np.isfinite(odds) or odds <= 1.01 or not np.isfinite(p):
            continue
        b = odds - 1.0
        k_full = (p * odds - 1.0) / b
        stake_frac = max(0.0, kelly_fraction * k_full)
        stake_frac = min(stake_frac, max_stake_pct)
        stake = bankroll * stake_frac
        pnl = stake * b if int(r["pick_correct"]) == 1 else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        dd = 0.0 if peak <= 0 else (peak - bankroll) / peak
        max_dd = max(max_dd, dd)
        total_staked += stake
        bets += 1

    profit = bankroll - INITIAL_BANKROLL
    roi_bank = (profit / INITIAL_BANKROLL) * 100.0
    yld = np.nan if total_staked <= 0 else (profit / total_staked) * 100.0
    return bets, bankroll, roi_bank, yld, max_dd * 100.0


def main():
    df = pd.read_csv(TEST_PRED_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", na_position="last").reset_index(drop=True)
    df["model_conf"] = df[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)
    df["pick_correct"] = np.where(
        (df["pick_side"] == "P1") & (df["target"] == 1), 1,
        np.where((df["pick_side"] == "P2") & (df["target"] == 0), 1, 0),
    )

    rows = []
    for t in THRESHOLDS:
        sel = df[(df["bet_flag"] == 1) & (df["model_conf"] >= t)].copy()
        if len(sel) == 0:
            continue
        hit = float(sel["pick_correct"].mean() * 100.0)
        flat_profit = np.where(sel["pick_correct"] == 1, sel["pick_odds"] - 1.0, -1.0)
        flat_roi = float(np.mean(flat_profit) * 100.0)

        for k in KELLYS:
            for cap in MAX_CAPS:
                bets, final_bank, roi_bank, yld, dd = simulate_bankroll(sel, float(k), float(cap))
                rows.append(
                    {
                        "threshold": float(t),
                        "kelly_fraction": float(k),
                        "max_stake_pct": float(cap),
                        "bets": int(bets),
                        "hit_rate_pct": round(hit, 2),
                        "flat_roi_pct": round(flat_roi, 2),
                        "final_bankroll": round(final_bank, 2),
                        "roi_on_bankroll_pct": round(roi_bank, 2),
                        "yield_on_staked_pct": round(float(yld), 2) if pd.notna(yld) else np.nan,
                        "max_drawdown_pct": round(dd, 2),
                    }
                )

    res = pd.DataFrame(rows)
    res["score_stable"] = res["roi_on_bankroll_pct"] - 1.2 * np.maximum(0.0, res["max_drawdown_pct"] - 20.0)
    res["score_balanced"] = res["roi_on_bankroll_pct"] - 0.6 * res["max_drawdown_pct"]
    res = res.sort_values(["roi_on_bankroll_pct", "max_drawdown_pct"], ascending=[False, True]).reset_index(drop=True)

    grid_path = OUT_DIR / "staking_grid_search.csv"
    best_path = OUT_DIR / "staking_best_settings.csv"
    res.to_csv(grid_path, index=False)

    aggressive = res.iloc[0]
    stable_pool = res[(res["max_drawdown_pct"] <= 25.0) & (res["bets"] >= 80)]
    stable = (
        stable_pool.sort_values(["score_stable", "roi_on_bankroll_pct"], ascending=[False, False]).iloc[0]
        if len(stable_pool) > 0
        else res.sort_values(["score_stable", "roi_on_bankroll_pct"], ascending=[False, False]).iloc[0]
    )
    balanced_pool = res[(res["max_drawdown_pct"] <= 35.0) & (res["bets"] >= 100)]
    balanced = (
        balanced_pool.sort_values(["score_balanced", "roi_on_bankroll_pct"], ascending=[False, False]).iloc[0]
        if len(balanced_pool) > 0
        else res.sort_values(["score_balanced", "roi_on_bankroll_pct"], ascending=[False, False]).iloc[0]
    )

    best = pd.DataFrame(
        [
            {"profile": "aggressive", **aggressive.to_dict()},
            {"profile": "balanced", **balanced.to_dict()},
            {"profile": "stable", **stable.to_dict()},
        ]
    )
    best.to_csv(best_path, index=False)

    print("Saved:")
    print(grid_path)
    print(best_path)
    print("\nBest profiles:")
    print(
        best[
            [
                "profile",
                "threshold",
                "kelly_fraction",
                "max_stake_pct",
                "bets",
                "roi_on_bankroll_pct",
                "max_drawdown_pct",
                "final_bankroll",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
