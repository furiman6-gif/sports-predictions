import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Lokalizuj root dynamicznie (TENISS_FINAL/build_extended_from_jeff.py -> root)
    root = Path(__file__).resolve().parent.parent
    base_path = root / "TENISS_FINAL" / "stats" / "csv" / "final_z_statami.csv"
    out_path = root / "TENISS_FINAL" / "stats" / "csv" / "final_z_statami_extended.csv"

    base = pd.read_csv(base_path, low_memory=False)
    base_cols = list(base.columns)

    files = sorted(glob.glob(str(root / "jeff" / "tennis_atp-master" / "atp_matches_*.csv")))
    use_files = []
    for f in files:
        b = os.path.basename(f)
        m = re.match(r"atp_matches_(\d{4})\.csv$", b)
        if not m:
            continue
        y = int(m.group(1))
        if 1968 <= y <= 1999:
            use_files.append(f)

    if not use_files:
        print("  Pomijam: brak plikow atp_matches_YYYY.csv (1968-1999) w jeff/ (gitignore).")
        return

    old = pd.concat([pd.read_csv(f, low_memory=False) for f in use_files], ignore_index=True)

    mapped = pd.DataFrame()
    mapped["ATP"] = np.nan
    mapped["Location"] = old.get("tourney_name")
    mapped["Tournament"] = old.get("tourney_name")
    mapped["Date"] = pd.to_datetime(old.get("tourney_date").astype(str), format="%Y%m%d", errors="coerce")
    mapped["Series"] = old.get("tourney_level")
    mapped["Court"] = np.nan
    mapped["Surface"] = old.get("surface")
    mapped["Round"] = old.get("round")
    mapped["Best of"] = pd.to_numeric(old.get("best_of"), errors="coerce")
    mapped["Winner"] = old.get("winner_name")
    mapped["Loser"] = old.get("loser_name")
    mapped["WRank"] = pd.to_numeric(old.get("winner_rank"), errors="coerce")
    mapped["LRank"] = pd.to_numeric(old.get("loser_rank"), errors="coerce")
    mapped["WPts"] = pd.to_numeric(old.get("winner_rank_points"), errors="coerce")
    mapped["LPts"] = pd.to_numeric(old.get("loser_rank_points"), errors="coerce")

    for c in ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5", "Wsets", "Lsets"]:
        mapped[c] = np.nan

    set_pattern = re.compile(r"^(\d+)-(\d+)")
    score_series = old.get("score").fillna("").astype(str)
    for i, s in enumerate(score_series.tolist()):
        wsets = 0
        lsets = 0
        set_idx = 0
        for tok in s.replace(",", " ").split():
            mt = set_pattern.match(tok.strip())
            if not mt:
                continue
            if set_idx >= 5:
                break
            w = float(mt.group(1))
            l = float(mt.group(2))
            mapped.at[i, f"W{set_idx + 1}"] = w
            mapped.at[i, f"L{set_idx + 1}"] = l
            if w > l:
                wsets += 1
            elif l > w:
                lsets += 1
            set_idx += 1
        if set_idx > 0:
            mapped.at[i, "Wsets"] = wsets
            mapped.at[i, "Lsets"] = lsets

    score_upper = score_series.str.upper()
    mapped["Comment"] = np.where(
        score_upper.str.contains("RET|W/O|DEF|ABN|WALKOVER", regex=True),
        "Retired",
        None,
    )
    mapped["_season"] = mapped["Date"].dt.year

    for c in base_cols:
        if c not in mapped.columns:
            mapped[c] = np.nan

    mapped = mapped[base_cols]
    mapped = mapped.dropna(subset=["Date", "Winner", "Loser"])
    mapped_pre2000 = mapped[mapped["Date"].dt.year < 2000].copy()

    base_cur = base.copy()
    base_cur["Date"] = pd.to_datetime(base_cur["Date"], errors="coerce")

    combined = pd.concat([mapped_pre2000, base_cur], ignore_index=True)
    combined = combined.dropna(subset=["Date", "Winner", "Loser"])
    combined = combined.sort_values("Date").drop_duplicates(
        subset=["Date", "Winner", "Loser", "Tournament"], keep="last"
    )

    combined.to_csv(out_path, index=False)

    y = pd.to_datetime(combined["Date"], errors="coerce").dt.year
    print("saved:", out_path)
    print("rows:", len(combined))
    print("year_min:", int(y.min()), "year_max:", int(y.max()))
    print("rows_lt_2000:", int((y < 2000).sum()))


if __name__ == "__main__":
    main()
