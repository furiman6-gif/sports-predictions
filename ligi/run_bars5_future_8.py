from pathlib import Path
from collections import deque
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).parent
BARS5_SCRIPT = ROOT / "bars5.py"
OUT_DIR = ROOT / "bars5_future_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "by_country").mkdir(parents=True, exist_ok=True)
MAX_LEAGUES = 8
PROGRESS_FILE = OUT_DIR / "progress_all.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_merged_csv(league_dir: Path) -> Path | None:
    candidates = [
        league_dir / "merged_data.csv",
        league_dir / "merged_data_ALL_features.csv",
        league_dir / "wszystkie_sezony.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def count_future_matches(csv_path: Path) -> int:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return 0
    if "Date" not in df.columns:
        return 0
    dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    today = pd.Timestamp.today().normalize()
    if "FTR" in df.columns:
        known_mask = df["FTR"].isin(["H", "D", "A"])
    else:
        gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
        ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
        known_mask = gh.notna() & ga.notna()
    future_mask = (dt >= today) & (~known_mask)
    return int(future_mask.sum())


def detect_future_leagues(limit: int) -> list[dict]:
    rows = []
    league_paths = list(ROOT.glob("*/*/gbm4.py"))
    log(f"Skanuje ligi: {len(league_paths)}")
    for idx, gbm4_path in enumerate(league_paths, start=1):
        if idx == 1 or idx % 50 == 0:
            log(f"  ... sprawdzone ligi: {idx}/{len(league_paths)}")
        league_dir = gbm4_path.parent
        csv_path = find_merged_csv(league_dir)
        if csv_path is None:
            continue
        future_count = count_future_matches(csv_path)
        if future_count <= 0:
            continue
        rel = league_dir.relative_to(ROOT)
        rows.append(
            {
                "league_dir": league_dir,
                "league_name": str(rel).replace("\\", "/"),
                "country": rel.parts[0],
                "csv_path": csv_path,
                "csv_name": csv_path.name,
                "csv_size": int(csv_path.stat().st_size),
                "csv_mtime_ns": int(csv_path.stat().st_mtime_ns),
                "future_count": future_count,
            }
        )
    rows.sort(key=lambda x: x["league_name"])
    return rows[:limit]


def save_progress(progress_rows: list[dict]) -> None:
    df = pd.DataFrame(progress_rows)
    df.to_csv(PROGRESS_FILE, index=False)
    if len(df) == 0:
        return
    for country, g in df.groupby("country"):
        g.to_csv(OUT_DIR / "by_country" / f"{country}.csv", index=False)


def load_progress() -> dict[str, dict]:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        df = pd.read_csv(PROGRESS_FILE)
    except Exception:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[str(row.get("league", ""))] = row.to_dict()
    return out


def run_one(league: dict) -> tuple[bool, str]:
    cmd = [sys.executable, "-u", str(BARS5_SCRIPT), league["csv_name"]]
    tail_lines = deque(maxlen=120)
    proc = subprocess.Popen(
        cmd,
        cwd=str(league["league_dir"]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log(f"    {line}")
        tail_lines.append(line)
    proc.wait()
    combined_tail = "\n".join(tail_lines)
    if proc.returncode == 0:
        return True, combined_tail[-4000:]
    return False, combined_tail[-4000:]


def main() -> None:
    log("Start run_bars5_future_8.py")
    leagues = detect_future_leagues(MAX_LEAGUES)
    log(f"Znalezione ligi z przyszlymi meczami: {len(leagues)}")
    for l in leagues:
        log(f" - {l['league_name']} ({l['csv_name']}, future={l['future_count']})")
    old = load_progress()
    progress_map = {k: v for k, v in old.items()}
    run_queue = []
    for league in leagues:
        old_row = progress_map.get(league["league_name"])
        already_done = False
        if old_row is not None and str(old_row.get("status", "")) == "ok":
            old_size = int(pd.to_numeric(old_row.get("csv_size"), errors="coerce") or -1)
            old_mtime = int(pd.to_numeric(old_row.get("csv_mtime_ns"), errors="coerce") or -1)
            if old_size == league["csv_size"] and old_mtime == league["csv_mtime_ns"]:
                already_done = True
        if already_done:
            log(f" - SKIP {league['league_name']} (bez zmian od ostatniego runu)")
            continue
        run_queue.append(league)

    if len(run_queue) == 0:
        save_progress(list(progress_map.values()))
        log("\nBrak lig do ponownego liczenia (wszystko aktualne).")
        log(str(PROGRESS_FILE))
        return

    run_queue.sort(key=lambda x: (x["country"], x["league_name"]))
    country_order = list(dict.fromkeys([x["country"] for x in run_queue]))
    total = len(run_queue)
    done = 0
    for country in country_order:
        log(f"\n=== KRAJ: {country} ===")
        country_leagues = [x for x in run_queue if x["country"] == country]
        for league in country_leagues:
            done += 1
            log(f"\n[{done}/{total}] {league['league_name']}")
            ok, tail = run_one(league)
            row = {
                "league": league["league_name"],
                "country": league["country"],
                "csv_name": league["csv_name"],
                "csv_size": league["csv_size"],
                "csv_mtime_ns": league["csv_mtime_ns"],
                "future_count": league["future_count"],
                "status": "ok" if ok else "error",
                "output_tail": tail,
            }
            progress_map[league["league_name"]] = row
            save_progress(list(progress_map.values()))
            if ok:
                log("  OK zapis postepu")
            else:
                log("  BLAD zapis postepu")
        save_progress(list(progress_map.values()))
        log(f"  Zapisano postep dla kraju: {country}")
    log("\nGotowe. Postep:")
    log(str(PROGRESS_FILE))
    log(str(OUT_DIR / "by_country"))


if __name__ == "__main__":
    main()
