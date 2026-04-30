from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

STATS_DIR = Path(__file__).resolve().parent
ROOT_DIR = STATS_DIR.parent
# stats/ musi być przed root/ żeby tennis_scraper2 z stats/ miał priorytet
for _p in [str(ROOT_DIR), str(STATS_DIR)]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(STATS_DIR))

from augment_charting_stats import ChartingWorkbookAugmenter
from tennis_scraper2 import update_excel_from_current_tournaments


LATEST_BASE_URL = "http://www.tennis-data.co.uk/2026/2026.xlsx"


def prune_project_columns(frame: pd.DataFrame, min_non_null_ratio: float = 0.5) -> pd.DataFrame:
    import re

    charting_columns = [
        column
        for column in frame.columns
        if column.startswith("winner_charting_") or column.startswith("loser_charting_")
    ]

    # BP columns are player-specific (e.g. winner_charting_key_points_ptsw_pct_rf_bp_faced).
    # Collapse all per-player BP columns into a single column per prefix+type.
    bp_patterns = {
        "winner_charting_bp_saved_pct": ("winner_charting_", "ptsw_pct", "bp_faced"),
        "winner_charting_bp_converted_pct": ("winner_charting_", "ptsw_pct", "bp_opps"),
        "loser_charting_bp_saved_pct": ("loser_charting_", "ptsw_pct", "bp_faced"),
        "loser_charting_bp_converted_pct": ("loser_charting_", "ptsw_pct", "bp_opps"),
    }

    collapsed: dict[str, pd.Series] = {}
    bp_columns: set[str] = set()
    for out_col, (prefix, metric, situation) in bp_patterns.items():
        matching = [
            c for c in charting_columns
            if c.startswith(prefix) and metric in c and situation in c
        ]
        bp_columns.update(matching)
        if matching:
            collapsed[out_col] = frame[matching].bfill(axis=1).iloc[:, 0]

    regular_columns = [column for column in charting_columns if column not in bp_columns]
    keep_columns = [
        column
        for column in regular_columns
        if frame[column].notna().mean() >= min_non_null_ratio
    ]

    base_columns = [column for column in frame.columns if column not in charting_columns]
    result = frame[base_columns + keep_columns].copy()
    for col_name, series in collapsed.items():
        result[col_name] = series.values
    return result


def download_latest_base_file(stats_dir: Path) -> Path:
    target_path = stats_dir / "2026.xlsx"
    request = urllib.request.Request(LATEST_BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        target_path.write_bytes(response.read())
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    stats_dir = Path(__file__).resolve().parent
    project_csv_dir = stats_dir / "csv" / "wersja_1_projektowa"
    csv_root_dir = stats_dir / "csv"

    project_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_root_dir.mkdir(parents=True, exist_ok=True)

    excel_path = stats_dir / "2026.xlsx"
    if not args.skip_download:
        excel_path = download_latest_base_file(stats_dir)
        print(excel_path)
    if not args.skip_live:
        update_excel_from_current_tournaments(excel_path, include_challengers=False)

    augmenter = ChartingWorkbookAugmenter(stats_dir=stats_dir)
    input_files = list(stats_dir.glob("20*.xlsx")) + list(stats_dir.glob("20*.xls"))
    input_files = sorted({path.resolve() for path in input_files}, key=lambda path: int(path.stem))
    missing_files: list[Path] = []
    for workbook_path in input_files:
        season_name = workbook_path.stem
        project_path = project_csv_dir / f"{season_name}_charting_project.csv"
        if not project_path.exists():
            missing_files.append(workbook_path)

    if missing_files:
        missing_names = augmenter._collect_player_names(missing_files)
        augmenter._warm_player_cache(missing_names)

    consolidated_project: list[pd.DataFrame] = []
    consolidated_mapping: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []

    final_project_csv = csv_root_dir / "final_2_projekt.csv"
    pruned_frames: list[pd.DataFrame] = []

    for workbook_path in input_files:
        season_name = workbook_path.stem
        season_frame = pd.read_excel(workbook_path, sheet_name=season_name)
        project_path = project_csv_dir / f"{season_name}_charting_project.csv"
        mapping_path = project_csv_dir / f"{season_name}_charting_mapping.csv"
        if project_path.exists() and mapping_path.exists():
            project_players_with_season = pd.read_csv(project_path, low_memory=False)
            mapping_frame_with_season = pd.read_csv(mapping_path, low_memory=False)
            project_players = project_players_with_season.drop(columns=["season"], errors="ignore")
        else:
            season_names = augmenter._collect_player_names_for_workbook(workbook_path)
            project_players = augmenter._build_player_stats_frame(season_names, mode="project")
            mapping_frame = augmenter._build_mapping_frame(season_names)
            project_players_with_season = project_players.assign(season=season_name)
            mapping_frame_with_season = mapping_frame.assign(season=season_name)
            project_players_with_season.to_csv(
                project_path,
                index=False,
                encoding="utf-8-sig",
            )
            mapping_frame_with_season.to_csv(
                mapping_path,
                index=False,
                encoding="utf-8-sig",
            )

        final_frame = augmenter._merge_match_frame_from_stats_frame(
            season_frame,
            project_players,
            left_col="Winner",
            right_col="Loser",
            left_prefix="winner",
            right_prefix="loser",
        )
        final_frames.append(final_frame)
        consolidated_project.append(project_players_with_season)
        consolidated_mapping.append(mapping_frame_with_season)

        pruned = prune_project_columns(final_frame)
        pruned_frames.append(pruned)
        print(f"  ✔ Sezon {season_name} przetworzony ({len(pruned)} meczów)")

    pd.concat(consolidated_project, ignore_index=True).to_csv(
        csv_root_dir / "all_seasons_charting_project.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(consolidated_mapping, ignore_index=True).to_csv(
        csv_root_dir / "all_seasons_charting_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(csv_root_dir / "all_seasons_charting_project.csv")
    print(csv_root_dir / "all_seasons_charting_mapping.csv")

    final_project = pd.concat(pruned_frames, ignore_index=True)
    final_project.to_csv(final_project_csv, index=False, encoding="utf-8-sig")
    print(final_project_csv)
    print(final_project_csv)


if __name__ == "__main__":
    main()
