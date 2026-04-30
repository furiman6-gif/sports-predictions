from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from augment_charting_stats import ChartingWorkbookAugmenter
from tennis_scraper2 import update_excel_from_current_tournaments


LATEST_BASE_URL = "http://www.tennis-data.co.uk/2026/2026.xlsx"


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
    parser.add_argument("--prune-min-ratio", type=float, default=0.0)
    args = parser.parse_args()

    stats_dir = Path(__file__).resolve().parent
    full_csv_dir = stats_dir / "csv" / "wersja_2_pelna"
    csv_root_dir = stats_dir / "csv"

    full_csv_dir.mkdir(parents=True, exist_ok=True)
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
    player_names = augmenter._collect_player_names(input_files)
    augmenter._warm_player_cache(player_names)

    consolidated_full_long: list[pd.DataFrame] = []
    consolidated_full_wide: list[pd.DataFrame] = []
    consolidated_mapping: list[pd.DataFrame] = []
    final_full_csv = csv_root_dir / "final_3_pelny.csv"
    if final_full_csv.exists():
        final_full_csv.unlink()

    stats_columns_union: list[str] = []
    stats_columns_seen: set[str] = set()

    base_columns_union: list[str] = []
    base_columns_seen: set[str] = set()

    for workbook_path in input_files:
        season_name = workbook_path.stem
        season_frame = pd.read_excel(workbook_path, sheet_name=season_name, nrows=1)
        for column in season_frame.columns.tolist():
            if column not in base_columns_seen:
                base_columns_union.append(column)
                base_columns_seen.add(column)

        season_full_wide_path = full_csv_dir / f"{season_name}_charting_full_wide.csv"
        if season_full_wide_path.exists():
            header = pd.read_csv(season_full_wide_path, nrows=0)
            for column in header.columns.tolist():
                if column not in stats_columns_seen:
                    stats_columns_union.append(column)
                    stats_columns_seen.add(column)

    always_keep = {"query_name", "matched", "player_name", "player_slug", "section", "directory_matches", "source_url"}
    stats_columns_union = [column for column in stats_columns_union if column != "season"]

    if args.prune_min_ratio and args.prune_min_ratio > 0:
        non_null_counts: dict[str, int] = {}
        total_rows = 0
        for workbook_path in input_files:
            season_name = workbook_path.stem
            season_full_wide_path = full_csv_dir / f"{season_name}_charting_full_wide.csv"
            if not season_full_wide_path.exists():
                continue
            stats_frame = pd.read_csv(season_full_wide_path, low_memory=False)
            total_rows += int(len(stats_frame))
            for column in stats_frame.columns:
                if column == "season":
                    continue
                non_null_counts[column] = non_null_counts.get(column, 0) + int(stats_frame[column].notna().sum())
        if total_rows > 0:
            stats_columns_union = [
                column
                for column in stats_columns_union
                if column in always_keep or (non_null_counts.get(column, 0) / total_rows) >= float(args.prune_min_ratio)
            ]

    for workbook_path in input_files:
        season_name = workbook_path.stem
        season_frame = pd.read_excel(workbook_path, sheet_name=season_name)

        season_full_long_path = full_csv_dir / f"{season_name}_charting_full_long.csv"
        season_full_wide_path = full_csv_dir / f"{season_name}_charting_full_wide.csv"
        season_mapping_path = full_csv_dir / f"{season_name}_charting_mapping.csv"

        if season_full_long_path.exists() and season_full_wide_path.exists() and season_mapping_path.exists():
            full_players_long_with_season = pd.read_csv(season_full_long_path, low_memory=False)
            full_players_wide_with_season = pd.read_csv(season_full_wide_path, low_memory=False)
            mapping_frame_with_season = pd.read_csv(season_mapping_path, low_memory=False)
            full_players_wide = full_players_wide_with_season.drop(columns=["season"], errors="ignore")
        else:
            workbook_player_names = augmenter._collect_player_names_for_workbook(workbook_path)
            full_players_long = augmenter._build_player_stats_long_frame(workbook_player_names)
            full_players_wide = augmenter._build_player_stats_frame(workbook_player_names, mode="full")
            mapping_frame = augmenter._build_mapping_frame(workbook_player_names)
            full_players_long_with_season = full_players_long.assign(season=season_name)
            full_players_wide_with_season = full_players_wide.assign(season=season_name)
            mapping_frame_with_season = mapping_frame.assign(season=season_name)

            full_players_long_with_season.to_csv(
                season_full_long_path,
                index=False,
                encoding="utf-8-sig",
            )
            full_players_wide_with_season.to_csv(
                season_full_wide_path,
                index=False,
                encoding="utf-8-sig",
            )
            mapping_frame_with_season.to_csv(
                season_mapping_path,
                index=False,
                encoding="utf-8-sig",
            )

        if stats_columns_union:
            keep_cols = [c for c in stats_columns_union if c in full_players_wide.columns]
            if "query_name" not in keep_cols and "query_name" in full_players_wide.columns:
                keep_cols = ["query_name"] + keep_cols
            full_players_wide = full_players_wide[keep_cols].copy()

        consolidated_full_long.append(full_players_long_with_season)
        consolidated_full_wide.append(full_players_wide_with_season)
        consolidated_mapping.append(mapping_frame_with_season)

        final_frame = augmenter._merge_match_frame_from_stats_frame(
            season_frame,
            full_players_wide,
            left_col="Winner",
            right_col="Loser",
            left_prefix="winner",
            right_prefix="loser",
        )

        expected_winner_cols = [f"winner_charting_{col}" for col in stats_columns_union if col != "query_name"]
        expected_loser_cols = [f"loser_charting_{col}" for col in stats_columns_union if col != "query_name"]
        expected_cols = base_columns_union + expected_winner_cols + expected_loser_cols
        for col in expected_cols:
            if col not in final_frame.columns:
                final_frame[col] = pd.NA
        final_frame = final_frame[expected_cols]
        final_frame.to_csv(
            final_full_csv,
            mode="a" if final_full_csv.exists() else "w",
            header=not final_full_csv.exists(),
            index=False,
            encoding="utf-8-sig",
        )

    pd.concat(consolidated_full_long, ignore_index=True).to_csv(
        csv_root_dir / "all_seasons_charting_full_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(consolidated_full_wide, ignore_index=True).to_csv(
        csv_root_dir / "all_seasons_charting_full_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(consolidated_mapping, ignore_index=True).to_csv(
        csv_root_dir / "all_seasons_charting_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(csv_root_dir / "all_seasons_charting_full_long.csv")
    print(csv_root_dir / "all_seasons_charting_full_wide.csv")
    print(csv_root_dir / "all_seasons_charting_mapping.csv")

    print(final_full_csv)


if __name__ == "__main__":
    main()
