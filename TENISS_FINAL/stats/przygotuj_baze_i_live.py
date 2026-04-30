from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from tennis_scraper2 import update_excel_from_current_tournaments

LATEST_BASE_URL = "http://www.tennis-data.co.uk/2026/2026.xlsx"


def download_latest_base_file(stats_dir: Path) -> Path:
    target_path = stats_dir / "2026.xlsx"
    request = urllib.request.Request(LATEST_BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        target_path.write_bytes(response.read())
    return target_path


def build_final_base_csv(stats_dir: Path, output_path: Path) -> Path:
    season_frames: list[pd.DataFrame] = []
    input_files = list(stats_dir.glob("20*.xlsx")) + list(stats_dir.glob("20*.xls"))
    input_files = sorted({path.resolve() for path in input_files}, key=lambda path: int(path.stem))
    for workbook_path in input_files:
        season_name = workbook_path.stem
        frame = pd.read_excel(workbook_path, sheet_name=season_name)
        season_frames.append(frame)
    final_frame = pd.concat(season_frames, ignore_index=True)
    final_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    stats_dir = Path(__file__).resolve().parent
    csv_root_dir = stats_dir / "csv"
    csv_root_dir.mkdir(parents=True, exist_ok=True)

    excel_path = stats_dir / "2026.xlsx"
    if not args.skip_download:
        excel_path = download_latest_base_file(stats_dir)
        print(excel_path)
    if not args.skip_live:
        update_excel_from_current_tournaments(excel_path, include_challengers=False)

    final_base_csv = csv_root_dir / "final_1_bazowy_live.csv"
    build_final_base_csv(stats_dir, final_base_csv)
    print(final_base_csv)


if __name__ == "__main__":
    main()
