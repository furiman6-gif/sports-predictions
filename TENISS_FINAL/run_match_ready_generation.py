from __future__ import annotations

from pathlib import Path

from augment_charting_stats import ChartingWorkbookAugmenter


def main() -> None:
    stats_dir = Path(__file__).resolve().parent / "stats"
    augmenter = ChartingWorkbookAugmenter(stats_dir=stats_dir)
    summary = augmenter.build_match_ready_from_existing_csv()
    print("Wygenerowano CSV match-ready:")
    for path in summary["match_ready_csv"]:
        print(path)


if __name__ == "__main__":
    main()
