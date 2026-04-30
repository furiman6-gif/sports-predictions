from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from core.scraper_layer import TennisAbstractChartingScraper


class ChartingWorkbookAugmenter:
    def __init__(self, stats_dir: str | Path, max_workers: int = 6):
        self.stats_dir = Path(stats_dir)
        self.scraper = TennisAbstractChartingScraper()
        self.directory = self.scraper.get_player_directory()
        self.max_workers = max_workers
        self.stats_cache: dict[str, dict[str, Any] | None] = {}

    def build_outputs(self) -> dict[str, Any]:
        input_files = sorted(self.stats_dir.glob("*.xlsx"))
        if not input_files:
            raise FileNotFoundError(f"Brak plików xlsx w {self.stats_dir}")

        player_names = self._collect_player_names(input_files)
        self._warm_player_cache(player_names)

        project_dir = self.stats_dir / "wersja_1_projektowa"
        full_dir = self.stats_dir / "wersja_2_pelna"
        csv_root_dir = self.stats_dir / "csv"
        project_csv_dir = csv_root_dir / "wersja_1_projektowa"
        full_csv_dir = csv_root_dir / "wersja_2_pelna"
        project_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)
        project_csv_dir.mkdir(parents=True, exist_ok=True)
        full_csv_dir.mkdir(parents=True, exist_ok=True)

        output_summary = {
            "project": [],
            "full": [],
            "project_csv": [],
            "full_csv": [],
            "full_wide_csv": [],
            "match_ready_csv": [],
            "players_total": len(player_names),
            "players_matched": 0,
        }
        matched_names = {name for name, payload in self.stats_cache.items() if payload}
        output_summary["players_matched"] = len(matched_names)
        consolidated_project: list[pd.DataFrame] = []
        consolidated_full_long: list[pd.DataFrame] = []
        consolidated_full_wide: list[pd.DataFrame] = []
        consolidated_mapping: list[pd.DataFrame] = []
        consolidated_match_project: list[pd.DataFrame] = []
        consolidated_match_full: list[pd.DataFrame] = []
        consolidated_upcoming_project: list[pd.DataFrame] = []
        consolidated_upcoming_full: list[pd.DataFrame] = []

        for workbook_path in input_files:
            workbook = pd.ExcelFile(workbook_path)
            project_out = project_dir / workbook_path.name
            full_out = full_dir / workbook_path.name
            season_name = workbook_path.stem
            workbook_player_names = self._collect_player_names_for_workbook(workbook_path)
            project_players = self._build_player_stats_frame(workbook_player_names, mode="project")
            full_players_long = self._build_player_stats_long_frame(workbook_player_names)
            full_players_wide = self._build_player_stats_frame(workbook_player_names, mode="full").assign(season=season_name)
            mapping_frame = self._build_mapping_frame(workbook_player_names)
            project_players_with_season = project_players.assign(season=season_name)
            full_players_long_with_season = full_players_long.assign(season=season_name)
            mapping_frame_with_season = mapping_frame.assign(season=season_name)
            project_lookup = self._build_prefixed_lookup_frame(workbook_player_names, mode="project")
            full_lookup = self._build_prefixed_lookup_frame(workbook_player_names, mode="full")

            with pd.ExcelWriter(project_out, engine="openpyxl") as project_writer, pd.ExcelWriter(full_out, engine="openpyxl") as full_writer:
                for sheet_name in workbook.sheet_names:
                    frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
                    frame.to_excel(project_writer, sheet_name=sheet_name, index=False)
                    frame.to_excel(full_writer, sheet_name=sheet_name, index=False)
                project_players.to_excel(project_writer, sheet_name="Charting_Project", index=False)
                mapping_frame.to_excel(project_writer, sheet_name="Charting_Mapping", index=False)
                full_players_long.to_excel(full_writer, sheet_name="Charting_Full_Long", index=False)
                mapping_frame.to_excel(full_writer, sheet_name="Charting_Mapping", index=False)

            match_frame = pd.read_excel(workbook_path, sheet_name=season_name)
            match_project = self._merge_match_frame(match_frame, project_lookup, left_col="Winner", right_col="Loser", left_prefix="winner", right_prefix="loser").assign(season=season_name)
            match_full = self._merge_match_frame(match_frame, full_lookup, left_col="Winner", right_col="Loser", left_prefix="winner", right_prefix="loser").assign(season=season_name)
            match_project.to_csv(project_csv_dir / f"{season_name}_match_ready_project.csv", index=False, encoding="utf-8-sig")
            match_full.to_csv(full_csv_dir / f"{season_name}_match_ready_full_wide.csv", index=False, encoding="utf-8-sig")
            output_summary["match_ready_csv"].append(str(project_csv_dir / f"{season_name}_match_ready_project.csv"))
            output_summary["match_ready_csv"].append(str(full_csv_dir / f"{season_name}_match_ready_full_wide.csv"))
            consolidated_match_project.append(match_project)
            consolidated_match_full.append(match_full)

            if "Upcoming" in workbook.sheet_names:
                upcoming_frame = pd.read_excel(workbook_path, sheet_name="Upcoming")
                upcoming_project = self._merge_match_frame(upcoming_frame, project_lookup, left_col="Player1", right_col="Player2", left_prefix="player1", right_prefix="player2").assign(season=season_name)
                upcoming_full = self._merge_match_frame(upcoming_frame, full_lookup, left_col="Player1", right_col="Player2", left_prefix="player1", right_prefix="player2").assign(season=season_name)
                upcoming_project.to_csv(project_csv_dir / f"{season_name}_upcoming_project.csv", index=False, encoding="utf-8-sig")
                upcoming_full.to_csv(full_csv_dir / f"{season_name}_upcoming_full_wide.csv", index=False, encoding="utf-8-sig")
                output_summary["match_ready_csv"].append(str(project_csv_dir / f"{season_name}_upcoming_project.csv"))
                output_summary["match_ready_csv"].append(str(full_csv_dir / f"{season_name}_upcoming_full_wide.csv"))
                consolidated_upcoming_project.append(upcoming_project)
                consolidated_upcoming_full.append(upcoming_full)

            project_players_with_season.to_csv(project_csv_dir / f"{season_name}_charting_project.csv", index=False, encoding="utf-8-sig")
            mapping_frame_with_season.to_csv(project_csv_dir / f"{season_name}_charting_mapping.csv", index=False, encoding="utf-8-sig")
            full_players_long_with_season.to_csv(full_csv_dir / f"{season_name}_charting_full_long.csv", index=False, encoding="utf-8-sig")
            full_players_wide.to_csv(full_csv_dir / f"{season_name}_charting_full_wide.csv", index=False, encoding="utf-8-sig")
            mapping_frame_with_season.to_csv(full_csv_dir / f"{season_name}_charting_mapping.csv", index=False, encoding="utf-8-sig")

            output_summary["project"].append(str(project_out))
            output_summary["full"].append(str(full_out))
            output_summary["project_csv"].append(str(project_csv_dir / f"{season_name}_charting_project.csv"))
            output_summary["project_csv"].append(str(project_csv_dir / f"{season_name}_charting_mapping.csv"))
            output_summary["full_csv"].append(str(full_csv_dir / f"{season_name}_charting_full_long.csv"))
            output_summary["full_csv"].append(str(full_csv_dir / f"{season_name}_charting_mapping.csv"))
            output_summary["full_wide_csv"].append(str(full_csv_dir / f"{season_name}_charting_full_wide.csv"))
            consolidated_project.append(project_players_with_season)
            consolidated_full_long.append(full_players_long_with_season)
            consolidated_full_wide.append(full_players_wide)
            consolidated_mapping.append(mapping_frame_with_season)

        all_project = pd.concat(consolidated_project, ignore_index=True)
        all_full_long = pd.concat(consolidated_full_long, ignore_index=True)
        all_full_wide = pd.concat(consolidated_full_wide, ignore_index=True)
        all_mapping = pd.concat(consolidated_mapping, ignore_index=True)
        all_match_project = pd.concat(consolidated_match_project, ignore_index=True)
        all_match_full = pd.concat(consolidated_match_full, ignore_index=True)
        all_project.to_csv(csv_root_dir / "all_seasons_charting_project.csv", index=False, encoding="utf-8-sig")
        all_full_long.to_csv(csv_root_dir / "all_seasons_charting_full_long.csv", index=False, encoding="utf-8-sig")
        all_full_wide.to_csv(csv_root_dir / "all_seasons_charting_full_wide.csv", index=False, encoding="utf-8-sig")
        all_mapping.to_csv(csv_root_dir / "all_seasons_charting_mapping.csv", index=False, encoding="utf-8-sig")
        all_match_project.to_csv(csv_root_dir / "all_seasons_match_ready_project.csv", index=False, encoding="utf-8-sig")
        all_match_full.to_csv(csv_root_dir / "all_seasons_match_ready_full_wide.csv", index=False, encoding="utf-8-sig")
        if consolidated_upcoming_project:
            pd.concat(consolidated_upcoming_project, ignore_index=True).to_csv(csv_root_dir / "all_seasons_upcoming_project.csv", index=False, encoding="utf-8-sig")
            output_summary["match_ready_csv"].append(str(csv_root_dir / "all_seasons_upcoming_project.csv"))
        if consolidated_upcoming_full:
            pd.concat(consolidated_upcoming_full, ignore_index=True).to_csv(csv_root_dir / "all_seasons_upcoming_full_wide.csv", index=False, encoding="utf-8-sig")
            output_summary["match_ready_csv"].append(str(csv_root_dir / "all_seasons_upcoming_full_wide.csv"))
        output_summary["project_csv"].append(str(csv_root_dir / "all_seasons_charting_project.csv"))
        output_summary["full_csv"].append(str(csv_root_dir / "all_seasons_charting_full_long.csv"))
        output_summary["project_csv"].append(str(csv_root_dir / "all_seasons_charting_mapping.csv"))
        output_summary["full_csv"].append(str(csv_root_dir / "all_seasons_charting_mapping.csv"))
        output_summary["full_wide_csv"].append(str(csv_root_dir / "all_seasons_charting_full_wide.csv"))
        output_summary["match_ready_csv"].append(str(csv_root_dir / "all_seasons_match_ready_project.csv"))
        output_summary["match_ready_csv"].append(str(csv_root_dir / "all_seasons_match_ready_full_wide.csv"))

        return output_summary

    def build_match_ready_from_existing_csv(self) -> dict[str, Any]:
        input_files = sorted(self.stats_dir.glob("20*.xlsx"))
        if not input_files:
            raise FileNotFoundError(f"Brak plików xlsx w {self.stats_dir}")

        csv_root_dir = self.stats_dir / "csv"
        project_csv_dir = csv_root_dir / "wersja_1_projektowa"
        full_csv_dir = csv_root_dir / "wersja_2_pelna"

        summary = {"match_ready_csv": []}
        consolidated_match_project: list[pd.DataFrame] = []
        consolidated_match_full: list[pd.DataFrame] = []
        consolidated_upcoming_project: list[pd.DataFrame] = []
        consolidated_upcoming_full: list[pd.DataFrame] = []

        for workbook_path in input_files:
            season_name = workbook_path.stem
            workbook = pd.ExcelFile(workbook_path)

            project_stats = pd.read_csv(project_csv_dir / f"{season_name}_charting_project.csv")
            full_stats = pd.read_csv(full_csv_dir / f"{season_name}_charting_full_wide.csv")

            match_frame = pd.read_excel(workbook_path, sheet_name=season_name)
            match_project = self._merge_match_frame_from_stats_frame(
                match_frame,
                project_stats,
                left_col="Winner",
                right_col="Loser",
                left_prefix="winner",
                right_prefix="loser",
            ).assign(season=season_name)
            match_full = self._merge_match_frame_from_stats_frame(
                match_frame,
                full_stats,
                left_col="Winner",
                right_col="Loser",
                left_prefix="winner",
                right_prefix="loser",
            ).assign(season=season_name)

            project_match_path = project_csv_dir / f"{season_name}_match_ready_project.csv"
            full_match_path = full_csv_dir / f"{season_name}_match_ready_full_wide.csv"
            match_project.to_csv(project_match_path, index=False, encoding="utf-8-sig")
            match_full.to_csv(full_match_path, index=False, encoding="utf-8-sig")
            summary["match_ready_csv"].append(str(project_match_path))
            summary["match_ready_csv"].append(str(full_match_path))
            consolidated_match_project.append(match_project)
            consolidated_match_full.append(match_full)

            if "Upcoming" in workbook.sheet_names:
                upcoming_frame = pd.read_excel(workbook_path, sheet_name="Upcoming")
                upcoming_project = self._merge_match_frame_from_stats_frame(
                    upcoming_frame,
                    project_stats,
                    left_col="Player1",
                    right_col="Player2",
                    left_prefix="player1",
                    right_prefix="player2",
                ).assign(season=season_name)
                upcoming_full = self._merge_match_frame_from_stats_frame(
                    upcoming_frame,
                    full_stats,
                    left_col="Player1",
                    right_col="Player2",
                    left_prefix="player1",
                    right_prefix="player2",
                ).assign(season=season_name)

                project_upcoming_path = project_csv_dir / f"{season_name}_upcoming_project.csv"
                full_upcoming_path = full_csv_dir / f"{season_name}_upcoming_full_wide.csv"
                upcoming_project.to_csv(project_upcoming_path, index=False, encoding="utf-8-sig")
                upcoming_full.to_csv(full_upcoming_path, index=False, encoding="utf-8-sig")
                summary["match_ready_csv"].append(str(project_upcoming_path))
                summary["match_ready_csv"].append(str(full_upcoming_path))
                consolidated_upcoming_project.append(upcoming_project)
                consolidated_upcoming_full.append(upcoming_full)

        all_match_project_path = csv_root_dir / "all_seasons_match_ready_project.csv"
        all_match_full_path = csv_root_dir / "all_seasons_match_ready_full_wide.csv"
        pd.concat(consolidated_match_project, ignore_index=True).to_csv(all_match_project_path, index=False, encoding="utf-8-sig")
        pd.concat(consolidated_match_full, ignore_index=True).to_csv(all_match_full_path, index=False, encoding="utf-8-sig")
        summary["match_ready_csv"].append(str(all_match_project_path))
        summary["match_ready_csv"].append(str(all_match_full_path))

        if consolidated_upcoming_project:
            all_upcoming_project_path = csv_root_dir / "all_seasons_upcoming_project.csv"
            pd.concat(consolidated_upcoming_project, ignore_index=True).to_csv(all_upcoming_project_path, index=False, encoding="utf-8-sig")
            summary["match_ready_csv"].append(str(all_upcoming_project_path))
        if consolidated_upcoming_full:
            all_upcoming_full_path = csv_root_dir / "all_seasons_upcoming_full_wide.csv"
            pd.concat(consolidated_upcoming_full, ignore_index=True).to_csv(all_upcoming_full_path, index=False, encoding="utf-8-sig")
            summary["match_ready_csv"].append(str(all_upcoming_full_path))

        return summary

    def _collect_player_names(self, input_files: list[Path]) -> list[str]:
        names: set[str] = set()
        for workbook_path in input_files:
            if "wersja_" in workbook_path.parent.name:
                continue
            workbook = pd.ExcelFile(workbook_path)
            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
                for column_name in ("Winner", "Loser", "Player1", "Player2", "Player"):
                    if column_name in frame.columns:
                        names.update(
                            str(value).strip()
                            for value in frame[column_name].dropna().tolist()
                            if str(value).strip()
                        )
        return sorted(names)

    def _collect_player_names_for_workbook(self, workbook_path: Path) -> list[str]:
        names: set[str] = set()
        workbook = pd.ExcelFile(workbook_path)
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
            for column_name in ("Winner", "Loser", "Player1", "Player2", "Player"):
                if column_name in frame.columns:
                    names.update(
                        str(value).strip()
                        for value in frame[column_name].dropna().tolist()
                        if str(value).strip()
                    )
        return sorted(names)

    def _warm_player_cache(self, player_names: list[str]) -> None:
        unresolved: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self.scraper.scrape_player_stats, name, self.directory): name for name in player_names}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    self.stats_cache[name] = future.result()
                except Exception:
                    self.stats_cache[name] = None
                    unresolved.append(name)
        if unresolved:
            for name in unresolved:
                self.stats_cache.setdefault(name, None)

    def _build_player_stats_frame(self, player_names: list[str], mode: str) -> pd.DataFrame:
        rows = [self._stats_for_player_row(name, mode) for name in player_names]
        return pd.DataFrame(rows)

    def _build_player_stats_long_frame(self, player_names: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for name in player_names:
            payload = self.stats_cache.get(name)
            base_row = {
                "query_name": name,
                "matched": bool(payload),
                "player_name": payload.get("player_name") if payload else None,
                "player_slug": payload.get("player_slug") if payload else None,
                "section": payload.get("section") if payload else None,
                "directory_matches": payload.get("charted_matches_directory") if payload else None,
                "source_url": payload.get("source_url") if payload else None,
            }
            if not payload:
                rows.append({**base_row, "stat_key": None, "stat_value": None})
                continue
            for stat_key, stat_value in payload.get("all_stats", {}).items():
                rows.append({**base_row, "stat_key": stat_key, "stat_value": stat_value})
        return pd.DataFrame(rows)

    def _build_mapping_frame(self, player_names: list[str]) -> pd.DataFrame:
        rows = []
        for name in player_names:
            payload = self.stats_cache.get(name)
            rows.append(
                {
                    "query_name": name,
                    "matched": bool(payload),
                    "player_name": payload.get("player_name") if payload else None,
                    "player_slug": payload.get("player_slug") if payload else None,
                    "section": payload.get("section") if payload else None,
                    "directory_matches": payload.get("charted_matches_directory") if payload else None,
                    "source_url": payload.get("source_url") if payload else None,
                }
            )
        return pd.DataFrame(rows)

    def _build_prefixed_lookup_frame(self, player_names: list[str], mode: str) -> pd.DataFrame:
        frame = self._build_player_stats_frame(player_names, mode=mode).copy()
        if frame.empty:
            return frame
        frame = frame.rename(columns={"query_name": "__query_name"})
        return frame

    def _merge_match_frame(
        self,
        frame: pd.DataFrame,
        lookup: pd.DataFrame,
        left_col: str,
        right_col: str,
        left_prefix: str,
        right_prefix: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        if lookup.empty:
            return result

        left_lookup = lookup.add_prefix(f"{left_prefix}_charting_").rename(
            columns={f"{left_prefix}_charting___query_name": left_col}
        )
        right_lookup = lookup.add_prefix(f"{right_prefix}_charting_").rename(
            columns={f"{right_prefix}_charting___query_name": right_col}
        )

        result = result.merge(left_lookup, on=left_col, how="left")
        result = result.merge(right_lookup, on=right_col, how="left")
        return result

    def _merge_match_frame_from_stats_frame(
        self,
        frame: pd.DataFrame,
        stats_frame: pd.DataFrame,
        left_col: str,
        right_col: str,
        left_prefix: str,
        right_prefix: str,
    ) -> pd.DataFrame:
        lookup = stats_frame.copy()
        if lookup.empty:
            return frame.copy()
        if "query_name" not in lookup.columns:
            raise KeyError("stats_frame musi zawierać kolumnę query_name")
        lookup = lookup.rename(columns={"query_name": "__query_name"})
        return self._merge_match_frame(frame, lookup, left_col, right_col, left_prefix, right_prefix)

    def _stats_for_player_row(self, raw_name: Any, mode: str) -> dict[str, Any]:
        name = str(raw_name).strip() if pd.notna(raw_name) else ""
        payload = self.stats_cache.get(name)
        row = {
            "query_name": name,
            "matched": bool(payload),
            "player_name": payload.get("player_name") if payload else None,
            "player_slug": payload.get("player_slug") if payload else None,
            "section": payload.get("section") if payload else None,
            "directory_matches": payload.get("charted_matches_directory") if payload else None,
            "source_url": payload.get("source_url") if payload else None,
        }
        if not payload:
            return row
        stats_key = "project_stats" if mode == "project" else "all_stats"
        row.update(payload.get(stats_key, {}))
        return row


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    stats_dir = root_dir / "stats"
    augmenter = ChartingWorkbookAugmenter(stats_dir=stats_dir)
    summary = augmenter.build_outputs()
    print("Wygenerowano wersje arkuszy:")
    print("Projektowa:")
    for path in summary["project"]:
        print(path)
    print("Pelna:")
    for path in summary["full"]:
        print(path)
    print("CSV projektowa:")
    for path in summary["project_csv"]:
        print(path)
    print("CSV pelna:")
    for path in summary["full_csv"]:
        print(path)
    print("CSV pelna wide:")
    for path in summary["full_wide_csv"]:
        print(path)
    print("CSV match ready:")
    for path in summary["match_ready_csv"]:
        print(path)
    print(f"Players total: {summary['players_total']}")
    print(f"Players matched: {summary['players_matched']}")


if __name__ == "__main__":
    main()
