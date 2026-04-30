import zipfile
import os
import csv
import io
import glob
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

# Mapping: CSV code -> (country_folder, league_folder)
LEAGUE_MAP = {
    "E0":  ("England", "Premier_League"),
    "E1":  ("England", "Championship"),
    "E2":  ("England", "League_One"),
    "E3":  ("England", "League_Two"),
    "EC":  ("England", "Conference"),
    "D1":  ("Germany", "Bundesliga_1"),
    "D2":  ("Germany", "Bundesliga_2"),
    "SP1": ("Spain",   "La_Liga"),
    "SP2": ("Spain",   "Segunda_Division"),
    "I1":  ("Italy",   "Serie_A"),
    "I2":  ("Italy",   "Serie_B"),
    "F1":  ("France",  "Ligue_1"),
    "F2":  ("France",  "Ligue_2"),
    "SC0": ("Scotland","Premiership"),
    "SC1": ("Scotland","Championship"),
    "SC2": ("Scotland","League_One"),
    "SC3": ("Scotland","League_Two"),
    "B1":  ("Belgium", "First_Division_A"),
    "N1":  ("Netherlands", "Eredivisie"),
    "P1":  ("Portugal",    "Primeira_Liga"),
    "G1":  ("Greece",      "Super_League"),
    "T1":  ("Turkey",      "Super_Lig"),
}

def season_label(zip_num: str) -> str:
    """Convert zip number (e.g. '00', '94') to season string like '0001' or '9495'."""
    n = int(zip_num)
    if n >= 94:
        start = 1900 + n
    else:
        start = 2000 + n
    end = start + 1
    return f"{str(start)[-2:]}{str(end)[-2:]}"

def normalize_date(date_str: str) -> str:
    """Normalize date to DD/MM/YYYY. Handles DD/MM/YY and DD/MM/YYYY."""
    if not date_str or not date_str.strip():
        return date_str
    s = date_str.strip()
    parts = s.split('/')
    if len(parts) != 3:
        return s
    dd, mm, yy = parts
    if len(yy) == 2:
        # 93-99 -> 1993-1999, 00-92 -> 2000-2092
        y = int(yy)
        full_year = 1900 + y if y >= 93 else 2000 + y
        return f"{dd}/{mm}/{full_year}"
    return s  # already DD/MM/YYYY


def date_sort_key(date_str: str):
    """Return a sortable key from a normalized DD/MM/YYYY date."""
    if not date_str or not date_str.strip():
        return datetime.min
    s = date_str.strip()
    parts = s.split('/')
    if len(parts) != 3:
        return datetime.min
    dd, mm, yyyy = parts
    try:
        return datetime(int(yyyy), int(mm), int(dd))
    except (ValueError, OverflowError):
        return datetime.min


def get_zip_paths():
    return sorted(BASE_DIR.glob("*.zip"))

def parse_csv(zf, name):
    """Read a CSV from a zipfile, return (normalized_header, data_rows)."""
    with zf.open(name) as f:
        content = f.read().decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    header = None
    data_rows = []
    for row in reader:
        if any(cell.strip() for cell in row):
            if header is None:
                header = [c.strip().lstrip("\ufeff") for c in row]
            else:
                data_rows.append(row)
    return header, data_rows

def main():
    # Pass 1: collect all data
    # league_data: {(country, league): [(season, header, data_rows), ...]}
    league_data = {}

    for zip_path in get_zip_paths():
        zip_num = zip_path.stem
        season = season_label(zip_num)
        print(f"Processing {zip_path.name} -> season {season}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                code = name[:-4]
                if code not in LEAGUE_MAP:
                    continue
                country, league = LEAGUE_MAP[code]
                header, data_rows = parse_csv(zf, name)
                if not header:
                    continue
                # Normalize Date column in-place
                if 'Date' in header:
                    date_idx = header.index('Date')
                    for row in data_rows:
                        if date_idx < len(row):
                            row[date_idx] = normalize_date(row[date_idx])
                key = (country, league)
                if key not in league_data:
                    league_data[key] = []
                league_data[key].append((season, header, data_rows))

    # Build GLOBAL union of all column names across all leagues and seasons
    global_seen = {}
    for seasons in league_data.values():
        for _, header, _ in seasons:
            for col in header:
                if col and col not in global_seen:
                    global_seen[col] = True
    global_cols = list(global_seen.keys())
    print(f"\nGlobalna unia kolumn: {len(global_cols)} kolumn\n")

    # Write output
    for (country, league), seasons in league_data.items():
        league_dir = BASE_DIR / country / league
        seasons_dir = league_dir / "sezony"
        league_dir.mkdir(parents=True, exist_ok=True)
        seasons_dir.mkdir(parents=True, exist_ok=True)

        seasons.sort(key=lambda x: x[0])

        # Write individual season files (original columns, no padding)
        for season, header, rows in seasons:
            season_file = seasons_dir / f"sezon_{season}.csv"
            with open(season_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

        # Write merged file with GLOBAL column set, sorted chronologically
        merged_file = league_dir / "wszystkie_sezony.csv"
        all_merged_rows = []
        date_col_idx = global_cols.index('Date') if 'Date' in global_cols else None
        for season, header, rows in seasons:
            for row in rows:
                row_dict = {col: (row[i] if i < len(row) else "") for i, col in enumerate(header)}
                out = [row_dict.get(col, "") for col in global_cols]
                all_merged_rows.append([season] + out)
        if date_col_idx is not None:
            # +1 because 'Sezon' is prepended
            all_merged_rows.sort(key=lambda r: date_sort_key(r[date_col_idx + 1]))
        with open(merged_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sezon"] + global_cols)
            writer.writerows(all_merged_rows)

        total_rows = sum(len(r) for _, _, r in seasons)
        print(f"  -> {country}/{league}: {len(seasons)} sezonów, {total_rows} meczów")

    print("\nGotowe!")

if __name__ == "__main__":
    main()
