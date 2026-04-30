"""
Naprawia wszystkie CSV w ligi/ ktore maja niedopasowane liczby kolumn.

Strategia line-based (nie uzywa csv module - zeby uniknac problemow z quotami):
1. Czyta plik jako tekst, splituje po liniach
2. Header: split po przecinku, strip trailing \r i empty cells
3. Kazdy wiersz: split po przecinku, strip \r, truncate/pad do len(header)
4. Zapisuje z LF endings
"""
import sys
from pathlib import Path


def fix_csv(path: Path) -> tuple[bool, int, int]:
    """Zwraca (zmieniony, n_naprawionych_wierszy, n_kolumn)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  BLAD odczytu {path}: {e}")
        return False, 0, 0

    lines = text.splitlines()
    if not lines:
        return False, 0, 0

    # Header: strip \r, strip trailing empty fields
    header_line = lines[0].rstrip("\r")
    header_fields = header_line.split(",")
    while header_fields and header_fields[-1].strip() == "":
        header_fields.pop()
    n_cols = len(header_fields)
    if n_cols == 0:
        return False, 0, 0

    fixed_lines = [",".join(header_fields)]
    n_fixed = 0

    for line in lines[1:]:
        line = line.rstrip("\r")
        if not line.strip():
            continue
        fields = line.split(",")
        original_len = len(fields)
        if original_len > n_cols:
            fields = fields[:n_cols]
            n_fixed += 1
        elif original_len < n_cols:
            fields = fields + [""] * (n_cols - original_len)
            n_fixed += 1
        fixed_lines.append(",".join(fields))

    # Sprawdz czy header sie zmienil albo cokolwiek bylo naprawione
    header_changed = lines[0].rstrip("\r") != ",".join(header_fields)
    if not header_changed and n_fixed == 0:
        return False, 0, n_cols

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(fixed_lines) + "\n")

    return True, n_fixed, n_cols


def main():
    root = Path("ligi")
    if not root.exists():
        print(f"BLAD: brak katalogu {root}")
        sys.exit(1)

    csv_files = sorted(root.rglob("*.csv"))
    print(f"Sprawdzam {len(csv_files)} plikow CSV w {root}/...")

    fixed_count = 0
    for path in csv_files:
        changed, n_fixed, n_cols = fix_csv(path)
        if changed:
            fixed_count += 1
            print(f"  [{n_cols} kol] naprawiono {n_fixed} wierszy: {path}")

    print(f"\nNaprawiono {fixed_count} z {len(csv_files)} plikow.")


if __name__ == "__main__":
    main()
