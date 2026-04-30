"""
Naprawia wszystkie CSV w ligi/ ktore maja niedopasowane liczby kolumn
(trailing commas, Windows line endings, ragged rows).

Strategia:
1. Czyta naglowek
2. Strip trailing empty columns z naglowka
3. Truncate kazdy wiersz do len(header) kolumn (lub pad pustymi)
4. Zapisuje z LF line endings, bez quoting jesli niepotrzebne
"""
import csv
import sys
from pathlib import Path


def fix_csv(path: Path) -> bool:
    """Zwraca True jesli plik zmieniony."""
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"  BLAD odczytu {path}: {e}")
        return False

    if not rows:
        return False

    header = rows[0]
    # Strip trailing empty columns z naglowka
    while header and header[-1].strip() == "":
        header.pop()
    n_cols = len(header)
    if n_cols == 0:
        return False

    # Sprawdz czy potrzebna naprawa
    needs_fix = False
    for r in rows[1:]:
        if len(r) != n_cols:
            needs_fix = True
            break

    if not needs_fix:
        return False

    # Naprawiaj: truncate lub pad
    fixed_rows = [header]
    for r in rows[1:]:
        if len(r) > n_cols:
            fixed_rows.append(r[:n_cols])
        elif len(r) < n_cols:
            fixed_rows.append(r + [""] * (n_cols - len(r)))
        else:
            fixed_rows.append(r)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)

    return True


def main():
    root = Path("ligi")
    if not root.exists():
        print(f"Brak katalogu {root}")
        return

    csv_files = list(root.rglob("*.csv"))
    print(f"Sprawdzam {len(csv_files)} plikow CSV w {root}/...")

    fixed = 0
    for path in csv_files:
        if fix_csv(path):
            fixed += 1
            print(f"  Naprawiono: {path}")

    print(f"\nNaprawiono {fixed} z {len(csv_files)} plikow.")


if __name__ == "__main__":
    main()
