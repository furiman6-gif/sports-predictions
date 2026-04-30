"""
Batch runner — odpala gbm4_tenis.py dla różnych konfiguracji i zbiera wyniki.
"""
import subprocess
import sys
import pandas as pd
from pathlib import Path
from io import StringIO

CONFIGS = [
    # (seasons, surface, split)
    (26, "",      "auto"),   # wszystkie nawierzchnie, wszystkie lata
    (15, "",      "auto"),   # ostatnie 15 lat
    (10, "",      "auto"),   # ostatnie 10 lat
    (26, "Clay",  "auto"),   # mączka
    (26, "Hard",  "auto"),   # twarda
    (26, "Grass", "auto"),   # trawa
    (15, "Clay",  "auto"),
    (15, "Hard",  "auto"),
    (10, "Clay",  "auto"),
    (10, "Hard",  "auto"),
    (26, "",      "80_10_10"),
    (26, "",      "75_12_5_12_5"),
]

SCRIPT = Path(__file__).parent / "gbm4_tenis.py"
RESULTS = []

for seasons, surface, split in CONFIGS:
    label = f"S={seasons} | {surface or 'ALL':6s} | {split}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Podaj odpowiedzi na pytania interaktywne przez stdin
    answers = "\n".join([
        str(seasons),   # ile sezonów
        split if split != "auto" else "",  # tryb splitu
        "",             # ile features (wszystkie)
        surface,        # nawierzchnia
    ]) + "\n"

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=answers,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = result.stdout + result.stderr

    # Wyciągnij kluczowe metryki z outputu
    def extract(keyword, lines):
        for line in lines:
            if keyword in line:
                parts = line.split()
                return parts[-1] if parts else "?"
        return "?"

    lines = output.splitlines()
    row = {
        "Sezony": seasons,
        "Nawierzchnia": surface or "ALL",
        "Split": split,
        "test_logloss": extract("test_logloss", lines),
        "test_accuracy": extract("test_accuracy", lines),
        "test_auc": extract("test_auc", lines),
        "test_ece": extract("test_ece", lines),
        "tg_mae": extract("tg_mae", lines),
        "tg_rmse": extract("tg_rmse", lines),
        "n_train": extract("n_train", lines),
        "n_test": extract("n_test", lines),
        "best_iter": extract("best_iteration", lines),
    }
    RESULTS.append(row)
    print(f"  → logloss={row['test_logloss']} | acc={row['test_accuracy']} | auc={row['test_auc']} | tg_mae={row['tg_mae']}")

print("\n\n" + "="*80)
print("RAPORT KOŃCOWY")
print("="*80)
df = pd.DataFrame(RESULTS)
print(df.to_string(index=False))

out_path = Path(__file__).parent / "outputs_tenis" / "batch_report.csv"
out_path.parent.mkdir(exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nZapisano: {out_path}")
