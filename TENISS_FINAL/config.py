from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(root: Path) -> None:
    """Wczytuje .env jeśli istnieje (bez zewnętrznych zależności)."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    odds_api_key: str
    bankroll: float
    kelly_fraction: float
    min_edge: float
    min_agreement: float
    n_simulations: int

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "AppConfig":
        root = Path(root_dir).resolve()
        _load_dotenv(root)
        return cls(
            root_dir=root,
            data_dir=root / "data",
            models_dir=root / "models",
            logs_dir=root / "logs",
            odds_api_key=os.getenv("TACTICAL_EDGE_ODDS_API_KEY", ""),
            bankroll=float(os.getenv("TACTICAL_EDGE_BANKROLL", "2000")),
            kelly_fraction=float(os.getenv("TACTICAL_EDGE_KELLY_FRACTION", "0.25")),
            min_edge=float(os.getenv("TACTICAL_EDGE_MIN_EDGE", "0.03")),
            min_agreement=float(os.getenv("TACTICAL_EDGE_MIN_AGREEMENT", "0.60")),
            n_simulations=int(os.getenv("TACTICAL_EDGE_N_SIMS", "10000")),
        )

    def validate(self) -> list[str]:
        """Zwraca listę błędów konfiguracji. Pusta lista = OK."""
        errors: list[str] = []
        if self.bankroll <= 0:
            errors.append(f"TACTICAL_EDGE_BANKROLL musi byc > 0, jest: {self.bankroll}")
        if not (0.0 < self.kelly_fraction <= 1.0):
            errors.append(f"TACTICAL_EDGE_KELLY_FRACTION musi byc w (0, 1], jest: {self.kelly_fraction}")
        if not (0.0 <= self.min_edge < 1.0):
            errors.append(f"TACTICAL_EDGE_MIN_EDGE musi byc w [0, 1), jest: {self.min_edge}")
        if not (0.0 < self.min_agreement <= 1.0):
            errors.append(f"TACTICAL_EDGE_MIN_AGREEMENT musi byc w (0, 1], jest: {self.min_agreement}")
        if self.n_simulations < 1000:
            errors.append(f"TACTICAL_EDGE_N_SIMS musi byc >= 1000, jest: {self.n_simulations}")
        if not self.odds_api_key:
            errors.append("TACTICAL_EDGE_ODDS_API_KEY nie jest ustawiony (wymagany do pobierania kursow live)")
        return errors

    def ensure_directories(self) -> None:
        for d in (self.root_dir, self.data_dir, self.models_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self, level: int = logging.INFO) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / "tactical_edge.log"
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
            force=True,
        )
