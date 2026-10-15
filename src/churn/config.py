"""Carga de la configuración YAML y rutas del proyecto."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Convierte una ruta relativa del config en absoluta respecto a la raíz."""
    return ROOT_DIR / relative
