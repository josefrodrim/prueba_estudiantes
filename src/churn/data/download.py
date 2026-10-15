"""Descarga el dataset Telco Customer Churn (IBM) a data/raw/."""
import urllib.request

from churn.config import load_config, resolve


def download() -> None:
    cfg = load_config()["data"]
    dest = resolve(cfg["raw_path"])
    if dest.exists():
        print(f"Dataset ya existe en {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {cfg['url']} ...")
    urllib.request.urlretrieve(cfg["url"], dest)
    print(f"Guardado en {dest}")


if __name__ == "__main__":
    download()
