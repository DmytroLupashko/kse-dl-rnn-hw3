from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = ROOT / "configs"
WANDB_DIR = DATA_DIR / "wandb"
KSE_DIR = DATA_DIR / "kse-ua-location-extraction-2026"
UK_GEO_PATH = KSE_DIR / "uk_geo_dataset.csv"
RU_GEO_PATH = KSE_DIR / "ru_geo_dataset.csv"

