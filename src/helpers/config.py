from pathlib import Path

import yaml
from transformers import HfArgumentParser

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = ROOT / "configs"
WANDB_DIR = DATA_DIR / "wandb"
KSE_DIR = DATA_DIR / "kse-ua-location-extraction-2026"
UK_GEO_PATH = KSE_DIR / "uk_geo_dataset.csv"
RU_GEO_PATH = KSE_DIR / "ru_geo_dataset.csv"


def parse_config(path, arg_types):
    """Fills arg_types dataclasses from YAML; output_dir is relative to ROOT, so cwd does not matter."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["output_dir"] = str(ROOT / cfg["output_dir"])
    return HfArgumentParser(arg_types).parse_dict(cfg)
