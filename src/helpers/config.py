from pathlib import Path

import yaml
from transformers import TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
CONFIG_DIR = ROOT / "configs"
WANDB_DIR = DATA_DIR / "wandb"
KSE_DIR = DATA_DIR / "kse-ua-location-extraction-2026"
UK_GEO_PATH = KSE_DIR / "uk_geo_dataset.csv"
RU_GEO_PATH = KSE_DIR / "ru_geo_dataset.csv"
TEST_PATH = KSE_DIR / "test.csv"
FASTTEXT_PATH = DATA_DIR / "fasttext" / "cc.uk.300.vec.gz"


def parse_config(path, model_cls, data_cls):
    """YAML: top `name`, `experiment`; sections `model`, `data`, `trainer`.
    Run folder PROCESSED_DIR/<experiment>/<name>, W&B project rnn-<experiment>."""
    cfg = yaml.safe_load(Path(path).read_text())
    name, experiment = cfg["name"], cfg["experiment"]
    training_args = TrainingArguments(
        **cfg["trainer"],
        output_dir=str(PROCESSED_DIR / experiment / name),
        run_name=name,
        project=f"rnn-{experiment}".replace("_", "-"),
    )
    return model_cls(**cfg["model"]), data_cls(**cfg["data"]), training_args
