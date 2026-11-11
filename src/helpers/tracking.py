import os

from helpers.config import WANDB_DIR


def setup_wandb(project):
    """HF WandbCallback reads project and local run dir from env, not from TrainingArguments."""
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_DIR"] = str(WANDB_DIR)
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
