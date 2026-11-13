"""Average tag probs of saved runs: python scripts/ensemble.py <name> <run_dir> <run_dir> ...
Prints LOC F1 on uk_geo valid per run and for average; writes data/submissions/<name>.csv from average."""

import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from core.datasets import load_uk_geo, tag_frame
from core.metrics import score_tags
from core.model import UniversalRNNTokenClassifier
from core.predict import probs_tag_ids, word_probs
from helpers.config import UK_GEO_PATH
from helpers.submission import write_submission


def load_run(run_dir):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UniversalRNNTokenClassifier.from_pretrained(run_dir).to(device).eval()
    return model, PreTrainedTokenizerFast.from_pretrained(run_dir)


def loc_f1(probs, valid, config):
    """LOC F1 of thresholded tags from word probs per text vs valid tags."""
    preds = [probs_tag_ids(p, config.label2id).numpy() for p in probs]
    labels = [np.array([config.label2id[t] for t in tags]) for _, tags in valid]
    return score_tags(preds, labels, config.id2label, entity="LOC")["f1"]


def main():
    name, run_dirs = sys.argv[1], sys.argv[2:]
    members = [load_run(d) for d in run_dirs]
    config = members[0][0].config
    valid = tag_frame(load_uk_geo(UK_GEO_PATH, train_rows=0))

    runs = [
        [word_probs(m, t, words) for words, _ in tqdm(valid, desc=d)]
        for d, (m, t) in zip(run_dirs, members)
    ]
    for run_dir, probs in zip(run_dirs, runs):
        print(f"{run_dir}: LOC F1 {loc_f1(probs, valid, config):.4f}")
    mean = [torch.stack(ps).mean(dim=0) for ps in zip(*runs)]
    print(f"average: LOC F1 {loc_f1(mean, valid, config):.4f}")

    print("Submission:", write_submission(members, name))


if __name__ == "__main__":
    main()
