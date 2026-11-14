"""Average tag probs of saved runs: python scripts/ensemble.py <name> <group> <group> ...
Group: comma-separated run dirs. Ensemble = mean over groups of mean within group.
Prints LOC F1 on uk_geo valid per run, per group and for ensemble;
writes data/submissions/<name>.csv from ensemble.
Valid probs cached per run in <run_dir>/valid_word_probs.pt, refreshed when model is newer."""

import sys
from pathlib import Path

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


def valid_probs(run_dir, member, valid):
    """Word probs per valid text, from cache when newer than model weights."""
    cache = Path(run_dir) / "valid_word_probs.pt"
    weights = Path(run_dir) / "model.safetensors"
    if cache.exists() and cache.stat().st_mtime > weights.stat().st_mtime:
        return torch.load(cache)
    model, tokenizer = member
    probs = [word_probs(model, tokenizer, words) for words, _ in tqdm(valid, desc=run_dir)]
    torch.save(probs, cache)
    return probs


def weighted(runs, weights):
    """Per text, weighted sum of runs' word probs."""
    return [sum(w * p for w, p in zip(weights, ps)) for ps in zip(*runs)]


def loc_f1(probs, valid, config):
    """LOC F1 of thresholded tags from word probs per text vs valid tags."""
    preds = [probs_tag_ids(p, config.label2id).numpy() for p in probs]
    labels = [np.array([config.label2id[t] for t in tags]) for _, tags in valid]
    return score_tags(preds, labels, config.id2label, entity="LOC")["f1"]


def main():
    name, groups = sys.argv[1], [g.split(",") for g in sys.argv[2:]]
    run_dirs = [d for g in groups for d in g]
    weights = [1 / (len(groups) * len(g)) for g in groups for _ in g]
    members = [load_run(d) for d in run_dirs]
    config = members[0][0].config
    valid = tag_frame(load_uk_geo(UK_GEO_PATH, train_rows=0))

    runs = dict(zip(run_dirs, (valid_probs(d, m, valid) for d, m in zip(run_dirs, members))))
    for d in run_dirs:
        print(f"{d}: LOC F1 {loc_f1(runs[d], valid, config):.4f}")
    for i, g in enumerate(groups):
        if len(g) > 1:
            group = weighted([runs[d] for d in g], [1 / len(g)] * len(g))
            print(f"group {i + 1} average: LOC F1 {loc_f1(group, valid, config):.4f}")
    ensemble = weighted([runs[d] for d in run_dirs], weights)
    print(f"ensemble: LOC F1 {loc_f1(ensemble, valid, config):.4f}")

    print("Submission:", write_submission(members, name, weights=weights))


if __name__ == "__main__":
    main()
