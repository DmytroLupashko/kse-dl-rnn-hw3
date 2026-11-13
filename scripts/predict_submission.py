"""Kaggle submission from saved run: python scripts/predict_submission.py <run_dir> <name> [loc_threshold]
Writes data/submissions/<name>.csv; no threshold: argmax."""

import sys

from transformers import PreTrainedTokenizerFast

from core.model import UniversalRNNTokenClassifier
from helpers.submission import write_submission


def main():
    run_dir, name = sys.argv[1], sys.argv[2]
    loc_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
    model = UniversalRNNTokenClassifier.from_pretrained(run_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(run_dir)
    print("Submission:", write_submission(model, tokenizer, name, loc_threshold))


if __name__ == "__main__":
    main()
