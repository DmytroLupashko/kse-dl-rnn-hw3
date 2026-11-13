"""LOC F1 on uk_geo valid for a grid of LOC thresholds, per trained run.
Usage: python scripts/tune_loc_threshold.py <out.csv> <run_dir>...
Run dir: Trainer output_dir with model and tokenizer. Valid encoded as in training (max_length 128)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from core.args import DataArguments
from core.datasets import (
    DataCollatorForTokenClassification,
    GeoNERDataset,
    load_uk_geo,
    tag_frame,
)
from core.metrics import score_tags
from core.model import UniversalRNNTokenClassifier
from core.predict import tag_ids
from helpers.config import UK_GEO_PATH

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def main():
    out, run_dirs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    valid = tag_frame(load_uk_geo(UK_GEO_PATH, train_rows=0))
    rows = []
    for run_dir in run_dirs:
        model = UniversalRNNTokenClassifier.from_pretrained(run_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(run_dir)
        ds = GeoNERDataset(valid, tokenizer, DataArguments.max_length)
        with tempfile.TemporaryDirectory() as tmp:
            # Batched like Trainer eval in training, so LOC_THRESHOLD row matches eval_results.json
            trainer = Trainer(
                model=model,
                args=TrainingArguments(tmp, per_device_eval_batch_size=64, report_to="none"),
                data_collator=DataCollatorForTokenClassification(tokenizer.pad_token_id),
            )
            pred = trainer.predict(ds)
        for t in THRESHOLDS:
            preds = tag_ids(pred.predictions, model.config.label2id, t).numpy()
            scores = score_tags(preds, pred.label_ids, model.config.id2label, "LOC")
            rows.append({"run": run_dir.name, "threshold": t, **scores})
            print(run_dir.name, t, round(scores["f1"], 4), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
