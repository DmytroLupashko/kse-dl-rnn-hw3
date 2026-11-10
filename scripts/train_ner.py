"""Train RNN token classifier on ukr-synth NER. Usage: python scripts/train_ner.py <config.yaml>"""

import sys

import numpy as np
import wandb
from transformers import Trainer, TrainingArguments

from core.args import NerDataArguments, NerModelArguments
from core.datasets import (
    PAD_TOKEN,
    UNK_TOKEN,
    DataCollatorForTokenClassification,
    TextDataset,
    build_word_tokenizer,
    get_label_distribution,
    read_conll_file,
    simple_tokenizer,
    yield_tokens,
)
from core.metrics import make_ner_compute_metrics
from core.model import RNNTokenClassifierConfig, UniversalRNNTokenClassifier
from helpers.config import DATA_DIR, parse_config
from helpers.tracking import setup_wandb


def main():
    model_args, data_args, training_args = parse_config(
        sys.argv[1], (NerModelArguments, NerDataArguments, TrainingArguments)
    )
    setup_wandb(training_args.project)

    ner_dir = DATA_DIR / "ukr-synth" / "data"
    train_data = read_conll_file(ner_dir / "train.conllu.gz", nrows=data_args.train_rows)
    valid_data = read_conll_file(ner_dir / "dev.conllu.gz", nrows=data_args.valid_rows)

    tokenizer = build_word_tokenizer(
        yield_tokens(train_data + valid_data), specials=(UNK_TOKEN, PAD_TOKEN)
    )
    label_names = list(get_label_distribution(train_data).keys())
    max_length = np.quantile(
        [len(simple_tokenizer(s)) for s in train_data], data_args.max_length_quantile
    )
    ds_train = TextDataset(train_data, tokenizer, label_names, max_length)
    ds_valid = TextDataset(valid_data, tokenizer, label_names, max_length)

    model = UniversalRNNTokenClassifier(
        RNNTokenClassifierConfig(
            **vars(model_args),
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            id2label=ds_train.id2label,
            label2id=ds_train.label2id,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=DataCollatorForTokenClassification(tokenizer.pad_token_id),
        processing_class=tokenizer,
        compute_metrics=make_ner_compute_metrics(ds_train.id2label),
    )
    trainer.train()
    # Best model, tokenizer and eval_results.json go to output_dir for the notebook
    trainer.save_model()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    print("Eval:", metrics)
    # Shared wandb-core service on Colab never closes run on exit, run shows crashed
    wandb.finish()


if __name__ == "__main__":
    main()
