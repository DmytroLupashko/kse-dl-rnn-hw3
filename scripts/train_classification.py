"""Train RNN classifier on Kaggle disaster tweets. Usage: python scripts/train_classification.py <config.yaml>"""

import sys

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from core.args import (
    ClassificationDataArguments,
    ClassificationModelArguments,
    ProbeArguments,
)
from core.datasets import (
    SeqDataset,
    build_bpe_tokenizer,
    build_word_tokenizer,
    make_collate_fn,
    word_tokenize,
)
from core.metrics import compute_metrics
from core.model import RNNClassifierConfig, UniversalRNNClassifier, count_params
from helpers.config import DATA_DIR, parse_config
from helpers.tracking import setup_wandb


def build_word_encoder(texts, vocab_max):
    tokenizer = build_word_tokenizer(map(word_tokenize, texts), max_size=vocab_max)

    def encode(text):
        return tokenizer.convert_tokens_to_ids(word_tokenize(text))

    return encode, tokenizer


def build_bpe_encoder(texts, vocab_size):
    tokenizer = build_bpe_tokenizer(texts, vocab_size)

    def encode(text):
        return tokenizer.encode(text.lower())

    return encode, tokenizer


def main():
    # ProbeArguments parsed only so the shared config validates; probe runs in the notebook
    model_args, data_args, _, training_args = parse_config(
        sys.argv[1],
        (
            ClassificationModelArguments,
            ClassificationDataArguments,
            ProbeArguments,
            TrainingArguments,
        ),
    )
    setup_wandb(training_args.project)

    df = pd.read_csv(DATA_DIR / "nlp-getting-started" / "train.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df.text,
        df.target,
        test_size=data_args.test_size,
        random_state=data_args.split_seed,
        shuffle=True,
        stratify=df.target,
    )

    # Vocab built on all texts, as in the lecture notebook
    if data_args.tokenizer == "word":
        encode, tokenizer = build_word_encoder(df.text, data_args.word_vocab_max)
    else:
        encode, tokenizer = build_bpe_encoder(df.text, data_args.bpe_vocab_size)

    model = UniversalRNNClassifier(
        RNNClassifierConfig(
            **vars(model_args),
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    print("Params:", count_params(model))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SeqDataset(X_train, y_train, encode),
        eval_dataset=SeqDataset(X_test, y_test, encode),
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
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
