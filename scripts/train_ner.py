"""Train RNN NER on uk_geo (+ ru_geo), score LOC F1 on uk_geo valid, write Kaggle submission.
Usage: python scripts/train_ner.py <config.yaml>"""

import sys

from transformers import Trainer

from core.args import DataArguments, ModelArguments
from core.datasets import (
    DataCollatorForTokenClassification,
    GeoNERDataset,
    build_tokenizer,
    load_geo,
    tag_frame,
)
from core.metrics import make_compute_metrics
from core.model import (
    RNNTokenClassifierConfig,
    UniversalRNNTokenClassifier,
    load_vec_embeddings,
)
from helpers.config import (
    FASTTEXT_PATH,
    RU_GEO_PATH,
    UK_GEO_PATH,
    parse_config,
)
from helpers.submission import write_submission
from helpers.tracking import setup_wandb


def main():
    model_args, data_args, training_args = parse_config(
        sys.argv[1], ModelArguments, DataArguments
    )
    if "wandb" in training_args.report_to:
        setup_wandb(training_args.project)

    geo = load_geo(
        UK_GEO_PATH,
        RU_GEO_PATH,
        train_rows=data_args.train_rows,
        ru_docs=data_args.ru_docs,
        ru_nrows=data_args.ru_nrows,
        seed=data_args.seed,
    )
    train = tag_frame(geo[geo["split"] == "train"])
    valid = tag_frame(geo[geo["split"] == "valid"])
    # Vocab from train words only: valid and test see same unknown-word rate
    tokenizer = build_tokenizer(
        data_args.tokenizer, [words for words, _ in train], data_args.vocab_size
    )
    ds_train = GeoNERDataset(train, tokenizer, data_args.max_length)
    ds_valid = GeoNERDataset(valid, tokenizer, data_args.max_length)

    model_kwargs = vars(model_args).copy()
    fasttext = model_kwargs.pop("fasttext")
    model = UniversalRNNTokenClassifier(
        RNNTokenClassifierConfig(
            **model_kwargs,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            id2label=ds_train.id2label,
            label2id=ds_train.label2id,
        )
    )
    if fasttext:
        hits = load_vec_embeddings(model.embed, FASTTEXT_PATH, tokenizer.get_vocab())
        print(f"fastText vectors: {hits} of {len(tokenizer)} tokens")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=DataCollatorForTokenClassification(tokenizer.pad_token_id),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(ds_train.id2label, entity="LOC"),
    )
    trainer.train()
    # Best model, tokenizer and eval_results.json go to output_dir
    trainer.save_model()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    print("Eval:", metrics)

    print("Submission:", write_submission([(model, tokenizer)], training_args.run_name))

    if "wandb" in training_args.report_to:
        import wandb

        # Shared wandb-core service on Colab never closes run on exit, run shows crashed
        wandb.finish()


if __name__ == "__main__":
    main()
