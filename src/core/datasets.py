import ast
import collections
import re

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


# ---- Tokenizers ----


def build_word_tokenizer(token_lists, specials=(PAD_TOKEN, UNK_TOKEN), max_size=None):
    """WordLevel vocab: specials first, then most common tokens; unknown tokens map to <unk>.
    Splitting text into tokens stays outside: callers pass tokens to convert_tokens_to_ids."""
    counts = collections.Counter(tok for toks in token_lists for tok in toks)
    for tok in specials:
        counts.pop(tok, None)
    n_words = max_size - len(specials) if max_size else None
    words = [w for w, _ in counts.most_common(n_words)]
    vocab = {tok: i for i, tok in enumerate([*specials, *words])}

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN
    )


# ---- NER: geo CSV (uk_geo, ru_geo) ----

ENTITY_TYPES = {"loc_markers": "LOC", "org_markers": "ORG", "per_markers": "PER"}
MARKER_COLS = tuple(ENTITY_TYPES)
NER_LABELS = ("O", *(f"{p}-{t}" for t in ENTITY_TYPES.values() for p in "BI"))


def _parse_markers(df):
    df = df.copy()
    for col in MARKER_COLS:
        df[col] = df[col].map(ast.literal_eval)  # "[(l, r), ...]" -> list of tuples
    return df


def load_uk_geo(path, train_rows=None, valid_rows=None):
    """uk_geo with `split` (train/valid from is_valid) and `lang`; first rows of each split.
    Markers parsed after subsetting: parsing is slow."""
    df = pd.read_csv(path)
    df["split"] = df.pop("is_valid").map({0: "train", 1: "valid"})
    df = pd.concat(
        [
            df[df["split"] == "train"].head(train_rows),
            df[df["split"] == "valid"].head(valid_rows),
        ],
        ignore_index=True,
    )
    df["lang"] = "uk"
    return _parse_markers(df)


def load_ru_geo(path, nrows=None, n_docs=None, seed=0):
    """ru_geo, all train; n_docs samples whole documents, so no document gets split."""
    df = pd.read_csv(path, nrows=nrows)
    if n_docs is not None:
        docs = df["doc_id"].drop_duplicates().sample(n_docs, random_state=seed)
        df = df[df["doc_id"].isin(docs)]
    df = df.assign(split="train", lang="ru").reset_index(drop=True)
    return _parse_markers(df)


def load_geo(uk_path, ru_path, ru_docs, seed=0):
    """uk_geo plus ru_docs sampled ru_geo documents; validation stays uk only."""
    frames = [load_uk_geo(uk_path), load_ru_geo(ru_path, n_docs=ru_docs, seed=seed)]
    return pd.concat(frames, ignore_index=True)


def row_spans(row):
    """All entity char spans of geo frame row as (l, r, type)."""
    return [(l, r, t) for col, t in ENTITY_TYPES.items() for l, r in row[col]]


def bio_tags(text, spans):
    """Whitespace words (same as str.split) and their BIO tags from (l, r, type) spans, [l, r)."""
    words, tags = [], []
    for m in re.finditer(r"\S+", text):
        s, e = m.span()
        hit = next(((l, r, t) for l, r, t in spans if s < r and e > l), None)
        words.append(m.group())
        tags.append("O" if hit is None else f"{'B' if s <= hit[0] else 'I'}-{hit[2]}")
    return words, tags


class GeoNERDataset(Dataset):
    """One split of geo frame; LOC/ORG/PER tags. Output matches DataCollatorForTokenClassification."""

    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.label2id = {k: i for i, k in enumerate(NER_LABELS)}
        self.id2label = dict(enumerate(NER_LABELS))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        words, tags = bio_tags(row["text"], row_spans(row))
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        labels = [self.label2id[t] for t in tags]
        return {
            "input_ids": input_ids[: self.max_length],
            "labels": labels[: self.max_length],
        }


# ---- Collator ----


class DataCollatorForTokenClassification:
    def __init__(self, pad_token_id, label_pad_token_id=-100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = (
            label_pad_token_id  # -100 to ignore tokens in loss computations
        )

    def __call__(self, features):
        # Extract input_ids and labels from the list of feature dictionaries.
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]

        # Determine the maximum sequence length in this batch.
        batch_max_length = max(len(ids) for ids in input_ids_list)

        # Pad input_ids and labels on the right.
        padded_input_ids = [
            ids + [self.pad_token_id] * (batch_max_length - len(ids))
            for ids in input_ids_list
        ]
        padded_labels = [
            labels + [self.label_pad_token_id] * (batch_max_length - len(labels))
            for labels in labels_list
        ]

        # Convert lists to tensors.
        batch_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)

        return {"input_ids": batch_input_ids, "labels": batch_labels}
