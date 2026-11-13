import ast
import collections
import gzip
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


# ---- NER: CoNLL-U ----


def read_conll_file(filepath, nrows=10):
    sentences = []
    current_sentence = []
    sentence_counts = 0
    with gzip.open(filepath, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # A blank line indicates the end of a sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    sentence_counts += 1
                    current_sentence = []
            else:
                # Split the line into its fields (e.g., token, POS, label, etc.)
                parts = line.split()
                current_sentence.append(parts)
            if sentence_counts >= nrows:
                break
        # If the file doesn't end with a newline, add the last sentence.
        if current_sentence:
            sentences.append(current_sentence)

    return sentences


def get_label_distribution(sentences, label_index=9):
    label_counter = collections.Counter()
    for sentence in sentences:
        for token in sentence:
            if token[0].startswith("#") or len(token) != 10:
                continue
            label = token[label_index]
            label_counter[label] += 1
    return label_counter


def yield_tokens(data):
    for sentence in data:
        tokens = [row[1] for row in sentence if not row[0].startswith("#")]
        yield tokens


# Define a tokenizer that extracts tokens from a given sentence (ignoring metadata rows)
def simple_tokenizer(sentence):
    return [row[1] for row in sentence if not row[0].startswith("#")]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, conll_data, tokenizer, label_names, max_length):
        self.conll_data = conll_data
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = int(max_length)
        self.label_names = label_names
        self.label2id = {k: ind for ind, k in enumerate(list(self.label_names))}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def simple_tokenizer(self, sentence):
        return [row[1] for row in sentence if not row[0].startswith("#")]

    def get_labels(self, sentence):
        return [
            self.label2id[row[-1]] for row in sentence if not row[0].startswith("#")
        ]

    def __getitem__(self, idx):
        current_item = self.conll_data[idx]
        input_ids = self.tokenizer.convert_tokens_to_ids(
            self.simple_tokenizer(current_item)
        )
        labels = self.get_labels(current_item)

        return {
            "input_ids": input_ids[: self.max_length],
            "labels": labels[: self.max_length],
        }

    def __len__(self):
        return len(self.conll_data)


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
