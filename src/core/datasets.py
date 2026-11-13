import ast
import collections
import re

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

# Word: letters/digits joined by hyphen or apostrophe; any other non-space char is own word.
# Punctuation split off, so "Києві," and "Києві" share one id and entity spans exclude it.
WORD_RE = re.compile(r"\w+(?:[-'’ʼ]\w+)*|[^\w\s]")


def split_words(text):
    """Words of text and their char spans [l, r)."""
    matches = list(WORD_RE.finditer(text))
    return [m.group() for m in matches], [m.span() for m in matches]


# ---- Tokenizers ----


def _fast(tokenizer):
    # Words come pre-split (is_split_into_words), WhitespaceSplit keeps each word whole
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN
    )


def build_word_tokenizer(token_lists, specials=(PAD_TOKEN, UNK_TOKEN), max_size=None):
    """WordLevel vocab: specials first, then most common tokens; unknown tokens map to <unk>.
    Splitting text into tokens stays outside: callers pass tokens to convert_tokens_to_ids."""
    counts = collections.Counter(tok for toks in token_lists for tok in toks)
    for tok in specials:
        counts.pop(tok, None)
    n_words = max_size - len(specials) if max_size else None
    words = [w for w, _ in counts.most_common(n_words)]
    vocab = {tok: i for i, tok in enumerate([*specials, *words])}
    return _fast(Tokenizer(models.WordLevel(vocab, unk_token=UNK_TOKEN)))


def build_bpe_tokenizer(token_lists, vocab_size, specials=(PAD_TOKEN, UNK_TOKEN)):
    """BPE trained on words; merges never cross word boundary, so each word maps to its subwords."""
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials))
    tokenizer.train_from_iterator((" ".join(toks) for toks in token_lists), trainer)
    return _fast(tokenizer)


def build_tokenizer(kind, token_lists, vocab_size):
    """kind: "word" (WordLevel) or "bpe"."""
    if kind == "word":
        return build_word_tokenizer(token_lists, max_size=vocab_size)
    if kind == "bpe":
        return build_bpe_tokenizer(token_lists, vocab_size)
    raise ValueError(f"unknown tokenizer {kind!r}")


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


def load_geo(uk_path, ru_path, train_rows=None, ru_docs=0, ru_nrows=None, seed=0):
    """First train_rows uk_geo train rows plus ru_docs documents sampled from first ru_nrows
    ru_geo rows (ru_docs 0: no ru); validation stays uk only."""
    frames = [load_uk_geo(uk_path, train_rows=train_rows)]
    if ru_docs:
        frames.append(load_ru_geo(ru_path, nrows=ru_nrows, n_docs=ru_docs, seed=seed))
    return pd.concat(frames, ignore_index=True)


def row_spans(row):
    """All entity char spans of geo frame row as (l, r, type)."""
    return [(l, r, t) for col, t in ENTITY_TYPES.items() for l, r in row[col]]


def bio_tags(text, spans):
    """Words (split_words) and their BIO tags from (l, r, type) spans, [l, r)."""
    words, tags = [], []
    for word, (s, e) in zip(*split_words(text)):
        hit = next(((l, r, t) for l, r, t in spans if s < r and e > l), None)
        words.append(word)
        tags.append("O" if hit is None else f"{'B' if s <= hit[0] else 'I'}-{hit[2]}")
    return words, tags


def tag_frame(df):
    """(words, tags) per row of geo frame; rows without words dropped."""
    pairs = (bio_tags(row["text"], row_spans(row)) for row in df.to_dict("records"))
    return [(w, t) for w, t in pairs if w]


def first_subword_labels(word_ids, word_labels, ignore=-100):
    """Label on first subword of each word, ignore on rest: loss and metric see one label per word."""
    labels, prev = [], None
    for wid in word_ids:
        labels.append(ignore if wid is None or wid == prev else word_labels[wid])
        prev = wid
    return labels


class GeoNERDataset(Dataset):
    """(words, tags) pairs encoded once; output matches DataCollatorForTokenClassification."""

    def __init__(self, tagged, tokenizer, max_length):
        self.label2id = {k: i for i, k in enumerate(NER_LABELS)}
        self.id2label = dict(enumerate(NER_LABELS))
        enc = tokenizer(
            [words for words, _ in tagged],
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,
            max_length=int(max_length),
        )
        self.input_ids = enc["input_ids"]
        self.labels = [
            first_subword_labels(enc.word_ids(i), [self.label2id[t] for t in tags])
            for i, (_, tags) in enumerate(tagged)
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


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
