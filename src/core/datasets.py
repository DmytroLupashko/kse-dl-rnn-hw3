import collections
import gzip
import re

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.nn.utils.rnn import pad_sequence
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


def build_bpe_tokenizer(texts, vocab_size, specials=(PAD_TOKEN, UNK_TOKEN)):
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials))
    tokenizer.train_from_iterator(text_iterator(texts), trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN
    )


# ---- Classification: tweets ----


def word_tokenize(text: str):
    # Lowercasing and basic tokenization: split on non-word characters
    text = text.lower()
    toks = re.findall(r"[a-z0-9_]+", text)
    return toks


# Prepare iterator over training texts
def text_iterator(series, batch_size=1000):
    batch = []
    for text in series:
        batch.append(text.lower())
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class SeqDataset(Dataset):
    def __init__(self, texts, labels, encode):
        self.texts = list(texts)
        self.labels = list(labels)
        self.encode = encode  # text -> list of token ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        y = int(self.labels[idx])
        ids = self.encode(str(t).lower().strip())
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_collate_fn(pad_id: int):
    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        return {
            "input_ids": padded,
            "lengths": lengths,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


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
