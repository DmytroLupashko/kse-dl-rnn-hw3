"""Config sections `model` and `data`; `trainer` section goes to TrainingArguments."""

from dataclasses import dataclass


@dataclass
class ModelArguments:
    rnn_type: str = "gru"
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 3
    bidirectional: bool = True
    dropout_rate: float = 0.3
    fasttext: bool = False  # init embeddings from FASTTEXT_PATH; needs embed_dim 300


@dataclass
class DataArguments:
    train_rows: int = 200_000  # first uk_geo train rows; validation: all uk_geo valid rows
    ru_docs: int = 0  # ru_geo documents, ~11 rows each, sampled from first ru_nrows rows
    ru_nrows: int = 1_000_000  # read only these: full ru_geo is 2 GB
    tokenizer: str = "word"  # "word" or "bpe"
    vocab_size: int = 50_000
    max_length: int = 128  # train/valid tokens; test texts never truncated
    seed: int = 0
