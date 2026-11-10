"""Config sections filled from configs/*.yaml by HfArgumentParser, next to TrainingArguments."""

from dataclasses import dataclass


@dataclass
class ClassificationModelArguments:
    rnn_type: str = "rnn"
    num_layers: int = 3
    embed_dim: int = 256
    hidden_size: int = 256
    bidirectional: bool = False


@dataclass
class ClassificationDataArguments:
    tokenizer: str = "word"  # "word" or "bpe"
    word_vocab_max: int = 20_000
    bpe_vocab_size: int = 1_000
    test_size: float = 0.2
    split_seed: int = 42


@dataclass
class ProbeArguments:
    probe_t: int = 120
    max_lag: int = 60


@dataclass
class NerModelArguments:
    rnn_type: str = "gru"
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 3
    bidirectional: bool = True
    dropout_rate: float = 0.3


@dataclass
class NerDataArguments:
    train_rows: int = 200_000
    valid_rows: int = 10_000
    max_length_quantile: float = 0.99
