"""Config sections filled from configs/*.yaml by HfArgumentParser, next to TrainingArguments."""

from dataclasses import dataclass


@dataclass
class ModelArguments:
    rnn_type: str = "gru"
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 3
    bidirectional: bool = True
    dropout_rate: float = 0.3


@dataclass
class DataArguments:
    train_rows: int = 200_000
    valid_rows: int = 10_000
    max_length_quantile: float = 0.99
