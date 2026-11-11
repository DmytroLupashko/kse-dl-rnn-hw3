from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

RNN_TYPES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RNNPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        pass  # keep PyTorch default init instead of HF normal(0.02)


class RNNTokenClassifierConfig(PreTrainedConfig):
    model_type = "rnn_token_classifier"

    vocab_size: int = 20_000
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 3
    rnn_type: str = "gru"  # key of RNN_TYPES
    bidirectional: bool = True
    dropout_rate: float = 0.3
    pad_token_id: int = 0


class UniversalRNNTokenClassifier(RNNPreTrainedModel):
    config_class = RNNTokenClassifierConfig

    def __init__(self, config: RNNTokenClassifierConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        self.embed = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        self.rnn = RNN_TYPES[config.rnn_type](
            input_size=config.embed_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(
            config.hidden_size * self.num_directions, config.num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[
            torch.LongTensor
        ] = None,  # not used by RNN; masking is via labels=-100
        **kwargs,
    ):
        # input_ids: [B, T]
        x = self.embed(input_ids)  # [B, T, E]
        x = self.dropout(x)

        # You can pack based on attention_mask or true lengths if you like.
        # For simplicity (and because labels carry -100), we run unpacked:
        out, _ = self.rnn(x)  # [B, T, H*D]
        out = self.dropout(out)
        logits = self.classifier(out)  # [B, T, C]

        loss = None
        if labels is not None:
            # CrossEntropy over time; ignore_index handles padding.
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        # Return the standard HF TokenClassifierOutput so Trainer can read loss/logits.
        return TokenClassifierOutput(loss=loss, logits=logits)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
