# Report

## EDA

## Tried models

Configs: `configs/cv_lb_check/`. Loss: cross-entropy on uk_geo valid, best epoch. Local validation: LOC F1 on uk_geo valid. Kaggle score: public LB.

| Model | Loss | Local validation | Kaggle score |
|---|---|---|---|
| `gru_uni` | 0.0818 | 0.8110 | 0.4602 |
| `bigru_uk_ru` | 0.0803 | 0.8322 | 0.4955 |
| `bigru_fasttext` | 0.0754 | 0.8446 | 0.5044 |
| `bigru_uk` | 0.0712 | 0.8523 | 0.5090 |
| `bigru_bpe` | 0.0454 | 0.8771 | 0.5225 |
| `bigru_bpe`, LOC threshold 0.5 | 0.0454 | 0.8778 | 0.5343 |
| `bigru_bpe_4k`, LOC threshold 0.5 | 0.0433 | 0.8843 | 0.5518 |

`bigru_bpe_4k`: `bigru_bpe` with BPE vocab size 4k instead of 30k, config `configs/bpe_vocab/` on branch `bpe-vocab`. BPE vocab size 4k optimal.

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
