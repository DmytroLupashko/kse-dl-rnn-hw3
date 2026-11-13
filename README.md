# Report

## EDA

## Tried models

Loss: cross-entropy on uk_geo valid, best epoch. Local validation: LOC F1 on uk_geo valid. Kaggle score: public LB.

uk_geo valid: all 10,000 uk_geo rows with `is_valid = 1`, same set every run, whatever `train_rows` and `ru_docs`. ru_geo goes to train only, never to validation.

### Validation check

Configs: `configs/cv_lb_check/`. Argmax tags.

| Model | Loss | Local validation | Kaggle score |
|---|---|---|---|
| `gru_uni` | 0.0818 | 0.8110 | 0.4602 |
| `bigru_uk_ru` | 0.0803 | 0.8322 | 0.4955 |
| `bigru_fasttext` | 0.0754 | 0.8446 | 0.5044 |
| `bigru_uk` | 0.0712 | 0.8523 | 0.5090 |
| `bigru_bpe` | 0.0454 | 0.8771 | 0.5225 |

### LOC threshold

Same `bigru_bpe` model; word tagged LOC when P(B-LOC) + P(I-LOC) ≥ 0.5.

| Tags | Loss | Local validation | Kaggle score |
|---|---|---|---|
| Argmax | 0.0454 | 0.8771 | 0.5225 |
| LOC threshold 0.5 | 0.0454 | 0.8778 | 0.5343 |

### uk only

Configs: `configs/uk_only/`. LOC threshold 0.5.

| Model | Loss | Local validation | Kaggle score |
|---|---|---|---|
| `bigru_fasttext_uk` | 0.0726 | 0.8622 | — |
| `bigru_bpe` | 0.0454 | 0.8778 | 0.5343 |
| `bigru_bpe_uk` | 0.0442 | 0.8831 | 0.5161 |

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
