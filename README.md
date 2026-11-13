# Report

## EDA

## Tried models

Configs: `configs/cv_lb_check/`. Loss: cross-entropy on uk_geo valid, best epoch. Local validation: LOC F1 on uk_geo valid. Kaggle score: public LB.

| Model | Loss | Local validation | Kaggle score |
|---|---|---|---|
| `gru_uni`, argmax | 0.0818 | 0.8110 | 0.4602 |
| `bigru_uk_ru`, argmax | 0.0803 | 0.8322 | 0.4955 |
| `bigru_fasttext`, argmax | 0.0754 | 0.8446 | 0.5044 |
| `bigru_uk`, argmax | 0.0712 | 0.8523 | 0.5090 |
| `bigru_bpe`, argmax | 0.0454 | 0.8771 | 0.5225 |
| `bigru_bpe` | 0.0454 | 0.8778 | 0.5343 |
| `bigru_bpe_4k` | 0.0433 | 0.8843 | 0.5518 |

`bigru_bpe_4k`: `bigru_bpe` with BPE vocab size 4k instead of 30k, config `configs/bpe_vocab/` on branch `bpe-vocab`. BPE vocab size 4k optimal.

### Ensemble

`scripts/ensemble.py`: tag probs of two models averaged per word, then LOC threshold 0.5. All rows scored by this script, one text per forward pass, so single-model numbers differ slightly from table above.

| Model | Local validation | Kaggle score |
|---|---|---|
| `bigru_fasttext` | 0.8500 | |
| `bigru_bpe_4k` | 0.8814 | 0.5518 |
| Average | 0.8867 | 0.5658 |

Averaging has effect. `bigru_fasttext` alone has no Kaggle score with LOC threshold, only argmax one above.

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
