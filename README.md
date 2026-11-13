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

### LOC threshold

Word tagged LOC when P(B-LOC) + P(I-LOC) ≥ threshold (B or I, whichever more likely); else most likely non-LOC tag. One fixed threshold for all models: 0.5, best mean LOC F1 over 5 models on uk_geo valid (grid 0.10–0.90, step 0.05; 0.55 ties). Every model gains over argmax, gain small: precision up, recall slightly down. Sweep: `scripts/tune_loc_threshold.py`.

| Model | Argmax | Threshold 0.5 |
|---|---|---|
| `gru_uni` | 0.8110 | 0.8138 |
| `bigru_uk_ru` | 0.8322 | 0.8348 |
| `bigru_fasttext` | 0.8446 | 0.8481 |
| `bigru_uk` | 0.8523 | 0.8543 |
| `bigru_bpe` | 0.8771 | 0.8778 |

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
