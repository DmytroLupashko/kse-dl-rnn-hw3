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

### Seeds and full uk_geo

Configs: `configs/seeds_full/`. `bigru_bpe_4k` with trainer and data seeds 1 to 5; `_full`: all 1M uk_geo train rows, ru kept 1:1, learning rate 1e-4 instead of 1e-3 (1e-3 diverged on long run). Loss: Trainer eval. Local validation: `scripts/ensemble.py`, one text per forward pass.

| Model | Loss | Local validation |
|---|---|---|
| `bigru_bpe_4k_s1` | 0.0444 | 0.8785 |
| `bigru_bpe_4k_s2` | 0.0439 | 0.8807 |
| `bigru_bpe_4k_s3` | 0.0433 | 0.8881 |
| `bigru_bpe_4k_s4` | 0.0436 | 0.8834 |
| `bigru_bpe_4k_s5` | 0.0444 | 0.8780 |

Seed noise comparable to gaps between earlier models.

| Model | Local validation | Kaggle score |
|---|---|---|
| `bigru_bpe_4k_full` | 0.9053 | 0.5569 |
| Average of 5 seeds | 0.8958 | 0.5483 |
| Average of 5 seeds and full | 0.9076 | 0.5631 |

Full uk_geo beats 5-seed average; adding seeds to full helps a little. Last row: mean of seed average and full.

`bigru_fasttext`, same setup.

| Model | Loss | Local validation |
|---|---|---|
| `bigru_fasttext_s1` | 0.0740 | 0.8477 |
| `bigru_fasttext_s2` | 0.0777 | 0.8455 |
| `bigru_fasttext_s3` | 0.0788 | 0.8485 |
| `bigru_fasttext_s4` | 0.0749 | 0.8460 |
| `bigru_fasttext_s5` | 0.0852 | 0.8442 |

| Model | Local validation | Kaggle score |
|---|---|---|
| `bigru_fasttext_full` | 0.8699 | |
| Average of 5 seeds | 0.8578 | |
| Average of 5 seeds and full | 0.8722 | |

Same pattern as BPE, all fastText rows below BPE.

BPE and fastText together: mean of two "average of 5 seeds and full" rows above.

| Model | Local validation | Kaggle score |
|---|---|---|
| Average of BPE and fastText | 0.9076 | 0.5803 |

No gain over BPE alone locally, but best Kaggle score so far.

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
