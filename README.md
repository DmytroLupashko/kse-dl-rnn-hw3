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
| `bigru_bpe_4k` + `bigru_fasttext` | | 0.8867 | 0.5658 |
| `bigru_bpe_4k`, 5 seeds | | 0.8958 | 0.5483 |
| `bigru_bpe_4k_full` | 0.0323 | 0.9053 | 0.5569 |
| `bigru_bpe_4k`, 5 seeds + full | | 0.9076 | 0.5631 |
| `bigru_fasttext`, 5 seeds | | 0.8578 | |
| `bigru_fasttext_full` | 0.0584 | 0.8699 | |
| `bigru_fasttext`, 5 seeds + full | | 0.8722 | |
| BPE + fastText, 5 seeds + full | | 0.9076 | 0.5803 |

`bigru_bpe_4k`: `bigru_bpe` with BPE vocab size 4k instead of 30k, config `configs/bpe_vocab/` on branch `bpe-vocab`. BPE vocab size 4k optimal.

Rows from `bigru_bpe_4k` + `bigru_fasttext` down: tag probs averaged per word, `scripts/ensemble.py`. Local validation scored one text per forward pass, without truncation and padding, so not directly comparable with rows above.

5 seeds: average of 5 runs with trainer and data seeds 1 to 5, 200k uk_geo rows as before. `_full`: all 1M uk_geo train rows, ru kept 1:1, learning rate 1e-4 (1e-3 diverged on long run). 5 seeds + full: mean of seed average and full run. BPE + fastText: mean of the two "5 seeds + full" rows. Configs: `configs/seeds_full/` on branch `seeds-full`.

Averaging has effect. Seed noise comparable to gaps between earlier models. Full uk_geo beats 5-seed average. Adding fastText gives no local gain over BPE alone, but best Kaggle score.

## Chosen hyperparameters

## Metrics and optimization curves

## Conclusions
