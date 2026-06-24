# Credit Card Fraud Detection

[中文说明](README.zh-CN.md)

A **run-out-of-the-box** credit card fraud detection project: synthetic data → train LightGBM → evaluate on a holdout set with AUC / KS / Precision-Recall. Built for learning and practicing payment risk control.

> Scenario: online-transaction anti-fraud for a Singapore e-commerce / card issuer. The methodology is country-agnostic; rules can be extended to meet MAS (Monetary Authority of Singapore) suspicious-transaction monitoring requirements.

## Why synthetic data

The industry-standard ULB credit card dataset requires a Kaggle login to download. So that anyone can clone and **run it immediately**, this project ships a synthetic data generator that mimics the ULB structure (`Time, V1..V28, Amount, Class`) and its most important property — **extreme class imbalance (fraud ≈ 0.3%)**. To switch to real data, see "Using the real ULB data" below.

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py          # generate data/creditcard_synthetic.csv
python src/train.py                  # train + evaluate, produce metrics and figures
```

Outputs:
- `reports/metrics.json` — validation / holdout AUC, KS, PR-AUC, plus TP/FP/FN/TN, precision and recall at the chosen threshold
- `reports/figures/roc_ks.png` — ROC curve with KS annotated
- `reports/figures/precision_recall.png` — PR curve (more informative under imbalance)
- `reports/figures/feature_importance.png` — feature importance

## Workflow

1. **Three-way split** `train 70% / validation 15% / holdout 15%` (stratified, fixed seed). The **holdout is used only once, at the end**, simulating "future, unseen transactions".
2. **Imbalance handling**: weight the rare fraud class via `scale_pos_weight`.
3. **Training**: LightGBM (gradient-boosted trees) with early stopping on validation.
4. **Threshold selection**: pick the cut-point at the maximum KS on validation (without peeking at the holdout).
5. **Evaluation** (holdout):
   - **AUC** — overall ranking ability
   - **KS** — maximum separation of the good/bad distributions = max(TPR − FPR)
   - **Precision / Recall** — at the chosen threshold, "how precise / how complete"
   - **PR-AUC (Average Precision)** — reflects real-world performance better than ROC-AUC under extreme imbalance

## How to read the metrics (cheat sheet)

| Metric | Formula / meaning | Focus |
|---|---|---|
| AUC | P(a random bad ranks above a random good) | Overall ranking power, robust to imbalance |
| KS | max(TPR − FPR) | Best cut-point separation; sets the approval line |
| Precision | TP/(TP+FP) | Share of true fraud among flagged (inverse of false alarms) |
| Recall (TPR) | TP/(TP+FN) | Share of true fraud that is caught |
| PR-AUC | Area under the PR curve | Real-world measure under low fraud rates |

## Using the real ULB data (optional)

1. Download `creditcard.csv` from the Kaggle `Credit Card Fraud Detection` (ULB) dataset;
2. Place it at `data/creditcard.csv`;
3. Run `python src/train.py --data data/creditcard.csv`.

The column structure is identical, so no code changes are needed.

## Full analysis report

For the in-depth "model selection × imbalance method" comparison, optimal-model business evaluation, error analysis and interpretability, see:

**[`docs/信用卡欺诈检测·模型选型与评估分析.md`](docs/)** (Chinese) / its English counterpart in `docs/`.

Key conclusion: on the real ULB data, **XGBoost + no resampling** is the best overall (PR-AUC 0.840, naturally well-calibrated); large-value fraud is a cross-model feature blind spot that requires richer features or amount-triggered rules.

## Project structure

```
src/                      analysis scripts
  generate_data.py        synthetic data generator
  train.py                train + evaluate + plot
  compare_imbalance.py    compare 5 imbalance methods
  model_imbalance_grid.py model × imbalance-method grid
  business_eval.py        calibration / cost / amount-weighted
  optimal_model_eval.py   optimal-model business eval + error analysis
  error_analysis.py       missed large-value fraud diagnosis
  interpretability.py     SHAP + permutation importance
docs/                     full analysis report (Markdown)
reports/                  metric CSV/JSON and figures (figures/)
data/                     data (generated at runtime, gitignored, not committed)
```

## Possible extensions

- Add a business rule engine (large amounts at night, cross-region, many txns in a short window) for a "rules + model" hybrid, covering the large-fraud blind spot
- Add behavioral features (velocity, device, geo, cardholder history) to improve large-fraud recall
- Apply the paper's method: sequential-neural-network interrogation → automated feature discovery (see the seq-feature-discovery playbook)
