---
name: payment-fraud-model-eval
description: >
  Model selection and evaluation methodology for payment fraud / chargeback risk control.
  On extremely imbalanced transaction-level tabular data, systematically compare a
  "model x imbalance-method" grid, pick the optimal model, and run business evaluation
  (probability calibration, amount-based cost/threshold optimization), error analysis
  (missed large-value fraud diagnosis) and interpretability (SHAP + permutation importance).
  Triggers when the user does anti-fraud/chargeback modeling, model selection, metric
  choice (AUC/KS/PR-AUC/F1), threshold/cost trade-offs, or asks "which model / which
  sampling / which metric to trust". Ships with runnable scripts and synthetic data.
---

# Payment Fraud / Chargeback - Model Selection & Evaluation

A modeling-and-evaluation methodology + runnable scripts for extremely imbalanced
(fraud rate typically 0.1%-1%) transaction-level tabular data. The deliverable is
"the chosen optimal model + its business value and blind spots", not a pile of metrics.

## When to use

- Imbalanced binary classification: anti-fraud / chargeback / account takeover / application fraud;
- You need to choose a model / an imbalance method / a threshold, or decide which metric to trust;
- Data is transaction-level tabular (numeric features + one label column + ideally an amount column).
- Not for: per-event sequences / cardholder history -> use the sequential-network method (see end).

## Iron rules (memorize first)

1. Three-way split: train / validation / holdout; the holdout is scored only once, at the end. Resampling on the training set only. With a meaningful time span, make the holdout out-of-time.
2. Select on PR-AUC (threshold-free), not AUC or accuracy - at 0.17% fraud, AUC is over-optimistic and "predict all normal" already scores 99.8% accuracy.
3. PR-AUC / AUC / KS are threshold-free; Precision / Recall / F1 depend on the threshold - always state at which threshold they are computed.
4. Set cost/threshold by business, not the default 0.5; and use amount recall (share of fraud amount recovered), not just count recall.

## Metric cheat sheet

| Metric | Meaning | Use |
|---|---|---|
| PR-AUC | Area under PR curve, baseline = fraud rate | Primary selection metric |
| AUC | Ranking ability, baseline 0.5 | Overall reference (over-optimistic under imbalance) |
| KS | max(TPR-FPR) | Sets the cut-point |
| Precision/Recall/F1 | Precise/complete/harmonic at a threshold | Look after fixing the threshold; do not use for selection |
| Brier | mean((p-y)^2) | Calibration quality (lower is better) |

## Workflow (scripts in scripts/, run by number)

Adapt to your data: scripts default to label column `Class` (0/1) and amount column `Amount`, treating the rest as numeric features. Shape your data this way and save as data/creditcard.csv, or run 00_generate_demo_data.py first to try on synthetic data.

    pip install -r scripts/requirements.txt
    python scripts/00_generate_demo_data.py          # (optional) generate synthetic fraud data
    python scripts/01_model_imbalance_grid.py --method None   # run once per imbalance method
    #   --method values: None / ClassWeight / Undersample / Oversample / SMOTE
    python scripts/02_optimal_model_eval.py          # optimal model: calibration + cost + amount-weighted + error analysis
    python scripts/03_error_analysis.py              # missed large-value fraud diagnosis
    python scripts/04_interpretability.py            # SHAP + permutation importance

Step 1 - First decide which models to compare based on the data (do not apply a fixed list)
Look at the data, then decide the candidate set:
- Medium-size, aggregated numeric tables (this skill's default): 1 linear baseline + tree ensembles, i.e. LogReg + RandomForest + LightGBM/XGBoost. Covers "linear vs non-linear" and "bagging vs boosting"; tree ensembles are the de facto strongest baseline for tabular data -> a sensible default for most payment-fraud cases.
- Many categorical features: add CatBoost (handles categoricals well).
- Strong interpretability / heavy regulation: lead with LogReg / scorecard, trees as a supporting comparison.
- Very few positives (fraud < a few thousand): prefer simple models + strong regularization; avoid deep trees / large n_estimators to prevent overfitting.
- Very large data or per-event sequences / cardholder history: only then is a neural network worth it (with representation learning, see upgrade path); otherwise MLP usually loses to GBM on flat tables.
- Engineering constraints: kernel SVM (O(n^2)) and KNN are impractical at large scale / high dimension - usually excluded.
Conclusion: the candidate set is "tailored to the data"; the scripts default to the 4 models of the first (most common) case above - add/remove models as needed.

Step 2 - Model x imbalance-method grid (01_*)
The chosen candidate models x 5 imbalance methods (None / ClassWeight / Undersample / Oversample / SMOTE).
- View at two operating points: KS-optimal threshold (recall-favoring) and F1-optimal threshold (balanced). PR-AUC is identical across both, serving as the constant anchor.
- GBMs must use early stopping (otherwise they overfit and PR-AUC collapses spuriously).

Step 3 - Pick the optimal model: primarily by PR-AUC, supported by KS/recall. Empirically, tree ensemble + no resampling tends to win and is naturally well-calibrated (no post-hoc calibration needed).

Step 4 - Business evaluation (02_*)
- Calibration (two kinds, optional - decide whether you need them first):
  - Do PROBABILITY calibration (Platt/Isotonic)? Only if you directly consume probability values: expected loss (prob x amount), multi-model fusion, fixed probability gates across populations/time, external/regulatory/pricing. Method: Platt for small samples, Isotonic for large; validate with Brier (raw vs calibrated). Note: resampling breaks calibration (SMOTE worst) -> if you resample you must calibrate; non-resampled tree models are usually well-calibrated and can skip it. If downstream only ranks or applies a threshold, you can skip it (monotonic transforms change neither ranking nor trade-offs).
  - Do TRUSTED-LAYER calibration? When you want to control the disturbance rate / blast radius (rather than accurate probabilities): use a good-user disturbance-rate (approx FPR) budget to back out the threshold, directly controlling the blast radius, independent of true probabilities. Pitfall: a too-"clean" trusted layer underestimates the disturbance rate -> the control group must be representative, use tiered thresholds, watch recall, roll out gradually.
  - Quick rule: want "accurate values" -> probability calibration; want "fewer disturbed good customers" -> trusted-layer calibration; only need "ranking / thresholding" -> neither is required.
- Cost/threshold: set missed = lost amount, false alarm = fixed cost C_FP, sweep thresholds for the minimum-total-cost operating point (often well below 0.5).
- Amount-weighted: report amount recall; it is often far below count recall -> exposes missed large-value fraud.

Step 5 - Error analysis (03_*): pull out the misses (especially large-value), and decide whether it is a threshold problem (scores just below the threshold -> tunable) or a feature blind spot (scores approx 0 -> the model cannot see them). Compare the feature profiles of "missed vs caught vs normal".

Step 6 - Interpretability (04_*): TreeSHAP (global beeswarm + per-transaction waterfall) + Permutation Importance (shuffle and watch the PR-AUC drop) to cross-validate importance; and check whether the SHAP values of missed large-value samples are approx 0 (confirming the blind spot).

## Frequent conclusions / pitfalls (from real ULB testing)

- Model choice matters >= imbalance method: choose the model first, then talk sampling.
- Class weighting is not a universal fix: it may help linear models but wreck tree models (LightGBM once dropped to PR-AUC 0.35).
- Undersampling is worst (large information loss); use only for fast iteration on huge data.
- F1 at the KS threshold is misleading (dragged down by low Precision, even making a high-precision low-recall weak model look best) - F1 only becomes meaningful after fixing a business threshold.
- Large-value fraud is often a cross-model feature blind spot: count recall is high but amount recall is only approx 50%, and lowering the threshold cannot fix it (scores approx 0). Fixes: add behavioral features (velocity, device, geo, cardholder history) or add amount-triggered rules / manual review on high-amount transactions (model catches the routine, rules guard the large ones).
- Trusted-layer threshold calibration: to control blast radius / disturbance rate only, set the threshold by a good-user FPR budget - no probability calibration needed; but don't make the trusted layer too "clean" or it underestimates the disturbance rate.

## Tooling notes

- shap vs xgboost compatibility: newer xgboost (3.x) serializes base_score as an array, which older shap fails to parse. Pin xgboost==2.1.4 for SHAP (already in requirements).
- In restricted environments, install with pip --break-system-packages.
- Background processes do not survive across calls in some sandboxes; run large grids in batches per --method.

## Out of scope / upgrade path

This method targets aggregated transaction-level tables. With per-event sequences / cardholder history, the real gains come from sequence modeling and representation learning - use "sequential-neural-network interrogation -> automated feature discovery" (seq-feature-discovery): treat the network as a feature search engine and feed interpretable features to a GBM.
