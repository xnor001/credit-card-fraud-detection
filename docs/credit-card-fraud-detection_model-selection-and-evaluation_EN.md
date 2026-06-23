# Credit Card Fraud Detection · Model Selection & Evaluation

> Data: real ULB credit card fraud dataset (284,807 transactions, 492 fraud, 0.173%)
> Pipeline: **model × imbalance-method grid → pick the optimal model → business evaluation → error analysis**
> Evaluation: a single holdout (never used in training, scored once); GBMs use early stopping, LogReg is standardized; resampling is applied to the training set only.

---

## Purpose

On real credit card fraud data, systematically compare all combinations of "**model selection × imbalance-handling method**", pick the optimal model, then run **business evaluation** (calibration / cost / amount-weighted), **error analysis** and **interpretability** on it — distilling a reusable payment-risk modeling and evaluation method, and answering "which metric to trust, which model to choose, and where the model's real business value and blind spots are".

## Key conclusions (TL;DR)

1. **Select on PR-AUC, not AUC/accuracy.** Under extreme imbalance (0.17% fraud) AUC is over-optimistic and accuracy is meaningless; both operating-point views (KS threshold, F1-optimal threshold) point to the same top models.
2. **Optimal = XGBoost + no resampling**, PR-AUC 0.840, ranked first overall; RandomForest + no resampling is a strong runner-up. Both are **naturally well-calibrated and need no resampling**.
3. **Model choice matters ≥ imbalance method**; undersampling is worst overall, class weighting is not a universal fix (helps LogReg, wrecks LightGBM).
4. **Business value**: at the cost-optimal threshold, high precision and good recall together (XGBoost P0.90 / R0.81, only 7 false positives), saving ≈49% of loss vs "block nothing".
5. **Core blind spot**: count recall is nearly 80%, but **amount recall is only ≈50%** — large-value fraud disguises itself as normal in feature space and the model scores it ≈0, so **lowering the threshold cannot recover it**; this is cross-model, and the fix is **richer behavioral features** or **rules / manual review on high-amount transactions**. SHAP and Permutation Importance agree that V14 dominates and confirm this blind spot from an interpretability angle.

> Below, the argument unfolds as "data → metrics → grid selection → optimal model → business eval → error analysis → interpretability".

---

## 1. Data structure at a glance

| Variable | Type | Notes |
|---|---|---|
| `Time` | numeric | Seconds since the first transaction, range 0–172,792 (exactly 2 days) |
| `V1`–`V28` | numeric | Anonymized features after **PCA transform**; mean≈0, decreasing variance, mutually uncorrelated; no business meaning but retain predictive power |
| `Amount` | numeric | Transaction amount, right-skewed; fraud mean (122) slightly above normal (88) |
| `Class` | label | 0=normal, 1=fraud; **0.173% extreme imbalance** |

The dataset is **31 columns, no missing values, no categorical fields**. Highest correlation with Class: V17, V14, V12, V10, V16, V3.

---

## 2. How to read the metrics

- **AUC (ROC-AUC)**: overall ranking ability. The FPR denominator includes the huge normal population, so it is **over-optimistic under imbalance**. Baseline 0.5.
- **KS**: maximum separation of good/bad distributions = max(TPR − FPR); indicates the best cut-point.
- **PR-AUC (Average Precision)**: area under the Precision-Recall curve; its denominator only considers "flagged as fraud", so it **truly reflects the cost of imbalance and is the key selection metric for fraud**. Baseline = fraud rate (≈0.0017).
- **Precision / Recall / F1**: at a given threshold, "how precise / how complete / their harmonic mean (F1=2PR/(P+R))". **In this grid all three use the KS-optimal threshold on validation** (recall-favoring, so Precision and F1 are low — a threshold choice, not a model flaw; **do not use F1 for selection**, see below).
- **Brier score**: = mean((p − y)²), the mean squared error of probabilities, measuring **how accurate the probability values are (calibration)**; lower is better. Under imbalance the values are inherently tiny (0.000x), so only relative comparison is meaningful.
- **Training set size**: resampling changes it (undersampling shrinks it greatly, oversampling/SMOTE roughly doubles it), indirectly reflecting a method's data cost.

**Core: select on PR-AUC (threshold-free); don't be fooled by a pretty AUC.**

---

## 3. Model selection × imbalance method grid (main analysis)

4 models (LogReg, RandomForest, LightGBM, XGBoost) × 5 imbalance methods = 20 combinations, real ULB, same holdout. (Why these models, see Appendix C.)

Since Precision/Recall/F1 all depend on the threshold, **view them at two operating points** to avoid being misled by a single threshold:

- **View 1: KS-optimal threshold** (max(TPR−FPR), a recall-favoring operating point)
- **View 2: F1-optimal threshold** (the threshold that maximizes F1 on validation, balancing precision and recall)

**AUC / KS / PR-AUC are identical across the two views (threshold-free)**; only P/R/F1 differ. PR-AUC, as the threshold-free selection anchor, appears in both tables.

### Table A: at the KS-optimal threshold

| Imbalance method | Model | Train size | AUC | KS | PR-AUC | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| None | LogReg | 199,364 | 0.957 | 0.844 | 0.750 | 0.051 | 0.851 | 0.097 |
| None | RandomForest | 199,364 | 0.968 | 0.860 | 0.827 | 0.021 | 0.892 | 0.040 |
| None | LightGBM | 199,364 | 0.904 | 0.823 | 0.737 | 0.555 | 0.824 | 0.663 |
| **None** | **XGBoost** | 199,364 | 0.966 | 0.853 | **0.840** | 0.075 | 0.851 | 0.138 |
| Class weight | LogReg | 199,364 | 0.968 | 0.864 | 0.793 | 0.080 | 0.865 | 0.146 |
| Class weight | RandomForest | 199,364 | 0.930 | 0.859 | 0.822 | 0.175 | 0.865 | 0.291 |
| Class weight | LightGBM | 199,364 | 0.901 | 0.826 | 0.347 | 0.107 | 0.838 | 0.190 |
| Class weight | XGBoost | 199,364 | 0.944 | 0.855 | 0.740 | 0.051 | 0.865 | 0.097 |
| Undersample | LogReg | 688 | 0.969 | 0.867 | 0.529 | 0.091 | 0.865 | 0.164 |
| Undersample | RandomForest | 688 | 0.976 | 0.850 | 0.739 | 0.022 | 0.905 | 0.042 |
| Undersample | LightGBM | 688 | 0.964 | 0.854 | 0.670 | 0.034 | 0.878 | 0.066 |
| Undersample | XGBoost | 688 | 0.969 | 0.861 | 0.694 | 0.060 | 0.865 | 0.113 |
| Oversample | LogReg | 398,040 | 0.967 | 0.861 | 0.788 | 0.073 | 0.878 | 0.135 |
| Oversample | RandomForest | 398,040 | 0.951 | 0.877 | 0.822 | 0.053 | 0.905 | 0.100 |
| Oversample | LightGBM | 398,040 | 0.967 | 0.866 | 0.772 | 0.045 | 0.892 | 0.086 |
| Oversample | XGBoost | 398,040 | 0.954 | 0.863 | 0.742 | 0.038 | 0.892 | 0.074 |
| SMOTE | LogReg | 398,040 | 0.967 | 0.856 | 0.796 | 0.065 | 0.865 | 0.122 |
| SMOTE | RandomForest | 398,040 | 0.977 | 0.871 | 0.819 | 0.037 | 0.878 | 0.071 |
| SMOTE | LightGBM | 398,040 | 0.971 | 0.852 | 0.780 | 0.036 | 0.878 | 0.069 |
| SMOTE | XGBoost | 398,040 | 0.949 | 0.869 | 0.731 | 0.027 | 0.892 | 0.052 |

> The KS threshold favors recall, so Precision/F1 are generally low (F1 0.04–0.29). **In this view F1 is misleading** — the "high-precision low-recall" LightGBM+None shows a deceptively high F1=0.663, yet its PR-AUC is only 0.737. **In this view, select on PR-AUC → XGBoost+None (0.840) is best.**

### Table B: at the F1-optimal threshold

| Imbalance method | Model | PR-AUC | P | R | F1 |
|---|---|---|---|---|---|
| None | LogReg | 0.750 | 0.815 | 0.716 | 0.763 |
| None | RandomForest | 0.827 | 0.950 | 0.770 | **0.851** |
| None | LightGBM | 0.737 | 0.943 | 0.676 | 0.787 |
| **None** | **XGBoost** | **0.840** | 0.894 | 0.797 | 0.843 |
| Class weight | LogReg | 0.793 | 0.906 | 0.784 | 0.841 |
| Class weight | RandomForest | 0.822 | 0.963 | 0.703 | 0.813 |
| Class weight | LightGBM | 0.347 | 0.432 | 0.770 | 0.553 |
| Class weight | XGBoost | 0.740 | 0.869 | 0.716 | 0.785 |
| Undersample | LogReg | 0.529 | 0.557 | 0.662 | 0.605 |
| Undersample | RandomForest | 0.739 | 0.883 | 0.716 | 0.791 |
| Undersample | LightGBM | 0.670 | 0.853 | 0.703 | 0.770 |
| Undersample | XGBoost | 0.694 | 0.898 | 0.716 | 0.797 |
| Oversample | LogReg | 0.788 | 0.931 | 0.730 | 0.818 |
| Oversample | RandomForest | 0.822 | 0.932 | 0.743 | 0.827 |
| Oversample | LightGBM | 0.772 | 0.849 | 0.757 | 0.800 |
| Oversample | XGBoost | 0.742 | 0.803 | 0.662 | 0.726 |
| SMOTE | LogReg | 0.796 | 0.931 | 0.730 | 0.818 |
| SMOTE | RandomForest | 0.819 | 0.949 | 0.757 | 0.842 |
| SMOTE | LightGBM | 0.780 | 0.871 | 0.730 | 0.794 |
| SMOTE | XGBoost | 0.731 | 0.877 | 0.770 | 0.820 |

> After moving to the F1-optimal threshold, Precision generally rises to 0.8–0.96 and **F1 becomes meaningful (0.55–0.85)**. **In this view, the best F1 = RandomForest+None (0.851), with XGBoost+None (0.843) right behind.**

![Model × imbalance-method PR-AUC grid](../reports/figures/model_imbalance_grid.png)

(Data: `reports/model_imbalance_grid.csv`)

**Observations:**

1. **Model choice matters ≥ imbalance method.** Swapping models within one method can change PR-AUC by 0.1–0.4.
2. **Tree ensembles are the most stable and strongest.** RandomForest / XGBoost take the top spots in both views; the LogReg baseline is surprisingly decent (≈0.79) but collapses to 0.53 under undersampling.
3. **Both views converge**: whether by PR-AUC or by the F1-optimal threshold, the top is "**None + XGBoost/RandomForest**".
4. **Undersampling is worst overall**; **class weighting is model-dependent** (helps LogReg, drops LightGBM to 0.347) — not a universal fix.

---

## 4. Key conclusion: the optimal model

**Both selection views point to "None + tree ensemble":**

- **View 1 (PR-AUC, threshold-free, the primary basis)**: XGBoost + None = **0.840**, first overall;
- **View 2 (F1-optimal threshold)**: RandomForest + None = **0.851** first, XGBoost + None = 0.843 right behind (0.008 gap).

**Overall, choose XGBoost + None as optimal**: first on PR-AUC, a hair behind on F1, with AUC 0.966 / KS 0.853 / Recall 0.851 all near the top; plus **no resampling** (saves data cost, doesn't break calibration) and **naturally well-calibrated** (see Section 5). **RandomForest + None is a strong runner-up / control model.** Not recommended: any model + undersampling, LightGBM + class weight (0.347).

> Note: this corrects the earlier "LightGBM-only" impression — at that time "SMOTE/oversampling looked better"; adding the model dimension shows the best is **tree ensemble + no resampling**, not LightGBM + SMOTE.

---

## 5. Optimal-model business evaluation (calibration / cost / amount-weighted)

Side-by-side business evaluation of the two candidates (optimal XGBoost+None and runner-up RandomForest+None). Cost assumptions: a missed fraud = the lost amount, a false alarm = ¥5 each; the "block nothing" loss baseline is ¥8,483.

| Dimension | XGBoost+None | RandomForest+None |
|---|---|---|
| AUC | 0.966 | **0.968** |
| KS | 0.853 | **0.860** |
| PR-AUC | **0.840** | 0.827 |
| Brier raw / calibrated | 0.00042 / 0.00041 | 0.00045 / 0.00043 |
| Cost-optimal threshold | 0.334 | 0.251 |
| Total cost (saving %) | ¥4,298 (save 49%) | **¥4,259 (save 50%)** |
| Precision | 0.896 | **0.935** |
| Count recall | **0.811** | 0.784 |
| Amount recall | 0.497 | 0.500 |
| F1 | 0.851 | **0.853** |
| TP / FP / FN | 60 / 7 / 14 | 58 / **4** / 16 |

![XGBoost calibration / cost / error-analysis triptych](../reports/figures/optimal_model_eval.png)

(The figure above is XGBoost's calibration curve / cost curve / amount-vs-score scatter triptych.)

**Conclusions:**

1. **The two models are nearly tied, with slight emphases.** XGBoost has slightly higher recall (catches 60 vs 58) and PR-AUC; RandomForest has higher precision (0.935, only 4 false alarms) and slightly better KS/AUC/cost. **Choose XGBoost for its first-place PR-AUC; if minimizing false alarms matters more, RandomForest is an equally reasonable choice.**
2. **Both are naturally well-calibrated.** Brier raw ≈ calibrated (both ≈0.0004) — **None + tree ensemble needs no post-hoc calibration**, a hidden advantage over resampled models.
3. **At the cost-optimal threshold, high precision and good recall together** (XGBoost P0.90/R0.81, RF P0.94/R0.78, single-digit false alarms). This is exactly "good calibration → the threshold becomes meaningful".
4. **Amount recall is only ≈50% in both**: count recall near 80% but only half the fraud amount recovered → large-value fraud is still missed (see Section 6), independent of model choice.
5. **How to choose calibration / threshold method**: probability calibration is only needed when you directly consume probability values (expected loss, multi-model fusion, fixed probability gates, external/regulatory use) — use Platt for small samples, Isotonic for large; if you only care about controlling the disturbance rate / blast radius, use a **trusted-layer threshold** (set the threshold by a good-user FPR budget), and be careful that the trusted layer is not too "clean" or it will underestimate the disturbance rate.

---

## 6. Optimal-model error analysis: the large-value fraud blind spot

At the optimal threshold, 14/74 fraud are missed, but those 14 account for **50% of total fraud loss**.

1. **It's a feature blind spot, not a threshold problem.** Of the 14, **12 are deep misses (score≈0)** — lowering the threshold cannot save them.
2. **The missed ones are exactly the large-value ones.** Median amount of caught fraud ¥2, of missed fraud ¥143.
3. **In feature space they look just like normal transactions.** On the most discriminative features, caught fraud sits at extreme values while missed fraud overlaps with normal:

   | | Missed fraud | Caught fraud | Normal |
   |---|---|---|---|
   | V17 | 0.43 | −8.20 | 0.01 |
   | V14 | −1.21 | −8.00 | 0.02 |
   | V12 | −0.66 | −7.21 | 0.01 |

**Meaning & remedy:** the model only catches fraud with an obvious feature signature (which happens to be small-value); large-value fraud disguises itself as normal and the model genuinely cannot see it — **this is cross-model; switching model/sampling won't fix it**. Threshold-based fixes (lowering the threshold / amount-tiered thresholds / expected-loss ranking) also fail, because the scores are ≈0. The real fixes: **add features** (velocity, device, geo, cardholder history — exactly what this data lacks); or **add amount-triggered manual review / rules on high-amount transactions** (model catches the routine, rules guard the large ones).

---

## 7. Interpretability analysis (SHAP + Permutation Importance)

Two complementary interpretability analyses on the optimal model XGBoost+None: **TreeSHAP** (global importance + per-transaction attribution) and **Permutation Importance** (shuffle a feature and measure the PR-AUC drop, model-agnostic). The two use different mechanisms, so their **agreement** reduces the risk of misreading.

> Tooling note: shap 0.49 is incompatible with xgboost 3.2's base_score serialization, so the SHAP analysis runs on xgboost 2.1.4 (same hyperparameters, essentially unchanged performance). SHAP values are in log-odds (margin) space; **the PCA-anonymized features have no business meaning, so the value of this analysis is "validating whether the model logic is stable and sensible" (governance use), not telling a business story.**

**Global importance (SHAP vs Permutation, top ranks highly consistent):**

| Feature | SHAP rank | Permutation rank | Permutation: PR-AUC drop when shuffled |
|---|---|---|---|
| V14 | 1 | 1 | **0.219** (0.84 → ≈0.62) |
| V10 | 2 | 2 | 0.047 |
| V17 | 4 | 3 | 0.036 |
| V12 | 3 | 4 | 0.023 |
| V7 | 5 | 5 | 0.015 |

**SHAP global importance (beeswarm):**

![SHAP beeswarm](../reports/figures/shap_beeswarm.png)

**Permutation Importance (PR-AUC drop when shuffled):**

![Permutation importance](../reports/figures/permutation_importance.png)

**Per-transaction explanation · the largest-amount fraud (¥1,097, predicted only 0.30):**

![SHAP waterfall - biggest fraud](../reports/figures/shap_waterfall_biggest_fraud.png)

**Conclusions:**

1. **The two methods' top 5 are identical** (V14, V10, V17, V12, V7), so the **importance conclusion is robust and credible** — exactly the value of cross-validating with two different mechanisms.
2. **V14 dominates absolutely**: shuffling it drops PR-AUC from 0.84 straight to ≈0.62; the contributions of other features fall sharply. The model relies heavily on a few V features.
3. **Consistent with the error analysis**: the largest-amount fraud (¥1,097) is predicted only **0.30** (below the cost threshold 0.334 → exactly the one that was missed), and its SHAP waterfall shows all features pushing only weakly — **confirming from an interpretability angle that "the model has no signal on large-value fraud"**, consistent with the Section 6 blind-spot conclusion.
4. **Governance value vs business limitation**: SHAP gives each decision a "complete receipt" (additivity), satisfying model governance / dispute explanation; but because V is PCA-anonymized, it cannot state the business reason. For business-level interpretability, raw features are needed.

---

## Appendix A: mechanisms and differences of the five imbalance methods

| Method | Acts on | Data volume | Information loss | Probability calibration | Speed |
|---|---|---|---|---|---|
| None | — | unchanged | none | **preserved** | medium |
| Class weight | algorithm (loss weighting) | unchanged | none | fairly good | fast |
| Random undersample | data (drop majority) | greatly reduced | large | broken | fastest |
| Random oversample | data (duplicate minority) | greatly increased | none | broken | slow |
| SMOTE | data (interpolated synthesis) | greatly increased | none | broken | slowest |

Key point: data-level methods (under/over/SMOTE) improve ranking but **break probability calibration**; the algorithm-level method (class weight) is gentler but over-doing it hurts PR-AUC; "None" is the cleanest on calibration — one of the reasons the optimal model uses it.

## Appendix B: when to do probability calibration vs trusted-layer calibration

- **Probability calibration (Platt/Isotonic)**: only needed when you **directly consume probability values** (expected loss, multi-model fusion, cross-population fixed gates, external/regulatory/pricing use). Both are monotonic transforms that do not change ranking or trade-offs — they only re-label the threshold scale. Use Platt for small samples, Isotonic for large.
- **Trusted-layer calibration**: calibrate the **operating point (threshold)** rather than the probability — set the threshold by the disturbance rate (≈FPR) budget on good users, directly controlling the blast radius, without relying on true probabilities. Pitfall: a trusted layer that is too "clean" underestimates the disturbance rate → the control group must be representative, use tiered thresholds, watch recall at the same time, and roll out gradually.

## Appendix C: model-selection rationale

**Why these 4 models (LogReg, RandomForest, LightGBM, XGBoost):** cover the main paradigms with the fewest models — **1 linear baseline + 3 tree ensembles**, with the tree ensembles spanning bagging and boosting:

- **LogReg (logistic regression)**: the classic credit scorecard — linear, interpretable, compliance-friendly — used as the baseline.
- **RandomForest**: tree **bagging** (parallel + voting), robust and resistant to overfitting.
- **LightGBM / XGBoost**: tree **boosting** (sequential error-correction), the strongest workhorse on tabular data.

This covers the "linear vs non-linear" and "bagging vs boosting" contrasts. Tree ensembles dominate because they are the **de facto strongest baseline for tabular/structured data**: they capture non-linearity and feature interactions natively, are robust to scale/outliers/missing values, directly produce feature importance/SHAP, and are fast and stable at medium scale.

**Why other models are not included for now** (not that they can't be compared, but their marginal value on this data is low or they are engineering-impractical):

- **Neural networks (MLP)**: usually lose to GBMs on flat tabular data (they shine on raw sequential/behavioral data);
- **Kernel SVM**: ≈O(n²) complexity, basically intractable at 280k samples;
- **KNN**: poor in high dimensions, slow at prediction;
- **Naive Bayes**: the independence assumption is too strong, accuracy usually below trees.

For a more complete comparison later, add **MLP and Naive Bayes** (both fast); if **raw behavioral/sequence features** become available, neural networks are worth a serious try (see the seq-feature-discovery method).
