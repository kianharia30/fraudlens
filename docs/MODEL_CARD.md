# Model card: FraudLens LightGBM fraud scorer

Format follows Mitchell et al., *Model Cards for Model Reporting* (2019). Numbers are generated
from [`metrics.json`](metrics.json) by `make train` / `make readme`. Until then they show as
placeholders.

## Model details

- **Developer:** Kian Haria (portfolio project; not a production system).
- **Model:** gradient-boosted trees (LightGBM, binary objective) with `scale_pos_weight` class
  weighting, hyperparameters chosen by a seeded Optuna TPE search on validation PR-AUC.
  An isotonic calibrator (fitted on validation) maps raw scores to probabilities, and two
  thresholds map probabilities to APPROVE / REVIEW / BLOCK. They minimise £ cost on validation
  within operational budgets (≤5% reviewed, ≤1% blocked).
- **Version:** the bundle directory name `YYYYmmdd-HHMMSS-<git>`. Full metadata (parameters,
  data window, config, git hash, Optuna trials) is in `artifacts/models/<version>/metadata.json`.
- **Baseline:** class-weighted logistic regression (median imputation and standard scaling).
- **Explanations:** SHAP TreeExplainer (log-odds), grouped into plain-English reasons.
  Anonymised features are labelled as such.
- **Licence:** MIT (code). The data is governed by the Kaggle competition rules.

## Intended use

- **Primary:** demonstrating a leakage-safe, cost-aware, explainable fraud-scoring workflow for
  education and portfolio purposes.
- **Users:** recruiters, reviewers, students, and ML practitioners exploring the approach.
- **Out of scope:** making real decisions about real customers or payments, any jurisdiction or
  merchant other than the dataset's, and any use without fresh validation, monitoring and a
  fairness review.

## Factors

- **Product code, card network and type, device type, email domain** shape the score.
  Performance is likely to differ across these subgroups but has not been audited per group.
- **Card history:** transactions on a proxy card UID with no earlier history have missing
  behaviour features. Performance on these "cold-start" cards may differ.
- **Time:** the model is evaluated on a later period than it was trained on. Performance is
  expected to decay further as time passes (concept drift).

## Metrics

- **Primary:** PR-AUC (average precision), suited to ~3.5% prevalence.
- **Secondary:** ROC-AUC, recall at 90% precision, Brier score and log-loss (calibration),
  confusion matrices for "blocked" and "flagged", and **total £ cost** under the configured
  cost matrix compared with do-nothing and the baseline.
- **Not used:** accuracy, which is about 96.5% for a model that approves everything.

## Training and evaluation data

- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
  (Vesta). `train_transaction.csv` (590,540 rows) left-joined with `train_identity.csv` on
  `TransactionID`. Unlabelled `test_*` files are not used.
- **Split (chronological on `TransactionDT`):** first 70% train, next 15% validation, final 15%
  test. Boundaries never split a timestamp.
  - train: fit the feature pipeline, the baseline and LightGBM.
  - validation: Optuna selection, early stopping, calibration and threshold selection.
  - test: final metrics, figures and the demo pool only.
- **Preprocessing:** compact dtypes; columns more than 95% missing in train dropped (listed in
  [`DATA_REPORT.md`](DATA_REPORT.md)); frequency and ordinal encodings fitted on train; past-only
  behaviour features on a proxy card UID; NaN kept for LightGBM.

## Quantitative analysis (held-out test block)

<!-- RESULTS:START -->
Model `20261002-211632-nogit` · trained 2026-10-02T21:16:32+00:00 · git `nogit` · source: [`docs/metrics.json`](metrics.json)

Held-out **test block**: 88,581 transactions, 3.48% fraud (never used for training, tuning, calibration or thresholds).

| Model | PR-AUC | ROC-AUC | Recall @ 90% precision | Brier (calibrated) |
|---|---|---|---|---|
| LightGBM | **0.5723** | 0.9066 | 23.2% | 0.0212 |
| Logistic regression | 0.1766 | 0.8252 | 0.0% | 0.1402 |

| Policy (test block) | Total cost | Fraud stopped | Fraud £ stopped | Genuine blocked | Genuine reviewed |
|---|---|---|---|---|---|
| Do nothing (approve all) | £469,609 | 0 / 3,083 (0.0%) | £0 | 0 | 0 |
| Logistic regression | £291,038 | 1,498 / 3,083 (48.6%) | £195,496 | 1,356 | 4,014 |
| LightGBM | £181,782 | 2,079 / 3,083 (67.4%) | £298,799 | 58 | 3,865 |

The LightGBM policy sends **6.0%** of test transactions to manual review and blocks **0.7%**. Thresholds were tuned to stay within budgets of review ≤ 5% and block ≤ 1% on *validation*; on the later test period the review budget is exceeded, a sign of drift that a live system would need to monitor. PR-AUC falls from 0.668 (validation) to 0.572 (the later test period), consistent with fraud patterns drifting over time.

LightGBM thresholds (chosen on validation): **REVIEW ≥ 0.105**, **BLOCK ≥ 0.921**. Savings on the test block: **£287,827** vs doing nothing, **£109,257** vs the logistic-regression policy.

Training time: 28.0 min total (8 Optuna trials, 26.9 min) on Darwin arm64, Python 3.11.5.
<!-- RESULTS:END -->

Figures: [PR](figures/pr_curve.png) · [ROC](figures/roc_curve.png) ·
[calibration](figures/calibration.png) · [score distribution](figures/score_distribution.png) ·
[feature importance](figures/feature_importance.png).

## Ethical considerations

- **Harms of errors:** a false BLOCK inconveniences a genuine customer (declined payment, possible
  embarrassment, lost trust). A missed fraud costs money and can harm the cardholder. The cost
  matrix encodes only crude £ values for these.
- **Proxy discrimination:** email domain, device, region and product can correlate with protected
  characteristics. No fairness audit has been done.
- **Privacy:** the data is anonymised by the provider. The model does not use names, card numbers
  or exact locations.
- **Explanations can mislead:** SHAP describes the model, not the causes of fraud, and anonymised
  features cannot be meaningfully explained.

## Caveats and recommendations

- Train and test come from one provider over about six months. Re-validate before any other use.
- The card UID is a heuristic. Behaviour features are approximate.
- The costs (£5 false alarm, £2 review, perfect reviewers) and the operational budgets (≤5% of
  transactions reviewed, ≤1% blocked) are illustrative.
  Replace them with real figures and re-run threshold selection.
- Any deployment would need drift monitoring, delayed-label handling (chargebacks), periodic
  retraining, human review of BLOCK decisions and a subgroup performance and fairness audit.
