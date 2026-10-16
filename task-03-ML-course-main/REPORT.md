# Model Card: Term Deposit Call Prioritisation

**Author:** Baizhan | **SEED:** 3101

## 1. Purpose

A Portuguese bank phones clients to sell a term deposit. In one campaign wave the call centre can phone only 10% of the clients on the list. The model gives every client a score (higher = more likely to say "yes") so that the 10% with the highest scores are called. The score is computed before any call is made, so it can use only information known before the call.

The model supports one decision: **whom to call first**. It is not a probability estimate for individual clients and must not be used to refuse service to anyone.

## 2. Data

UCI Bank Marketing, file `bank-additional-full.csv` (Moro, Rita & Cortez, 2014; CC BY 4.0), calls made between May 2008 and November 2010. After removing 12 exact duplicates the instructor split it at random into `train.csv` (32 940 rows, 11.27% "yes") and `test.csv` (8 236 rows, no answers). This is a teaching dataset: no conclusions about real banks or clients.

- `duration` (call length) exists only in train. It is known only after the call, so it is **not used**; with it the scores would look better than they could ever be in real use.
- `id` is an identifier and is not used.
- 19 features remain: age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome and four economic indicators.
- Missing information: "unknown" appears in default (20.9% of rows), education (4.3%), housing and loan (2.4% each), job (0.8%), marital (0.2%). `pdays = 999` (96.2% of rows) means "not contacted before".

## 3. Pipeline

Everything is one scikit-learn `Pipeline` (no custom code), so it accepts the raw columns of `test.csv` (without `id` and `duration`) and is fitted on training folds only during cross-validation.

- `pdays`: 999 treated as missing, median imputation, missing-indicator column, StandardScaler.
- Other numeric columns: median imputation + StandardScaler.
- Categorical columns: OneHotEncoder (`handle_unknown="ignore"`); "unknown" kept as its own category. Replacing it by the most frequent value gave the same score (ROC-AUC 0.7934 vs 0.7936, inside the fold spread).
- Model: RandomForestClassifier (200 trees, `min_samples_leaf=20`, `max_depth=None`), chosen by RandomizedSearchCV (ROC-AUC) on the development part only.
- Validation: 20% hold-out (6 588 rows) kept aside until the end; 5-fold StratifiedKFold (shuffled, SEED 3101) on the remaining 26 352 rows. Final model refitted on all of `train.csv`.

## 4. Metrics and results

Cross-validation on the development part (ROC-AUC, mean ± std over 5 folds):

| Model | ROC-AUC |
|---|---|
| Majority-class baseline | 0.5000 ± 0.0000 (accuracy 0.8873) |
| LogisticRegression (C=1) | 0.7936 ± 0.0087 (AP 0.4568 ± 0.0172) |
| LinearSVC (C=0.01) | 0.7912 ± 0.0095 |
| DecisionTree (depth 8, leaf 50) | 0.7868 ± 0.0062 |
| **RandomForest (chosen)** | **0.7998 ± 0.0069** |

The RandomForest, LogisticRegression and LinearSVC differ by less than or about one fold standard deviation, so I do not claim that RandomForest is truly better; only the single tree looks clearly weaker. Class weighting did not help (0.7934 vs 0.7936). Accuracy is misleading on these data: always answering "no" already gives 0.887.

Own hold-out, scored once:

| Metric | Value |
|---|---|
| ROC-AUC | 0.7968 |
| Average precision | 0.4665 (base rate 0.113) |
| Top 10% called | 659 of 6 588 clients (cutoff 0.3451) |
| Precision@10% | 0.513 (338 buyers among 659 calls) |
| Recall@10% | 0.456 (338 of 742 buyers reached) |
| Lift over random calling | 4.55 (about 4.2 to 4.9 given sampling error) |

In plain words: about every second call ends in a sale, and the model finds about 4.5 times more buyers than random calling. Still, 404 of 742 buyers (54%) are not reached because only 10% can be called.

`predictions.csv` holds `id`, `y_pred` (1 for the 824 highest-scoring test clients = 10.0%) and `y_score` (probability of "yes") for all 8 236 test rows.

## 5. Limits and ways the model can fail

1. **Random split flatters the numbers.** Train and hold-out come from the same months and economic conditions. Real use is "train on the past, call in the future", so true performance is likely lower than 0.797 ROC-AUC and lift 4.55.
2. **Time drift.** The data are from 2008-2010. Interest rates (`euribor3m`, `nr.employed`) are among the strongest signals, but they mostly tell *when* the call was made. In another economic period these features take values the model has not seen and the ranking can degrade.
3. **Previous-contact history.** `poutcome = success` gives 64.8% "yes", but it applies to few clients; for the 96% never contacted before, the model has only weak signals.
4. **Only the top 10% was optimised.** Ranking quality below the cutoff was not checked, and the cutoff is a score quantile, not a probability, so it must be recomputed on each new list.
5. **Missing data.** "unknown" in `default` (20.9%) may mean different things in different periods; changes in how the call centre records answers would shift the scores.
6. **Small tuning search** (at most 6 random combinations per model) and one validation design; the reported scores are slightly optimistic for the chosen model.
7. **Fairness.** Age, job, marital status and education are features. A bank should check that prioritisation does not disadvantage groups before using it.

## 6. Monitoring after launch

- Share of "yes" among called clients (expected precision about 0.5) compared with the uncalled sample; if lift falls toward 1 the model is no longer useful.
- Distribution of `y_score` and of key inputs (`euribor3m`, `nr.employed`, `month`, share of "unknown") against the training data; alert when they drift.
- Count of categories unseen in training (the encoder ignores them silently).
- Refit on recent campaigns on a schedule, and validate on a time-ordered split ("train on the past, test on the following months") rather than a random one.