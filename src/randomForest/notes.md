# 🌲 Random Forest

## 1. Why Random Forest?

A Decision Tree is powerful because it can model:

- Non-linear relationships
- Feature interactions
- Numerical and categorical patterns after appropriate preprocessing
- Complex decision boundaries

But a major weakness of a Decision Tree is **high variance**.

A small change in the training data can produce:

- A different root split
- Different subsequent splits
- A substantially different tree structure
- Different predictions

### Core idea

Instead of trusting one unstable tree:

> Train many different Decision Trees and aggregate their predictions.

If the trees make sufficiently different errors, their individual fluctuations can partially cancel during aggregation.

Therefore, Random Forest mainly tries to:

> **Reduce the variance of Decision Trees while preserving their predictive strength.**

---

# 2. High-Level Architecture

```text
                    Training Dataset
                           |
            --------------------------------
            |              |               |
       Bootstrap 1    Bootstrap 2     Bootstrap N
            |              |               |
          Tree 1          Tree 2          Tree N
            |              |               |
            --------------------------------
                           |
                       Aggregate
                           |
                    Final Prediction
```

Random Forest introduces diversity through:

1. **Bootstrap sampling of rows**
2. **Random feature selection at each split**

Then it aggregates many trees.

---

# 3. Bootstrap Sampling

Suppose the original dataset contains `N` rows.

For every tree:

> Draw `N` observations randomly **with replacement**.

Because sampling is with replacement, the same observation can appear multiple times.

Example:

```text
Original dataset:

A B C D E

Bootstrap sample:

A A B D D
```

The tree trains on:

```text
A A B D D
```

Unique observations seen:

```text
A B D
```

Never seen:

```text
C E
```

Different trees receive different bootstrap samples:

```text
Tree 1 → A A B D D
Tree 2 → B C C E A
Tree 3 → D D E B E
```

This creates diversity between trees.

---

# 4. Bootstrap vs Bagging vs Random Forest

These terms are related but NOT identical.

### Bootstrap

Sampling rows **with replacement**.

```text
Original data
     ↓
Bootstrap sample
```

### Bagging

**Bootstrap Aggregating**

```text
Bootstrap samples
      +
Train multiple models
      +
Aggregate predictions
```

Conceptually:

```text
Bootstrap + Models + Aggregation
             =
           Bagging
```

### Random Forest

Random Forest is essentially:

```text
Bagged Decision Trees
        +
Random feature selection at each split
```

So:

> Bootstrap sampling alone is NOT bagging.

---

# 5. Why Bootstrap Alone Is Not Enough

Imagine one feature is extremely predictive.

Example:

```text
Sex
```

Even with different bootstrap samples:

```text
Tree 1 → Sex becomes root
Tree 2 → Sex becomes root
Tree 3 → Sex becomes root
Tree 4 → Sex becomes root
```

The trees may still develop similar structures.

Therefore their errors can remain highly correlated.

Random Forest introduces another source of randomness.

---

# 6. Random Feature Selection

Suppose the dataset contains:

```text
10 features
```

Instead of allowing all 10 features to compete at every split, Random Forest randomly chooses a subset.

Example:

```text
Node 1 candidates:
Age, Fare, Pclass

Node 2 candidates:
Sex, SibSp, Fare

Node 3 candidates:
Parch, Age, Embarked
```

The best split is chosen **only among the randomly selected candidate features**.

Important:

> The feature subset is typically selected again at every split.

It is NOT necessarily one fixed subset for the entire tree.

This encourages different trees to discover different structures.

Result:

```text
Random feature subsets
        ↓
different splits
        ↓
different trees
        ↓
lower tree correlation
        ↓
aggregation becomes more useful
```

---

# 7. Splitting Inside Each Tree

Random Forest does not invent a new tree-splitting algorithm.

Each tree still behaves like a normal Decision Tree.

For classification, a common criterion is Gini impurity.

## Gini Impurity

\[
Gini = 1-\sum_{k=1}^{K}p_k^2
\]

For binary classification:

\[
Gini = 1-(p_0^2+p_1^2)
\]

### Pure node

```text
100% Class 1
0% Class 0
```

\[
Gini=1-(1^2+0^2)=0
\]

`Gini = 0` means completely pure.

---

## Evaluating a Candidate Split

Suppose:

```text
Parent = 100 observations

       Split
      /     \
    70       30
```

Calculate Gini for both children.

Then:

\[
Gini_{split}
=
\frac{70}{100}Gini_L
+
\frac{30}{100}Gini_R
\]

Impurity reduction:

\[
\Delta Gini
=
Gini_{parent}-Gini_{split}
\]

The tree prefers splits producing greater impurity reduction.

Random Forest changes **which features are allowed to compete**, not the fundamental splitting logic.

---

# 8. Why Averaging Trees Works

Suppose individual regression trees make errors:

```text
Tree 1 → +10
Tree 2 → -8
Tree 3 → +2
```

Their errors move in different directions.

Aggregation can partially cancel them.

But suppose:

```text
Tree 1 → +10
Tree 2 → +9
Tree 3 → +11
```

All trees make similar errors.

Averaging does not help much.

Therefore:

> Random Forest needs many trees AND sufficiently low correlation between them.

---

# 9. Tree Correlation — ρ

Let:

```text
ρ = average correlation between trees
```

Intuition:

```text
ρ ≈ 0
→ trees behave differently
→ aggregation is highly useful

ρ ≈ 1
→ trees behave similarly
→ aggregation provides little variance reduction
```

Random Forest does NOT normally make trees completely independent.

They still:

- Originate from the same underlying dataset
- Solve the same target
- Use the same learning algorithm
- Often share useful features

The objective is:

> **Reduce correlation enough to make aggregation effective.**

---

# 10. Random Forest Variance Formula

A useful simplified theoretical expression is:

\[
\boxed{
Var(RF)
\approx
\rho\sigma^2
+
\frac{(1-\rho)\sigma^2}{N}
}
\]

Where:

```text
σ² = variance/instability of an individual tree
ρ  = average correlation between trees
N  = number of trees
```

Think of an individual tree's variance as having two conceptual components.

### Shared/correlated instability

\[
\rho\sigma^2
\]

This is instability that trees tend to share.

Aggregation struggles to remove it.

### Unshared instability

\[
(1-\rho)\sigma^2
\]

Averaging `N` trees reduces this component:

\[
\frac{(1-\rho)\sigma^2}{N}
\]

Therefore:

\[
Var(RF)
=
\rho\sigma^2
+
\frac{(1-\rho)\sigma^2}{N}
\]

---

## Extreme Cases

### Completely uncorrelated trees

If:

\[
\rho=0
\]

then:

\[
Var(RF)=\frac{\sigma^2}{N}
\]

Adding trees substantially reduces variance.

### Perfectly correlated trees

If:

\[
\rho=1
\]

then:

\[
Var(RF)=\sigma^2
\]

Adding more trees gives essentially no variance reduction.

---

## What Random Forest Attacks

From:

\[
Var(RF)
\approx
\rho\sigma^2
+
\frac{(1-\rho)\sigma^2}{N}
\]

Random Forest has two important strategies:

```text
N ↑
→ more trees
→ more averaging

ρ ↓
→ bootstrap rows
→ random feature selection
→ trees become less correlated
```

Even if:

\[
N\rightarrow\infty
\]

the second term approaches zero, but approximately:

\[
Var(RF)\rightarrow\rho\sigma^2
\]

Therefore:

> Adding unlimited trees cannot completely fix highly correlated trees.

This is why Random Forest needs **diversity**, not merely a large number of trees.

---

# 11. Out-of-Bag (OOB) Samples

Bootstrap sampling produces a useful side effect.

For a dataset containing `N` observations, consider one particular observation `A`.

Probability of selecting `A` in one draw:

\[
\frac{1}{N}
\]

Probability of NOT selecting `A`:

\[
1-\frac{1}{N}
\]

There are `N` bootstrap draws.

Probability that `A` is never selected:

\[
\left(1-\frac{1}{N}\right)^N
\]

For sufficiently large `N`:

\[
\left(1-\frac{1}{N}\right)^N
\approx
e^{-1}
\approx
0.368
\]

Therefore approximately:

```text
63.2% of original observations
→ appear at least once

36.8% of original observations
→ never appear
```

The observations never used by a particular tree are its:

> **Out-of-Bag samples**

Important:

`36.8% OOB` does NOT mean "36.8% are duplicates."

Duplicates exist inside the bootstrap sample.

OOB observations are original observations that were **not selected at all**.

---

# 12. OOB Prediction

Consider Passenger A.

Suppose:

```text
Tree 1 → trained on A
Tree 2 → did NOT train on A
Tree 3 → trained on A
Tree 4 → did NOT train on A
Tree 5 → did NOT train on A
```

For Passenger A's OOB evaluation:

```text
Tree 1 ❌
Tree 3 ❌

Tree 2 ✅
Tree 4 ✅
Tree 5 ✅
```

Only trees that never trained on A are allowed to contribute.

Suppose:

```text
Tree 2 → Survived
Tree 4 → Died
Tree 5 → Survived
```

Aggregate:

```text
OOB prediction → Survived
```

If:

```text
Actual A → Survived
```

the OOB prediction is correct.

Repeat for all eligible training observations and compute an evaluation metric.

---

# 13. Why OOB Is Useful

OOB asks:

> Can trees that never trained on this observation correctly predict it using patterns learned from other observations?

This approximates a generalization test without requiring a separate validation split purely for that purpose.

The OOB trees are NOT making random guesses.

They learned patterns from other training observations and apply those learned patterns to an unseen observation.

---

# 14. Wrong OOB Predictions

Suppose:

```text
OOB prediction → Died
Actual         → Survived
```

Random Forest does NOT:

- Restructure those trees
- Retrain the wrong trees
- Increase the row's importance
- Force later trees to correct the mistake

The error is simply recorded for evaluation.

This distinction becomes important when comparing **bagging vs boosting**.

Random Forest:

```text
Build trees largely independently
        ↓
Aggregate
```

Boosting:

```text
Later learners explicitly focus on
errors/residuals left by earlier learners
```

---

# 15. OOB vs Production Inference

Suppose the forest contains:

```text
500 trees
```

Passenger A was OOB for roughly some subset of those trees.

For A's OOB prediction:

```text
Only trees that did NOT train on A
→ participate
```

For a completely new production observation:

```text
NONE of the trees trained on it
        ↓
ALL 500 trees can predict
        ↓
aggregate
```

OOB is useful for model development/evaluation, but an untouched test set is still valuable for final evaluation after model-selection decisions.

---

# 16. Bias and Variance

## Bias

Bias represents systematic error caused by insufficient flexibility or incorrect assumptions.

Example:

```text
Very shallow tree
→ cannot learn complicated relationship
→ underfitting
→ high bias
```

Typical symptoms:

```text
Training performance   → poor
Validation performance → poor
Gap                    → relatively small
```

---

## Variance

Variance represents sensitivity of the learned model to changes in training data.

Deep Decision Trees can have:

```text
Training set A → Tree A
slightly different set B → very different Tree B
```

Typical symptoms:

```text
Training performance   → excellent
Validation performance → substantially worse
Gap                    → large
```

Decision Trees commonly have:

```text
Low bias
High variance
```

Random Forest mainly attacks the **variance problem**.

---

# 17. Why Random Forest Can Intentionally Weaken Individual Trees

Random feature selection may prevent a tree from using the globally strongest feature at a particular split.

Therefore an individual tree might become slightly weaker.

Why would we intentionally do this?

Because:

```text
Extremely strong but similar trees
        ↓
high correlation
        ↓
aggregation less useful
```

Whereas:

```text
Reasonably strong + diverse trees
        ↓
lower correlation
        ↓
aggregation more effective
```

The objective is NOT:

> Build the strongest possible individual tree.

It is:

> **Build many sufficiently strong trees with sufficiently different errors so that the ensemble generalizes well.**

---

# 18. Random Forest Prediction

## Regression

Each tree predicts a numerical value.

Final prediction:

\[
\boxed{
\hat y
=
\frac{1}{N}
\sum_{i=1}^{N}T_i(x)
}
\]

Example:

```text
Tree 1 → 72
Tree 2 → 80
Tree 3 → 76

Final ≈ 76
```

---

## Classification

Conceptually, classification can be understood as majority voting.

```text
Tree 1 → Survived
Tree 2 → Died
Tree 3 → Survived

Final → Survived
```

Implementations such as scikit-learn's Random Forest classifier can average class probabilities produced by individual trees and select the class with the highest mean probability.

Conceptually:

\[
P(y=k|x)
=
\frac{1}{N}
\sum_{i=1}^{N}P_i(y=k|x)
\]

Then choose the class with the largest aggregated probability.

---

# 19. Decision Threshold

For binary classification, suppose the model produces:

```text
P(Fraud) = 0.72
```

A threshold converts the score/probability into a decision.

Example:

```text
threshold = 0.50

0.72 > 0.50
→ Fraud
```

The threshold does NOT necessarily need to remain `0.5`.

It should depend on:

- Cost of false positives
- Cost of false negatives
- Business requirements
- Precision/recall trade-off

The model score and the business decision are separate concepts.

---

# 20. Important Hyperparameters

## `n_estimators`

Number of trees.

```text
n_estimators ↑
→ more trees
→ more averaging
→ predictions generally stabilize
→ training/inference cost ↑
```

Eventually performance usually plateaus.

More trees are not equivalent to deeper trees.

---

## `max_features`

Number/fraction of candidate features considered at each split.

```text
max_features ↓
→ more randomness
→ tree diversity ↑
→ correlation ρ ↓
→ individual trees may become weaker
```

Too high:

```text
strong trees
but potentially high correlation
```

Too low:

```text
high diversity
but weak individual trees
```

Goal:

> Balance tree strength and diversity.

---

## `max_depth`

Maximum depth of individual trees.

```text
max_depth ↑
→ complexity ↑
→ bias ↓
→ variance ↑
→ training fit ↑
```

```text
max_depth ↓
→ simpler trees
→ variance ↓
→ bias ↑
```

Random Forest can tolerate relatively deep trees better than a single Decision Tree because aggregation reduces variance.

---

## `min_samples_split`

Minimum number of observations required for a node to be eligible for splitting.

```text
min_samples_split ↑
→ fewer splits
→ simpler trees
→ stronger regularization
```

Question answered:

> Does this parent node contain enough observations to attempt another split?

---

## `min_samples_leaf`

Minimum number of observations allowed in a resulting leaf.

Example:

```text
Parent = 100

Split:
99 / 1
```

If:

```text
min_samples_leaf = 5
```

this split is invalid.

Increasing it:

```text
→ prevents tiny leaves
→ reduces memorization
→ reduces variance
→ can increase bias
```

Difference:

```text
min_samples_split
→ Can the parent split?

min_samples_leaf
→ Are resulting leaves large enough?
```

---

## `bootstrap`

Controls whether row sampling with replacement is used.

```text
bootstrap=True
→ standard bootstrap behavior
→ enables natural OOB observations
```

---

## `max_samples`

Controls how many training observations are drawn for each tree when bootstrapping.

Smaller values can:

```text
→ increase diversity
→ reduce training cost/tree
```

But too little data can weaken individual trees.

---

## `criterion`

Controls how candidate splits are evaluated.

Classification examples include:

```text
Gini
Entropy / log-loss based criteria
```

Regression uses regression-specific criteria such as squared-error based splitting.

---

## `class_weight`

Useful when classes are imbalanced.

Example:

```text
99% Class 0
1%  Class 1
```

Higher class weight can make errors on the minority/important class matter more during learning.

`class_weight="balanced"` can derive weights from class frequencies.

It does NOT automatically solve the entire class-imbalance problem.

---

## `oob_score`

Enables OOB-based evaluation.

It is an:

```text
evaluation mechanism
```

not a tree-training/correction mechanism.

---

## `max_leaf_nodes`

Limits the maximum number of terminal leaves.

```text
lower value
→ simpler tree

higher value
→ more complex tree
```

---

## `random_state`

Controls reproducibility of random operations such as:

- Bootstrap sampling
- Feature randomness

The value `42` has no special ML meaning.

---

## `n_jobs`

Controls computational parallelism.

Random Forest trees can largely be trained independently, making RF easy to parallelize.

`n_jobs` affects computation speed, not model theory.

---

# 21. Hyperparameter Mental Map

```text
n_estimators
→ How many trees?


max_samples
→ How much data per tree?


max_features
→ How many candidate features per split?


max_depth
→ How deep can a tree become?


min_samples_split
→ Can this node split?


min_samples_leaf
→ How small can resulting leaves become?


bootstrap
→ Should rows be sampled with replacement?


criterion
→ How do we evaluate candidate splits?


class_weight
→ How important are errors from each class?


oob_score
→ Should OOB evaluation be calculated?
```

---

# 22. Advantages of Random Forest

- Handles non-linear relationships
- Learns feature interactions automatically
- Usually much more stable than a single Decision Tree
- Requires relatively little feature scaling
- Works for classification and regression
- Robust baseline for tabular data
- Parallelizable
- OOB evaluation available with bootstrap sampling
- Less sensitive to individual noisy observations than one unrestricted tree
- Can provide feature-importance measures

---

# 23. Limitations / Failure Cases

## Model Size and Latency

Hundreds/thousands of trees can increase:

- Memory usage
- Training cost
- Inference latency
- Model size

---

## Interpretability

One Decision Tree can be visualized.

A forest containing hundreds of trees is much harder to explain globally.

---

## Extrapolation in Regression

Tree models partition observed feature space.

They generally do not naturally extrapolate trends beyond training regions the way a correctly specified parametric relationship might.

---

## Correlated Trees

If trees remain highly correlated, adding more trees gives diminishing variance reduction.

---

## High-Cardinality / Identifier Features

Columns such as:

```text
CustomerID
UUID
TransactionID
```

can encourage meaningless memorization/splits.

Feature meaning must be considered before training.

---

## Leakage

Random Forest can exploit leaked information extremely effectively.

Check for:

- Target-derived features
- Future information
- Duplicate entities across splits
- Temporal leakage
- Preprocessing leakage

Suspiciously excellent performance should trigger leakage investigation.

---

## Class Imbalance

Random Forest does not automatically solve imbalance.

Consider:

- Appropriate evaluation metrics
- Class weights
- Sampling strategies
- Threshold tuning
- Business costs of FP/FN

---

## Small Datasets

Bootstrap sampling creates different samples, not new information.

Training 10,000 trees from 50 original observations does not turn the dataset into a large dataset.

---

# 24. Random Forest — Complete Mental Model

```text
Single Decision Tree
        ↓
powerful but unstable
        ↓
high variance

Need multiple trees
        ↓
But similar trees make similar errors
        ↓
Need diversity

       ┌──────────────────┐
       │                  │
Bootstrap rows      Random features
       │                  │
       └────────┬─────────┘
                ↓
        Different trees
                ↓
       Lower correlation ρ
                ↓
         Aggregate trees
                ↓
        Variance reduction
                ↓
       Better generalization
```

---

# 📊 Classification Evaluation Metrics

# 25. Confusion Matrix

For binary classification:

```text
                         ACTUAL
                    Positive   Negative

Predicted Positive     TP         FP
Predicted Negative     FN         TN
```

### True Positive — TP

```text
Actual    = Positive
Predicted = Positive
```

Correct positive prediction.

### True Negative — TN

```text
Actual    = Negative
Predicted = Negative
```

Correct negative prediction.

### False Positive — FP

```text
Actual    = Negative
Predicted = Positive
```

Model raised a false positive alarm.

### False Negative — FN

```text
Actual    = Positive
Predicted = Negative
```

Model missed a real positive.

---

# 26. Accuracy

\[
\boxed{
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
}
\]

Meaning:

> Of all predictions, what proportion were correct?

Useful when:

- Classes are reasonably balanced
- FP and FN costs are similar

Can be misleading with severe class imbalance.

Example:

```text
99% legitimate
1% fraud

Predict everything as legitimate
→ 99% accuracy
→ 0 fraud detected
```

Always inspect class distribution before trusting accuracy.

---

# 27. Precision

\[
\boxed{
Precision=
\frac{TP}{TP+FP}
}
\]

Denominator:

```text
TP + FP
=
everything predicted positive
```

Meaning:

> Of everything the model predicted as positive, how much was actually positive?

Mental shortcut:

> **Can I trust a positive prediction?**

High precision means relatively few false positives.

Useful when false positives are expensive.

---

# 28. Recall / Sensitivity

\[
\boxed{
Recall=
\frac{TP}{TP+FN}
}
\]

Denominator:

```text
TP + FN
=
all actual positives
```

Meaning:

> Of all actual positives, how many did the model find?

Mental shortcut:

> **How much of the positive class did I catch?**

High recall means relatively few false negatives.

Useful when missing positives is expensive.

---

# 29. Precision vs Recall

Changing the classification threshold changes model behavior.

## Lower Threshold

```text
Predict positive more easily
        ↓
TP often ↑
FN often ↓
        ↓
Recall tends ↑

But:

FP may ↑
        ↓
Precision may ↓
```

## Higher Threshold

```text
Become more conservative about positive predictions
        ↓
FP often ↓
        ↓
Precision tends ↑

But:

FN may ↑
        ↓
Recall tends ↓
```

This is the **precision-recall trade-off**.

The exact behavior need not change monotonically at every tiny threshold step.

---

# 30. F1 Score

Sometimes we want one number that balances Precision and Recall.

F1 uses the harmonic mean:

\[
\boxed{
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
}
\]

Why harmonic mean?

It penalizes cases where one metric is high and the other is very low.

Example:

```text
Precision = 0.90
Recall    = 0.20
```

Arithmetic mean:

```text
0.55
```

F1:

```text
≈ 0.327
```

So excellent precision cannot completely hide terrible recall.

F1 is useful when:

- Both Precision and Recall matter
- Class imbalance makes accuracy insufficient
- A single summary score is useful

But F1 is NOT automatically the correct metric for every imbalanced problem.

If FN is dramatically more costly than FP, Recall may matter more.

If FP is dramatically more costly, Precision may matter more.

---

# 31. Metric Selection Is a Problem Decision

Do NOT choose a metric because of the algorithm.

Random Forest does not imply:

```text
use Accuracy
```

Logistic Regression does not imply:

```text
use F1
```

Metrics depend on:

```text
Class distribution
Business objective
Cost of FP
Cost of FN
Whether ranking matters
Whether probability quality matters
```

Examples:

```text
Balanced classification
→ Accuracy may be informative

Fraud detection
→ Precision / Recall / F1 / PR-oriented metrics
→ threshold based on business cost

Disease screening
→ Recall may be particularly important
→ while still monitoring false positives

Spam filtering
→ Precision may be important if false positives are costly
```

---

# 32. Training vs Validation Diagnosis

Metrics also help diagnose bias and variance.

### Possible High Variance

```text
Training performance   → excellent
Validation/OOB         → substantially worse
```

Large generalization gap.

Investigate:

- Tree complexity
- Noise
- Data size
- Leakage
- Validation strategy
- `max_depth`
- `min_samples_leaf`
- `max_features`

---

### Possible High Bias

```text
Training performance   → poor
Validation performance → poor
```

Both may be relatively close.

Investigate:

- Model capacity
- Feature quality
- Excessive regularization
- Missing predictive information
- Alternative models

---

# 33. Current Metric Map

```text
CONFUSION MATRIX
       ↓
TP / TN / FP / FN
       │
       ├── Accuracy
       │   Overall correctness
       │
       ├── Precision
       │   Trustworthiness of positive predictions
       │
       ├── Recall
       │   Coverage of actual positives
       │
       └── F1
           Balance between Precision and Recall
```

Additional metrics still worth learning separately:

```text
ROC Curve
ROC-AUC
Precision-Recall Curve
PR-AUC / Average Precision
Log Loss
Probability Calibration
Regression metrics:
MAE / MSE / RMSE / R²
```

---

# 🧠 Final Random Forest Summary

Random Forest solves a major weakness of Decision Trees:

> **High variance and instability.**

It builds many Decision Trees using:

```text
Bootstrap sampling of observations
+
Random feature candidates at each split
```

This creates sufficiently diverse trees.

Then:

```text
Classification → aggregate class evidence/probabilities
Regression     → average numerical predictions
```

The theoretical intuition is:

\[
Var(RF)
\approx
\rho\sigma^2+
\frac{(1-\rho)\sigma^2}{N}
\]

Therefore Random Forest benefits from:

```text
More useful trees
        +
Lower correlation between trees
        +
Reasonably strong individual trees
```

The goal is NOT:

> Make every tree perfect.

The goal is:

> **Build many useful, sufficiently decorrelated trees whose aggregation generalizes better than relying on one unstable Decision Tree.**

# Advanced Random Forest — Practical Notes

## Feature Importance

Random Forest provides feature importance to understand which features the fitted model relied on.

### 1. Mean Decrease in Impurity (MDI)

Available in scikit-learn through:

```python
rf.feature_importances_
```

A feature is considered important when its splits produce large reductions in impurity across the forest.

For classification:

\[
Gini = 1-\sum p_k^2
\]

Impurity reduction:

\[
\Delta Gini =
Gini_{parent} - Gini_{children}^{weighted}
\]

Splits affecting more samples contribute more importance.

### Problems with MDI

**High-cardinality features**

Features with many possible split points can receive inflated importance because the tree has more opportunities to find useful-looking splits.

Examples:

```text
PassengerId
CustomerID
continuous/high-cardinality variables
```

**Correlated features**

If two features contain similar information:

```text
MonthlySalary
AnnualSalary
```

the model may use one instead of the other, causing importance to be divided or concentrated unpredictably.

Therefore:

> MDI describes how the fitted forest used features; it is not an objective measurement of real-world importance.

---

## 2. Permutation Importance

Instead of examining tree splits, permutation importance asks:

> How much does model performance deteriorate when the information in one feature is destroyed?

Procedure:

```text
1. Calculate validation score
2. Shuffle one feature
3. Predict again
4. Measure performance drop
5. Repeat for other features
```

Conceptually:

\[
Importance_j =
Score_{baseline} - Score_{permuted(j)}
\]

Example:

```text
Baseline accuracy        = 90%
Shuffle Age              = 87%
Importance of Age        ≈ 3 percentage-point drop

Shuffle PassengerId      = 90%
Importance               ≈ 0
```

A large performance drop suggests the fitted model depends strongly on that feature.

### Advantages

- Model-agnostic
- Directly connected to predictive performance
- Can be measured on validation/test data
- Avoids some problems of impurity-based importance

### Limitation: Correlated Features

Suppose:

```text
AnnualSalary
MonthlySalary
```

contain nearly identical information.

Shuffle `AnnualSalary`:

```text
Model can still use MonthlySalary
→ little performance drop
```

Shuffle `MonthlySalary`:

```text
Model can still use AnnualSalary
→ little performance drop
```

Both may appear individually unimportant even though the underlying salary information is highly predictive.

---

## MDI vs Permutation Importance

| MDI | Permutation |
|---|---|
| Based on impurity reduction | Based on performance degradation |
| Tree-specific | Model-agnostic |
| Very fast | More computationally expensive |
| Calculated from fitted tree structure | Preferably evaluated on held-out data |
| Can favor high-cardinality features | Less affected by split-count opportunity |
| Correlated features can distort importance | Correlated features can hide importance |

---

## Feature Importance ≠ Causality

A feature being important means:

> The fitted model found the feature useful for prediction given the available data/features.

It does NOT mean:

\[
X \rightarrow Y
\]

or prove that the feature causes the target.

---

# Practical Random Forest Tuning Strategy

Do not blindly tune every hyperparameter.

First compare:

```text
Training performance
vs
Validation / OOB performance
```

### High Variance / Overfitting

```text
Training score   → very high
Validation score → significantly lower
```

Possible changes:

```text
max_depth ↓
min_samples_leaf ↑
min_samples_split ↑
max_features ↓ / tune
max_samples ↓ / tune
```

Also investigate:

- Noise
- Leakage
- Insufficient data
- Distribution differences

### High Bias / Underfitting

```text
Training score   → poor
Validation score → similarly poor
```

Possible changes:

```text
max_depth ↑
min_samples_leaf ↓
min_samples_split ↓
```

Also investigate:

- Weak/missing features
- Excessive regularization
- Feature engineering
- Whether RF is appropriate for the problem

### `n_estimators`

Increase trees until validation/OOB performance becomes stable.

More trees generally:

```text
Variance stability ↑
Training cost ↑
Inference cost ↑
Memory ↑
```

Eventually there are diminishing returns.

---

# Hyperparameter Search

## Grid Search

Tests predefined combinations exhaustively.

Useful when:

- Search space is small
- Important parameter ranges are already known

Problem:

```text
Many parameters × many values
→ combinations explode
```

## Randomized Search

Samples random hyperparameter combinations.

Useful when:

- Search space is large
- Some parameters matter much more than others
- Compute budget is limited

For large search spaces, Randomized Search is often a better starting point than exhaustive Grid Search.

Use cross-validation where appropriate rather than selecting parameters from test-set performance.

The test set should remain untouched until final evaluation.

---

# Random Forest Preprocessing

## Feature Scaling

Usually NOT required.

Decision Trees make rules such as:

```text
Age < 30
Salary > 50000
```

Changing scale:

```text
Salary = 50000
→ standardized Salary = 0.72
```

does not fundamentally change the ordering used by threshold splits.

Therefore RF generally does not require:

```text
StandardScaler
MinMaxScaler
```

unlike distance/gradient-sensitive models where scaling may matter substantially.

---

## Categorical Features

Handling depends on the implementation.

With scikit-learn workflows, categorical variables are commonly encoded before fitting Random Forest.

Possible techniques include:

```text
One-Hot Encoding
Ordinal Encoding — only when appropriate
```

Avoid accidentally introducing fake ordinal meaning into nominal categories.

---

## High-Cardinality Features

Be careful with:

```text
CustomerID
PassengerId
UUID
TransactionID
```

They may encourage memorization or meaningless splits.

Ask:

> Does this feature contain generalizable predictive information?

---

# Production Considerations

## Model Size

More trees mean:

```text
Memory ↑
Model artifact size ↑
```

A forest with thousands of deep trees can become large.

## Inference Latency

Prediction requires traversing many trees:

```text
New observation
      ↓
Tree 1
Tree 2
...
Tree N
      ↓
Aggregate
```

Therefore:

```text
n_estimators ↑
→ inference cost ↑
```

Balance accuracy/stability against latency requirements.

## Parallelism

Trees can largely be trained independently.

In scikit-learn:

```python
n_jobs=-1
```

can use available CPU cores for parallel computation.

## Reproducibility

Use:

```python
random_state=42
```

or another fixed seed for reproducible experiments.

The value `42` itself has no ML significance.

## Data Drift

Production data can change over time.

Monitor:

```text
Input feature distributions
Prediction distributions
Performance when labels become available
Class balance
Missing-value patterns
```

A strong historical validation score does not guarantee permanent production performance.

---

# Interpretability

### Global Feature Importance

Answers:

> Which features does the model generally rely on?

Tools:

```text
MDI
Permutation Importance
```

### Local Explanation

Answers:

> Why did the model make THIS particular prediction?

Tools such as:

```text
SHAP
```

can be used for local/global model explanations.

SHAP should be treated as an explanation of model behavior, NOT proof of causal relationships.

Partial Dependence Plots (PDP) can help inspect how predictions change as a feature changes on average, but correlated features can make interpretation unreliable.

---

# Random Forest — 2+ Year ML Engineer Checklist

You should be comfortable explaining:

- Why a single Decision Tree has high variance
- Bootstrap sampling
- Bagging
- Random feature selection
- Why tree correlation matters
- Gini/impurity-based splitting
- Classification vs regression aggregation
- OOB samples and OOB evaluation
- Why approximately 36.8% of observations are OOB
- Bias vs variance
- Forest variance intuition:

\[
Var(RF)
\approx
\rho\sigma^2+
\frac{(1-\rho)\sigma^2}{N}
\]

- Important hyperparameters and their trade-offs
- Train vs validation/OOB diagnosis
- MDI feature importance
- Permutation importance
- Correlated-feature importance problems
- Why feature importance does not imply causality
- Why scaling is generally unnecessary
- Class imbalance handling
- Leakage risks
- Model size and inference latency
- Reproducibility and data drift

At this point, Random Forest is sufficiently covered for a practical ~2-year ML-engineer level.