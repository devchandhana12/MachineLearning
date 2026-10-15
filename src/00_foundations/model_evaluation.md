# Machine Learning — Complete Evaluation, Validation & Testing Notes

**Level:** ML Engineer — 2 Years Experience  
**Classification Example:** Titanic Survival Prediction  
**Regression Example:** House Price Prediction

---

# PART 1 — UNDERSTANDING MODEL EVALUATION

## 1. Evaluation Metrics vs Validation Metrics vs Test Metrics

These aren't fundamentally different mathematical categories.

**Evaluation metrics** measure model performance.

Examples:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- MAE
- MSE
- RMSE
- R²

The names **training metric**, **validation metric**, and **test metric** describe the dataset on which a metric is calculated.

| Dataset    | Purpose                                                      |
| ---------- | ------------------------------------------------------------ |
| Training   | Learn model parameters                                       |
| Validation | Tune hyperparameters, select models, and monitor overfitting |
| Test       | Estimate performance on unseen data after model selection    |

Example:

Training accuracy = 95%

Validation accuracy = 83%

Test accuracy = 81%

All three use the same accuracy formula.

**Important:** Don't repeatedly tune models using the test set. Doing so leaks test information into model selection and makes the final performance estimate less reliable.

---

# PART 2 — CLASSIFICATION EVALUATION METRICS

## 2. Confusion Matrix

A confusion matrix describes how predicted classes compare with actual classes.

Titanic example:

Positive class (1) = Survived.

Negative class (0) = Did not survive.

| Actual / Predicted | Survived (1) | Died (0) |
| ------------------ | ------------ | -------- |
| Survived (1)       | TP = 30      | FN = 10  |
| Died (0)           | FP = 10      | TN = 50  |

Definitions:

**True Positive (TP):** Predicted survived, actually survived.

**True Negative (TN):** Predicted died, actually died.

**False Positive (FP):** Predicted survived, actually died.

**False Negative (FN):** Predicted died, actually survived.

Total samples:

N = TP + TN + FP + FN = 100

Remember:

- True/False = Whether prediction is correct.
- Positive/Negative = Which class the model predicted.

---

## 3. Accuracy

Accuracy measures the fraction of predictions that are correct.

**Formula:**

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Titanic example:

Accuracy = (30 + 50) / 100

**Accuracy = 80%**

### When to use

Useful when:

- Class distribution is reasonably balanced.
- Different classification mistakes have similar costs.

### Limitation

Accuracy can be misleading on imbalanced datasets.

Example:

99% genuine transactions and 1% fraud.

A model predicting every transaction as genuine achieves 99% accuracy but detects zero fraud.

**Interview point:** High accuracy doesn't guarantee good minority-class detection.

---

## 4. Error Rate

Error Rate measures the fraction of incorrect predictions.

**Formula:**

Error Rate = (FP + FN) / N

Alternatively:

Error Rate = 1 - Accuracy

Titanic:

Error Rate = (10 + 10) / 100

**Error Rate = 20%**

---

## 5. Precision

Precision measures the correctness of positive predictions.

**Formula:**

Precision = TP / (TP + FP)

Titanic:

Precision = 30 / (30 + 10)

**Precision = 75%**

Interpretation:

Among passengers predicted to survive, 75% actually survived.

### When to prioritize Precision

When False Positives are particularly costly.

Examples:

- Automatically deleting spam emails.
- Flagging legitimate transactions as fraudulent.
- Taking expensive action based on a positive prediction.

**Memory trick:** When the model says YES, how often is it correct?

---

## 6. Recall (Sensitivity / True Positive Rate)

Recall measures how many actual positive cases were detected.

**Formula:**

Recall = TP / (TP + FN)

Titanic:

Recall = 30 / (30 + 10)

**Recall = 75%**

Interpretation:

The model identified 75% of actual survivors.

### When to prioritize Recall

When False Negatives are particularly costly.

Examples:

- Disease screening.
- Fraud detection when missed fraud is very costly.
- Safety monitoring systems.

**Memory trick:** Out of all actual YES cases, how many did we find?

---

## 7. Specificity (True Negative Rate)

Specificity measures how many actual negative cases were correctly identified.

**Formula:**

Specificity = TN / (TN + FP)

Titanic:

Specificity = 50 / (50 + 10)

**Specificity = 83.33%**

Interpretation:

The model correctly identified 83.33% of passengers who did not survive.

### Recall vs Specificity

Recall focuses on detecting positive cases.

Specificity focuses on correctly identifying negative cases.

Both are particularly relevant in screening, detection, and diagnostic applications.

---

## 8. False Positive Rate (FPR)

FPR measures the fraction of actual negative cases incorrectly classified as positive.

**Formula:**

FPR = FP / (FP + TN)

Alternatively:

FPR = 1 - Specificity

Titanic:

FPR = 10 / 60

**FPR = 16.67%**

FPR is important when building ROC curves.

---

## 9. False Negative Rate (FNR)

FNR measures the fraction of actual positive cases incorrectly classified as negative.

**Formula:**

FNR = FN / (FN + TP)

Alternatively:

FNR = 1 - Recall

Titanic:

FNR = 10 / 40

**FNR = 25%**

---

## 10. F1-Score

F1 is the harmonic mean of Precision and Recall.

**Formula:**

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Alternative:

F1 = 2TP / (2TP + FP + FN)

Titanic:

Precision = 0.75

Recall = 0.75

F1 = 2 × (0.75 × 0.75) / 1.5

**F1 = 0.75**

### Why harmonic mean?

F1 penalizes situations in which one metric is much smaller than the other.

Example:

Model A:

- Precision = 100%
- Recall = 20%
- F1 = 33.33%

Model B:

- Precision = 80%
- Recall = 80%
- F1 = 80%

Model B has the better F1-score.

### When to use

Useful when Precision and Recall are both important.

Limitation:

F1 doesn't include True Negatives directly and doesn't represent the actual monetary or operational costs of mistakes.

---

## 11. F-Beta Score

F-beta generalizes F1 by allowing us to assign different importance to Precision and Recall.

**Formula:**

Fβ = (1 + β²) × (P × R) / (β²P + R)

Where:

- β = 1: Equal emphasis on Precision and Recall.
- β > 1: Greater emphasis on Recall.
- β < 1: Greater emphasis on Precision.

Common scores:

**F1:** Balanced emphasis.

**F2:** Greater importance to Recall.

**F0.5:** Greater importance to Precision.

### Example

For disease screening, F2 may be appropriate when detecting actual disease cases matters more than avoiding false alarms.

For automated spam deletion, F0.5 may be useful when false positives are particularly expensive.

**Interview point:** Choose beta based on the relative consequences of false positives and false negatives.

---

# PART 3 — CLASSIFICATION THRESHOLDS

## 12. Classification Threshold

Binary classification models often generate probabilities.

Example:

Passenger survival probability = 0.72

At threshold = 0.50:

Predict Survived (1).

At threshold = 0.80:

Predict Died (0).

### Lowering the threshold

Typically:

- More positive predictions.
- Recall increases or stays unchanged.
- False Positives may increase.
- Precision can decrease, but this isn't guaranteed.

### Raising the threshold

Typically:

- Fewer positive predictions.
- Recall decreases or stays unchanged.
- False Positives decrease or stay unchanged.
- Precision may improve, but this isn't guaranteed.

Changing the threshold does not require retraining the model.

### Interview scenario

A fraud model has excellent Precision but poor Recall.

Possible solution:

Lower the decision threshold to detect more fraudulent transactions, while checking how many additional false alarms are created.

Choose the threshold using validation data, not the final test set.

---

# PART 4 — ROC, AUC AND PR CURVES

## 13. ROC Curve

ROC stands for Receiver Operating Characteristic.

It plots:

**Y-axis:** True Positive Rate (Recall)

**X-axis:** False Positive Rate

Each point represents the classifier's performance at a particular decision threshold.

### Interpretation

A model with strong discrimination tends to achieve:

- High True Positive Rate.
- Low False Positive Rate.

The ideal upper-left point is:

TPR = 1

FPR = 0

### Why ROC is useful

It shows how detecting positive cases trades off against false alarms as the decision threshold changes.

---

## 14. ROC-AUC

AUC stands for Area Under the Curve.

ROC-AUC summarizes the ROC curve into a single number.

Typical interpretation:

| ROC-AUC   | Meaning                               |
| --------- | ------------------------------------- |
| 1.0       | Perfect ranking                       |
| 0.5       | Random-ranking performance            |
| Below 0.5 | Worse-than-random ranking orientation |

Higher ROC-AUC generally indicates better discrimination between positive and negative cases.

A probabilistic interpretation:

ROC-AUC estimates the probability that a randomly selected positive instance receives a higher score than a randomly selected negative instance, with half credit for ties.

### Important

ROC-AUC measures ranking performance across thresholds.

It does not directly tell us:

- The best classification threshold.
- Whether predicted probabilities are calibrated.
- The business cost of errors.

---

## 15. Precision-Recall Curve

A PR curve plots:

**Y-axis:** Precision

**X-axis:** Recall

Each point corresponds to a decision threshold.

### When useful

PR curves are particularly informative when the positive class is rare and positive detection is the main concern.

Examples:

- Fraud detection.
- Rare disease detection.
- Equipment failure prediction.

### ROC-AUC vs PR-AUC

ROC-AUC summarizes discrimination using TPR and FPR.

PR-AUC summarizes the relationship between Precision and Recall.

For highly imbalanced datasets, PR curves often reveal positive-class performance more clearly.

**Important distinction:** Average Precision (AP) and trapezoidal PR-AUC are related but not mathematically identical.

---

# PART 5 — PROBABILITY EVALUATION METRICS

## 16. Log Loss (Binary Cross-Entropy)

Log Loss evaluates how well predicted probabilities agree with actual labels.

Unlike Accuracy, it considers prediction confidence.

**Formula:**

Log Loss = -1/N × Σ[y log(p) + (1-y) log(1-p)]

Where:

- y = Actual binary label.
- p = Predicted probability of positive class.

### Example

Passenger actually survived (y = 1).

Model A predicts survival probability = 0.90.

Model B predicts survival probability = 0.55.

Both produce the correct class at threshold 0.5.

But Model A receives a lower Log Loss for this observation because it assigned higher probability to the correct outcome.

Conversely, confidently wrong predictions receive large penalties.

### Key Takeaways

- Lower Log Loss is better.
- It evaluates probability predictions, not just class labels.
- It penalizes confident mistakes heavily.
- Commonly used for Logistic Regression and neural-network classification.

---

## 17. Brier Score

Brier Score measures the mean squared difference between predicted probabilities and actual binary outcomes.

**Formula:**

Brier Score = 1/N × Σ(p - y)²

Example:

Actual survival = 1.

Predicted probability = 0.8.

Squared error:

(0.8 - 1)² = 0.04

Lower Brier Score is better.

Unlike Accuracy, Brier Score evaluates the probability itself.

### Log Loss vs Brier Score

Both evaluate probabilistic predictions.

Log Loss heavily penalizes predictions that are confidently wrong.

Brier Score applies a quadratic penalty.

Both reflect probability quality, incorporating calibration and discrimination.

---

## 18. Probability Calibration

Calibration checks whether predicted probabilities match observed frequencies.

Example:

Among passengers assigned approximately 70% survival probability, roughly 70% should actually survive if the probabilities are well calibrated.

A model can have excellent ROC-AUC but poor calibration.

### Common techniques

- Reliability diagrams.
- Calibration curves.
- Sigmoid calibration.
- Isotonic regression.

Calibration is especially important when probabilities drive decisions or risk estimates.

---

# PART 6 — MULTICLASS CLASSIFICATION METRICS

## 19. Macro, Micro and Weighted Averaging

For multiclass classification, Precision, Recall, and F1 can be computed separately for each class.

We then combine these class-specific scores.

### Macro Average

Calculate the metric for each class, then take their simple average.

Every class receives equal importance.

Useful when minority-class performance matters.

### Micro Average

Aggregate TP, FP, and FN across classes before calculating the metric.

Every individual prediction contributes to the total.

For ordinary single-label multiclass classification:

Micro Precision = Micro Recall = Micro F1 = Accuracy.

### Weighted Average

Calculate each class's score, then average scores weighted by class support (number of actual observations per class).

Larger classes contribute more.

### Interview point

Macro F1 gives minority classes equal weight.

Weighted F1 reflects class frequencies.

Micro F1 reflects aggregate performance across observations.

---

## 20. Balanced Accuracy

Balanced Accuracy addresses unequal class distributions.

For binary classification:

**Balanced Accuracy = (Recall + Specificity) / 2**

Titanic:

Recall = 75%

Specificity = 83.33%

Balanced Accuracy:

(75 + 83.33) / 2

**Balanced Accuracy ≈ 79.17%**

Useful when classes are imbalanced.

For multiclass classification, balanced accuracy is the unweighted average of per-class recall.

---

## 21. Matthews Correlation Coefficient (MCC)

MCC summarizes binary classification performance using all four confusion matrix values.

**Formula:**

MCC = (TP × TN - FP × FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

### Range

- +1: Perfect prediction.
- 0: No correlation between predictions and outcomes.
- -1: Perfect inverse prediction.

MCC can be useful for imbalanced classification because it incorporates TP, TN, FP and FN.

---

# PART 7 — REGRESSION EVALUATION METRICS

Use house-price prediction as our example.

Suppose:

| Actual price | Predicted price | Absolute error |
| ------------ | --------------- | -------------- |
| 100          | 90              | 10             |
| 150          | 160             | 10             |
| 200          | 170             | 30             |

All prices are in ₹ lakhs.

---

## 22. MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Formula:**

MAE = (1/N) × Σ|y - ŷ|

Example:

MAE = (10 + 10 + 30) / 3

**MAE ≈ 16.67 lakhs**

### Advantages

- Easy to interpret.
- Same units as the target.
- Less sensitive to large errors than MSE.

### Limitation

It treats errors linearly rather than disproportionately penalizing large mistakes.

---

## 23. MSE — Mean Squared Error

Measures average squared prediction error.

**Formula:**

MSE = (1/N) × Σ(y - ŷ)²

Example:

MSE = (10² + 10² + 30²) / 3

MSE = (100 + 100 + 900) / 3

**MSE ≈ 366.67 (lakhs)²**

### Why square errors?

- Prevents positive and negative errors from canceling.
- Penalizes large errors more strongly.
- Provides a smooth, differentiable objective useful in optimization.

### Limitation

Sensitive to outliers.

Units are squared, making interpretation less intuitive.

---

## 24. RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

**Formula:**

RMSE = √MSE

Example:

RMSE = √366.67

**RMSE ≈ 19.15 lakhs**

### Why use RMSE?

- Penalizes large errors more heavily than MAE.
- Returns error to the target's original units.
- Common regression evaluation metric.

### MAE vs RMSE

MAE = 16.67 lakhs.

RMSE = 19.15 lakhs.

A large difference between RMSE and MAE can indicate the presence of relatively large errors.

---

## 25. R² — Coefficient of Determination

R² evaluates performance relative to a baseline that predicts the mean target value.

**Formula:**

R² = 1 - [Σ(y - ŷ)² / Σ(y - ȳ)²]

Where:

ȳ = Mean actual target value.

### Interpretation

R² = 1:

Perfect predictions.

R² = 0:

Same total squared error as predicting the target mean.

R² below 0:

Worse than predicting the target mean.

### Important

Negative R² is possible.

A high R² does not automatically mean the model has acceptably small errors.

Always interpret R² with domain requirements and an error metric such as MAE or RMSE.

---

## 26. Adjusted R²

Adjusted R² adjusts the ordinary R² score based on the number of predictors.

**Formula:**

Adjusted R² = 1 - (1-R²)(N-1)/(N-p-1)

Where:

- N = Number of observations.
- p = Number of predictors.

### Why use it?

Ordinary training R² cannot decrease when predictors are added to an unregularized OLS linear regression model.

Adjusted R² penalizes the inclusion of unnecessary predictors.

It is primarily useful for comparing conventional linear regression models with different predictor counts under suitable assumptions.

It is not a universal metric for evaluating arbitrary ML algorithms.

---

## 27. MAPE — Mean Absolute Percentage Error

Measures absolute prediction error relative to the actual value.

**Formula:**

MAPE = (100/N) × Σ| (y - ŷ) / y |

### Advantage

Expresses error as a percentage.

### Limitations

- Undefined when actual values equal zero.
- Unstable near zero.
- Can behave asymmetrically for overprediction and underprediction.

Use with caution.

---

## 28. Median Absolute Error

Median Absolute Error is the median of all absolute prediction errors.

**Formula:**

MedianAE = median(|y - ŷ|)

Useful when we want a regression metric less influenced by a small number of extreme errors.

---

# PART 8 — VALIDATION TECHNIQUES

## 29. Train / Validation / Test Split

A common starting arrangement is:

- 70% Training.
- 15% Validation.
- 15% Testing.

These proportions are examples, not universal rules.

### Training Set

Used to learn model parameters.

### Validation Set

Used for:

- Hyperparameter tuning.
- Model selection.
- Early stopping.
- Classification threshold selection.

### Test Set

Used for final performance estimation.

**Important:** Fit preprocessing transformations only on the training data, then apply them to validation/test data.

Use pipelines to reduce data leakage.

---

## 30. K-Fold Cross-Validation

K-Fold divides training data into K subsets.

For each iteration:

1. Train using K-1 folds.
2. Validate using the remaining fold.
3. Repeat until each fold has been used for validation.
4. Aggregate validation scores.

Example:

K = 5.

We obtain five validation scores.

Final CV score = Mean of those five scores.

### Advantages

- Makes more complete use of limited data.
- Helps estimate variability across splits.
- Useful for comparing hyperparameters.

### Limitations

- More computationally expensive.
- Random K-Fold may be inappropriate for time-series or grouped data.

---

## 31. Stratified K-Fold

Stratified K-Fold approximately preserves class proportions across folds.

Example:

Titanic training data:

- 60% did not survive.
- 40% survived.

Each fold maintains approximately the same distribution.

Useful especially for imbalanced classification.

---

## 32. Group K-Fold

Used when samples belong to related groups.

Examples:

- Multiple medical records from the same patient.
- Multiple images from one subject.
- Repeated measurements from the same machine.

It ensures a group doesn't appear in both training and validation portions of a fold.

This reduces group-based leakage.

---

## 33. Time-Series Validation

Ordinary random splitting may leak future information into training.

For time-series data, use chronological splitting.

Example:

Train: January–March.

Validate: April.

Test: May.

In rolling or expanding-window validation, training uses past observations and validation uses later observations.

Avoid using future information that wouldn't exist at prediction time.

---

## 34. Nested Cross-Validation

Nested CV separates two processes:

**Inner cross-validation:** Hyperparameter tuning.

**Outer cross-validation:** Estimating generalization performance.

Useful when:

- Datasets are small.
- Model selection is extensive.
- A less biased estimate of the model-selection procedure is required.

It is computationally expensive.

---

# PART 9 — MODEL DIAGNOSTICS

## 35. Overfitting

The model learns patterns specific to training data that fail to generalize.

Example:

Training accuracy = 98%.

Validation accuracy = 70%.

Possible solutions:

- Reduce model complexity.
- Add regularization.
- Use early stopping.
- Collect more representative data.
- Reduce leakage and investigate unstable features.

---

## 36. Underfitting

The model fails to learn enough useful relationships.

Example:

Training accuracy = 60%.

Validation accuracy = 59%.

Assuming those results are poor relative to the problem's baseline, underfitting may be present.

Possible solutions:

- Increase model capacity.
- Reduce excessive regularization.
- Improve features.
- Train sufficiently.
- Test more suitable algorithms.

**Important:** A small training-validation gap alone does not prove underfitting.

---

## 37. Bias-Variance Tradeoff

**High Bias:**

Model assumptions are too restrictive.

Often associated with underfitting.

**High Variance:**

Model is overly sensitive to training-set details.

Often associated with overfitting.

Goal:

Balance sufficient model capacity with strong generalization.

---

## 38. Learning Curves

Learning curves can plot training and validation performance against:

- Training set size.
- Training epochs.
- Boosting rounds.

### Interpretation

Both poor and similar:

Possible underfitting.

Training strong, validation much worse:

Possible overfitting.

Both strong and close:

Potentially good generalization.

Learning curves help diagnose whether additional data, regularization, or capacity changes may be useful.

---

## 39. Early Stopping

Early stopping stops training when validation performance fails to improve sufficiently.

Example:

An XGBoost model trains for up to 500 boosting rounds.

Validation Log Loss stops improving after round 120.

With suitable early stopping settings, training stops instead of continuing through all 500 rounds.

### Why useful?

- Limits unnecessary training.
- Can reduce overfitting.
- Helps select boosting rounds or epochs.

The best stopping criterion should match the task and evaluation priorities.

---

## 40. Data Leakage

Data leakage occurs when model development uses information that would not legitimately be available when making predictions.

Examples:

- Fitting a scaler on the entire dataset before splitting.
- Imputing missing values using information from the test set.
- Including target-derived features.
- Allowing future information into time-series training.
- Repeatedly choosing models based on test performance.

Data leakage can produce misleadingly high validation or test scores.

**Interview point:** A suspiciously high score should trigger checks for leakage before celebrating model performance.

---

# PART 10 — PYTHON IMPLEMENTATION

## 41. Classification Metrics with Scikit-Learn

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss
)

# True classes
y_true = [1, 1, 0, 0, 1, 0]

# Predicted probabilities
y_prob = [0.9, 0.8, 0.6, 0.2, 0.7, 0.1]

# Apply classification threshold
threshold = 0.5
y_pred = [int(p >= threshold) for p in y_prob]

print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("F2:", fbeta_score(y_true, y_pred, beta=2))
print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
print("MCC:", matthews_corrcoef(y_true, y_pred))

# Probability-based metrics
print("ROC-AUC:", roc_auc_score(y_true, y_prob))
print("Average Precision:", average_precision_score(y_true, y_prob))
print("Log Loss:", log_loss(y_true, y_prob))
print("Brier Score:", brier_score_loss(y_true, y_prob))
```

### Important

Classification labels are typically used for:

- Accuracy.
- Precision.
- Recall.
- F1.
- Confusion Matrix.

Probability or continuous decision scores are used for:

- ROC-AUC.
- Average Precision.
- ROC and PR curves.

Predicted probabilities are used for:

- Log Loss.
- Brier Score.
- Probability calibration.

---

## 42. Regression Metrics with Scikit-Learn

```python
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)

y_true = np.array([100, 150, 200])
y_pred = np.array([90, 160, 170])

print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", root_mean_squared_error(y_true, y_pred))
print("R2:", r2_score(y_true, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
print("Median AE:", median_absolute_error(y_true, y_pred))
```

---

# PART 11 — METRIC SELECTION FOR REAL-WORLD PROJECTS

| Problem                | Useful metrics                                | Why                                               |
| ---------------------- | --------------------------------------------- | ------------------------------------------------- |
| Titanic survival       | Accuracy, F1, Recall, ROC-AUC                 | Understand overall and positive-class performance |
| Fraud detection        | Precision, Recall, PR-AUC, cost-based metrics | Rare positives and costly mistakes                |
| Disease screening      | Recall, Specificity, ROC-AUC, calibration     | Missed cases and false alarms matter              |
| Spam classification    | Precision, Recall, F1                         | False positives and missed spam                   |
| House-price prediction | MAE, RMSE, R²                                 | Prediction error and baseline comparison          |
| Sales forecasting      | MAE, RMSE, WAPE where appropriate             | Forecast errors and business scale                |
| Credit-risk prediction | ROC-AUC, PR-AUC, Log Loss, calibration        | Ranking and probability reliability               |
| Image classification   | Accuracy, Macro F1, per-class Recall          | Class performance and imbalance                   |

No single metric is best for every project.

Choose metrics based on:

1. The prediction problem.
2. The class distribution.
3. The real-world cost of mistakes.
4. Whether class labels, rankings, or calibrated probabilities are needed.

---

# PART 12 — INTERVIEW QUESTIONS AND ANSWERS

**Q1. Can a model have 99% accuracy and still be bad?**

Yes. In an imbalanced dataset, a model can predict only the majority class and achieve high accuracy while completely missing the minority class.

**Q2. Precision vs Recall?**

Precision measures how many predicted positives are correct.

Recall measures how many actual positives are detected.

**Q3. When would you prioritize Recall?**

When missing positive cases is expensive or dangerous, such as in certain screening or fraud-detection tasks.

**Q4. When would you prioritize Precision?**

When false-positive predictions lead to costly or harmful actions.

**Q5. Why F1 instead of Accuracy?**

F1 balances Precision and Recall and can be more informative when positive-class detection matters in imbalanced datasets.

**Q6. Why use ROC-AUC?**

To measure how well a model ranks positive cases above negative cases across classification thresholds.

**Q7. ROC-AUC vs PR-AUC?**

ROC-AUC summarizes TPR versus FPR.

PR-AUC summarizes Precision versus Recall and is often especially informative when the positive class is rare.

**Q8. Does changing threshold require retraining?**

No. Threshold tuning changes how scores or probabilities are converted to classes.

**Q9. MAE vs MSE?**

MAE penalizes errors linearly.

MSE squares errors, penalizing large mistakes more strongly.

**Q10. Why RMSE instead of MSE?**

RMSE returns errors to the original target units, making interpretation easier.

**Q11. Can R² be negative?**

Yes. Negative R² indicates worse squared-error performance than predicting the mean target value on the evaluation dataset.

**Q12. What is data leakage?**

Using information during model development that would not legitimately be available when making predictions.

**Q13. Why use Stratified K-Fold?**

To preserve approximately the same class proportions across classification folds.

**Q14. Why shouldn't we tune on the test dataset?**

Because tuning on test results biases the final performance estimate.

**Q15. Can high ROC-AUC coexist with poor probability calibration?**

Yes. ROC-AUC evaluates ranking, while calibration evaluates whether predicted probabilities correspond to observed frequencies.

---

# FINAL REVISION CHEAT SHEET

## Classification

| Metric            | Formula                |
| ----------------- | ---------------------- |
| Accuracy          | (TP+TN)/N              |
| Error Rate        | (FP+FN)/N              |
| Precision         | TP/(TP+FP)             |
| Recall            | TP/(TP+FN)             |
| Specificity       | TN/(TN+FP)             |
| FPR               | FP/(FP+TN)             |
| FNR               | FN/(FN+TP)             |
| F1                | 2PR/(P+R)              |
| F-beta            | (1+β²)PR/(β²P+R)       |
| Balanced Accuracy | (Recall+Specificity)/2 |

## Regression

| Metric      | Formula                        |
| ----------- | ------------------------------ |
| MAE         | Mean absolute error            |
| MSE         | Mean squared error             |
| RMSE        | Square root of MSE             |
| R²          | 1 - SSE/SST                    |
| Adjusted R² | 1-(1-R²)(N-1)/(N-p-1)          |
| MAPE        | Mean absolute percentage error |
| MedianAE    | Median absolute error          |

## What to Remember

**Accuracy:** How many predictions are correct?

**Precision:** When I predict positive, how often am I right?

**Recall:** Of all actual positives, how many did I find?

**F1:** How well do I balance Precision and Recall?

**Specificity:** Of all actual negatives, how many did I correctly reject?

**ROC-AUC:** How well can the model rank positives above negatives?

**PR-AUC:** How well are Precision and Recall balanced across thresholds?

**Log Loss:** How good are predicted probabilities, with strong penalties for confident mistakes?

**MAE:** How far off are predictions on average?

**RMSE:** How large are errors, with extra emphasis on large mistakes?

**R²:** How does squared-error performance compare with predicting the target mean?

**Cross-Validation:** How consistently does the model perform across different training/validation splits?

**Test Evaluation:** How well does the finalized model perform on held-out data?

---

## Final Interview Principle

A good ML engineer doesn't simply say:

"My model achieved 95% accuracy."

They explain:

- Why that metric was selected.
- What errors the model makes.
- How class imbalance affects evaluation.
- Whether the model generalizes.
- How thresholds and hyperparameters were selected.
- Whether data leakage was prevented.
- Whether the model meets actual business requirements.

**Model evaluation isn't just calculating metrics. It's determining whether a model is reliable and useful for the problem it was built to solve.**
