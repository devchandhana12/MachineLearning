# K-Nearest Neighbors (KNN) — Complete ML Notes

**Level:** Beginner to 2–3 Years ML Engineer  
**Coverage:** Intuition, Mathematics, Classification, Regression, Hyperparameters, Evaluation, Optimization, Production, and Interviews

---

# Chapter 1 — What Is KNN?

## 1.1 Definition

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that predicts the output for a new observation by examining the K most similar observations in its training dataset.

**K** represents the number of neighbors considered.

KNN supports:

- **Classification:** Predicting categories.
- **Regression:** Predicting continuous numerical values.

Unlike Linear Regression or Logistic Regression, KNN does not learn a global mathematical equation during training.

Instead, it stores the training observations and uses them when making predictions.

This is why KNN is called a **lazy learning algorithm**.

## 1.2 Why Do We Need KNN?

Imagine predicting whether a Titanic passenger survived.

We know:

- Passenger's age
- Ticket fare
- Passenger class
- Sex
- Whether they survived

Now suppose a new passenger arrives.

Instead of learning an equation for survival, KNN asks:

**Which existing passengers are most similar to this new passenger, and what happened to them?**

If most nearby passengers survived, KNN predicts survival.

If most nearby passengers did not survive, KNN predicts non-survival.

This is the central idea.

## 1.3 KNN vs Other Algorithms

| Algorithm           | How it learns                                     |
| ------------------- | ------------------------------------------------- |
| Linear Regression   | Learns coefficients for a numerical prediction    |
| Logistic Regression | Learns coefficients for classification            |
| Decision Tree       | Learns feature-based splitting rules              |
| Random Forest       | Combines predictions from many trees              |
| XGBoost             | Builds trees sequentially to improve an objective |
| KNN                 | Uses nearby stored training observations          |

KNN assumes that observations close to each other in the chosen feature space tend to have similar targets.

This assumption is useful in some datasets and unreliable in others.

## 1.4 Properties of KNN

**Supervised:** Requires labeled training data.

**Non-parametric:** Does not assume a fixed functional form, such as a straight-line relationship.

**Instance-based:** Uses stored examples for prediction.

**Lazy learning:** Performs relatively little model fitting but substantial work at prediction time.

**Distance-based:** Depends on how similarity between observations is measured.

---

# Chapter 2 — How KNN Classification Works

## 2.1 Our Titanic Example

Suppose we have a new passenger:

- Age: 25
- Fare: ₹50 (hypothetical value for illustration)
- Survival: Unknown

For simplicity, assume we're using only Age and Fare.

After applying appropriate feature scaling, we find these five nearby training passengers.

| Neighbor | Survived? |
| -------- | --------- |
| A        | Yes       |
| B        | Yes       |
| C        | No        |
| D        | Yes       |
| E        | No        |

If K = 5:

- Survival votes = 3
- Non-survival votes = 2

Therefore:

**Predicted class = Survived (1)**

## 2.2 Classification Formula

For binary classification with uniform voting:

\[
\hat y = \operatorname{mode}(y_1,y_2,\ldots,y_K)
\]

Where:

- \(\hat y\): Predicted class
- \(y_i\): Actual class of neighbor \(i\)
- \(K\): Number of nearest neighbors
- Mode: Most frequent class

For the above example:

\[
\hat y=\operatorname{mode}(1,1,0,1,0)=1
\]

## 2.3 What Happens If K Changes?

Suppose the ordered neighbors have labels:

\[
[1,1,0,1,0,0,0]
\]

If K = 1:

\[
\hat y=1
\]

If K = 3:

\[
\hat y=\operatorname{mode}(1,1,0)=1
\]

If K = 7:

\[
\hat y=\operatorname{mode}(1,1,0,1,0,0,0)=0
\]

The prediction changes when K changes.

Therefore, K is an important hyperparameter.

## 2.4 What About Ties?

With K = 4, suppose:

- Two neighbors survived.
- Two neighbors did not survive.

There is no clear majority.

Tie-breaking depends on the implementation and voting rule.

Using an odd K can reduce the possibility of ties in binary classification, but does not guarantee that every issue disappears, especially with multiclass classification or distance-weighted voting.

---

# Chapter 3 — How KNN Measures Distance

## 3.1 Why Distance Matters

KNN cannot identify similar observations without a definition of similarity.

For numerical features, distance metrics provide one such definition.

The most common metrics are:

1. Euclidean distance
2. Manhattan distance
3. Minkowski distance

## 3.2 Euclidean Distance

Euclidean distance measures straight-line distance between two points.

For two features:

\[
d(A,B)=\sqrt{(x*{A1}-x*{B1})^2+(x*{A2}-x*{B2})^2}
\]

Suppose we have two passengers represented using already-scaled features:

Passenger A:

\[
A=(2,3)
\]

Passenger B:

\[
B=(5,7)
\]

Then:

\[
d(A,B)=\sqrt{(5-2)^2+(7-3)^2}
\]

\[
=\sqrt{9+16}
\]

\[
=\sqrt{25}=\boxed{5}
\]

For n numerical features:

\[
d(x,z)=\sqrt{\sum\_{j=1}^{n}(x_j-z_j)^2}
\]

Euclidean distance uses squared differences, so relatively large differences can contribute strongly.

## 3.3 Manhattan Distance

Manhattan distance measures the sum of absolute differences across features.

\[
d(A,B)=|x*{A1}-x*{B1}|+|x*{A2}-x*{B2}|
\]

Using the same points:

\[
A=(2,3),\quad B=(5,7)
\]

\[
d(A,B)=|5-2|+|7-3|
\]

\[
=3+4=\boxed{7}
\]

The idea resembles moving along horizontal and vertical city blocks.

## 3.4 Euclidean vs Manhattan

| Euclidean                                   | Manhattan                              |
| ------------------------------------------- | -------------------------------------- |
| Straight-line distance                      | Sum of absolute coordinate differences |
| Uses squares and square root                | Uses absolute values                   |
| Can emphasize larger coordinate differences | Adds coordinate differences linearly   |
| Minkowski p = 2                             | Minkowski p = 1                        |

Neither metric is universally superior.

The appropriate metric depends on the dataset, feature representation, and validation results.

## 3.5 Minkowski Distance

Minkowski distance generalizes Euclidean and Manhattan distance:

\[
d(x,z)=\left(\sum\_{j=1}^{n}|x_j-z_j|^p\right)^{1/p}
\]

When:

\[
p=1
\]

We obtain Manhattan distance.

When:

\[
p=2
\]

We obtain Euclidean distance.

For standard Minkowski distance, \(p\geq1\).

### Interview Question

**Q: What does the p parameter control in KNN?**

It determines the Minkowski distance formulation.

- p = 1 → Manhattan distance
- p = 2 → Euclidean distance
- Higher p → Larger coordinate differences receive increasing emphasis

---

# Chapter 4 — Why Feature Scaling Is Important

## 4.1 The Problem

Suppose we predict Titanic survival using:

- Age: 0–80
- Fare: 0–500

Take two passengers:

| Feature | Passenger A | Passenger B |
| ------- | ----------: | ----------: |
| Age     |          20 |          30 |
| Fare    |          20 |         200 |

Euclidean distance:

\[
d=\sqrt{(30-20)^2+(200-20)^2}
\]

\[
=\sqrt{100+32400}
\]

\[
=\sqrt{32500}\approx180.28
\]

The fare difference contributes much more to the squared distance than the age difference.

KNN might effectively prioritize Fare simply because of its numerical scale.

That is not necessarily what we want.

## 4.2 Standardization

Standardization transforms a feature as:

\[
z=\frac{x-\mu}{\sigma}
\]

Where:

- \(x\): Original value
- \(\mu\): Mean of that feature in the training data
- \(\sigma\): Standard deviation in the training data

After standardization, features are expressed relative to their own distributions.

This often makes numerical features more comparable for distance calculations.

## 4.3 Min-Max Scaling

Min-Max Scaling transforms a feature into a selected range, commonly [0,1].

\[
x*{\text{scaled}}=\frac{x-x*{\min}}{x*{\max}-x*{\min}}
\]

For example, if Age ranges from 0 to 80:

\[
Age=20
\]

Then:

\[
Age\_{\text{scaled}}=\frac{20-0}{80-0}=0.25
\]

## 4.4 StandardScaler vs MinMaxScaler

| Standardization                                                      | Min-Max Scaling                                                   |
| -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Centers and rescales using mean and standard deviation               | Uses minimum and maximum                                          |
| Often useful for approximately standardized numerical feature spaces | Often useful when bounded scaling is appropriate                  |
| Sensitive to extreme observations                                    | Particularly sensitive to observed extreme minimum/maximum values |
| Does not guarantee a fixed value range                               | Training values usually map within chosen range                   |

Robust scaling may be useful when outliers distort the mean and standard deviation.

## 4.5 Data Leakage Warning

Scaling statistics must be learned from training data only.

Incorrect approach:

1. Scale the entire dataset.
2. Split into training and test sets.

Correct approach:

1. Split into training and test sets.
2. Fit scaling statistics using training data.
3. Transform training data.
4. Transform validation/test data using those same training statistics.

During cross-validation, preprocessing must be fitted independently within each training fold.

Otherwise, validation information leaks into the training process and evaluation can become overly optimistic.

## 4.6 Interview Question

**Q: Why does KNN require scaling while Decision Trees generally do not?**

KNN explicitly calculates distances between feature values. Features with larger scales can dominate.

Decision Trees use threshold-based splits. Ordinary positive linear rescaling generally preserves the feature ordering and does not change the essential splitting possibilities.

---

# Chapter 5 — Choosing K: Bias–Variance Trade-off

## 5.1 What Happens When K = 1?

The nearest training observation determines the prediction.

Advantages:

- Captures highly local patterns.
- Can represent complicated decision boundaries.

Disadvantages:

- Sensitive to noise.
- Sensitive to mislabeled data.
- High variance.
- Higher overfitting risk.

If the nearest Titanic passenger is incorrectly labeled, the prediction may be wrong.

## 5.2 What Happens When K Is Very Large?

Suppose 100 training passengers have:

- 70 non-survivors.
- 30 survivors.

If K = 100 for every prediction, uniform-voting KNN always predicts non-survival on that reference set.

The model loses much of its ability to capture local patterns.

Consequences:

- Higher bias.
- Greater underfitting risk.
- Majority-class dominance.

## 5.3 Bias and Variance

| K          | Bias     | Variance | Typical concern  |
| ---------- | -------- | -------- | ---------------- |
| Very small | Low      | High     | Overfitting      |
| Moderate   | Balanced | Balanced | Often preferable |
| Very large | High     | Low      | Underfitting     |

These are general tendencies, not guarantees for every dataset.

## 5.4 How Should K Be Selected?

Do not assume K = 3 or K = 5 is always best.

Instead:

1. Define candidate K values.
2. Evaluate candidates using cross-validation.
3. Choose the K that performs best on the chosen validation metric.
4. Evaluate the selected model once on an untouched test set.

For example:

\[
K\in\{3,5,7,9,11,15\}
\]

The optimal K depends on the data.

---

# Chapter 6 — Uniform vs Distance-Weighted KNN

## 6.1 Uniform Voting

Every selected neighbor contributes one equally weighted vote.

Suppose:

| Neighbor | Distance | Class    |
| -------- | -------: | -------- |
| A        |      0.1 | Survived |
| B        |      0.2 | Died     |
| C        |      0.3 | Died     |

With uniform weights:

- Survived = 1 vote
- Died = 2 votes

Prediction:

\[
\boxed{0\text{ (Died)}}
\]

## 6.2 Distance-Weighted Voting

Nearby observations receive larger influence.

A common weighting formula is:

\[
w_i=\frac{1}{d_i}
\]

Where \(d_i\) is the distance to neighbor \(i\).

Using the same example:

| Neighbor | Distance | Weight | Class    |
| -------- | -------: | -----: | -------- |
| A        |      0.1 |     10 | Survived |
| B        |      0.2 |      5 | Died     |
| C        |      0.3 |   3.33 | Died     |

Weighted votes:

\[
Score(1)=10
\]

\[
Score(0)=5+3.33=8.33
\]

Therefore:

\[
\boxed{\hat y=1}
\]

The prediction changes because the closest survivor receives more influence.

## 6.3 Weighted Classification Formula

For class \(c\):

\[
Score(c)=\sum\_{i=1}^{K}w_i I(y_i=c)
\]

Where:

- \(w_i\): Neighbor weight
- \(I(y_i=c)\): 1 if neighbor belongs to class \(c\), otherwise 0

The predicted class is:

\[
\hat y=\arg\max_c Score(c)
\]

## 6.4 When Is Distance Weighting Useful?

Distance weighting may help when:

- The closest observations are especially informative.
- More distant neighbors are less representative.
- Uniform voting ignores meaningful distance differences.

However, if data contains noisy close neighbors, distance weighting may amplify their influence.

### Special Case: Zero Distance

If a query exactly matches one or more training observations, the inverse-distance formula would involve division by zero.

Practical implementations handle exact matches specially, such as assigning influence only to zero-distance matching observations.

---

# Chapter 7 — KNN Regression

## 7.1 Classification vs Regression

Classification predicts a category.

Example:

\[
Survived\in\{0,1\}
\]

Regression predicts a continuous value.

Example:

\[
HousePrice=₹55,00,000
\]

In KNN Regression, the prediction is obtained from the target values of nearby training observations.

## 7.2 House Price Example

Suppose a new house has three nearest neighbors.

| Neighbor |    Price |
| -------- | -------: |
| A        | ₹40 lakh |
| B        | ₹50 lakh |
| C        | ₹60 lakh |

If K = 3:

\[
\hat y=\frac{40+50+60}{3}
\]

\[
\boxed{\hat y=₹50\text{ lakh}}
\]

The general uniform-weight regression formula:

\[
\hat y=\frac{1}{K}\sum\_{i=1}^{K}y_i
\]

## 7.3 Distance-Weighted Regression

Suppose:

| House | Price | Distance |
| ----- | ----: | -------: |
| A     |  ₹40L |      0.1 |
| B     |  ₹50L |      0.2 |
| C     |  ₹60L |      0.3 |

Inverse-distance weights:

\[
w_1=10,\quad w_2=5,\quad w_3\approx3.333
\]

Weighted prediction:

\[
\hat y=\frac{\sum_iw_iy_i}{\sum_iw_i}
\]

\[
=\frac{10(40)+5(50)+3.333(60)}{10+5+3.333}
\]

\[
\boxed{\hat y\approx₹46.36L}
\]

Because the nearest house costs ₹40L, the weighted prediction shifts toward that price.

## 7.4 How Does K Affect Regression?

**Small K:**

- Fits local variations closely.
- Can be noisy.
- May overfit.

**Large K:**

- Produces smoother predictions.
- May ignore local differences.
- May underfit.

In regression, K controls the degree of local smoothing.

---

# Chapter 8 — Neighbor Search Algorithms

KNN must determine which training observations are closest to a query.

The method used to find them affects prediction latency and scalability.

## 8.1 Brute Force

Brute-force search computes the distance from the query to every stored training observation.

Suppose:

- 1,000,000 training passengers
- 20 numerical features

A straightforward search examines all 1,000,000 candidates to find the nearest neighbors.

Advantages:

- Simple.
- Exact nearest-neighbor retrieval.
- No spatial indexing structure required.

Disadvantages:

- Expensive for large datasets.
- Prediction latency can grow with training-set size.

Approximate distance-calculation work for one query:

\[
O(nd)
\]

Where:

- \(n\): Number of training observations
- \(d\): Number of features

Additional work may be required to select the K smallest distances.

## 8.2 KD-Tree

KD-Tree stands for K-Dimensional Tree.

It partitions feature space using axis-aligned splits.

Intuition:

Instead of examining all passengers, the tree attempts to eliminate regions that cannot contain closer neighbors.

Advantages:

- Can accelerate exact neighbor search in suitable low-dimensional data.
- Reduces unnecessary distance comparisons when pruning is effective.

Limitations:

- Often loses efficiency as dimensionality increases.
- May require examining many branches.
- Index building and storage have costs.

## 8.3 Ball Tree

Ball Trees organize observations into nested regions defined by metric-space distance bounds.

Instead of partitioning only with axis-aligned splits, Ball Trees group observations into distance-bounded regions.

Advantages:

- Can work well for some metric spaces and distributions.
- May accelerate exact nearest-neighbor retrieval.

Limitations:

- Not universally faster than brute force.
- Performance depends on dimensionality, metric, and data distribution.
- Index construction has overhead.

## 8.4 Comparison

| Algorithm   | Main technique                    | Suitable situations                                                        |
| ----------- | --------------------------------- | -------------------------------------------------------------------------- |
| Brute Force | Checks all candidates             | Small datasets, higher-dimensional cases, efficient vectorized computation |
| KD-Tree     | Axis-aligned spatial partitions   | Certain low-dimensional numerical datasets                                 |
| Ball Tree   | Hierarchical metric-space regions | Certain metric/dimensionality combinations                                 |

The search method affects computational efficiency, not the underlying KNN voting rule.

## 8.5 Approximate Nearest Neighbors

In very large systems, exact neighbor retrieval may be too costly.

Approximate Nearest Neighbor (ANN) methods trade some retrieval accuracy for speed and scalability.

Examples of broader ANN approaches include:

- Graph-based methods such as HNSW
- Quantization-based methods
- Approximate indexing structures

These are retrieval techniques, not new voting formulas for KNN.

A production team may evaluate retrieval recall, latency, memory usage, and indexing costs before adopting ANN.

---

# Chapter 9 — Cross-Validation and Hyperparameter Tuning

## 9.1 Why Do We Need Cross-Validation?

Suppose we split Titanic into:

- 80% training data
- 20% validation data

K = 5 gives 85% validation accuracy.

Does that guarantee K = 5 is best?

No.

A different validation split might produce a different result.

Cross-validation measures performance across several training/validation partitions.

## 9.2 Five-Fold Cross-Validation

Divide the training dataset into five folds.

Perform five rounds:

1. Fold 1 validates; folds 2–5 train.
2. Fold 2 validates; remaining folds train.
3. Fold 3 validates; remaining folds train.
4. Fold 4 validates; remaining folds train.
5. Fold 5 validates; remaining folds train.

Suppose the validation accuracies are:

\[
[0.82,\ 0.85,\ 0.84,\ 0.83,\ 0.86]
\]

Mean validation accuracy:

\[
\frac{0.82+0.85+0.84+0.83+0.86}{5}
\]

\[
=\boxed{0.84}
\]

Average CV accuracy = 84%.

The variation across folds is also informative.

## 9.3 Stratified Cross-Validation

Titanic survival classes may be imbalanced.

Stratified K-Fold aims to keep the class proportions approximately consistent across folds.

For classification tasks, stratification is commonly preferable to ordinary shuffled K-fold splitting.

For temporal or grouped datasets, time-aware or group-aware validation may instead be necessary.

## 9.4 Grid Search

Grid Search evaluates combinations of hyperparameters.

For example:

\[
K\in\{3,5,7\}
\]

\[
Weights\in\{uniform,distance\}
\]

\[
p\in\{1,2\}
\]

Total combinations:

\[
3\times2\times2=\boxed{12}
\]

With five-fold cross-validation:

\[
12\times5=\boxed{60}
\]

There are 60 validation fits across the grid, plus a possible final refit using the selected hyperparameters.

## 9.5 How Do We Choose the Best Combination?

Choose the hyperparameters that maximize or minimize the relevant mean cross-validation score.

Examples:

- Balanced classification: Accuracy or macro F1, depending on the goal.
- Minority-class detection: Recall, F1, or PR-AUC-based measures.
- Regression: MAE, RMSE, or another business-relevant regression loss.

Grid Search does not eliminate overfitting risk.

Trying many configurations can overfit the validation process itself, which is another reason to retain an untouched test set.

---

# Chapter 10 — KNN Classification Evaluation

## 10.1 Confusion Matrix

For Titanic:

- Positive class (1): Survived
- Negative class (0): Did not survive

Suppose our held-out evaluation gives:

|                 | Predicted Survived | Predicted Died |
| --------------- | -----------------: | -------------: |
| Actual Survived |            TP = 30 |        FN = 10 |
| Actual Died     |             FP = 5 |        TN = 55 |

Total passengers:

\[
30+10+5+55=100
\]

## 10.2 Accuracy

Accuracy measures the proportion of correct predictions.

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
=\frac{30+55}{100}
\]

\[
\boxed{85\%}
\]

## 10.3 Precision

Precision asks:

Of everyone predicted to survive, how many actually survived?

\[
Precision=\frac{TP}{TP+FP}
\]

\[
=\frac{30}{30+5}
\]

\[
\boxed{85.71\%}
\]

## 10.4 Recall

Recall asks:

Of everyone who actually survived, how many were identified?

\[
Recall=\frac{TP}{TP+FN}
\]

\[
=\frac{30}{30+10}
\]

\[
\boxed{75\%}
\]

## 10.5 F1 Score

F1 is the harmonic mean of precision and recall.

\[
F1=\frac{2PR}{P+R}
\]

Equivalently:

\[
F1=\frac{2TP}{2TP+FP+FN}
\]

\[
=\frac{60}{60+5+10}
\]

\[
=\boxed{80\%}
\]

## 10.6 Other Useful Metrics

**Specificity**

Measures how well the model identifies actual negatives.

\[
Specificity=\frac{TN}{TN+FP}
\]

**Balanced Accuracy**

Averages sensitivity (recall) and specificity.

\[
BalancedAccuracy=\frac{Recall+Specificity}{2}
\]

**ROC-AUC**

Measures ranking discrimination across classification thresholds using true-positive and false-positive rates.

**PR-AUC / Average Precision**

Summarizes the precision–recall trade-off, especially relevant when the positive class is rare.

Average Precision and trapezoidal PR-AUC are related but not identical calculations.

**Log Loss**

Measures the quality of predicted probabilities and penalizes confident incorrect predictions heavily.

KNN's neighbor-vote fractions can provide probability-like estimates, but they are not automatically well-calibrated probabilities.

## 10.7 Which Metric Should We Choose?

| Situation                             | Useful metrics                              |
| ------------------------------------- | ------------------------------------------- |
| Approximately balanced classes        | Accuracy, F1, ROC-AUC                       |
| Rare positive class                   | Precision, Recall, PR-AUC                   |
| False negatives are expensive         | Recall and cost-sensitive evaluation        |
| False positives are expensive         | Precision and cost-sensitive evaluation     |
| Both errors matter                    | F1, balanced accuracy, business-cost metric |
| Probabilities are consumed downstream | Log loss, Brier score, calibration          |

Always choose evaluation metrics based on the actual decision-making objective.

---

# Chapter 11 — KNN Regression Evaluation

Suppose we predict prices for three houses.

All prices below are in lakhs.

| Actual | Predicted | Absolute error |
| ------ | --------- | -------------- |
| 40     | 45        | 5              |
| 50     | 50        | 0              |
| 60     | 50        | 10             |

## 11.1 Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum\_{i=1}^{n}|y_i-\hat y_i|
\]

\[
=\frac{5+0+10}{3}
\]

\[
\boxed{MAE=5\text{ lakh}}
\]

Interpretation:

Predictions differ from actual prices by ₹5 lakh on average in absolute terms.

## 11.2 Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum\_{i=1}^{n}(y_i-\hat y_i)^2
\]

\[
=\frac{25+0+100}{3}
\]

\[
\boxed{MSE\approx41.67\text{ lakh}^2}
\]

Squaring increases the contribution of larger errors.

## 11.3 Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

\[
=\sqrt{41.67}
\]

\[
\boxed{RMSE\approx6.45\text{ lakh}}
\]

RMSE is expressed in the same units as the target.

## 11.4 R-Squared

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

Where \(\bar y\) is the mean actual target.

R² compares the model against a constant predictor that always predicts the mean of the evaluated targets.

Interpretation:

- R² = 1: Perfect predictions.
- R² = 0: Same squared-error performance as that mean baseline.
- R² below 0: Worse squared-error performance than that mean baseline.

KNN regression uses the same regression evaluation formulas as other regression algorithms.

---

# Chapter 12 — Curse of Dimensionality

## 12.1 What Is Dimensionality?

Dimensionality refers to the number of features.

For Titanic:

- Age only → 1 dimension.
- Age + Fare → 2 dimensions.
- Age + Fare + Pclass → 3 dimensions.
- 500 numerical features → 500 dimensions.

## 12.2 Why Is High Dimensionality Difficult?

KNN depends on finding nearby observations.

As the number of dimensions increases, the available space grows rapidly.

With a fixed number of training observations, the data can become extremely sparse relative to that space.

Distances can also become less discriminative, depending on feature distributions and the distance metric.

As a result, the nearest neighbor may not be meaningfully similar in the features that matter for prediction.

## 12.3 Simple Intuition

Imagine dividing each numerical feature range into 10 intervals.

With one feature:

\[
10^1=10
\]

Possible bins.

With two features:

\[
10^2=100
\]

Possible bins.

With ten features:

\[
10^{10}
\]

Possible bins.

The number of observations needed to populate these regions densely can grow dramatically.

This is an illustration of sparsity, not a literal requirement that KNN must fill every bin.

## 12.4 Remedies

Possible approaches:

- Remove irrelevant features.
- Select informative features.
- Reduce dimensionality.
- Use appropriate feature representations.
- Choose or learn a meaningful distance metric.
- Increase representative training data when feasible.
- Compare against algorithms less dependent on raw high-dimensional distances.

Feature scaling alone does not solve the curse of dimensionality.

---

# Chapter 13 — Overfitting and Underfitting

## 13.1 Overfitting

KNN may overfit when:

- K is too small.
- The dataset has noisy observations.
- Some labels are incorrect.
- Features contain irrelevant patterns.
- Nearby observations are not representative.

Typical symptom:

Training performance is high while validation performance is noticeably worse.

Possible remedies:

- Increase K.
- Improve data quality.
- Remove irrelevant features.
- Select appropriate distance metrics.
- Validate with suitable cross-validation.

## 13.2 Underfitting

KNN may underfit when:

- K is too large.
- Local distinctions are averaged away.
- The distance metric fails to reflect meaningful similarity.
- Features are poorly represented.

Typical symptom:

Both training and validation performance are poor relative to a useful baseline.

Possible remedies:

- Reduce K.
- Improve feature representation.
- Consider distance weighting.
- Compare alternative algorithms.

## 13.3 Important Interview Distinction

High training accuracy alone does not guarantee overfitting.

Low training and validation scores do not automatically prove underfitting without considering the task difficulty and baseline.

Diagnosis requires comparing training and validation behavior, class distributions, data quality, and achievable performance.

---

# Chapter 14 — Production-Grade KNN

## 14.1 Prediction Latency

KNN often performs substantial computation when a prediction request arrives.

With brute-force search, a new query may require comparing its features with every stored training observation.

Approximate cost per query:

\[
O(nd)
\]

Where:

- n = Number of stored observations.
- d = Number of features.

If n is extremely large, inference can become expensive.

### Production Scenario

Suppose there are five million user records, each represented by 300 numerical features.

Ordinary brute-force nearest-neighbor search may be too slow for a strict real-time latency requirement.

Potential solutions include:

- More efficient neighbor-search indexes.
- Approximate nearest-neighbor retrieval.
- Reducing unnecessary dimensions.
- Reducing the reference dataset when appropriate.
- Batching queries or optimizing distance computations.
- Using an alternative predictive architecture.

The right solution depends on measured accuracy, latency, and system constraints.

## 14.2 Memory Consumption

KNN usually needs access to its reference observations during prediction.

With n training observations and d numerical features, storing a dense feature matrix alone requires storage proportional to:

\[
O(nd)
\]

Additional memory is needed for labels, preprocessing information, indexes, and serving infrastructure.

For example, five million observations with 300 float32 features require approximately:

\[
5{,}000{,}000\times300\times4
\]

\[
=6{,}000{,}000{,}000\text{ bytes}
\]

Approximately 6 GB (decimal) for the raw dense feature values alone.

This excludes labels, indexes, replicas, and other system overhead.

## 14.3 Training Cost vs Inference Cost

KNN is called lazy because it does not fit a traditional global predictive function.

However, practical KNN systems may still need to:

- Clean data.
- Calculate preprocessing statistics.
- Build neighbor-search indexes.
- Maintain reference data.
- Validate and select hyperparameters.

Therefore, lazy learning does not mean absolutely zero preparation cost.

### Interview Question

**Q: Why can KNN train quickly but predict slowly?**

Because KNN postpones much of the work until prediction time, when it retrieves nearby training observations and aggregates their labels or targets.

## 14.4 Updating Training Data

Since KNN predictions depend on reference observations, introducing new examples can change future predictions.

But a production system must still manage:

- Reference-data freshness.
- Feature transformations.
- Index rebuilding or incremental updates.
- Serving consistency.
- Data quality.
- Versioning.

Adding new data is not automatically free or safe.

## 14.5 Missing Values

Distance calculations require a consistent feature representation.

Missing values can make distances undefined or misleading.

Potential solutions:

- Imputation.
- Missing-value indicators where useful.
- Domain-appropriate feature handling.
- Consistent training and inference preprocessing.

Imputation parameters must be learned without test-data leakage.

## 14.6 Categorical Variables

Titanic includes categorical variables such as Sex and Pclass.

Standard Euclidean distance does not directly operate on arbitrary text categories.

Possible approaches include:

- One-hot encoding.
- Binary encoding where appropriate.
- Domain-specific distance measures.
- Mixed-type distance metrics.

Encoding choices affect the geometry of the feature space and consequently the nearest neighbors.

Treating category identifiers as ordinary numerical measurements can introduce an unjustified ordering or spacing.

## 14.7 Class Imbalance

Suppose our training data contains:

- 90% non-survivors.
- 10% survivors.

For large neighborhoods, majority-class labels may dominate uniform voting.

Consequences:

- High overall accuracy may hide poor minority-class recall.
- A large K can suppress local minority-class signals.

Possible remedies:

- Tune K based on the intended metric.
- Consider class-aware methods where appropriate.
- Examine feature quality and data representation.
- Evaluate precision, recall, F1, and PR-AUC.
- Use appropriate resampling within training folds if justified.

There is no guarantee that distance weighting alone resolves class imbalance.

## 14.8 Probability Estimates and Calibration

For uniform-vote binary KNN, a common estimated positive-class probability is:

\[
\hat P(Y=1\mid x)=
\frac{\text{positive neighbors}}{K}
\]

Example:

Three of five neighbors survived.

\[
\hat P(Y=1\mid x)=\frac35=0.6
\]

This is a vote fraction, not automatically a calibrated 60% real-world event probability.

Calibration should be evaluated using:

- Reliability diagrams.
- Brier score.
- Log loss.
- Calibration methods if necessary.

Calibration fitting must use suitable held-out or cross-validated data.

## 14.9 Data Drift

Suppose KNN is deployed on passenger-like records, and the incoming population changes substantially.

Potential drift types:

**Feature drift:** Age, Fare, or other feature distributions change.

**Concept drift:** The relationship between features and survival changes.

Possible production monitoring:

- Input feature distributions.
- Missing-value rates.
- Distance-to-nearest-neighbor distributions.
- Class prediction proportions.
- Inference latency.
- Model evaluation metrics once actual outcomes arrive.
- Calibration over time.

Retraining or refreshing the reference dataset may be necessary, but the response should depend on the cause of degradation.

## 14.10 Outliers and Noisy Data

Outliers can affect distance relationships and neighbor selection.

Mislabeled observations may be especially harmful when they are extremely close to a prediction query.

Potential safeguards:

- Data validation.
- Outlier inspection.
- Comparing K values.
- Robust feature preprocessing.
- Monitoring nearest-neighbor behavior.

## 14.11 Privacy and Security

KNN frequently retains representations of individual training examples.

This creates privacy and security considerations, including:

- Access control over reference data.
- Protection of stored embeddings or feature records.
- Appropriate retention and deletion policies.
- Avoiding unnecessary exposure of individual neighbors.
- Controls against malicious or poisoned reference examples.

A system that explains predictions by showing nearest training examples must ensure those examples do not leak sensitive user information.

---

# Chapter 15 — Important Hyperparameters

| Hyperparameter  | Meaning                                           | Main effect                                            |
| --------------- | ------------------------------------------------- | ------------------------------------------------------ |
| n_neighbors / K | Number of nearest neighbors                       | Controls locality and bias–variance trade-off          |
| weights         | Uniform or distance-based                         | Controls neighbor influence                            |
| metric          | Distance function                                 | Determines which observations are considered similar   |
| p               | Minkowski distance power                          | p=1 Manhattan, p=2 Euclidean                           |
| algorithm       | Brute, KD-Tree, Ball Tree, or automatic selection | Controls retrieval strategy                            |
| leaf_size       | Spatial-tree leaf-size setting                    | Can affect index construction, query speed, and memory |

These hyperparameters must be assessed on appropriate validation data.

Scaling and feature representation are often as consequential as tuning K itself.

---

# Chapter 16 — KNN vs Other Algorithms

## 16.1 KNN vs Logistic Regression

| KNN                                | Logistic Regression                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| Local, example-based prediction    | Learns global coefficients                                                    |
| Nonlinear boundaries are possible  | Linear decision boundary in original features unless features are transformed |
| No explicit global equation fitted | Fits a parametric classification model                                        |
| Can become expensive at inference  | Usually inexpensive at inference                                              |
| Sensitive to chosen distance       | Depends on modeled feature relationships                                      |

## 16.2 KNN vs Decision Trees

| KNN                                       | Decision Trees                                  |
| ----------------------------------------- | ----------------------------------------------- |
| Uses distance                             | Uses feature thresholds                         |
| Typically needs numerical feature scaling | Usually insensitive to ordinary feature scaling |
| Stores reference examples                 | Stores learned tree structure                   |
| Local similarity reasoning                | Hierarchical rule-based partitioning            |

## 16.3 KNN vs Random Forest / XGBoost

KNN can be useful when local similarity is informative and the dataset is manageable.

Random Forest and XGBoost can be preferable for many structured tabular prediction problems because they learn predictive patterns rather than performing a full neighbor search for every query.

However, there is no universal winner.

Compare algorithms using suitable evaluation metrics, latency, memory, reliability, and maintainability.

---

# Chapter 17 — Production and Interview Scenarios

## Scenario 1 — Only 100 Training Observations

**Question:** Can we train KNN with just 100 observations?

**Answer:**

Yes.

KNN does not require a specific minimum dataset size beyond having enough observations for the chosen K.

However, 100 observations might provide poor neighborhood coverage, especially with many features.

Recommended reasoning:

1. Examine the number of features.
2. Inspect class distribution.
3. Avoid unnecessarily large K.
4. Use cross-validation carefully.
5. Compare against simple baselines.
6. Report uncertainty due to limited sample size.

A small dataset does not automatically mean KNN is the best algorithm.

## Scenario 2 — One Million Observations

**Question:** Why might KNN become slow?

**Answer:**

Exact brute-force retrieval may require distance comparisons with all one million examples for each query.

Possible remedies include indexing, ANN search, feature reduction, and architecture changes.

## Scenario 3 — Accuracy Is 95%, but Minority Recall Is 20%

**Question:** Is the model good?

**Answer:**

Not necessarily.

High accuracy may be driven by predicting the majority class.

Review confusion matrix, recall, precision, F1, and PR-AUC.

Choose metrics according to the business impact of errors.

## Scenario 4 — K = 1 Has 100% Training Accuracy

**Question:** Is that good?

**Answer:**

Not enough information.

For ordinary training-set evaluation, one-nearest-neighbor classification often reproduces the training labels because an observation is its own nearest neighbor.

This does not establish generalization.

Measure performance on unseen observations.

## Scenario 5 — K = 50 Performs Worse Than K = 5

**Question:** Why?

**Answer:**

A larger K may average across observations that belong to different local patterns.

This can increase bias and cause underfitting.

Validate alternative K values rather than assuming a particular one is best.

## Scenario 6 — Adding a New Feature Reduces Accuracy

**Question:** How is that possible?

**Answer:**

KNN measures similarity across chosen features.

A noisy or irrelevant feature can alter pairwise distances and change which observations are selected as neighbors.

Potential solution:

- Feature selection.
- Improved representation.
- Suitable scaling.
- Validation of each change.

More features do not automatically mean better predictions.

## Scenario 7 — Two Passengers Have Identical Feature Values but Different Labels

**Question:** What happens?

**Answer:**

The observations occupy the same location in feature space but disagree on the outcome.

KNN cannot distinguish them using the available features alone.

The prediction depends on the selected neighbors, labels, and voting rule.

This may indicate missing predictive features, genuine randomness, or label noise.

## Scenario 8 — Training Performance Is High but Validation Performance Is Low

**Question:** What do you investigate?

**Answer:**

Potential overfitting.

Check:

- Whether K is too small.
- Label quality.
- Data leakage.
- Feature representation.
- Relevant train/validation distribution differences.
- Hyperparameter choices.

Increase K if validation evidence supports it.

## Scenario 9 — KNN Is Used in a Real-Time API

**Question:** What would you monitor?

**Answer:**

At minimum:

- Prediction latency, especially p95/p99.
- Query throughput.
- Memory usage.
- Index or reference-data freshness.
- Prediction distribution.
- Input feature drift.
- Nearest-neighbor distance patterns.
- Quality metrics when labels become available.

The model should be evaluated as part of a complete serving system, not only by offline accuracy.

## Scenario 10 — When Would You Avoid Ordinary KNN?

You might avoid it when:

- Training data is extremely large and low-latency prediction is mandatory.
- Feature space is high-dimensional and raw distances are uninformative.
- Reference data is expensive to store.
- The chosen distance metric does not capture meaningful similarity.
- The deployment environment cannot support necessary retrieval infrastructure.

You should still test alternatives rather than ruling KNN out solely from dataset size.

---

# Chapter 18 — Final Interview Questions and Answers

**Q1. What is KNN?**

A supervised, non-parametric, instance-based algorithm that predicts from the labels or target values of nearby stored observations.

**Q2. Why is KNN called lazy learning?**

Because it performs relatively little fitting of a global model and defers much of the computation to prediction time.

**Q3. Is KNN classification or regression?**

Both.

Classification uses neighbor voting; regression uses neighbor averaging or weighted averaging.

**Q4. Why do we need scaling?**

Because features with larger numerical magnitudes may disproportionately influence distance.

**Q5. How do you select K?**

Evaluate candidate K values using appropriate cross-validation and a task-relevant metric.

**Q6. What happens with K = 1?**

The classifier becomes highly local and potentially sensitive to noise.

**Q7. What happens when K becomes very large?**

Predictions become smoother, with increasing risk of underfitting and majority-class dominance.

**Q8. Euclidean vs Manhattan?**

Euclidean uses the square root of summed squared differences; Manhattan uses summed absolute differences.

**Q9. What is Minkowski distance?**

A generalized distance family that includes Manhattan and Euclidean distance for p = 1 and p = 2.

**Q10. What are uniform weights?**

Every selected neighbor contributes equally.

**Q11. What are distance weights?**

Nearby observations receive greater voting or averaging influence.

**Q12. Why does KNN struggle in high dimensions?**

Observations become sparse relative to the feature space, and distances may become less informative.

**Q13. Why is KNN potentially expensive in production?**

Prediction may require searching a large reference dataset; it also incurs data storage and indexing costs.

**Q14. Does KNN always need a KD-Tree?**

No. Brute force, KD-Tree, Ball Tree, and approximate methods have different performance characteristics.

**Q15. Is KNN suitable for imbalanced data?**

It can be used, but class imbalance may distort voting. Metrics and validation strategy must account for the class distribution.

**Q16. Can KNN output probabilities?**

It can produce class-vote fractions or normalized weighted-vote scores. These are not necessarily calibrated probabilities.

**Q17. Why use cross-validation?**

To estimate performance across multiple validation partitions and select hyperparameters more reliably than from one arbitrary split.

**Q18. What is data leakage in KNN?**

Using information from validation/test observations when fitting preprocessing or selecting model configurations, leading to optimistic evaluation.

**Q19. Is KNN sensitive to irrelevant features?**

Yes. Irrelevant dimensions can distort which observations appear nearest.

**Q20. What is the biggest practical limitation of KNN?**

There isn't one universal limitation, but inference cost, memory requirements, and the quality of high-dimensional distance measurements are frequent concerns.

---

# Chapter 19 — Final Revision Sheet

| Topic                   | Remember                                               |
| ----------------------- | ------------------------------------------------------ |
| Full form               | K-Nearest Neighbors                                    |
| Learning type           | Supervised                                             |
| Model family            | Non-parametric, instance-based                         |
| Learning strategy       | Lazy learning                                          |
| Classification          | Majority or weighted vote                              |
| Regression              | Mean or weighted mean                                  |
| K                       | Number of neighbors                                    |
| K = 1                   | Very local; potentially high variance                  |
| Large K                 | Smoother; potentially higher bias                      |
| Euclidean               | Straight-line distance                                 |
| Manhattan               | Sum of absolute differences                            |
| Minkowski p = 1         | Manhattan                                              |
| Minkowski p = 2         | Euclidean                                              |
| Scaling                 | Usually essential for ordinary numerical distance      |
| Uniform weights         | Equal neighbor influence                               |
| Distance weights        | Closer neighbors receive more influence                |
| Brute force             | Direct distance comparisons                            |
| KD-Tree                 | Axis-aligned spatial partitioning                      |
| Ball Tree               | Distance-bounded region hierarchy                      |
| Cross-validation        | More robust hyperparameter selection                   |
| Grid Search             | Tests defined parameter combinations                   |
| Classification metrics  | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC       |
| Regression metrics      | MAE, MSE, RMSE, R²                                     |
| Curse of dimensionality | Sparse spaces and potentially less useful distances    |
| Production constraints  | Inference latency, memory, data freshness              |
| Probability calibration | Vote proportions aren't automatically calibrated       |
| Model monitoring        | Performance, drift, latency, neighbor-distance changes |

## Final Mental Model

When predicting for a new observation, KNN asks five questions:

1. **How do we represent this observation?**  
   Select and preprocess meaningful features.

2. **What does similar mean?**  
   Choose a suitable distance metric.

3. **Who are the nearest observations?**  
   Retrieve the K closest training examples.

4. **How should those neighbors influence the result?**  
   Use majority voting, averaging, or distance weighting.

5. **Can we trust the prediction?**  
   Validate performance, check data quality, assess calibration where needed, and monitor production behavior.

**One-line summary:** KNN learns by using the outcomes of nearby examples rather than fitting a global prediction equation. Its success depends heavily on feature representation, distance quality, K selection, and the ability to retrieve useful neighbors efficiently.

**Status: KNN complete — conceptual, mathematical, evaluation, production, and interview coverage.**
