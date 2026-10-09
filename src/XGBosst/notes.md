# XGBoost — Complete Study Notes

**Level:** ML Engineer (2 Years Experience)  
**Example:** Titanic Survival Prediction (Binary Classification)

---

## 1. What is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**.

It is an optimized Gradient Boosting algorithm that builds decision trees sequentially.

Each new tree learns corrections to the predictions made by the existing ensemble.

### Why XGBoost when Gradient Boosting already exists?

Traditional Gradient Boosting can suffer from:

- Overfitting due to complex trees.
- Computationally expensive sequential training.
- Lack of explicit regularization in conventional implementations.

XGBoost introduces:

- Regularization (Gamma, Lambda, Alpha).
- First-order gradients and second-order Hessians.
- Efficient tree construction.
- Parallelized split calculations.
- Native missing-value handling.

**Interview Answer:** XGBoost is an optimized and regularized Gradient Boosting framework that uses gradients and Hessians to construct trees efficiently while controlling model complexity.

---

## 2. How XGBoost Works

Consider the Titanic dataset.

**Target:** Survived (1) or Did Not Survive (0).

**Features:** Sex, Age, Pclass, Fare, etc.

### Step 1: Initial Prediction

For our simplified example, assume every passenger initially receives a survival probability of 0.50.

Note: This is a teaching assumption. The actual initial prediction depends on XGBoost's configuration and initial score estimation.

### Step 2: Calculate Gradients

The gradient tells us the direction in which the prediction needs to change to reduce loss.

For binary classification with logistic loss:

**g = p - y**

Where:

- p = Predicted survival probability.
- y = Actual survival (0 or 1).

Example:

| Passenger | Actual | Predicted | Gradient |
| --------- | ------ | --------- | -------- |
| Female A  | 1      | 0.5       | -0.5     |
| Female B  | 1      | 0.5       | -0.5     |
| Male C    | 0      | 0.5       | +0.5     |
| Male D    | 0      | 0.5       | +0.5     |

**Negative gradient:** Increase the raw survival score.

**Positive gradient:** Decrease the raw survival score.

XGBoost updates its predictions in the direction opposite to the gradient.

---

## 3. Hessians

A Hessian is the second derivative of the loss function.

It measures how the gradient changes as the model's raw prediction score changes.

For binary logistic classification:

**h = p × (1 - p)**

Example:

p = 0.5

h = 0.5 × (1 - 0.5)

**h = 0.25**

If three female passengers belong to one leaf:

H = 0.25 + 0.25 + 0.25

**H = 0.75**

Important distinction:

- g = Gradient of an individual passenger.
- G = Sum of gradients in a node.
- h = Hessian of an individual passenger.
- H = Sum of Hessians in a node.

---

## 4. Calculating Leaf Output

A leaf produces a correction to the model's current prediction.

Without regularization:

**w = -G / H**

Example: Female passengers.

G = -1.5

H = 0.75

w = -(-1.5) / 0.75

**Leaf output = +2**

This means the tree wants to increase the raw survival score by 2.

Important: The leaf output is NOT directly added to the probability.

For binary logistic classification, the correction is added to the raw score (logit).

The sigmoid function converts the score into a probability:

**p = 1 / (1 + e^(-z))**

If the initial score is 0:

New score = 0 + 2 = 2

New survival probability ≈ 88.1%.

---

## 5. Learning Rate

The learning rate determines how much of each tree's correction is applied.

**New Score = Old Score + Learning Rate × Leaf Output**

Formula:

**Fₘ(x) = Fₘ₋₁(x) + η × fₘ(x)**

Where:

- η = Learning rate.
- fₘ(x) = Current tree output.
- Fₘ₋₁(x) = Previous ensemble prediction.

Example:

Old raw score = 0

Leaf output = +2

Learning rate = 0.1

New score = 0 + (0.1 × 2)

**New score = 0.2**

After sigmoid:

**Survival probability ≈ 55%.**

### Why is learning rate necessary?

Without learning rate, trees may make large corrections and fit training patterns too aggressively.

A smaller learning rate:

- Makes gradual updates.
- Often improves generalization when tuned appropriately.
- Typically requires more trees.

A very small learning rate does not automatically prevent overfitting.

---

## 6. How Tree 2 Learns

Tree 2 does not simply receive Tree 1's output as its target.

Instead:

1. Tree 1 updates the ensemble predictions.
2. XGBoost recalculates gradients and Hessians using those updated predictions.
3. Tree 2 uses the original features and the new gradient/Hessian values.
4. Tree 2 calculates new leaf outputs.
5. Its learning-rate-scaled corrections are added to the existing ensemble.

### Titanic Example

Initial female survival probability = 0.50.

After Tree 1 = approximately 0.55.

New gradient:

g = 0.55 - 1 = -0.45

For three female passengers:

G = -0.45 × 3 = -1.35

New Hessian:

h = 0.55 × 0.45 = 0.2475

H = 0.2475 × 3 = 0.7425

Tree 2 leaf output:

w = -(-1.35) / 0.7425

**w ≈ +1.82**

Learning rate = 0.1.

Correction = 0.1 × 1.82 = 0.182.

Previous raw score = 0.2.

New raw score = 0.2 + 0.182 = 0.382.

After sigmoid:

**New survival probability ≈ 59.4%.**

Note: Values are rounded, and regularization is omitted for this example.

---

## 7. How XGBoost Chooses Splits

XGBoost evaluates candidate splits and calculates their improvement using a metric called **Gain**.

Gain measures how much a split improves the approximate objective compared with keeping the node unsplit.

Without regularization:

**Gain = ½ × [(Gₗ²/Hₗ) + (Gᵣ²/Hᵣ) - (Gₚ²/Hₚ)]**

Where:

- Gₗ, Hₗ = Left node gradient/Hessian sums.
- Gᵣ, Hᵣ = Right node gradient/Hessian sums.
- Gₚ, Hₚ = Parent node gradient/Hessian sums.

### Titanic Example

Parent:

- G = 0
- H = 1.5

Female node:

- G = -1.5
- H = 0.75

Male node:

- G = +1.5
- H = 0.75

Gain = ½ × [(2.25/0.75) + (2.25/0.75) - (0/1.5)]

Gain = ½ × (3 + 3 - 0)

**Gain = 3**

Suppose another split based on Passenger Class gives Gain = 0.

XGBoost prefers the Sex split because it provides greater improvement.

**Important:** XGBoost chooses the best eligible split using the gain calculation, not simply the largest gradient sum.

---

## 8. Gamma (γ) — Split Regularization

Gamma controls whether a new split provides enough improvement to justify making the tree more complex.

**Net Gain = Gain Before Gamma - γ**

Example:

Sex split Gain = 3

Gamma = 0.5

Net Gain = 3 - 0.5 = 2.5

Split is worthwhile.

Another split:

Age Gain = 0.02

Gamma = 0.5

Net Gain = 0.02 - 0.5 = -0.48

Reject the split.

### Key Takeaways

- Higher Gamma makes XGBoost more selective about splitting.
- Gamma discourages unnecessary branches.
- Excessively high Gamma can cause underfitting.
- Low Gamma allows more potential splits, increasing overfitting risk.

**Interview Answer:** Gamma is the minimum loss reduction required for a split. It controls tree complexity by discouraging splits that produce insufficient improvement.

---

## 9. Lambda (λ) — L2 Regularization

Lambda controls the magnitude of leaf outputs.

Without Lambda:

**w = -G/H**

With Lambda:

**w = -G/(H + λ)**

### Titanic Example

Female leaf:

G = -1.5

H = 0.75

Without Lambda:

w = -(-1.5)/0.75 = +2

With Lambda = 1:

w = -(-1.5)/(0.75 + 1)

**w ≈ +0.857**

Lambda reduces the size of the leaf correction.

### Key Takeaways

- Higher Lambda generally produces smaller leaf weights.
- It discourages excessively large corrections.
- It helps control overfitting.
- Excessively high Lambda can cause underfitting.
- Lambda also influences split Gain.

**Interview Answer:** Lambda applies L2 regularization to leaf weights, reducing overly aggressive corrections and controlling model complexity.

---

## 10. Alpha (α) — L1 Regularization

Alpha also regularizes leaf outputs.

However, unlike Lambda, Alpha can reduce a leaf output to exactly zero.

### Intuition

Suppose a Titanic leaf has:

G = -0.3

H = 0.75

Without regularization:

w = -(-0.3)/0.75 = +0.4

Now suppose:

Alpha = 0.5

Since |G| = 0.3 is less than Alpha = 0.5:

**Leaf output = 0**

The leaf contributes no correction.

### Key Takeaways

- Alpha is L1 regularization.
- It shrinks leaf weights.
- Weak leaf corrections can become exactly zero.
- It can help limit overfitting to noisy patterns.
- Excessively high Alpha can suppress useful corrections and cause underfitting.
- Alpha does not directly identify or remove noisy data points.

**Interview Answer:** Alpha introduces L1 regularization, which can shrink small leaf weights to zero and help reduce unnecessary model complexity.

---

## 11. Gamma vs Lambda vs Alpha

| Parameter | Purpose              | Main Effect                      |
| --------- | -------------------- | -------------------------------- |
| Gamma     | Split regularization | Discourages unnecessary branches |
| Lambda    | L2 regularization    | Shrinks leaf weights smoothly    |
| Alpha     | L1 regularization    | Can shrink leaf weights to zero  |

Remember:

**Gamma → Should I split?**

**Lambda → How strongly should the leaf correct?**

**Alpha → Is a weak correction worth keeping?**

These are intuition-based summaries. All three can affect tree construction through the regularized objective.

---

## 12. Overfitting vs Underfitting

### Underfitting

The model hasn't learned enough useful patterns.

Illustrative example:

Training accuracy = 65%

Validation accuracy = 64%

If both are poor relative to the baseline, the model may be underfitting.

Possible causes:

- Excessively strong regularization.
- Trees too shallow.
- Too few boosting rounds.
- Insufficient predictive features.

### Overfitting

The model learns training-specific patterns that don't generalize well.

Illustrative example:

Training accuracy = 98%

Validation accuracy = 70%

The large gap suggests overfitting.

Possible causes:

- Trees too deep.
- Too many boosting rounds.
- Weak regularization.
- Noisy data or small dataset.

### Important Distinction

Low training + low validation performance → Possible underfitting.

High training + much lower validation performance → Possible overfitting.

High training + high validation performance → Potentially good generalization.

Always compare against appropriate baselines and verify that the evaluation split is reliable.

---

## 13. Complete XGBoost Workflow

1. Initialize model predictions.
2. Calculate gradients and Hessians.
3. Evaluate candidate feature splits using Gain.
4. Apply regularization when evaluating splits.
5. Choose the best eligible split.
6. Continue splitting nodes when worthwhile.
7. Calculate regularized leaf weights.
8. Multiply tree outputs by the learning rate.
9. Update ensemble predictions.
10. Recalculate gradients and Hessians.
11. Build the next tree.
12. Repeat until training ends or early stopping occurs.

---

## 14. Most Important Equations

**Gradient (Binary Logistic):**

g = p - y

**Hessian (Binary Logistic):**

h = p(1 - p)

**Gradient Sum:**

G = Σgᵢ

**Hessian Sum:**

H = Σhᵢ

**Leaf Weight Without Regularization:**

w = -G/H

**Leaf Weight With Lambda (no Alpha):**

w = -G/(H + λ)

**Leaf Weight With Alpha and Lambda:**

w = -sign(G) × max(|G| - α, 0) / (H + λ)

**Split Gain With Lambda and Gamma (no Alpha):**

Gain = ½ × [Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) - Gₚ²/(Hₚ+λ)] - γ

**Ensemble Update:**

Fₘ(x) = Fₘ₋₁(x) + ηfₘ(x)

**Sigmoid:**

p = 1/(1 + e^(-z))

---

## 15. Interview Questions — Quick Revision

**Q1. Is XGBoost bagging or boosting?**

Boosting. Trees are trained sequentially, each contributing corrections to the current ensemble.

**Q2. Why does XGBoost use gradients and Hessians?**

Gradients describe the direction of loss change, while Hessians describe its curvature. Together, they enable efficient second-order approximations for leaf weights and split Gain.

**Q3. What does a leaf predict?**

A correction to the current raw model score. The meaning of the raw score depends on the objective.

**Q4. How does XGBoost determine the best split?**

It calculates the regularized Gain of eligible candidate splits and chooses the best one.

**Q5. Why is learning rate needed?**

It scales each tree's contribution so the ensemble updates gradually rather than applying every correction at full strength.

**Q6. What is Gamma?**

A penalty on adding leaves, used to discourage insufficiently beneficial splits.

**Q7. What is Lambda?**

L2 regularization that discourages large leaf weights.

**Q8. What is Alpha?**

L1 regularization that can shrink weak leaf weights to exactly zero.

**Q9. How does Tree 2 know what to learn?**

XGBoost recalculates gradients and Hessians from the updated ensemble predictions and uses them to build Tree 2.

**Q10. Does increasing regularization always improve accuracy?**

No. Excessive regularization can cause underfitting. Hyperparameters must be tuned using validation data.

---

## Final Mental Model

XGBoost follows this cycle:

**Current Predictions → Gradients + Hessians → Best Splits → Leaf Corrections → Learning Rate → Updated Predictions → Next Tree**

Remember these five ideas:

1. **Gradient:** Which direction should we correct?
2. **Hessian:** How does the gradient change?
3. **Gain:** Is this split worth making?
4. **Gamma, Lambda, Alpha:** How do we control complexity?
5. **Learning Rate:** How much of the new correction should we apply?

**Final Summary:** XGBoost is a regularized gradient boosting algorithm that sequentially builds trees using first- and second-order derivatives of a loss function. It chooses splits by estimated objective improvement, calculates optimal leaf corrections, and adds learning-rate-scaled tree predictions to improve the ensemble.
