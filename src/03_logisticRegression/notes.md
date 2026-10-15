# Logistic Regression — From First Principles

> **Core idea**
>
> I am Logistic Regression.
>
> Give me some features `X` and a categorical target `y`.
>
> My job is not to predict an unrestricted number.
>
> My job is to estimate:
>
> **How likely is this observation to belong to a particular class?**

---

# 1. What Problem Am I Solving?

Suppose you give me historical data:

| Hours Studied | Passed |
| ------------: | -----: |
|             1 |     No |
|             2 |     No |
|             3 |     No |
|             5 |    Yes |
|             6 |    Yes |
|             8 |    Yes |

Then a new student arrives:

```text
Hours Studied = 4.5
Passed = ???
```

This isn't a regression target like:

```text
Salary = ₹70,000
House Price = ₹80 lakh
Temperature = 32.4°C
```

There are two possible classes:

```text
No
Yes
```

We usually encode them:

```text
No  = 0
Yes = 1
```

So this is:

> **Binary Classification**

---

# 2. Why Can't I Just Use Linear Regression?

Linear Regression already gives me:

```text
z = wx + b
```

Why not simply train that and say:

```text
z < 0.5 → class 0
z >= 0.5 → class 1
```

Seems reasonable.

But there is a problem.

Linear Regression can output:

```text
-4.7
0.2
0.8
3.6
27
```

Classification probability must stay between:

```text
0 and 1
```

A probability of:

```text
3.6
```

makes no sense.

Neither does:

```text
-4.7
```

So I need something that takes an unrestricted number:

```text
-∞ < z < +∞
```

and converts it into:

```text
0 < probability < 1
```

---

# 3. Keep the Linear Part

I don't throw away Linear Regression's useful idea.

I still calculate:

```text
z = wx + b
```

For multiple features:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Or:

```text
z = wᵀx + b
```

But now:

> `z` is NOT my final prediction.

Think of `z` as my **raw score**.

---

# 4. What Exactly Is `z`?

Suppose:

```text
z = -8.73
```

That doesn't mean:

```text
Probability = -8.73
```

It is simply the score produced by the weighted combination of features.

Conceptually:

```text
Features
   ↓
Weights
   ↓
Weighted sum
   ↓
z
```

Example:

```text
Age      = 25
Income   = 50
Debt     = 10

w₁ = 0.4
w₂ = 0.2
w₃ = -0.8
b  = -5
```

Then:

```text
z = w₁Age + w₂Income + w₃Debt + b
```

`z` tells me which direction and how strongly the evidence is leaning.

Roughly:

```text
Very negative z
→ strong evidence toward class 0

z around 0
→ uncertain region

Very positive z
→ strong evidence toward class 1
```

But I still need to convert this score into probability.

---

# 5. Enter the Sigmoid Function

I pass `z` through:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

I usually write:

```text
p = sigmoid(z)
```

where:

```text
p = estimated probability of class 1
```

So my complete prediction process becomes:

```text
X
 ↓
z = wᵀx + b
 ↓
sigmoid(z)
 ↓
p
```

---

# 6. Why Sigmoid?

Look at what happens.

If:

```text
z → very large positive number
```

then:

```text
sigmoid(z) → 1
```

If:

```text
z → very large negative number
```

then:

```text
sigmoid(z) → 0
```

If:

```text
z = 0
```

then:

```text
sigmoid(0) = 0.5
```

So sigmoid maps:

```text
(-∞, +∞)
```

into:

```text
(0, 1)
```

Exactly what I need for a probability-like output.

---

# 7. Example — z = -8.73

Suppose:

```text
z = -8.73
```

Then:

```text
p = sigmoid(-8.73)
```

Approximately:

```text
p ≈ 0.00016
```

So:

```text
P(y = 1 | x) ≈ 0.016%
```

That means I strongly lean toward:

```text
class 0
```

Notice the distinction:

```text
z = -8.73
```

is the raw score.

```text
p ≈ 0.00016
```

is the estimated probability for class 1.

---

# 8. What Probability Am I Actually Predicting?

This is important.

When I output:

```text
p = 0.8
```

I mean:

```text
P(y = 1 | x) = 0.8
```

Since binary classification has only two possibilities:

```text
P(y = 0 | x) = 1 - p
```

Therefore:

```text
P(y=1) = p

P(y=0) = 1-p
```

Example:

```text
p = 0.8

Class 1 probability = 0.8
Class 0 probability = 0.2
```

---

# 9. Probability Is NOT Yet a Class Prediction

Suppose:

```text
p = 0.73
```

That's a probability.

But maybe the application needs:

```text
YES / NO
Fraud / Not Fraud
Disease / No Disease
```

So I need a **decision threshold**.

A common default is:

```text
threshold = 0.5
```

Then:

```text
p >= 0.5 → predict class 1

p < 0.5  → predict class 0
```

---

# 10. Why 0.5?

Because:

```text
sigmoid(0) = 0.5
```

So:

```text
z >= 0
```

corresponds to:

```text
p >= 0.5
```

This makes `z = 0` the default binary decision boundary.

But:

> **0.5 is NOT a law of nature.**

We can choose another threshold.

For example:

```text
Fraud detection:

threshold = 0.30
```

might intentionally catch more suspicious transactions.

Changing the threshold changes the trade-off between:

```text
False Positives
False Negatives
Precision
Recall
```

The model produces a score/probability.

**We choose the operating threshold based on the problem.**

---

# 11. Decision Boundary

Suppose:

```text
p = 0.5
```

Since:

```text
sigmoid(0) = 0.5
```

the boundary occurs where:

```text
z = 0
```

But:

```text
z = wᵀx + b
```

Therefore the default decision boundary is:

```text
wᵀx + b = 0
```

With two features:

```text
w₁x₁ + w₂x₂ + b = 0
```

This describes a line.

With three features:

```text
a plane
```

With many features:

```text
a hyperplane
```

---

# 12. Important Realization

Sigmoid is nonlinear.

But the standard Logistic Regression decision boundary is still linear in the original features:

```text
wᵀx + b = 0
```

So Logistic Regression does NOT magically create arbitrary nonlinear decision boundaries.

If the true classes look like:

```text
⭕⭕⭕
⭕XXX⭕
⭕XXX⭕
⭕⭕⭕
```

a plain linear boundary may struggle.

We would need:

```text
feature engineering
polynomial features
interactions
or a nonlinear model
```

---

# 13. Okay — What Am I Actually Learning?

My model is:

```text
z = wᵀx + b

p = sigmoid(z)
```

You already give me:

```text
X
y
```

What don't I know?

```text
w
b
```

Therefore training means:

> **Find `w` and `b` that assign high probability to the correct classes.**

---

# 14. Could I Just Use the Raw Error?

Suppose:

```text
Actual y = 1
Predicted p = 0.7
```

Why not simply use:

```text
y - p
```

which gives:

```text
1 - 0.7 = 0.3
```

There's a fundamental distinction:

```text
Error
```

and:

```text
Loss Function
```

are not necessarily the same thing.

`y - p` tells me:

```text
direction and size of prediction error
```

But I need a training objective with useful mathematical behavior that strongly penalizes confident wrong predictions and corresponds naturally to probabilistic classification.

That's where **Binary Cross-Entropy** enters.

---

# 15. Think About What I Want From a Loss

Suppose the true label is:

```text
y = 1
```

Compare these predictions:

```text
p = 0.9
p = 0.6
p = 0.1
p = 0.0001
```

I want:

```text
p = 0.9
→ small punishment

p = 0.6
→ moderate punishment

p = 0.1
→ large punishment

p = 0.0001
→ HUGE punishment
```

Why?

Because:

```text
y = 1
p = 0.0001
```

means:

> I was almost completely certain that the correct answer was impossible.

That deserves a strong penalty.

---

# 16. Loss When the Actual Class Is 1

When:

```text
y = 1
```

we use:

```text
Loss = -log(p)
```

Look at the behavior:

```text
p → 1
Loss → 0

p → 0
Loss → very large
```

Exactly what we want.

Example:

```text
Actual y = 1
Prediction p = 0.99

Loss = -log(0.99)
→ tiny
```

But:

```text
Actual y = 1
Prediction p = 0.01

Loss = -log(0.01)
→ large
```

---

# 17. Loss When the Actual Class Is 0

Now suppose:

```text
y = 0
```

`p` still means:

```text
probability of class 1
```

Therefore the probability assigned to the correct class 0 is:

```text
1 - p
```

So:

```text
Loss = -log(1-p)
```

Now:

```text
p → 0
Loss → 0
```

Good.

And:

```text
p → 1
Loss → huge
```

Correct again.

---

# 18. So Do We Have Two Different Loss Functions?

Conceptually:

```text
If y = 1:

Loss = -log(p)


If y = 0:

Loss = -log(1-p)
```

But we don't want to write an `if` statement mathematically for every row.

We can combine both into one equation:

```text
Loss = -[y log(p) + (1-y) log(1-p)]
```

This is:

> **Binary Cross-Entropy — BCE**

also commonly called:

> **Log Loss**

---

# 19. Why Does the Combined BCE Equation Work?

This was one of the important "hold on, what?" moments.

Let's substitute the actual labels.

## Case 1 — y = 1

```text
Loss = -[1 × log(p) + (1-1) × log(1-p)]
```

Since:

```text
1 - 1 = 0
```

we get:

```text
Loss = -log(p)
```

The class-0 term disappears.

---

## Case 2 — y = 0

```text
Loss = -[0 × log(p) + (1-0) × log(1-p)]
```

The first term disappears.

So:

```text
Loss = -log(1-p)
```

Therefore the single equation automatically selects the correct loss based on `y`.

That's why `y` and `(1-y)` appear.

They're acting like switches.

---

# 20. Why the Negative Sign?

Probabilities are between:

```text
0 and 1
```

For numbers in that range:

```text
log(p) <= 0
```

Example:

```text
log(0.9)  → negative
log(0.5)  → negative
log(0.01) → very negative
```

But we want loss to be positive.

So we use:

```text
-log(...)
```

---

# 21. Where Did BCE Actually Come From?

It isn't an arbitrary punishment someone invented.

Logistic Regression is fundamentally a probabilistic model.

For one observation:

```text
P(y | x)
```

can be written as:

```text
p^y × (1-p)^(1-y)
```

Why?

If:

```text
y = 1
```

then:

```text
p¹ × (1-p)⁰ = p
```

If:

```text
y = 0
```

then:

```text
p⁰ × (1-p)¹ = 1-p
```

So this expression gives the probability the model assigned to the observed class.

---

# 22. Maximum Likelihood

During training, I want parameters that make the labels we actually observed as probable as possible.

For many observations:

```text
Likelihood =
product of the probabilities assigned
to all observed outcomes
```

Conceptually:

```text
L = P(y₁|x₁) × P(y₂|x₂) × ... × P(yₙ|xₙ)
```

I want:

```text
Maximum Likelihood
```

In other words:

> Find `w,b` under which the observed training labels are most probable.

---

# 23. Why Take the Log?

Products of many probabilities become extremely tiny.

Example:

```text
0.8 × 0.7 × 0.9 × 0.6 × ...
```

With thousands or millions of observations, numerical underflow becomes a problem.

Logs convert:

```text
multiplication
```

into:

```text
addition
```

Because:

```text
log(a × b) = log(a) + log(b)
```

So instead of maximizing likelihood, we maximize:

```text
log-likelihood
```

And maximizing log-likelihood is equivalent to minimizing:

```text
negative log-likelihood
```

For binary Logistic Regression, this gives us BCE.

So:

```text
Probability model
      ↓
Likelihood
      ↓
Log-Likelihood
      ↓
Negative Log-Likelihood
      ↓
Binary Cross-Entropy
```

BCE isn't random.

It falls naturally out of the probability model.

---

# 24. Cost Across the Whole Dataset

For one observation:

```text
Lossᵢ =
-[yᵢ log(pᵢ) + (1-yᵢ)log(1-pᵢ)]
```

For the entire training dataset:

```text
Cost =
average BCE across all training examples
```

So my objective becomes:

> **Find `w,b` that minimize Binary Cross-Entropy.**

---

# 25. Why Not Just Use MSE Like Linear Regression?

Technically, we could combine sigmoid with MSE.

But it is generally not the natural objective for Logistic Regression.

With:

```text
sigmoid + BCE
```

we get:

```text
a likelihood-based objective
clean optimization behavior
strong penalties for confident wrong predictions
simple gradients
```

BCE matches the Bernoulli probability model behind binary Logistic Regression.

So:

```text
Linear Regression
→ Gaussian-style squared-error modeling
→ MSE / least squares

Logistic Regression
→ Bernoulli probability modeling
→ BCE / log loss
```

---

# 26. How Do I Learn `w` and `b`?

Same broad optimization idea as before.

I have:

```text
Cost(w,b)
```

I need:

```text
w,b that minimize Cost
```

So I can use gradient-based optimization.

Conceptually:

```text
Initialize w,b
      ↓
Calculate z
      ↓
Sigmoid
      ↓
Get probabilities p
      ↓
Calculate BCE
      ↓
Calculate gradients
      ↓
Update w,b
      ↓
Repeat
```

---

# 27. The Beautiful Gradient Result

After combining:

```text
Linear score
+
Sigmoid
+
Binary Cross-Entropy
```

the gradient simplifies nicely.

For one feature:

```text
dw = average of ((p - y) × x)
```

and:

```text
db = average of (p - y)
```

Notice something familiar.

Linear Regression gave us something structurally similar:

```text
(prediction - actual) × feature
```

Logistic Regression gives:

```text
(probability - actual) × feature
```

So again:

```text
p - y
```

tells me:

> How wrong was my probability?

And:

```text
× x
```

tells me:

> How strongly was this feature involved?

---

# 28. Wait — Didn't We Say `y - p` Isn't the Loss?

Correct.

This distinction is extremely important.

```text
p - y
```

appearing inside the gradient does NOT mean:

```text
Loss = p - y
```

The loss is:

```text
Binary Cross-Entropy
```

After differentiating BCE through sigmoid, part of the derivative simplifies to:

```text
p - y
```

So:

```text
LOSS
≠
p - y
```

but:

```text
GRADIENT
contains
p - y
```

Don't mix them.

---

# 29. I Am Logistic Regression — One Training Iteration

Suppose I initialize:

```text
w = 0
b = 0
```

For every row:

### Step 1 — Calculate raw score

```text
z = wᵀx + b
```

### Step 2 — Convert score to probability

```text
p = sigmoid(z)
```

### Step 3 — Compare with actual class

```text
Actual = y
Predicted probability = p
```

### Step 4 — Calculate BCE

```text
Loss =
-[y log(p) + (1-y)log(1-p)]
```

### Step 5 — Calculate gradients

```text
dw = average((p-y)x)

db = average(p-y)
```

### Step 6 — Update parameters

```text
w = w - αdw

b = b - αdb
```

### Step 7 — Repeat

Eventually my parameters settle near values that minimize the objective.

---

# 30. What Happens During Inference?

This was another important question.

Suppose I trained using 100 rows.

Now a **101st row** arrives.

We know its features:

```text
x₁₀₁
```

But obviously we don't know:

```text
y₁₀₁
```

If we knew the answer, there would be nothing to predict.

During inference I simply do:

```text
z = wᵀx₁₀₁ + b
```

then:

```text
p = sigmoid(z)
```

then optionally:

```text
if p >= threshold:
    class = 1
else:
    class = 0
```

That's it.

---

# 31. Do My Weights Update During Inference?

No.

Absolutely not.

Training:

```text
X + y
 ↓
Loss
 ↓
Gradients
 ↓
Update weights
```

Inference:

```text
New X
 ↓
Existing learned weights
 ↓
z
 ↓
sigmoid
 ↓
probability
 ↓
class
```

No actual `y`.

No BCE.

No gradient.

No parameter update.

My weights remain fixed until someone explicitly retrains or updates the model.

---

# 32. Why Is Logistic Regression Called "Regression"?

This name confuses almost everyone initially.

Logistic Regression is normally used for:

```text
classification
```

So why "regression"?

Because internally I model a continuous quantity related to the class probability.

More precisely, I model the **log-odds** as a linear function of the features.

---

# 33. Odds

Suppose:

```text
p = 0.8
```

Probability of class 1:

```text
0.8
```

Probability of class 0:

```text
1 - p = 0.2
```

Odds are:

```text
odds = p / (1-p)
```

So:

```text
odds = 0.8 / 0.2
     = 4
```

Meaning:

```text
class 1 is 4 times as likely as class 0
```

according to the model.

---

# 34. Log-Odds / Logit

Odds range from:

```text
0 → +∞
```

Take the logarithm:

```text
log-odds = log(p / (1-p))
```

Now the range becomes:

```text
-∞ → +∞
```

And Logistic Regression assumes:

```text
log(p / (1-p)) = wᵀx + b
```

Notice:

```text
right side = z
```

Therefore:

```text
log-odds = z
```

Solving this equation for `p` gives us the sigmoid function.

So sigmoid isn't merely a random S-shaped squashing function.

It naturally appears when we assume:

> **The log-odds of class 1 are a linear function of the features.**

---

# 35. Interpreting a Logistic Regression Weight

We have:

```text
log-odds = w₁x₁ + w₂x₂ + ... + b
```

Suppose:

```text
w₁ = 0.7
```

Increasing `x₁` by one unit, while holding other features constant, increases the log-odds by:

```text
0.7
```

Exponentiate:

```text
e^0.7 ≈ 2.01
```

So the odds are multiplied by approximately:

```text
2.01
```

This is why Logistic Regression can be highly interpretable.

---

# 36. Multiple Features

Suppose we're predicting whether someone buys a product.

Features:

```text
Age
Income
WebsiteVisits
PreviousPurchases
```

Then:

```text
z =
w₁Age
+ w₂Income
+ w₃WebsiteVisits
+ w₄PreviousPurchases
+ b
```

Then:

```text
p = sigmoid(z)
```

Every feature contributes to the raw score according to its learned weight.

---

# 37. Feature Scaling

Logistic Regression can benefit substantially from feature scaling.

Suppose:

```text
Age    → 18–80
Income → 20,000–2,000,000
```

Gradient-based optimization can become poorly conditioned.

Scaling makes optimization easier.

This becomes especially important when using regularization.

Typical preprocessing:

```text
StandardScaler
```

Fit the scaler only on training data.

Then use that same fitted scaler to transform validation/test/production data.

---

# 38. Regularization

Logistic Regression can overfit, especially with:

```text
many features
correlated features
small datasets
noisy predictors
```

So we can penalize very large coefficients.

---

## L2 Regularization

Add a penalty related to:

```text
Σw²
```

This encourages smaller weights.

Often useful when many features contain some signal.

---

## L1 Regularization

Add a penalty related to:

```text
Σ|w|
```

This can push some coefficients exactly to zero.

Therefore it can also act as a form of feature selection.

---

## Regularization Strength

In many sklearn Logistic Regression configurations:

```text
C
```

controls inverse regularization strength.

So:

```text
small C
→ stronger regularization

large C
→ weaker regularization
```

This inverse relationship is easy to forget.

---

# 39. Class Imbalance

Suppose:

```text
99% → class 0
1%  → class 1
```

A useless model could predict:

```text
class 0
```

for everyone and achieve:

```text
99% accuracy
```

So accuracy alone can be dangerously misleading.

We need to examine:

```text
Confusion Matrix
Precision
Recall
F1
ROC-AUC
PR-AUC
```

depending on the problem.

---

# 40. Confusion Matrix

For binary classification:

```text
                 Predicted
                0         1

Actual 0       TN        FP

Actual 1       FN        TP
```

Where:

```text
TP = correctly predicted positive

TN = correctly predicted negative

FP = predicted positive but actually negative

FN = predicted negative but actually positive
```

---

# 41. Precision

Precision asks:

> Of everything I predicted as positive, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

Useful when false positives are expensive.

Example:

```text
Spam filtering
```

depending on the product requirements.

---

# 42. Recall

Recall asks:

> Of all actual positive cases, how many did I find?

```text
Recall = TP / (TP + FN)
```

Useful when missing a positive case is expensive.

Example:

```text
disease screening
fraud detection
```

depending on the application.

---

# 43. Precision vs Recall and the Threshold

Suppose I lower my threshold:

```text
0.5 → 0.3
```

Now more observations become class 1.

Usually:

```text
Recall ↑

but

False Positives may ↑
Precision may ↓
```

Raise the threshold:

```text
0.5 → 0.8
```

Now I'm stricter about predicting class 1.

Often:

```text
Precision ↑

but

Recall ↓
```

So threshold selection is a **product/business decision**, not simply a fixed property of Logistic Regression.

---

# 44. Multiclass Classification

Binary Logistic Regression handles:

```text
0 vs 1
```

But suppose we have:

```text
Cat
Dog
Horse
```

Now we need multiple class probabilities.

One approach is:

```text
One-vs-Rest
```

Another important approach is:

> **Multinomial Logistic Regression using Softmax**

---

# 45. From Sigmoid to Softmax

For binary classification, one score is enough to represent class probability.

For multiclass classification, calculate one score per class.

Example:

```text
z_cat   = 2.1
z_dog   = 4.5
z_horse = 1.2
```

These raw scores aren't probabilities.

Softmax converts them into probabilities that:

```text
are all between 0 and 1
```

and:

```text
sum to 1
```

Conceptually:

```text
softmax(zᵢ) = e^(zᵢ) / Σe^(zⱼ)
```

So we may get:

```text
Cat   → 0.08
Dog   → 0.88
Horse → 0.04
```

Then:

```text
Predicted class = Dog
```

---

# 46. Why Exponentials in Softmax?

Exponentials guarantee positive values:

```text
e^z > 0
```

Then dividing each exponential by their total guarantees:

```text
all probabilities sum to 1
```

It also preserves ordering:

```text
larger z
→ larger probability
```

So Softmax converts arbitrary class scores into a valid probability distribution.

---

# 47. Binary vs Multiclass

```text
BINARY LOGISTIC REGRESSION

X
↓
one linear score z
↓
sigmoid
↓
P(class 1)
↓
threshold
↓
class 0 / class 1
```

Multiclass:

```text
MULTINOMIAL LOGISTIC REGRESSION

X
↓
one score per class
↓
[z₁, z₂, ..., zₖ]
↓
softmax
↓
[p₁, p₂, ..., pₖ]
↓
highest probability
↓
predicted class
```

---

# 48. When Logistic Regression Works Well

Logistic Regression is strong when:

```text
Decision boundary is approximately linear

Interpretability matters

You need probabilities

Dataset is not extremely complex

You want a strong baseline

Features contain meaningful linear signal
```

It is often surprisingly competitive.

Simple does not mean useless.

---

# 49. Where Logistic Regression Struggles

Plain Logistic Regression struggles when:

```text
Decision boundary is highly nonlinear

Complex feature interactions dominate

Important relationships aren't represented in features

Data contains severe outliers

Classes require complicated geometric boundaries
```

Feature engineering can help.

Otherwise, a more expressive model may be appropriate.

---

# 50. A Subtle Point About "Probability"

Sigmoid gives me a value between 0 and 1.

But:

> **A number between 0 and 1 is not automatically a well-calibrated probability.**

If I predict:

```text
0.8
```

for many observations, ideally roughly 80% of those observations should actually belong to class 1.

That's **probability calibration**.

Calibration can be checked separately.

So distinguish:

```text
discrimination
→ can I rank/separate classes?

calibration
→ do my probability values correspond well to observed frequencies?
```

---

# 51. Data Leakage Still Matters

Correct workflow:

```text
Raw Data
   ↓
Train / Validation / Test split
   ↓
Fit preprocessing ONLY on train
   ↓
Transform train
   ↓
Transform validation/test using fitted preprocessing
   ↓
Train Logistic Regression
   ↓
Evaluate
```

Do not let test information influence:

```text
Scaling
Imputation
Feature selection
Encoding that learns statistics
Model training
```

---

# 52. I Am Logistic Regression — Final Mental Model

```text
You give me:

X = features
y = class labels

        ↓

I encode binary labels:

0 / 1

        ↓

I calculate a linear score:

z = wᵀx + b

        ↓

But z can range from:

-∞ to +∞

        ↓

I need something probability-like

        ↓

Sigmoid

        ↓

p = P(y=1 | x)

        ↓

During training I know actual y

        ↓

I ask:

How much probability did I assign
to the class that actually happened?

        ↓

Binary Cross-Entropy

        ↓

Correct confident prediction
→ tiny loss

Wrong confident prediction
→ huge loss

        ↓

Calculate gradients

        ↓

Update w,b

        ↓

Repeat

        ↓

Learn parameters

        ↓

TRAINING ENDS

        ↓

New observation arrives

        ↓

I DO NOT know y

        ↓

Use fixed learned w,b

        ↓

z = wᵀx + b

        ↓

sigmoid(z)

        ↓

probability p

        ↓

apply chosen threshold

        ↓

predicted class
```

---

# 53. The Doubts Worth Remembering

These are more useful than memorizing equations.

### "Why not just use Linear Regression?"

Because its output is unrestricted and does not naturally represent binary class probability.

---

### "What the hell is z?"

`z` is the raw weighted score:

```text
z = wᵀx + b
```

It is NOT yet a probability.

---

### "Why sigmoid?"

Because we need to map an unrestricted score into `(0,1)`, and sigmoid arises naturally when modeling log-odds linearly.

---

### "Why can't y-p itself be my loss?"

Because raw prediction error is not the probabilistically appropriate training objective.

BCE comes from maximum likelihood and properly punishes confident wrong predictions.

---

### "Then why does p-y appear during training?"

Because after differentiating BCE through sigmoid, the gradient simplifies and contains:

```text
p - y
```

Gradient ≠ loss.

---

### "Why does BCE look different for y=0 and y=1?"

Because we're penalizing the model according to the probability it assigned to the **actual class**.

```text
y=1 → correct-class probability = p

y=0 → correct-class probability = 1-p
```

The combined BCE equation handles both automatically.

---

### "What happens when the 101st row comes and y is unknown?"

That's inference.

We don't need `y`.

We only calculate:

```text
X
↓
z
↓
sigmoid
↓
probability
↓
class
```

---

### "Do the weights update for that 101st row?"

No.

Weights update during training.

During normal inference they remain fixed.

---

### "Does sigmoid create a nonlinear decision boundary?"

Not by itself.

With original linear features, the default boundary remains:

```text
wᵀx + b = 0
```

Sigmoid makes the **output mapping nonlinear**, while the standard decision boundary remains linear.

---

# 54. If I Forget Everything Else

Reconstruct me from the problem:

```text
I need classification
        ↓
Linear Regression gives unrestricted output
        ↓
Keep its useful linear score
        ↓
z = wᵀx + b
        ↓
Convert score into probability
        ↓
Sigmoid
        ↓
Need to learn w,b
        ↓
Need a loss appropriate for probabilities
        ↓
Binary Cross-Entropy
        ↓
BCE comes from Maximum Likelihood
        ↓
Minimize BCE
        ↓
Gradient-based optimization
        ↓
Learn w,b
        ↓
At inference:
NO y
NO BCE
NO weight update
        ↓
z → sigmoid → probability → threshold → class
```

> If you can reconstruct this chain instead of memorizing isolated formulas, you understand Logistic Regression.
