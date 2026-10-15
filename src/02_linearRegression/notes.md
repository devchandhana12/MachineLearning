# Linear Regression — First Principles

> **Core idea**
>
> I am a Linear Regression algorithm.
>
> Give me historical examples containing some input `X` and a continuous numerical target `y`.
>
> My job is to learn the relationship between them and predict `y` for new data.

---

# 1. What problem am I solving?

Suppose you give me:

| Experience | Salary |
| ---------: | -----: |
|     1 year |     30 |
|    2 years |     40 |
|    3 years |     50 |
|    4 years |     60 |

Then you ask:

```text
Experience = 5 years
Salary = ???
```

I need to predict a **continuous number**.

Therefore this is a:

> **Regression problem**

Looking at the historical data, I notice:

```text
Experience ↑
     ↓
Salary tends to ↑
```

My first hypothesis is:

> Maybe I can approximate this relationship using a straight line.

---

# 2. My Model

I choose the equation:

```text
ŷ = wx + b
```

Where:

```text
x  = input feature
y  = actual target
ŷ  = my predicted target
w  = weight / slope
b  = bias / intercept
```

Suppose I eventually learn:

```text
w = 10
b = 20
```

Then:

```text
ŷ = 10x + 20
```

For:

```text
x = 3
```

I predict:

```text
ŷ = 10(3) + 20
ŷ = 50
```

---

# 3. What Exactly Am I Learning?

This is important.

You already gave me:

```text
x
y
```

And I already decided my model structure:

```text
ŷ = wx + b
```

What don't I know?

```text
w = ?
b = ?
```

Therefore:

> **Training Linear Regression means finding good values for `w` and `b`.**

These are called the **model parameters**.

```text
DATA
 ↓
Find w and b
 ↓
Use them for future predictions
```

---

# 4. What Does `w` Mean?

Consider:

```text
ŷ = wx + b
```

Suppose:

```text
w = 10
```

If:

```text
x increases by 1
```

then my prediction increases by approximately:

```text
10
```

So `w` controls the direction and strength of the relationship.

```text
w > 0
→ prediction increases as x increases

w < 0
→ prediction decreases as x increases

w ≈ 0
→ x has little linear influence on the prediction
```

Geometrically:

> `w` controls the slope of my line.

---

# 5. What Does `b` Mean?

Again:

```text
ŷ = wx + b
```

Set:

```text
x = 0
```

Then:

```text
ŷ = b
```

So `b` determines where my line crosses the y-axis.

It shifts the entire line up or down.

> **Important**
>
> `b` does not always have a meaningful real-world interpretation.
>
> For example, predicting salary at `experience = 0` may or may not be meaningful depending on the problem.
>
> But mathematically, the bias is still useful for positioning the fitted line.

---

# 6. How Do I Find `w` and `b`?

Initially, I don't know them.

I can start with something simple:

```text
w = 0
b = 0
```

Then:

```text
ŷ = 0x + 0
ŷ = 0
```

Every prediction becomes:

```text
0
```

Obviously terrible.

But that's okay.

I now have something extremely important:

> **A prediction that I can compare against reality.**

---

# 7. Residual / Prediction Error

Suppose:

```text
Actual y     = 50
Predicted ŷ  = 35
```

The residual is:

```text
Residual = y - ŷ

Residual = 50 - 35
Residual = 15
```

If I overpredict:

```text
Actual     = 50
Prediction = 60

Residual = 50 - 60
Residual = -10
```

So residuals can be:

```text
positive
negative
zero
```

Zero means:

```text
prediction = actual value
```

---

# 8. Why Can't I Simply Add All Residuals?

Suppose my errors are:

```text
+10
-10
+5
-5
```

Add them:

```text
10 - 10 + 5 - 5 = 0
```

It looks like:

```text
Total error = 0
```

But I clearly made four mistakes.

Positive and negative errors cancelled each other.

So I need a way to measure errors without cancellation.

---

# 9. Mean Squared Error — MSE

I square every error.

```text
Error      Squared Error

+10   →    100
-10   →    100
+5    →     25
-5    →     25
```

Then average them.

```text
MSE = average of (y - ŷ)²
```

Mathematically:

```text
MSE = Σ(yᵢ - ŷᵢ)² / n
```

Now positive and negative errors cannot cancel.

---

# 10. Why Do We Square Errors?

There are several reasons.

### Reason 1 — Prevent cancellation

```text
+10 → 100
-10 → 100
```

Both mistakes contribute positively.

### Reason 2 — Large errors are punished more

```text
Error = 2
Squared = 4

Error = 10
Squared = 100
```

So being very wrong becomes expensive.

### Reason 3 — Smooth optimization

Squared error gives us a smooth differentiable function.

That makes gradient-based optimization convenient.

### Reason 4 — Convexity

For ordinary Linear Regression, squared-error optimization is convex.

That means there is one global minimum rather than many bad local minima.

We'll come back to this.

---

# 11. Loss vs Cost

For one training example:

```text
Loss = (y - ŷ)²
```

For the entire training dataset:

```text
Cost = average loss across all examples
```

So:

```text
One row        → Loss
Whole dataset  → Cost
```

People sometimes use these words loosely, but this distinction is useful.

---

# 12. My Cost Function

Since:

```text
ŷ = wx + b
```

my cost depends on `w` and `b`.

Conceptually:

```text
J(w,b) = average of (prediction - actual)²
```

or:

```text
J(w,b) = Σ(wxᵢ + b - yᵢ)² / n
```

You may also see:

```text
J(w,b) = Σ(wxᵢ + b - yᵢ)² / 2n
```

Why `2n` instead of `n`?

The `2` is just mathematical convenience.

When we differentiate the squared term, a `2` appears.

The `1/2` cancels it.

It does **not** change where the minimum occurs.

---

# 13. What Is My Actual Objective?

Now my entire training objective can be stated in one sentence:

> **Find the values of `w` and `b` that minimize the cost.**

```text
Bad w,b
 ↓
Bad predictions
 ↓
Large MSE


Good w,b
 ↓
Better predictions
 ↓
Small MSE
```

So I need an optimization algorithm.

---

# 14. Cost Function Is NOT Gradient Descent

Very common confusion.

### Cost Function

Answers:

```text
"How bad are my current parameters?"
```

Example:

```text
J(w,b) = 500
```

It measures the problem.

### Gradient Descent

Answers:

```text
"How should I change w and b
to reduce that cost?"
```

It solves the optimization problem.

So:

```text
Cost Function
     ↓
Measures error

Gradient Descent
     ↓
Updates parameters
```

They are not the same thing.

---

# 15. Imagine My Cost Surface

Suppose I try many values of `w`.

Each produces a different cost.

Conceptually:

```text
Cost
  ^
  |
  | \                 /
  |  \               /
  |   \             /
  |    \           /
  |     \         /
  |      \_______/
  |          ★
  +----------------------> w

             ★
        minimum cost
```

With both `w` and `b`, this becomes a multidimensional surface.

I may currently be somewhere up the slope.

I need to move downhill.

---

# 16. How Do I Know Which Direction Is Downhill?

This is where calculus enters.

I ask:

```text
If I slightly change w,
how does my cost change?
```

That is:

```text
∂J / ∂w
```

Then:

```text
If I slightly change b,
how does my cost change?
```

That is:

```text
∂J / ∂b
```

These derivatives tell me the slope of the cost function with respect to my parameters.

Together, they form the **gradient**.

---

# 17. What Does the Gradient Tell Me?

Suppose:

```text
∂J/∂w > 0
```

Increasing `w` increases the cost.

So I should decrease `w`.

If:

```text
∂J/∂w < 0
```

Increasing `w` decreases the cost.

So I should increase `w`.

The gradient points toward:

> **steepest increase in cost**

But I want:

> **decrease in cost**

Therefore I move in the opposite direction.

---

# 18. Gradient Descent Update Rule

For weight:

```text
w_new = w_old - learning_rate × ∂J/∂w
```

For bias:

```text
b_new = b_old - learning_rate × ∂J/∂b
```

Or shorter:

```text
w = w - α × dw
b = b - α × db
```

Where:

```text
α = learning rate
```

---

# 19. Why Do We Subtract the Gradient?

Because the gradient points uphill.

Imagine:

```text
            HIGH COST
                ↑
                |
        gradient points here
                ↑

             ● current position

                ↓
         we want to move here

            LOW COST
```

So:

```text
parameter = parameter - gradient
```

moves us toward lower cost.

---

# 20. Derivative of the Cost

For Linear Regression:

```text
Prediction:

ŷᵢ = wxᵢ + b
```

The gradient with respect to `w` becomes:

```text
dw = average of ((ŷᵢ - yᵢ) × xᵢ)
```

The gradient with respect to `b` becomes:

```text
db = average of (ŷᵢ - yᵢ)
```

---

# 21. Why Does `dw` Contain `x`?

This is important.

```text
dw = average of ((ŷ - y) × x)
```

The first part:

```text
ŷ - y
```

means:

> How wrong was I?

The second part:

```text
× x
```

means:

> How strongly did this feature participate in producing that prediction?

Remember:

```text
ŷ = wx + b
```

`w` is multiplied by `x`.

So changing `w` has a larger effect when `x` is larger.

That's why `x` appears in the derivative.

---

# 22. Why Doesn't `db` Contain `x`?

Because:

```text
ŷ = wx + b
```

`b` is added directly.

It isn't multiplied by `x`.

Changing:

```text
b → b + 1
```

shifts every prediction by:

```text
+1
```

Therefore:

```text
db = average of (ŷ - y)
```

No `x` is required.

---

# 23. I Am the Algorithm — Full Training Loop

Now put everything together.

You give me training data.

I initialize:

```text
w = 0
b = 0
```

### Step 1 — Make predictions

```text
ŷ = wx + b
```

### Step 2 — Compare predictions with reality

```text
error = ŷ - y
```

### Step 3 — Calculate cost

```text
MSE = average(error²)
```

### Step 4 — Calculate gradients

```text
dw = average(error × x)

db = average(error)
```

### Step 5 — Update my parameters

```text
w = w - α × dw

b = b - α × db
```

### Step 6 — Predict again

My line has moved slightly.

### Step 7 — Calculate the new cost

Hopefully:

```text
new cost < old cost
```

### Step 8 — Repeat

```text
Initialize w,b
      ↓
Predict
      ↓
Calculate error
      ↓
Calculate cost
      ↓
Calculate gradients
      ↓
Update w,b
      ↓
Predict again
      ↓
...
      ↓
Converge
```

That is Linear Regression training using Gradient Descent.

---

# 24. What Is the Learning Rate?

The learning rate `α` controls:

> **How large a step I take during each parameter update.**

### Very small learning rate

```text
●
 \
  ●
   \
    ●
     \
      ●
       \
        ★
```

I'll probably reach the minimum.

But slowly.

### Good learning rate

```text
●
   \
      ●
         \
            ★
```

Efficient convergence.

### Too large

```text
      ●
     / \
    /   \
   ●     ●
    \   /
     \ /
      ★
```

I may repeatedly overshoot the minimum.

If extremely large, the cost can even diverge.

> **Key distinction**
>
> Gradient = direction
>
> Learning rate = step size

---

# 25. Why Does Linear Regression Have a Global Minimum?

With squared-error Linear Regression, the cost function is convex with respect to the model parameters.

Think:

```text
\                 /
 \               /
  \             /
   \           /
    \         /
     \_______/
         ★
```

There is one global valley.

So, with an appropriate learning rate, Gradient Descent can converge to the global minimum.

This is very different from deep neural networks, whose optimization landscapes can be much more complicated.

---

# 26. Multiple Linear Regression

So far:

```text
ŷ = wx + b
```

uses one feature.

Real datasets usually contain many.

For house prices:

```text
x₁ = size
x₂ = bedrooms
x₃ = house age
x₄ = distance from city
```

Then:

```text
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
```

Generally:

```text
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Every feature gets its own weight.

---

# 27. What Does a Weight Mean in Multiple Regression?

Suppose:

```text
Price =
0.05 × Size
+ 4 × Bedrooms
- 0.3 × Age
+ 10
```

Then:

```text
w_size = 0.05
```

means approximately:

> Holding the other included features constant, increasing size by one unit is associated with a 0.05-unit increase in predicted price.

Similarly:

```text
w_age = -0.3
```

means:

> Holding other features constant, increasing house age by one unit is associated with a 0.3-unit decrease in predicted price.

> **Warning**
>
> Coefficient ≠ causation.
>
> Linear Regression learns statistical relationships from the available data.
>
> It does not automatically prove that changing a feature causes the target to change.

---

# 28. Vector Form

Instead of writing:

```text
w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
```

we can represent the features and weights as vectors.

```text
x = [x₁, x₂, ..., xₙ]

w = [w₁, w₂, ..., wₙ]
```

Then:

```text
ŷ = wᵀx + b
```

For an entire dataset:

```text
ŷ = Xw + b
```

This is the same Linear Regression model.

Just written using Linear Algebra.

---

# 29. Why Vectorization Matters

Suppose I have:

```text
1,000,000 rows
100 features
```

Writing Python loops over every feature and row would be slow.

Matrix operations allow optimized numerical libraries to perform these calculations efficiently.

So production ML code usually relies heavily on vectorized operations.

---

# 30. Gradient Descent Variants

## Batch Gradient Descent

Use the entire training dataset before making one update.

```text
ALL rows
   ↓
calculate gradient
   ↓
update parameters
```

Advantages:

```text
Stable gradient
```

Disadvantages:

```text
Expensive for huge datasets
```

---

## Stochastic Gradient Descent — SGD

Use one training example at a time.

```text
Row 1 → update
Row 2 → update
Row 3 → update
...
```

Advantages:

```text
Very frequent updates
Can scale to large/streaming datasets
```

Disadvantages:

```text
Very noisy optimization path
```

---

## Mini-Batch Gradient Descent

Use small groups:

```text
32 rows
64 rows
128 rows
...
```

Then update.

This combines:

```text
efficient computation
+
more frequent updates
+
less noise than pure SGD
```

Mini-batches are especially common in deep learning.

---

# 31. Do I Always Need Gradient Descent?

No.

Linear Regression has another solution.

The least-squares optimum can be obtained algebraically.

A common form is:

```text
w = (XᵀX)⁻¹Xᵀy
```

This is commonly called the **Normal Equation**.

Modern numerical implementations typically avoid literally computing the matrix inverse and instead use more numerically stable linear algebra methods.

---

# 32. Gradient Descent vs Closed-Form Solution

### Closed-form / Linear Algebra Solver

Useful when:

```text
number of features is manageable
```

Advantages:

```text
No learning rate
No iterative convergence
Direct least-squares solution
```

Potential problems:

```text
Matrix operations become expensive with many features
Numerical issues can occur
```

---

### Gradient-Based Optimization

Useful when:

```text
dataset / feature space becomes large
or iterative optimization is preferred
```

Advantages:

```text
Scales better in many large problems
Works naturally with SGD / mini-batches
```

Requires:

```text
learning rate
iterations
convergence monitoring
```

---

# 33. What If XᵀX Cannot Be Inverted?

This can happen when features are linearly dependent.

Example:

```text
x₂ = 2 × x₁
```

Then one feature contains information already perfectly represented by another.

This creates **perfect multicollinearity**.

A robust solution is the:

```text
Moore-Penrose Pseudoinverse
```

Conceptually:

```text
w = X⁺y
```

Libraries generally use numerically stable methods rather than expecting you to manually invert matrices.

---

# 34. Multicollinearity

Suppose:

```text
x₁ = house size in square feet

x₂ = house size in square meters
```

These contain essentially the same information.

The model can struggle to determine:

```text
How much weight belongs to x₁?
How much belongs to x₂?
```

Predictions may still be reasonable.

But individual coefficients can become unstable.

This matters particularly when you want to **interpret coefficients**.

---

# 35. Does Linear Regression Need Feature Scaling?

This needs a nuanced answer.

### For the mathematical model itself:

No.

Linear Regression can fit features with different scales.

### For Gradient Descent:

Scaling can be very useful.

Imagine:

```text
Age       → 20–80
Salary    → 20,000–2,000,000
Distance  → 0–100
```

The cost surface can become badly stretched.

Gradient Descent may zig-zag and converge slowly.

Scaling features can make optimization much easier.

Common options:

```text
StandardScaler
MinMaxScaler
```

> **Interview answer**
>
> Linear Regression itself does not fundamentally require scaling, but scaling is often useful when optimization is performed using Gradient Descent and is also important when regularization is used.

---

# 36. Outliers

Linear Regression with MSE is sensitive to outliers.

Why?

Because errors are squared.

```text
Error = 10
Penalty = 100

Error = 100
Penalty = 10,000
```

One extreme observation can strongly influence the fitted line.

So:

> Never delete an observation merely because an IQR rule called it an outlier.

First ask:

```text
Is it a data error?

or

Is it a legitimate extreme observation?
```

Domain reasoning matters.

---

# 37. Assumptions of Ordinary Linear Regression

For classical statistical interpretation/inference, important assumptions include:

### 1. Linearity

The expected relationship between predictors and target is appropriately linear in the parameters.

### 2. Independence

Observations/errors should not have problematic dependence unless explicitly modeled.

### 3. Homoscedasticity

Residual variance should be reasonably constant across fitted values.

Bad pattern:

```text
Residual spread

small → medium → HUGE
```

This is heteroscedasticity.

### 4. No perfect multicollinearity

Features should not be exact linear combinations of each other.

### 5. Residual normality

Normal residuals are especially relevant for classical confidence intervals and hypothesis tests.

> **Important**
>
> These assumptions are especially important for statistical inference.
>
> Prediction can sometimes still be useful even when some assumptions are imperfect.

---

# 38. Residual Analysis

After training, don't only look at one metric.

Inspect:

```text
Residual = actual - predicted
```

Ideally residuals should look roughly random around zero.

Example:

```text
Residual
   ^
 + |    .   .      .
   | .       .
 0 |-------------------
   |      .      .
 - |  .        .
   +--------------------> Prediction
```

Bad patterns can indicate:

```text
curvature
heteroscedasticity
missing features
outliers
time dependence
incorrect model specification
```

Residuals tell us **how the model is failing**.

---

# 39. Evaluation Metrics

## MAE

```text
MAE = average of |y - ŷ|
```

Easy to interpret.

Less sensitive to extreme errors than MSE.

---

## MSE

```text
MSE = average of (y - ŷ)²
```

Strongly penalizes large mistakes.

Units are squared.

---

## RMSE

```text
RMSE = √MSE
```

Returns error to the same unit as the target.

---

## R²

R² asks:

> How much better am I than simply predicting the target mean?

Conceptually:

```text
R² = 1 - model error / baseline mean-model error
```

Interpretation:

```text
R² = 1
→ perfect fit

R² = 0
→ no better than predicting the mean

R² < 0
→ worse than the mean baseline
```

> **Important**
>
> High R² does not automatically mean the model is good.
>
> Always consider validation performance, residuals, leakage, domain relevance, and other appropriate metrics.

---

# 40. Adjusted R²

Regular R² generally does not decrease just because you add another feature.

That can encourage useless features.

Adjusted R² adds a penalty for unnecessary predictors.

Conceptually:

```text
Useful new feature
→ Adjusted R² may increase

Useless new feature
→ Adjusted R² may decrease
```

This is especially useful when comparing regression models with different numbers of predictors.

---

# 41. Polynomial Features — Is That Still Linear Regression?

Suppose the relationship is curved.

We create:

```text
x
x²
x³
```

Then fit:

```text
ŷ = w₁x + w₂x² + w₃x³ + b
```

The curve is nonlinear with respect to `x`.

But the model is still **linear in its parameters**:

```text
w₁
w₂
w₃
```

Therefore it is still a form of Linear Regression.

> Linear Regression means **linear in the coefficients**, not necessarily a visually straight relationship in the original feature.

---

# 42. Overfitting

Suppose training performance is excellent:

```text
Train RMSE = 2
```

but test performance is poor:

```text
Test RMSE = 20
```

The model learned patterns that don't generalize.

Possible causes include:

```text
too many unnecessary features
high-degree polynomial features
data leakage
small/noisy dataset
```

One solution family is:

```text
Regularization
```

which leads to:

```text
Ridge
Lasso
Elastic Net
```

---

# 43. Underfitting

Suppose:

```text
Train error = high
Test error  = high
```

The model may simply be too simple.

Example:

Trying to fit:

```text
strongly curved relationship
```

using only:

```text
ŷ = wx + b
```

Potential solutions:

```text
better features
interaction terms
polynomial features
more expressive model
```

---

# 44. Training vs Inference

## Training

During training:

```text
X + y
 ↓
Predictions
 ↓
Cost
 ↓
Gradients / solver
 ↓
Learn w,b
```

Parameters change.

---

## Inference

A new row arrives:

```text
x_new
```

I calculate:

```text
ŷ = wx_new + b
```

using the parameters learned during training.

No target `y` is required.

No cost needs to be calculated.

No gradient descent occurs.

No parameter update occurs.

> **Inference = use learned parameters, not learn them again.**

---

# 45. Data Leakage

Suppose you scale the entire dataset before train/test splitting.

Then the scaler learned:

```text
mean
standard deviation
```

using information from the test set.

Your training pipeline has indirectly seen test information.

That's leakage.

Correct approach:

```text
Split
 ↓
Fit preprocessing on TRAIN
 ↓
Transform TRAIN
 ↓
Transform TEST using the SAME fitted preprocessing
```

General rule:

> Anything that **learns from data** should generally be fitted only on the training set.

Examples:

```text
Scaler
Imputer
Feature selector
PCA
Target encoder
Model
```

---

# 46. Production Thinking

Training a Linear Regression model is only part of the job.

In production I also care about:

```text
Data validation
Missing values
Feature consistency
Train-serving skew
Outliers
Distribution drift
Prediction latency
Metric monitoring
Model versioning
Retraining strategy
```

Suppose training data used:

```text
Salary in ₹
```

but production sends:

```text
Salary in dollars
```

The model isn't mathematically broken.

The pipeline is.

Production ML failures are often **data/system failures**, not algorithm failures.

---

# 47. When Should I Use Linear Regression?

Good choice when:

```text
Target is continuous

Relationship is reasonably approximated
by a linear model

Interpretability matters

You need a strong/simple baseline

Dataset isn't demanding complex nonlinear interactions
```

Examples:

```text
House price
Revenue
Demand
Temperature
Salary
Sales
```

depending on the actual data-generating problem.

---

# 48. When Might Linear Regression Be a Bad Choice?

Be suspicious when:

```text
Relationship is strongly nonlinear

Complex feature interactions dominate

Extreme outliers dominate the fit

Target behavior violates important assumptions

Prediction is bounded in a way linear output handles poorly

Another model generalizes substantially better
```

Do not use an algorithm because:

```text
"I know Linear Regression."
```

Use it because its assumptions and inductive bias make sense for the problem.

---

# 49. I Am Linear Regression — Final Mental Model

```text
You give me:

X = features
y = continuous target

        ↓

I assume:

ŷ = wᵀx + b

        ↓

But I don't know:

w, b

        ↓

I initialize them

        ↓

I predict:

ŷ

        ↓

I compare:

ŷ vs y

        ↓

I calculate:

MSE

        ↓

I ask:

How should w and b change
to reduce MSE?

        ↓

Gradient

        ↓

Update:

w = w - αdw
b = b - αdb

        ↓

Repeat

        ↓

Cost converges

        ↓

Training complete

        ↓

New unseen X arrives

        ↓

Use learned w,b

        ↓

ŷ = predicted continuous value
```

---

# 50. One-Line Interview Definition

> **Linear Regression is a supervised regression algorithm that models a continuous target as a linear combination of input features and learns the coefficients by minimizing a loss, typically squared error.**

---

# 51. What I Should Actually Remember

Do NOT memorize the algorithm as:

```text
ŷ = wx + b
MSE
Gradient Descent
Done.
```

Remember the reasoning:

```text
I need to predict a number
        ↓
Assume a linear relationship
        ↓
I don't know the best line
        ↓
Represent the line using parameters
        ↓
Measure how wrong those parameters are
        ↓
Need an objective that doesn't cancel errors
        ↓
Squared error
        ↓
Need to minimize that objective
        ↓
Use optimization
        ↓
Gradient tells me uphill
        ↓
Move downhill
        ↓
Learn parameters
        ↓
Validate on unseen data
        ↓
Use fixed parameters during inference
```

If you understand that chain, you understand Linear Regression rather than merely remembering its equation.
