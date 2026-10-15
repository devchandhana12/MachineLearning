# Gradient Boosting — Deep Notes with Titanic Example

## 1. What problem is Gradient Boosting trying to solve?

A single decision tree can learn useful patterns, but it has limitations.

If the tree is shallow, it may underfit.

If the tree is very deep, it may overfit.

Gradient Boosting takes a different approach:

Instead of building one very powerful tree, we build many small trees sequentially.

Each new tree tries to correct what the current model is still getting wrong.

Core idea:

```text
Start with a simple model
        ↓
Make predictions
        ↓
Measure what is still wrong
        ↓
Train a new tree to learn that error
        ↓
Add a small correction
        ↓
Repeat
```

So Gradient Boosting is:

$$
\boxed{
\text{Current Model}
+
\text{Small Corrective Tree}
+
\text{Small Corrective Tree}
+
\cdots
}
$$

---

# 2. Titanic Example

Suppose we have this tiny Titanic dataset:

| Passenger | Sex    | Age | Pclass | Survived |
| --------- | ------ | --: | -----: | -------: |
| A         | Male   |  22 |      3 |        0 |
| B         | Female |  38 |      1 |        1 |
| C         | Female |  26 |      3 |        1 |
| D         | Male   |  35 |      1 |        0 |
| E         | Male   |   8 |      3 |        1 |
| F         | Male   |  40 |      3 |        0 |

Target:

$$
y=[0,1,1,0,1,0]
$$

where:

```text
1 = survived
0 = did not survive
```

The features are:

$$
X=(Sex,Age,Pclass)
$$

Our goal is to estimate:

$$
P(Survived=1)
$$

---

# 3. Base Model

Gradient Boosting does not begin with a random prediction.

It begins with a very simple model.

For regression with MSE, the starting prediction is often the mean of the target.

For binary classification, the starting point is based on the proportion of positive examples.

In our tiny Titanic dataset:

```text
Survived = 3
Did not survive = 3
```

Therefore:

$$
P(y=1)=\frac{3}{6}=0.5
$$

So initially, before looking at any feature, the model can predict:

$$
p=0.5
$$

for every passenger.

Initial predictions:

| Passenger | Actual y | Initial p |
| --------- | -------: | --------: |
| A         |        0 |       0.5 |
| B         |        1 |       0.5 |
| C         |        1 |       0.5 |
| D         |        0 |       0.5 |
| E         |        1 |       0.5 |
| F         |        0 |       0.5 |

This is the base model.

---

# 4. What Error Does the Next Tree Learn?

In ordinary regression, we often use residual:

$$
r=y-\hat y
$$

For binary classification using log loss, the negative-gradient signal becomes:

$$
\boxed{r_i=y_i-p_i}
$$

For our Titanic examples:

Passenger A:

$$
y=0,\quad p=0.5
$$

$$
r=0-0.5=-0.5
$$

Passenger B:

$$
y=1,\quad p=0.5
$$

$$
r=1-0.5=+0.5
$$

Doing this for all passengers:

| Passenger |   y |   p | y - p |
| --------- | --: | --: | ----: |
| A         |   0 | 0.5 |  -0.5 |
| B         |   1 | 0.5 |  +0.5 |
| C         |   1 | 0.5 |  +0.5 |
| D         |   0 | 0.5 |  -0.5 |
| E         |   1 | 0.5 |  +0.5 |
| F         |   0 | 0.5 |  -0.5 |

So Tree 1 receives:

```text
Input features X:
Sex, Age, Pclass

Target:
[-0.5, +0.5, +0.5, -0.5, +0.5, -0.5]
```

This is a key point.

Tree 1 is not directly trying to predict:

```text
0 or 1
```

It is trying to learn:

```text
How should the current model be corrected?
```

---

# 5. How Does Tree 1 Split the Data?

Tree 1 behaves like a regression tree.

It tries different feature splits such as:

```text
Sex = Female?
Age < 15?
Pclass < 2?
```

Important:

The tree does not directly split using residual values.

It cannot say:

```text
if residual > 0 → right
```

because residual is the target.

The tree must split using features.

So:

$$
\boxed{\text{Features determine the split}}
$$

while:

$$
\boxed{\text{Residual values determine whether the split is good}}
$$

---

# 6. Example Split: Sex = Female

Suppose Tree 1 tests:

```text
Sex = Female?
```

Female passengers:

```text
B → +0.5
C → +0.5
```

Male passengers:

```text
A → -0.5
D → -0.5
E → +0.5
F → -0.5
```

Female side is very clean.

Male side is mixed.

The tree measures this variation using a regression-tree criterion such as SSE.

For a node:

$$
SSE=\sum(r_i-\bar r)^2
$$

where:

$$
\bar r
$$

is the mean target value inside that node.

---

# 7. SSE Example

Female residuals:

$$
[0.5,0.5]
$$

Mean:

$$
\bar r=0.5
$$

SSE:

$$
(0.5-0.5)^2+(0.5-0.5)^2=0
$$

Perfectly grouped.

Male residuals:

$$
[-0.5,-0.5,+0.5,-0.5]
$$

Mean:

$$
\frac{-0.5-0.5+0.5-0.5}{4}
=
-0.25
$$

Then:

$$
SSE_{male}
=
(-0.5+0.25)^2
+
(-0.5+0.25)^2
+
(0.5+0.25)^2
+
(-0.5+0.25)^2
$$

$$
=
(-0.25)^2
+
(-0.25)^2
+
(0.75)^2
+
(-0.25)^2
$$

$$
=
0.0625+0.0625+0.5625+0.0625
$$

$$
=0.75
$$

Total:

$$
SSE_{split}=0+0.75=0.75
$$

The tree compares this against other candidate splits.

The split with the lowest resulting error is preferred.

---

# 8. The Tree Can Split Again

Male passengers were:

| Passenger | Age | Residual |
| --------- | --: | -------: |
| A         |  22 |     -0.5 |
| D         |  35 |     -0.5 |
| E         |   8 |     +0.5 |
| F         |  40 |     -0.5 |

Now the tree may notice that passenger E is very young.

It could test:

```text
Age < 15?
```

This gives:

```text
Age < 15
→ [+0.5]

Age >= 15
→ [-0.5, -0.5, -0.5]
```

Now both leaves are extremely clean.

So the tree may become:

```text
                Sex = Female?
               /             \
            Yes               No
           +0.5           Age < 15?
                           /        \
                        +0.5       -0.5
```

Conceptually:

```text
Female       → positive correction
Young male   → positive correction
Adult male   → negative correction
```

This is what the tree has learned from the current model's errors.

---

# 9. What Is h_m(x)?

The m-th tree is written as:

$$
h_m(x)
$$

It is simply the prediction made by that tree.

For example:

```text
Female       → +0.5
Male age <15 → +0.5
Adult male   → -0.5
```

Then for passenger B:

$$
h_1(x_B)=+0.5
$$

For passenger A:

$$
h_1(x_A)=-0.5
$$

The entire regression tree is the function:

$$
\boxed{h_m(x)}
$$

There is no separate mystery formula for it.

---

# 10. Regression Leaf Values

For regression with squared-error loss, each leaf usually predicts the average target value inside that leaf.

Example:

Residuals inside a leaf:

$$
[0.4,0.3]
$$

Leaf output:

$$
\frac{0.4+0.3}{2}=0.35
$$

So:

$$
h_m(x)=0.35
$$

for any sample landing in that leaf.

Important:

The tree does not reproduce each individual residual exactly.

It learns an approximation.

If one passenger has:

$$
r=0.4
$$

and another has:

$$
r=0.3
$$

the same leaf may output:

$$
0.35
$$

for both.

---

# 11. Why Do We Need Learning Rate?

This is one of the most important ideas in Gradient Boosting.

Suppose the true correction needed for one sample is:

$$
20
$$

but the tree predicts:

$$
25
$$

If we fully trust the tree:

$$
F_{new}=F_{old}+25
$$

we may overshoot.

The tree is only an approximation of the correct residual pattern.

It can contain:

```text
real signal
+
noise
```

So instead of applying the whole correction, we shrink it.

The update becomes:

$$
\boxed{
F_m(x)=F_{m-1}(x)+\eta h_m(x)
}
$$

where:

$$
\eta
$$

is the learning rate.

---

# 12. Learning Rate Example

Suppose:

$$
h_m(x)=25
$$

If:

$$
\eta=1
$$

then contribution is:

$$
25
$$

If:

$$
\eta=0.1
$$

then contribution is:

$$
0.1\times25=2.5
$$

So:

$$
F_{new}=F_{old}+2.5
$$

The model then recalculates the new remaining error and lets later trees continue learning.

The learning rate therefore controls:

$$
\boxed{\text{How much influence each tree has}}
$$

---

# 13. Why Smaller Learning Rate Helps

Suppose a tree learns some training noise.

With:

$$
\eta=1
$$

that noisy rule gets full influence.

With:

$$
\eta=0.05
$$

only 5% of that tree's suggested correction is added.

So a smaller learning rate:

```text
reduces aggressive corrections
reduces sensitivity to noisy trees
acts as regularization
often improves generalization
```

But there is a trade-off.

Smaller learning rate usually needs more trees.

Therefore:

$$
\boxed{
\text{Smaller learning rate}
\Rightarrow
\text{More trees}
}
$$

---

# 14. Connection to Gradient Descent

Gradient Descent update:

$$
w_{new}=w_{old}-\eta\nabla L
$$

Gradient tells us:

```text
Which direction reduces the loss?
```

Learning rate tells us:

```text
How far should we move?
```

Gradient Boosting follows the same principle.

Instead of directly updating a numerical parameter like a weight, we add a new function:

$$
h_m(x)
$$

So:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Gradient Boosting is often described as:

```text
Gradient descent in function space.
```

---

# 15. Classification Uses Raw Scores

For Titanic, we are doing binary classification.

The model internally works with a raw score:

$$
F(x)
$$

This is converted into probability using sigmoid:

$$
p(x)=\frac{1}{1+e^{-F(x)}}
$$

Example:

If:

$$
F(x)=0
$$

then:

$$
p=0.5
$$

If:

$$
F(x)=1
$$

then:

$$
p\approx0.731
$$

If:

$$
F(x)=-1
$$

then:

$$
p\approx0.269
$$

So positive raw scores increase survival probability.

Negative raw scores decrease survival probability.

---

# 16. Why y - p Appears

For binary classification, we normally optimize binary log loss:

$$
L=
-\left[
y\log(p)+(1-y)\log(1-p)
\right]
$$

When we differentiate this loss with respect to the model score, the negative gradient becomes:

$$
\boxed{y-p}
$$

This is why the next tree gets trained on:

$$
X \rightarrow y-p
$$

Example:

If:

$$
y=1,\quad p=0.4
$$

then:

$$
y-p=0.6
$$

Meaning:

```text
Increase the model score.
```

If:

$$
y=0,\quad p=0.7
$$

then:

$$
y-p=-0.7
$$

Meaning:

```text
Decrease the model score.
```

---

# 17. Important Difference Between Regression and Classification

For regression with MSE:

```text
Residual = y - prediction
```

and leaf output is usually the mean residual.

For binary classification:

```text
Negative gradient = y - p
```

but the final leaf correction is not simply the average of these values.

Why?

Because classification is optimizing log loss, not MSE.

---

# 18. Classification Leaf Value

For binary Gradient Boosting with log loss, a common leaf correction is:

$$
\boxed{
\gamma
=
\frac{\sum(y_i-p_i)}
{\sum p_i(1-p_i)}
}
$$

Do not memorize this blindly.

Understand the pieces.

Numerator:

$$
\sum(y_i-p_i)
$$

means:

```text
How strongly does this leaf want to move upward or downward?
```

Denominator:

$$
\sum p_i(1-p_i)
$$

adjusts the step based on the curvature of log loss.

---

# 19. Titanic Leaf Example

Suppose one leaf contains these passengers:

| Passenger |   y |   p |
| --------- | --: | --: |
| B         |   1 | 0.4 |
| C         |   1 | 0.6 |
| D         |   0 | 0.3 |

Calculate:

$$
y-p
$$

Passenger B:

$$
1-0.4=0.6
$$

Passenger C:

$$
1-0.6=0.4
$$

Passenger D:

$$
0-0.3=-0.3
$$

Numerator:

$$
0.6+0.4-0.3=0.7
$$

Now denominator:

Passenger B:

$$
0.4(1-0.4)=0.24
$$

Passenger C:

$$
0.6(1-0.6)=0.24
$$

Passenger D:

$$
0.3(1-0.3)=0.21
$$

Total:

$$
0.24+0.24+0.21=0.69
$$

Therefore:

$$
\gamma=
\frac{0.7}{0.69}
\approx1.014
$$

So this leaf wants to push the raw model score upward by roughly:

$$
1.014
$$

---

# 20. Apply Learning Rate

Suppose:

$$
\eta=0.1
$$

Then actual contribution is:

$$
0.1\times1.014
=
0.1014
$$

So:

$$
F_{new}(x)
=
F_{old}(x)+0.1014
$$

Important:

We are updating the raw score.

We are not directly adding 0.1014 to probability.

After updating the raw score, probability is recalculated:

$$
p_{new}
=
\sigma(F_{new})
$$

---

# 21. Full Titanic Gradient Boosting Loop

For binary classification:

### Step 1

Start with a base raw score:

$$
F_0(x)
$$

### Step 2

Convert raw score to probability:

$$
p_i=\sigma(F_0(x_i))
$$

### Step 3

Calculate negative gradients:

$$
r_i=y_i-p_i
$$

### Step 4

Train a regression tree:

$$
X\rightarrow r
$$

### Step 5

Use features to create splits.

The tree searches for groups with similar gradient values.

### Step 6

For every final leaf, calculate its correction:

$$
\gamma
=
\frac{\sum(y-p)}
{\sum p(1-p)}
$$

### Step 7

Update model:

$$
F_m(x)
=
F_{m-1}(x)+\eta\gamma
$$

### Step 8

Convert new raw scores to probability:

$$
p_m(x)=\sigma(F_m(x))
$$

### Step 9

Calculate new:

$$
y-p
$$

### Step 10

Train the next tree.

Repeat.

---

# 22. Final Ensemble

After M trees:

$$
\boxed{
F_M(x)
=
F_0(x)
+
\eta h_1(x)
+
\eta h_2(x)
+
\cdots
+
\eta h_M(x)
}
$$

The final prediction is produced by the combined contribution of all trees.

For classification:

$$
P(y=1|x)=\sigma(F_M(x))
$$

---

# 23. Why It Is Called Gradient Boosting

Boosting means:

```text
Combine many weak learners sequentially to make a strong learner.
```

Gradient means:

```text
Each new learner tries to move the model in the direction that reduces the loss.
```

So Gradient Boosting means:

```text
Sequentially add weak learners that follow the negative gradient of the loss.
```

---

# 24. Important Hyperparameters

## n_estimators

Number of trees.

More trees allow more corrections.

Too many can overfit.

---

## learning_rate

Controls how much each tree contributes.

Example:

$$
0.01,\ 0.05,\ 0.1
$$

Smaller learning rate usually requires more trees.

---

## max_depth

Controls the complexity of each tree.

Shallow trees are commonly used.

Example:

```text
max_depth = 2
max_depth = 3
```

A shallow tree acts as a weak learner.

---

## min_samples_split

Minimum number of samples required to split a node.

---

## min_samples_leaf

Minimum number of samples allowed inside a leaf.

Helps avoid overly specific leaves.

---

## subsample

Fraction of rows used for each tree.

Example:

```text
subsample = 0.8
```

Using less than 1 can reduce overfitting.

This creates stochastic Gradient Boosting.

---

# 25. Evaluation Metrics

Gradient Boosting does not have special evaluation metrics.

Metrics depend on the task.

For Titanic classification:

### Accuracy

$$
Accuracy=
\frac{Correct\ Predictions}{Total\ Predictions}
$$

Useful when classes are reasonably balanced.

---

### Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

Of everyone predicted as survived, how many actually survived?

---

### Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

Of everyone who actually survived, how many did the model identify?

---

### F1-score

$$
F1=
2\frac{Precision\times Recall}{Precision+Recall}
$$

Useful when we care about both precision and recall.

---

### ROC-AUC

Measures how well the model ranks positive cases above negative cases across many thresholds.

Higher is better.

---

### Log Loss

Evaluates the quality of predicted probabilities.

It penalizes confidently wrong predictions heavily.

Since Gradient Boosting classification often optimizes log loss, this is an especially relevant training/evaluation metric.

---

# 26. Loss Function vs Evaluation Metric

These are not the same thing.

Loss function:

```text
Used during training to decide how the model should improve.
```

Evaluation metric:

```text
Used to judge model performance.
```

For Titanic:

```text
Training loss:
Log Loss

Evaluation:
Accuracy
F1
ROC-AUC
Log Loss
```

---

# 27. Random Forest vs Gradient Boosting

Random Forest:

```text
Many trees
Trained independently
Usually deep-ish trees
Predictions combined by averaging/voting
Main goal: reduce variance
```

Gradient Boosting:

```text
Trees trained sequentially
Each tree learns remaining errors
Usually shallow trees
Predictions are added
Main goal: gradually reduce loss
```

Core difference:

$$
\boxed{
\text{Random Forest = independent trees}
}
$$

$$
\boxed{
\text{Gradient Boosting = corrective sequential trees}
}
$$

---

# 28. Very Important Interview Points

### Why are trees shallow?

Because each tree only needs to learn one small correction.

We do not want one tree to solve the whole problem.

---

### Why learning rate?

Because each tree is only an approximation.

Learning rate prevents one tree from changing the ensemble too aggressively.

---

### Why does Gradient Boosting overfit?

Possible reasons:

```text
too many trees
trees too deep
learning rate too high
very small leaf sizes
too much fitting to noise
```

---

### How to reduce overfitting?

```text
smaller learning rate
shallower trees
early stopping
subsampling
larger min_samples_leaf
fewer trees
```

---

### What does Tree m learn?

It learns the negative gradient of the current loss.

For binary log loss:

$$
\boxed{y-p}
$$

---

### What is h_m(x)?

It is simply the prediction function of tree m.

---

# 29. Final Mental Model

Do not memorize Gradient Boosting as formulas first.

Remember this:

```text
Current model makes predictions
            ↓
Find what is still wrong
            ↓
Represent that mistake using negative gradients
            ↓
Train a small tree to recognize patterns in those mistakes
            ↓
Each leaf proposes a correction
            ↓
Shrink the correction using learning rate
            ↓
Add it to the current model
            ↓
Recalculate errors
            ↓
Repeat
```

That is Gradient Boosting.

---

# 30. One-Line Interview Answer

Gradient Boosting is a sequential ensemble method in which each weak tree is trained to approximate the negative gradient of the current loss, and its prediction is added to the existing ensemble using a learning rate so that the model gradually improves.
