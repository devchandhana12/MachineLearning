# Machine Learning from First Principles

A practical, code-first tutorial for learning supervised machine learning. The goal is to understand the path from data to prediction:

```text
data -> model -> loss / impurity -> optimization -> evaluation -> inference
```

Each topic starts with intuition, introduces the mathematics, and connects the ideas to Python code.

## Learning path

Follow the algorithms in this order:

| Step | Topic | Main problem | Core ideas |
| ---: | --- | --- | --- |
| 1 | [Linear Regression](src/linearRegression/notes.md) | Predict a continuous number | Linear model, MSE, derivatives, gradient descent |
| 2 | Multiple Linear Regression | Predict from many features | Vectors, matrices, scaling, evaluation |
| 3 | [Logistic Regression](src/logisticRegression/notes.md) | Predict one of two classes | Sigmoid, probability, BCE, classification metrics |
| 4 | [Multiclass / Softmax Regression](src/softmax/notes.md) | Predict one of several classes | Softmax, argmax, one-hot targets, cross-entropy |
| 5 | [Decision Trees](src/decisionTrees/notes.md) | Make rule-based predictions | Splits, Gini impurity, information gain, pruning |
| 6 | Random Forest | Combine many trees | Bootstrap samples, feature randomness, voting |

Multiple linear regression and multiclass regression are different ideas:

```text
multiple features -> multiple linear regression
multiple classes  -> softmax regression
```

## Mathematics to learn first

You do not need advanced mathematics before starting. You need a small set of ideas and the ability to follow them carefully.

### 1. Notation and shapes

```text
x_i = features for observation i
y_i = actual target for observation i
y_hat_i = model prediction for observation i
n = number of observations
```

Be comfortable with scalars, vectors, matrices, sums, averages, powers, square roots, and percentages. In a dataset, one row is usually one sample and one column is one feature.

### 2. Functions

A model is a function that maps inputs to outputs:

```text
y_hat = f(X; parameters)
```

For one-feature linear regression:

```text
y_hat = wx + b
```

For many features:

```text
y_hat = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b
```

### 3. Linear algebra

Learn vector shape, matrix shape, dot products, matrix multiplication, and transpose. The compact form of a linear model is:

```text
y_hat = Xw + b
```

NumPy uses these operations to calculate predictions for many samples at once.

### 4. Derivatives and gradients

A derivative tells us how much a value changes when an input changes. A gradient is a collection of derivatives, one for each parameter. Gradient descent uses:

```text
parameter <- parameter - learning_rate * gradient
```

The gradient points toward increasing cost, so the update moves in the opposite direction.

### 5. Probability and logarithms

Classification models use probabilities between `0` and `1`. A probability becomes a class through a threshold or `argmax`. Logarithms are important because they penalize confident wrong predictions and turn products into sums in likelihood calculations.

These ideas lead to binary cross-entropy for logistic regression and categorical cross-entropy for softmax regression.

### 6. Evaluation and generalization

Always distinguish:

```text
training data   -> used to learn parameters or splits
validation data -> used to compare model choices
test data       -> used once for final unseen evaluation or prediction
```

Learn overfitting, underfitting, data leakage, and why preprocessing must be fitted on training data only.

## The algorithms

### 1. Linear Regression

Linear regression predicts a continuous target with a weighted sum of features:

```text
y_hat = Xw + b
```

It measures error with mean squared error:

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

[`linearRegression.py`](linearRegression.py) shows the training loop directly: predict, calculate error, calculate gradients, update `w` and `b`, and repeat. The reusable example in [`src/linearRegression`](src/linearRegression) uses preprocessing and evaluates with MAE, MSE, RMSE, and R2.

Read the detailed explanation in [`src/linearRegression/notes.md`](src/linearRegression/notes.md).

### 2. Multiple Linear Regression

The one-feature equation becomes:

```text
y_hat = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b
```

The model is still linear in its parameters even when it uses many features. This is the right place to learn matrices, vectorization, scaling, multicollinearity, residuals, and regularization.

### 3. Logistic Regression

Logistic regression starts with a linear score:

```text
z = Xw + b
```

and converts it to a probability with the sigmoid:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Binary cross-entropy trains the probabilities:

```text
J = -(1/n) * sum(y_i log(p_i) + (1-y_i) log(1-p_i))
```

The repository includes a small NumPy implementation in [`rawLogisticRegression.py`](src/logisticRegression/rawLogisticRegression.py) and a scikit-learn customer-churn pipeline in [`model.py`](src/logisticRegression/model.py).

### 4. Multiclass / Softmax Regression

For `K` classes, the model produces one score per class:

```text
Z = XW + b
```

Softmax turns those scores into a probability distribution:

```text
softmax(z_k) = e^(z_k) / sum(e^(z_j))
```

The predicted class is the index of the largest probability. Training uses categorical cross-entropy; `argmax` is only the final decision operation.

Use [`rawSoftmax.py`](src/softmax/rawSoftmax.py) to follow the calculations and [`softmax.py`](src/softmax/softmax.py) for the Iris workflow.

### 5. Decision Trees

A tree repeatedly asks questions such as:

```text
Is Age <= 30?
Is Fare <= 50?
Is Sex == female?
```

For class proportions `p_1, ..., p_K`, Gini impurity is:

```text
Gini = 1 - sum(p_k^2)
```

Pure nodes have Gini `0`. The tree searches for a split that minimizes the weighted impurity of its children and repeats recursively until a stopping rule is reached.

The runnable Titanic example is [`decision_tree.py`](src/decisionTrees/decision_tree.py). It trains with `criterion="gini"`, validates on a held-out part of `train.csv`, predicts the unlabeled `test.csv`, and writes:

- `decision_tree_results.log` for Gini and evaluation results;
- `test_predictions.csv` for predictions.

The included `test.csv` does not contain `Survived`, so test accuracy or test Gini cannot be calculated from it. Validation metrics and final test predictions are kept separate.

### 6. Random Forest

Random Forest is the next step after understanding one tree. It trains many trees on bootstrapped samples and random subsets of features, then combines their predictions:

```text
many diverse trees -> vote / average -> final prediction
```

Important ideas are variance reduction, diversity, bootstrap sampling, feature randomness, out-of-bag evaluation, and feature importance. [`src/randomForest/notes.md`](src/randomForest/notes.md) is reserved for this section; the implementation is the next planned addition.

## Repository map

```text
.
├── linearRegression.py              # Minimal linear regression from scratch
├── src/
│   ├── linearRegression/
│   │   ├── notes.md                 # First-principles linear regression
│   │   ├── preprocessing.py         # Advertising dataset pipeline
│   │   └── linearRegression.py      # sklearn regression example
│   ├── logisticRegression/
│   │   ├── notes.md                 # Binary classification theory
│   │   ├── rawLogisticRegression.py # NumPy implementation
│   │   └── model.py                 # Customer churn pipeline
│   ├── softmax/
│   │   ├── notes.md                 # Multiclass theory
│   │   ├── rawSoftmax.py            # NumPy implementation
│   │   └── softmax.py               # Iris classification workflow
│   ├── decisionTrees/
│   │   ├── notes.md                 # Tree and Gini theory
│   │   ├── decision_tree.py         # Training, validation, and prediction
│   │   ├── decisionTree.ipynb       # Notebook entry point
│   │   ├── train.csv                # Titanic training data
│   │   └── test.csv                 # Titanic unseen data
│   └── randomForest/
│       └── notes.md                 # Planned ensemble section
└── pyproject.toml                   # Python dependencies
```

## Running the examples

Install the dependencies:

```bash
uv sync
```

Run the minimal linear regression example:

```bash
python linearRegression.py
```

Run the decision-tree workflow:

```bash
python src/decisionTrees/decision_tree.py
```

Open the decision-tree notebook from the repository root:

```bash
jupyter lab src/decisionTrees/decisionTree.ipynb
```

Some older examples use relative data paths. Run those files from their own directory or adjust the path before executing them.

## A repeatable workflow for every model

1. Identify the features `X` and target `y`.
2. Decide whether the task is regression or classification.
3. Split data before fitting preprocessing steps.
4. Understand the model equation and its parameters.
5. Understand the loss function or impurity measure.
6. Train the model.
7. Evaluate on data not used for fitting.
8. Inspect errors, not only one summary metric.
9. Predict on genuinely unseen data.
10. Check for overfitting and data leakage.

## What to learn next

After Random Forest, continue with regularization, cross-validation, hyperparameter search, gradient boosting, unsupervised learning, neural networks, and production concerns such as packaging, monitoring, and data drift.

