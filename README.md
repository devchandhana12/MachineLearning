# Machine Learning, from Mathematics to Models

A beginner-friendly, code-first path through supervised machine learning. Start with the mathematics, then study how each model turns data into predictions.

```text
math foundations -> data and evaluation -> models -> prediction
```

The notes explain the ideas and equations; the Python examples show how those ideas appear in code. You do not need advanced math to begin.

## Start here: mathematics

Begin with [Mathematics for Machine Learning](src/00_foundations/mathematics/README.md). Its chapters are arranged in reading order:

1. Scalars, vectors, and dimensions
2. Vector addition and subtraction
3. Scalar multiplication, magnitude, and norms
4. Dot products and angles
5. Linear combinations, span, and independence
6. Matrices, shapes, and operations

The current math chapters focus on linear algebra. The math index also lists the algebra, calculus, probability, and statistics topics that support later lessons. Read [Model Evaluation](src/00_foundations/model_evaluation.md) before comparing models, and explore the [telecom churn notebook](src/01_dataAnalysis/telecom_churn_exploration.ipynb) to practice examining a dataset.

## Algorithm study order

Work through these topics in this order:

| Step | Topic | What changes from the previous step |
| ---: | --- | --- |
| 1 | [Linear Regression](src/02_linearRegression/notes.md) | Learn to predict a continuous value from one feature. |
| 2 | [Multiple Linear Regression](src/02_linearRegression/notes.md#26-multiple-linear-regression) | Extend the same model to several input features. |
| 3 | [Logistic Regression](src/03_logisticRegression/notes.md) | Turn a linear score into a probability for a binary class. |
| 4 | [Multiclass / Softmax Regression](src/04_softmaxRegression/notes.md) | Predict probabilities across several classes. |
| 5 | [Decision Trees](src/05_decisionTrees/notes.md) | Replace one global equation with learned feature-based rules. |
| 6 | [Random Forest](src/06_randomForest/notes.md) | Combine many randomized trees to improve stability. |

“Multiple” and “multiclass” describe different things:

```text
multiple features -> multiple linear regression
multiple classes  -> softmax regression
```

### After the core sequence

Once you understand the six topics above, continue with:

1. [K-Nearest Neighbors](src/07_knn/notes.md) — distance and neighborhood-based prediction.
2. [Gradient Boosting](src/08_gradientBoosting/notes.md) — build trees sequentially to correct earlier errors.
3. [XGBoost](src/09_xgboost/notes.md) — a regularized, efficient gradient-boosted tree method.

These later folders currently contain study notes; the main runnable examples are in the earlier model folders.

## What each section contains

### Linear Regression

The [notes](src/02_linearRegression/notes.md) derive the model, mean squared error, gradients, gradient descent, multiple features, and evaluation. The [from-scratch NumPy example](src/02_linearRegression/linear_regression_from_scratch.py) shows the training loop directly. The [Advertising example](src/02_linearRegression/linearRegression.py) uses scikit-learn, while [house price preprocessing](src/02_linearRegression/house_prices/house_price_preprocessing.py) demonstrates a larger tabular dataset and a training-only preprocessing pipeline.

### Logistic Regression

The [notes](src/03_logisticRegression/notes.md) explain sigmoid probabilities, log-odds, binary cross-entropy, and classification metrics. The folder includes a [NumPy example](src/03_logisticRegression/rawLogisticRegression.py) and a [customer churn workflow](src/03_logisticRegression/model.py).

### Multiclass / Softmax Regression

The [notes](src/04_softmaxRegression/notes.md) introduce class scores, softmax, and categorical cross-entropy. Compare the [NumPy implementation](src/04_softmaxRegression/rawSoftmax.py) with the [Iris example](src/04_softmaxRegression/softmax.py).

### Decision Trees

The [notes](src/05_decisionTrees/notes.md) explain recursive splitting, Gini impurity, and pruning. The [Titanic workflow](src/05_decisionTrees/decision_tree.py) trains a Gini-criterion classifier, measures validation results, and writes predictions for the provided test data.

The [notebook entry point](src/05_decisionTrees/decision_tree_tutorial.ipynb) runs the same workflow. The Titanic `test.csv` has no `Survived` labels, so it can be used to generate predictions but not to calculate test accuracy or test Gini. The script logs validation metrics and Gini results in `decision_tree_results.log` and writes predictions to `test_predictions.csv` beside the script.

### Random Forest

The [notes](src/06_randomForest/notes.md) cover bootstrap sampling, random feature selection, voting, and why combining trees can reduce variance. This section currently has theory notes; a runnable Random Forest example is a future addition.

## Repository layout

```text
.
├── README.md
└── src/
    ├── 00_foundations/
    │   ├── mathematics/        # Math chapter index and chapters 1-6
    │   └── model_evaluation.md
    ├── 01_dataAnalysis/        # Telecom churn exploration notebook and data
    ├── 02_linearRegression/    # Linear and multiple linear regression
    ├── 03_logisticRegression/
    ├── 04_softmaxRegression/
    ├── 05_decisionTrees/       # Includes decision_tree_tutorial.ipynb
    ├── 06_randomForest/
    ├── 07_knn/
    ├── 08_gradientBoosting/
    └── 09_xgboost/
```

## Run the examples

Install project dependencies:

```bash
uv sync
```

Run the compact from-scratch linear regression example:

```bash
python src/02_linearRegression/linear_regression_from_scratch.py
```

Run the House Prices preprocessing example:

```bash
python src/02_linearRegression/house_prices/house_price_preprocessing.py
```

Run the Titanic decision tree:

```bash
python src/05_decisionTrees/decision_tree.py
```

Run the other examples by passing their script paths to Python. Data files are resolved relative to the script where needed.

## A useful habit for every model

For each algorithm, identify the features and target, understand the model and its loss, fit preprocessing using training data only, evaluate on held-out data, inspect errors, and then predict on genuinely unseen examples. This makes it easier to spot overfitting and data leakage before trusting a result.

