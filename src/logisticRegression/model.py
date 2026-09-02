# ============================================================
#                   LOGISTIC REGRESSION
#              TELCO CUSTOMER CHURN
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/telecomChurn.csv")


# ============================================================
# BASIC DATA INSPECTION
# ============================================================

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ============================================================
# NULL CHECK
# ============================================================

null_count = df.isnull().sum()

null_percentage = (
    df.isnull().sum() / len(df)
) * 100


# ============================================================
# CHECK SUSPICIOUS / MISSING VALUES
# ============================================================

for column in df.columns:

    if pd.api.types.is_string_dtype(df[column]):

        empty_strings = (
            df[column]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        )

    zero_count = (df[column] == 0).sum()


# ============================================================
# INVESTIGATE BLANK TotalCharges
# ============================================================

blank_total_charges = df[
    df["TotalCharges"].str.strip() == ""
]

# We found:
#
# TotalCharges has 11 blank values.
# All 11 corresponding customers have tenure = 0.
#
# This strongly suggests these are new customers with
# no accumulated TotalCharges yet.

print(
    "All blank TotalCharges have tenure = 0:",
    (blank_total_charges["tenure"] == 0).all()
)


# ============================================================
# CONVERT TotalCharges TO NUMERIC
# ============================================================

# TotalCharges is currently stored as a string because
# of those 11 blank values.
#
# Convert it to numeric.
# Blank strings become NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Since the 11 missing TotalCharges belong to customers
# with tenure = 0, we will treat their accumulated
# TotalCharges as 0 for this baseline model.

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ============================================================
# OUTLIER DETECTION USING IQR
# ============================================================

numeric_columns_for_analysis = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns_for_analysis:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    ]

    print(f"\n===== {column} =====")
    print("Number of Outliers:", len(outliers))


# ============================================================
# CHECK SKEWNESS
# ============================================================

for column in numeric_columns_for_analysis:

    skewness = df[column].skew()

    print(f"{column} skewness: {skewness}")


# Results approximately:
#
# tenure          ->  0.24
# MonthlyCharges  -> -0.22
# TotalCharges    ->  0.96
#
# No log transformation for baseline.
#
# Logistic Regression does NOT require features
# themselves to follow a normal distribution.


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

# customerID is an identifier and should not be used
# as a predictive feature.
#
# Churn is our target.

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    # 80% training, 20% testing
    test_size=0.2,

    # Makes our split reproducible
    random_state=42,

    # Preserve approximately the same churn ratio
    # in train and test sets.
    stratify=y
)


# ============================================================
# IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# PREPROCESSING
# ============================================================

# CATEGORICAL:
#
# OneHotEncoder converts categories such as:
#
# InternetService:
#
# DSL
# Fiber optic
# No
#
# into separate numerical indicator features.
#
#
# NUMERICAL:
#
# StandardScaler puts numerical features onto
# comparable scales.


preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        ),

        (
            "numerical",

            StandardScaler(),

            numeric_columns
        )
    ]
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = LogisticRegression(
    class_weight={
        "No":1,
        "Yes":1.5
    },
    max_iter=1000
)

# For punishing the model aggressively for churns
# | Model          | Accuracy | Yes Precision | Yes Recall |  Yes F1 | ROC-AUC |
# | -------------- | -------: | ------------: | ---------: | ------: | ------: |
# | Baseline `1:1` |  **81%** |       **66%** |        56% |     60% |  0.8416 |
# | Custom `1:1.5` |      78% |           57% |    **66%** | **61%** |  0.8420 |
# | `"balanced"`   |      74% |           50% |    **78%** |     61% |  0.8416 |



# ============================================================
# COMPLETE ML PIPELINE
# ============================================================

# Raw data
#    ↓
# ColumnTransformer
#    ↓
# ┌─────────────────────────────┐
# │ categorical → OneHotEncoder │
# │ numerical   → StandardScaler│
# └─────────────────────────────┘
#    ↓
# Logistic Regression
#    ↓
# Prediction


model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "logistic_regression",
            logistic_model
        )
    ]
)


# ============================================================
# TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# MAKE CLASS PREDICTIONS
# ============================================================

# predict() returns the final class:
#
# Yes / No

y_pred = model.predict(
    X_test
)


# ============================================================
# GET PROBABILITIES
# ============================================================

# predict_proba() gives probabilities for BOTH classes.
#
# Example:
#
# [0.82, 0.18]
#
# meaning roughly:
#
# P(No)  = 0.82
# P(Yes) = 0.18

y_proba = model.predict_proba(
    X_test
)


# Let's verify which probability column belongs
# to which class.

print(
    "\nClass order:",
    model.named_steps[
        "logistic_regression"
    ].classes_
)


# ============================================================
# EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)


print("\n==============================")
print("       MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
)

print("\nConfusion Matrix:")
print(cm)


# With labels=["No", "Yes"], sklearn arranges it as:
#
#                 Predicted
#                 No     Yes
#
# Actual No       TN      FP
# Actual Yes      FN      TP
#
#
# [[TN, FP],
#  [FN, TP]]


# ============================================================
# FULL CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# SHOW SOME ACTUAL PREDICTIONS
# ============================================================

results = X_test.copy()

results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred


# Find probability corresponding specifically to "Yes".
#
# We DON'T blindly assume [:, 1] without first knowing
# the model's class ordering.

classes = model.named_steps[
    "logistic_regression"
].classes_

yes_index = list(classes).index("Yes")

results["Churn_Probability"] = (
    y_proba[:, yes_index]
)


print("\nSample Predictions:")

print(
    results[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Actual_Churn",
            "Predicted_Churn",
            "Churn_Probability"
        ]
    ].head(10)
)



# Accuracy 81%
# Churn precision 66%
# Churn recall 56%
# Churn F1 60%

# Model accuracy is just 81% because it was provided with imbalanced data. It has more example for non-churn and very less examples for churn
# Let's try -- class weight="balanced" and ROC_Auc score for metrics

classess = model.named_steps["logistic_regression"].classes_
yes_index = list(classess).index("Yes")

y_probe_yes = model.predict_proba(X_test)[:, yes_index]

baseline_roc_auc = roc_auc_score(y_test, y_probe_yes)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probe_yes))


# Accuracy    = 81%
# Yes F1      = 60%
# Yes Recall  = 56%
# ROC-AUC     = 84%