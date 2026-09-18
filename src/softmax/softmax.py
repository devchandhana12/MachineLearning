import pandas as pd

df = pd.read_csv('data/iris.csv')


# # print the first 5 rows of the dataframe
# print(df.head())

# # print the info of the dataframe
# print(df.info())

# # print the describe of the dataframe
# print(df.describe())

# # print the null values of the dataframe
# print(df.isnull().sum())

# ----------------------- CHECKING FOR DUPLICATES -----------------------
# print("Duplicated rows:", df.duplicated().sum())  ----> Nothing dupliated so far

# print("Nan and emppty value checking:")

# print(df.isna().sum())

# for col in df.columns:
#     if df[col].dtype == 'object':
#         count = df[col].str.strip().eq("").sum()
#         print(f"{col}: {count} empty values")

# ----------------------- CHECKING FOR OUTLIERS -----------------------
print("Outliers checking:")

numeric_cols = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

for column in numeric_cols:
    if column == "SepalWidthCm":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
        ]
        # print(f"\n{column}")
        # print(f"Q1: {Q1}")
        # print(f"Q3: {Q3}")
        # print(f"IQR: {IQR}")
        # print(f"Lower Bound: {lower_bound}")
        # print(f"Upper Bound: {upper_bound}")
        # print(f"Outliers: {len(outliers)}")
        if len(outliers) > 0:
            pass
            # print(f"Outliers in {column}:")
            # print(outliers[column])

print(df["Species"].value_counts())

print("\nPercentages:")
print(df["Species"].value_counts(normalize=True) * 100)


# ============================================================
# MULTICLASS LOGISTIC REGRESSION - SKLEARN
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 1. CREATE X AND y
# ------------------------------------------------------------

# Id is only an identifier.
# It should NOT be used by the model.
#
# Species is our target, so it should not be inside X.

X = df.drop(columns=["Id", "Species"])

y = df["Species"]


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------------------

# stratify=y preserves our class distribution.
#
# Original dataset:
#
# Setosa      -> 33.33%
# Versicolor  -> 33.33%
# Virginica   -> 33.33%
#
# stratify makes train/test approximately preserve this ratio.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 3. CREATE PIPELINE
# ------------------------------------------------------------

# StandardScaler:
#
# Fits ONLY on X_train when model.fit() is called.
#
# It learns:
#
# mean
# standard deviation
#
# from the TRAINING DATA.
#
# Then the exact same learned transformation is used
# on X_test.
#
#
# Pipeline protects us from accidentally doing:
#
# scaler.fit(full_dataset)
#
# which would leak information from test data.


model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------

# Internally:
#
# X_train
#     ↓
# StandardScaler
#     ↓
# Logistic Regression
#     ↓
# class-specific logits
#     ↓
# multiclass probabilities
#     ↓
# loss
#     ↓
# optimizer learns weights
#
#
# sklearn handles the optimization internally.

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 5. MAKE CLASS PREDICTIONS
# ------------------------------------------------------------

# predict() gives the FINAL class.
#
# Conceptually:
#
# X_test
#    ↓
# logits
#    ↓
# probabilities
#    ↓
# highest probability
#    ↓
# predicted class

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 6. GET CLASS PROBABILITIES
# ------------------------------------------------------------

# predict_proba() lets us inspect the probabilities BEFORE
# the final class is selected.
#
# Example:
#
# [
#   0.02,
#   0.15,
#   0.83
# ]
#
# sklearn then chooses the class corresponding to 0.83.

y_proba = model.predict_proba(
    X_test
)


# ------------------------------------------------------------
# 7. CHECK CLASS ORDER
# ------------------------------------------------------------

# VERY IMPORTANT.
#
# Never assume:
#
# column 0 = Setosa
# column 1 = Versicolor
# column 2 = Virginica
#
# Ask the trained model.

classes = model.named_steps[
    "logistic_regression"
].classes_


print("\nClass order:")
print(classes)


# Therefore:
#
# y_proba[:, 0]
#
# belongs to classes[0]
#
# y_proba[:, 1]
#
# belongs to classes[1]
#
# etc.


# ------------------------------------------------------------
# 8. LOOK AT SOME PROBABILITY PREDICTIONS
# ------------------------------------------------------------

print("\nSample predictions:\n")


for i in range(5):

    print("Actual:")
    print(y_test.iloc[i])

    print("Probabilities:")

    for class_name, probability in zip(
        classes,
        y_proba[i]
    ):

        print(
            f"  {class_name}: "
            f"{probability:.4f}"
        )

    print(
        "Predicted:",
        y_pred[i]
    )

    print("-" * 50)


# ------------------------------------------------------------
# 9. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nAccuracy:",
    accuracy
)


# ------------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ------------------------------------------------------------

# Gives us:
#
# Precision
# Recall
# F1-score
# Support
#
# FOR EACH SPECIES.

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

# For multiclass classification the confusion matrix becomes
# a 3 x 3 matrix.
#
#
#                PREDICTED
#
#              S     V     Vi
#
# Actual S     ?     ?     ?
#
# Actual V     ?     ?     ?
#
# Actual Vi    ?     ?     ?
#
#
# Diagonal values = correct predictions.
#
# Off-diagonal values = mistakes.

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


print("\nConfusion Matrix:")

print(cm)


# ------------------------------------------------------------
# 12. INSPECT LEARNED WEIGHTS
# ------------------------------------------------------------

logistic_model = model.named_steps[
    "logistic_regression"
]


print("\nWeights shape:")
print(
    logistic_model.coef_.shape
)


print("\nLearned weights:")
print(
    logistic_model.coef_
)


print("\nBiases:")
print(
    logistic_model.intercept_
)


# With:
#
# 3 classes
# 4 features
#
# coef_ will contain one coefficient vector per class.
#
# Conceptually:
#
#                  SepalL  SepalW  PetalL  PetalW
#
# Setosa             w       w       w       w
#
# Versicolor         w       w       w       w
#
# Virginica          w       w       w       w
#
#
# This is sklearn's version of the weight matrix
# we manually built using NumPy.


# ------------------------------------------------------------
# 13. PREDICT ONE NEW FLOWER
# ------------------------------------------------------------

# IMPORTANT:
#
# We give raw measurements here.
#
# Do NOT manually scale them.
#
# Pipeline automatically applies the SAME StandardScaler
# learned from X_train.

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)


new_probability = model.predict_proba(
    new_flower
)[0]


new_prediction = model.predict(
    new_flower
)[0]


print("\nNew flower probabilities:")


for class_name, probability in zip(
    classes,
    new_probability
):

    print(
        f"{class_name}: "
        f"{probability:.4f}"
    )


print(
    "\nFinal prediction:",
    new_prediction
)