"""Train and evaluate a Gini-criterion decision tree for the Titanic data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"
LOG_PATH = BASE_DIR / "decision_tree_results.log"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("decision_tree")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create stable, useful features shared by train and test data."""
    result = data.copy()
    result["Title"] = (
        result["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
        .replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            }
        )
    )
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    result["CabinDeck"] = result["Cabin"].fillna("Unknown").str[0]
    result["FarePerPerson"] = result["Fare"] / result["FamilySize"].clip(lower=1)
    return result.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])


def gini_impurity(labels: pd.Series | np.ndarray) -> float:
    """Return Gini impurity for one node."""
    values = pd.Series(labels).dropna()
    if values.empty:
        return 0.0
    probabilities = values.value_counts(normalize=True).to_numpy()
    return float(1.0 - np.sum(probabilities**2))


def weighted_leaf_gini(labels: pd.Series, leaf_ids: np.ndarray) -> float:
    """Return sample-weighted Gini impurity across the reached leaves."""
    labels = pd.Series(labels).reset_index(drop=True)
    leaf_ids = np.asarray(leaf_ids)
    total = len(labels)
    return float(
        sum(
            (len(group) / total) * gini_impurity(labels.iloc[group.index])
            for _, group in labels.groupby(leaf_ids)
        )
    )


def make_model(max_depth: int = 6) -> Pipeline:
    categorical_columns = ["Sex", "Embarked", "Title", "CabinDeck"]
    numerical_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numerical",
                SimpleImputer(strategy="median"),
                numerical_columns,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_leaf=4,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    logger = configure_logging()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if "Survived" not in train.columns:
        raise ValueError("train.csv must contain the 'Survived' target column")
    if "Survived" in test.columns:
        raise ValueError("test.csv should not contain the 'Survived' target column")

    X = prepare_features(train.drop(columns=["Survived"]))
    y = train["Survived"].astype(int)
    X_test = prepare_features(test)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    validation_model = make_model()
    validation_model.fit(X_train, y_train)
    validation_predictions = validation_model.predict(X_validation)
    validation_leaf_ids = validation_model.named_steps["classifier"].apply(
        validation_model.named_steps["preprocessor"].transform(X_validation)
    )

    logger.info("Decision tree criterion: gini")
    logger.info("Training rows: %d | Test rows: %d", len(train), len(test))
    logger.info("Root Gini impurity: %.6f", gini_impurity(y_train))
    logger.info(
        "Validation accuracy: %.4f", accuracy_score(y_validation, validation_predictions)
    )
    logger.info(
        "Validation weighted leaf Gini impurity: %.6f",
        weighted_leaf_gini(y_validation, validation_leaf_ids),
    )
    logger.info("Validation confusion matrix:\n%s", confusion_matrix(y_validation, validation_predictions))
    logger.info(
        "Validation classification report:\n%s",
        classification_report(y_validation, validation_predictions, zero_division=0),
    )

    final_model = make_model()
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test).astype(int)
    prediction_output = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": test_predictions}
    )
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    logger.info("Final training root Gini impurity: %.6f", gini_impurity(y))
    logger.info("Test prediction counts: %s", prediction_output["Survived"].value_counts().to_dict())
    logger.info("Test predictions written to: %s", PREDICTIONS_PATH)
    logger.info("Gini results written to: %s", LOG_PATH)


if __name__ == "__main__":
    main()
