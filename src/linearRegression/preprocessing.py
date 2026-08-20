import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def preprocess_data():

    # ========================================================
    # LOAD DATA
    # ========================================================

    url = "https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv"

    df = pd.read_csv(
        url,
        index_col=0
    )


    # ========================================================
    # FEATURES / TARGET
    # ========================================================

    X = df.drop(columns=["Sales"])
    y = df["Sales"]


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # ========================================================
    # PREPROCESSING PIPELINE
    # ========================================================

    preprocessor = Pipeline([
        (
            "scaler",
            StandardScaler()
        )
    ])


    # ========================================================
    # FIT ON TRAINING DATA
    # ========================================================

    X_train_processed = preprocessor.fit_transform(
        X_train
    )


    # ========================================================
    # TRANSFORM VALIDATION DATA
    # ========================================================

    X_val_processed = preprocessor.transform(
        X_val
    )


    # ========================================================
    # RETURN PROCESSED DATA
    # ========================================================

    return (
        X_train_processed,
        X_val_processed,
        y_train,
        y_val
    )