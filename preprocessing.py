import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("train.csv")


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# STRUCTURAL MISSING VALUES
# ============================================================

def apply_structural_rules(df):
    df = df.copy()

    df["Alley"] = df["Alley"].fillna("NoAlley")
    df["Fence"] = df["Fence"].fillna("NoFence")
    df["MiscFeature"] = df["MiscFeature"].fillna(
        "NoMiscFeature"
    )

    df["PoolQC"] = df["PoolQC"].fillna("NoPool")

    df["FireplaceQu"] = df["FireplaceQu"].fillna(
        "NoFireplace"
    )

    for col in [
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
    ]:
        df[col] = df[col].fillna("NoGarage")

    for col in [
        "BsmtQual",
        "BsmtCond",
        "BsmtFinType1",
    ]:
        df[col] = df[col].fillna("NoBasement")

    df.loc[
        (df["GarageType"] == "NoGarage") &
        (df["GarageYrBlt"].isna()),
        "GarageYrBlt",
    ] = 0

    df.loc[
        (df["TotalBsmtSF"] == 0) &
        (df["BsmtExposure"].isna()),
        "BsmtExposure",
    ] = "NoBasement"

    df.loc[
        (df["TotalBsmtSF"] == 0) &
        (df["BsmtFinType2"].isna()),
        "BsmtFinType2",
    ] = "NoBasement"

    df.loc[
        (df["MasVnrArea"] == 0) &
        (df["MasVnrType"].isna()),
        "MasVnrType",
    ] = "NoVeneer"

    return df


X_train = apply_structural_rules(X_train)
X_val = apply_structural_rules(X_val)


# ============================================================
# LEARN IMPUTATION VALUES FROM TRAINING DATA ONLY
# ============================================================

electrical_mode = X_train["Electrical"].mode()[0]

lot_frontage_by_neighborhood = (
    X_train.groupby("Neighborhood")["LotFrontage"]
    .median()
)

global_lot_frontage_median = (
    X_train["LotFrontage"].median()
)

veneer_mode = X_train.loc[
    (X_train["MasVnrArea"] > 0) &
    (X_train["MasVnrType"].notna()),
    "MasVnrType",
].mode()[0]

veneer_area_median = X_train.loc[
    X_train["MasVnrArea"] > 0,
    "MasVnrArea",
].median()

bsmt_exposure_mode = X_train.loc[
    (X_train["TotalBsmtSF"] > 0) &
    (X_train["BsmtExposure"].notna()),
    "BsmtExposure",
].mode()[0]

bsmt_fin2_mode = X_train.loc[
    (X_train["TotalBsmtSF"] > 0) &
    (X_train["BsmtFinType2"].notna()),
    "BsmtFinType2",
].mode()[0]


# ============================================================
# APPLY LEARNED IMPUTATION
# ============================================================

def apply_learned_imputation(df):
    df = df.copy()

    df["Electrical"] = df["Electrical"].fillna(
        electrical_mode
    )

    neighborhood_frontage = (
        df["Neighborhood"]
        .map(lot_frontage_by_neighborhood)
    )

    df["LotFrontage"] = (
        df["LotFrontage"]
        .fillna(neighborhood_frontage)
        .fillna(global_lot_frontage_median)
    )

    df.loc[
        (df["MasVnrArea"] > 0) &
        (df["MasVnrType"].isna()),
        "MasVnrType",
    ] = veneer_mode

    df["MasVnrArea"] = df["MasVnrArea"].fillna(
        veneer_area_median
    )

    df["MasVnrType"] = df["MasVnrType"].fillna(
        "Unknown"
    )

    df.loc[
        (df["TotalBsmtSF"] > 0) &
        (df["BsmtExposure"].isna()),
        "BsmtExposure",
    ] = bsmt_exposure_mode

    df.loc[
        (df["TotalBsmtSF"] > 0) &
        (df["BsmtFinType2"].isna()),
        "BsmtFinType2",
    ] = bsmt_fin2_mode

    return df


X_train = apply_learned_imputation(X_train)
X_val = apply_learned_imputation(X_val)


# ============================================================
# SEMANTIC CLEANUP
# ============================================================

X_train = X_train.drop(columns=["Id"])
X_val = X_val.drop(columns=["Id"])

X_train["MSSubClass"] = (
    X_train["MSSubClass"].astype(str)
)

X_val["MSSubClass"] = (
    X_val["MSSubClass"].astype(str)
)


# ============================================================
# LOG-TRANSFORM FEATURES
# ============================================================

log_features = [
    "LotArea",
    "MasVnrArea",
    "GrLivArea",
    "WoodDeckSF",
    "OpenPorchSF",
]


# ============================================================
# FEATURE GROUPS
# ============================================================

categorical_features = (
    X_train
    .select_dtypes(include=["object", "str"])
    .columns
    .tolist()
)

numeric_features = (
    X_train
    .select_dtypes(include=["int64", "float64"])
    .columns
    .tolist()
)

regular_numeric_features = [
    col
    for col in numeric_features
    if col not in log_features
]


# ============================================================
# NUMERIC LOG PIPELINE
# ============================================================

log_pipeline = Pipeline([
    (
        "log",
        FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one",
        ),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


# ============================================================
# REGULAR NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
        ),
    ),
])


# ============================================================
# COMPLETE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([
    (
        "log_numeric",
        log_pipeline,
        log_features,
    ),
    (
        "numeric",
        numeric_pipeline,
        regular_numeric_features,
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])


# ============================================================
# FIT TRAIN / TRANSFORM VALIDATION
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)


# ============================================================
# LOG TARGET
# ============================================================

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)


# ============================================================
# CHECK OUTPUT
# ============================================================

print(
    "X_train:",
    X_train.shape,
    "->",
    X_train_processed.shape,
)

print(
    "X_val:",
    X_val.shape,
    "->",
    X_val_processed.shape,
)

print(
    "y_train:",
    y_train_log.shape,
)

print(
    "y_val:",
    y_val_log.shape,
)