import pandas as pd

df = pd.read_csv("train.csv")

# ====================== EXPLORATION ======================

# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.info())


# ====================== DIAGNOSIS ======================

# Duplicate check
# print(df.duplicated().sum())

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
# print(missing)


# ====================== DEPENDENCY / STRUCTURAL NULL CHECKS ======================

# Garage
garage_count = df[
    (df["GarageType"].isnull()) &
    ((df["GarageArea"] != 0) | (df["GarageCars"] != 0))
]
# print("Garage inconsistent rows:", len(garage_count))


# Fireplace
fireplace_count = df[
    (df["FireplaceQu"].isnull()) &
    (df["Fireplaces"] != 0)
]
# print("Fireplace inconsistent rows:", len(fireplace_count))


# Pool
pool_count = df[
    (df["PoolQC"].isnull()) &
    (df["PoolArea"] != 0)
]
# print("Pool inconsistent rows:", len(pool_count))


# Masonry veneer
mas_count = df[
    (df["MasVnrType"].isnull()) &
    (df["MasVnrArea"] != 0)
]
# print("Masonry veneer inconsistent rows:", len(mas_count))


# Basement quality
bsmt_qual_count = df[
    (df["BsmtQual"].isnull()) &
    (df["TotalBsmtSF"] != 0)
]
# print("BsmtQual inconsistent rows:", len(bsmt_qual_count))


# Basement condition
bsmt_cond_count = df[
    (df["BsmtCond"].isnull()) &
    (df["TotalBsmtSF"] != 0)
]
# print("BsmtCond inconsistent rows:", len(bsmt_cond_count))


# Basement exposure
bsmt_exposure_count = df[
    (df["BsmtExposure"].isnull()) &
    (df["TotalBsmtSF"] != 0)
]
# print("BsmtExposure inconsistent rows:", len(bsmt_exposure_count))


# Basement finish type 1
bsmt_fin1_count = df[
    (df["BsmtFinType1"].isnull()) &
    (df["TotalBsmtSF"] != 0)
]
# print("BsmtFinType1 inconsistent rows:", len(bsmt_fin1_count))


# Basement finish type 2
bsmt_fin2_count = df[
    (df["BsmtFinType2"].isnull()) &
    (df["TotalBsmtSF"] != 0)
]
# print("BsmtFinType2 inconsistent rows:", len(bsmt_fin2_count))


# Masonry area missing but type exists
mas_area_count = df[
    (df["MasVnrArea"].isnull()) &
    (df["MasVnrType"].notnull())
]
# print("MasVnrArea inconsistent rows:", len(mas_area_count))


# ====================== HANDLING NULLS ======================

# Electrical: genuine missing categorical value
df["Electrical"] = df["Electrical"].fillna(
    df["Electrical"].mode()[0]
)


# Structural absence categories
df["Alley"] = df["Alley"].fillna("NoAlley")
df["Fence"] = df["Fence"].fillna("NoFence")
df["MiscFeature"] = df["MiscFeature"].fillna("NoMiscFeature")

df["BsmtQual"] = df["BsmtQual"].fillna("NoBasement")
df["BsmtCond"] = df["BsmtCond"].fillna("NoBasement")
df["BsmtFinType1"] = df["BsmtFinType1"].fillna("NoBasement")

df["PoolQC"] = df["PoolQC"].fillna("NoPool")

df["GarageType"] = df["GarageType"].fillna("NoGarage")
df["GarageFinish"] = df["GarageFinish"].fillna("NoGarage")
df["GarageQual"] = df["GarageQual"].fillna("NoGarage")
df["GarageCond"] = df["GarageCond"].fillna("NoGarage")

df["FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")


# ====================== LOT FRONTAGE ======================

# Fill missing frontage using median frontage
# of houses in the same neighborhood
df["LotFrontage"] = df["LotFrontage"].fillna(
    df.groupby("Neighborhood")["LotFrontage"].transform("median")
)


# ====================== GARAGE YEAR BUILT ======================

# If garage doesn't exist, garage year built becomes 0
df.loc[
    (df["GarageType"] == "NoGarage") &
    (df["GarageYrBlt"].isnull()),
    "GarageYrBlt"
] = 0


# ====================== MASONRY VENEER ======================

# Case 1:
# No veneer area + missing type = no veneer
df.loc[
    (df["MasVnrArea"] == 0) &
    (df["MasVnrType"].isnull()),
    "MasVnrType"
] = "NoVeneer"


# Case 2:
# Veneer area exists but type is missing
# Calculate mode ONLY among houses that actually have veneer
veneer_mode = df.loc[
    (df["MasVnrArea"] > 0) &
    (df["MasVnrType"].notnull()),
    "MasVnrType"
].mode()[0]

df.loc[
    (df["MasVnrArea"] > 0) &
    (df["MasVnrType"].isnull()),
    "MasVnrType"
] = veneer_mode


# Case 3:
# MasVnrArea itself is missing
# Use median among houses with positive veneer area
positive_veneer_area_median = df.loc[
    df["MasVnrArea"] > 0,
    "MasVnrArea"
].median()

df.loc[
    df["MasVnrArea"].isnull(),
    "MasVnrArea"
] = positive_veneer_area_median


# If type is still unknown after all logical checks
df["MasVnrType"] = df["MasVnrType"].fillna("Unknown")


# ====================== BASEMENT FIN TYPE 2 ======================

# Structural absence
df.loc[
    (df["TotalBsmtSF"] == 0) &
    (df["BsmtFinType2"].isnull()),
    "BsmtFinType2"
] = "NoBasement"


# Genuine missing value among houses WITH basement
bsmt_fin2_mode = df.loc[
    (df["TotalBsmtSF"] > 0) &
    (df["BsmtFinType2"].notnull()),
    "BsmtFinType2"
].mode()[0]

df.loc[
    (df["TotalBsmtSF"] > 0) &
    (df["BsmtFinType2"].isnull()),
    "BsmtFinType2"
] = bsmt_fin2_mode


# ====================== BASEMENT EXPOSURE ======================

# Structural absence
df.loc[
    (df["TotalBsmtSF"] == 0) &
    (df["BsmtExposure"].isnull()),
    "BsmtExposure"
] = "NoBasement"


# Genuine missing value among houses WITH basement
bsmt_exposure_mode = df.loc[
    (df["TotalBsmtSF"] > 0) &
    (df["BsmtExposure"].notnull()),
    "BsmtExposure"
].mode()[0]

df.loc[
    (df["TotalBsmtSF"] > 0) &
    (df["BsmtExposure"].isnull()),
    "BsmtExposure"
] = bsmt_exposure_mode


# ====================== FINAL NULL CHECK ======================

missing_after_cleaning = df.isnull().sum()
missing_after_cleaning = missing_after_cleaning[
    missing_after_cleaning > 0
]

print(missing_after_cleaning)