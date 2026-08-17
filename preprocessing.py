import pandas as pd

df = pd.read_csv('train.csv')

# --------- EXPLORATION ---------
# print(df.shape)
print(df.columns)
# print(df.dtypes)
# print(df.info())


# -------- DIAGNOSIS-------------
# print(df.duplicated().sum())
missing = df.isnull().sum()
missing = missing[missing > 0]

# -------- DEPENDENCY / STRUCTURAL NULL CHECKS --------

# Garage
garage_count = df[
    (df['GarageType'].isnull()) &
    ((df['GarageArea'] != 0) | (df['GarageCars'] != 0))
]
print("Garage inconsistent rows:", len(garage_count))


# Fireplace
fireplace_count = df[
    (df['FireplaceQu'].isnull()) &
    (df['Fireplaces'] != 0)
]
print("Fireplace inconsistent rows:", len(fireplace_count))


# Pool
pool_count = df[
    (df['PoolQC'].isnull()) &
    (df['PoolArea'] != 0)
]
print("Pool inconsistent rows:", len(pool_count))


# Masonry veneer
mas_count = df[
    (df['MasVnrType'].isnull()) &
    (df['MasVnrArea'] != 0)
]
print("Masonry veneer inconsistent rows:", len(mas_count))


# Basement quality
bsmt_qual_count = df[
    (df['BsmtQual'].isnull()) &
    (df['TotalBsmtSF'] != 0)
]
print("BsmtQual inconsistent rows:", len(bsmt_qual_count))


# Basement condition
bsmt_cond_count = df[
    (df['BsmtCond'].isnull()) &
    (df['TotalBsmtSF'] != 0)
]
print("BsmtCond inconsistent rows:", len(bsmt_cond_count))


# Basement exposure
bsmt_exposure_count = df[
    (df['BsmtExposure'].isnull()) &
    (df['TotalBsmtSF'] != 0)
]
print("BsmtExposure inconsistent rows:", len(bsmt_exposure_count))


# Basement finish type 1
bsmt_fin1_count = df[
    (df['BsmtFinType1'].isnull()) &
    (df['TotalBsmtSF'] != 0)
]
print("BsmtFinType1 inconsistent rows:", len(bsmt_fin1_count))


# Basement finish type 2
bsmt_fin2_count = df[
    (df['BsmtFinType2'].isnull()) &
    (df['TotalBsmtSF'] != 0)
]
print("BsmtFinType2 inconsistent rows:", len(bsmt_fin2_count))


# Masonry area missing but type exists
mas_area_count = df[
    (df['MasVnrArea'].isnull()) &
    (df['MasVnrType'].notnull())
]
print("MasVnrArea inconsistent rows:", len(mas_area_count))
