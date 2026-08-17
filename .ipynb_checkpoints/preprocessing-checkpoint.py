import pandas as pd

df = pd.read_csv('train.csv')

# --------- EXPLORATION ---------
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.info())


# -------- DIAGNOSIS-------------
# print(df.duplicated().sum())
print(df.isnull().sum())