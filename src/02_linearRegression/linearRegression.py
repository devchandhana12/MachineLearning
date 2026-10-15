import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import preprocess_data


# Get the data returned by preprocessing.py
X_train, X_val, y_train, y_val = preprocess_data()


# Create model
model = LinearRegression()


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_val)


# Evaluate
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)


print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)