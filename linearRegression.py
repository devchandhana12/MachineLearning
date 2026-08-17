import numpy as np

# 1. Training data
x = np.array([1,2,3,4,5,6], dtype=float)
y = np.array([3,5,7,9,11,13], dtype=float)


# 2. Initialize parameters
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000
m = len(x)

# 3. Training loop

for epohch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y

    # MSE
    loss = np.mean(error ** 2)

    # gradients
    dw = (2 / m) * np.sum(error * x)
    db = (2 / m) * np.sum(error)

    # update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # print progress occasionally
    if epohch * 100 == 0:
        print(
            f"Epoch: {epohch:4d} |"
            f"Loss: {loss:.6f}"
            f"w: {w:.4f}"
            f"b: {b:.4f}"
        )

# 4. Final learned model

print("\nTraining completed")
print(f"Learned weight: {w:.4f}")
print(f"Learned bias:   {b:.4f}")

# 5. Prediction on new data
new_x = 7
prediction = w * new_x + b

print(f"\nPrediction for x = {new_x}: {prediction:.2f}")
