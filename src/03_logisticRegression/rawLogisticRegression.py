import numpy as np


# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH USING NUMPY
# ============================================================
#
# Problem:
# Predict whether a student will PASS or FAIL
# based on the number of hours they studied.
#
# X = Hours studied
# y = Actual result
#
# y = 0 -> FAIL
# y = 1 -> PASS
#
#
# COMPLETE TRAINING FLOW:
#
# X
# ↓
# z = Xw + b
# ↓
# sigmoid(z)
# ↓
# predicted probability p
# ↓
# Binary Cross Entropy Loss
# ↓
# calculate gradients
# ↓
# update weights and bias using Gradient Descent
# ↓
# repeat for many epochs
#
# ============================================================



# ============================================================
# 1. CREATE THE TRAINING DATA
# ============================================================

# Each inner list represents ONE student.
#
# We currently have only ONE feature:
#
# hours_studied
#
# Student 1 studied 1 hour
# Student 2 studied 2 hours
# ...
# Student 8 studied 8 hours
#
# X therefore has:
#
# 8 rows    -> 8 students / samples
# 1 column  -> 1 feature
#
# Shape:
#
# X.shape = (8, 1)

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
], dtype=float)


# These are the ACTUAL labels.
#
# First four students failed.
# Last four students passed.
#
# Hours:  1  2  3  4  5  6  7  8
# Actual: 0  0  0  0  1  1  1  1

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=float)



# ============================================================
# 2. GET NUMBER OF SAMPLES AND FEATURES
# ============================================================

# X.shape returns:
#
# (number_of_rows, number_of_columns)
#
# Here:
#
# X.shape = (8, 1)
#
# Therefore:
#
# n_samples  = 8 --> Rows
# n_features = 1 --> Inputs/ features 

n_samples, n_features = X.shape



# ============================================================
# 3. INITIALIZE MODEL PARAMETERS
# ============================================================

# Logistic Regression learns:
#
# weights
# bias
#
#
# If we had:
#
# x1 = hours studied
# x2 = attendance
# x3 = previous marks
#
# we would need THREE weights:
#
# w1
# w2
# w3
#
# because every feature gets its own weight.
#
#
# But currently:
#
# n_features = 1
#
# Therefore:
#
# np.zeros(n_features)
#
# becomes:
#
# np.zeros(1)
#
# which gives:
#
# [0.]
#
# So our model starts with weight = 0.

weights = np.zeros(n_features)


# Bias is also initialized to zero.
#
# Initially:
#
# weight = 0
# bias   = 0
#
# Therefore:
#
# z = Xw + b
#
# becomes:
#
# z = X(0) + 0
#
# so every student initially gets z = 0.

bias = 0.0



# ============================================================
# 4. DEFINE HYPERPARAMETERS
# ============================================================

# Learning rate determines HOW LARGE each Gradient Descent
# update should be.
#
# Gradient tells us:
#
# "Which direction increases the loss?"
#
# Gradient Descent moves in the opposite direction.
#
# Learning rate controls how big that movement is.
#
#
# Too large:
#
# We may overshoot the minimum or make training unstable.
#
#
# Too small:
#
# Training may take a very long time.

learning_rate = 0.2


# One epoch means:
#
# Perform one complete:
#
# forward pass
# +
# loss calculation
# +
# gradient calculation
# +
# parameter update
#
# We will repeat that process 5000 times.

epochs = 5000



# ============================================================
# 5. SIGMOID FUNCTION
# ============================================================

def sigmoid(z):

    # Our linear equation produces:
    #
    # z = Xw + b
    #
    # z is NOT a probability.
    #
    # It can theoretically be:
    #
    # -100
    # -10
    # -2
    # 0
    # 4
    # 20
    # 100
    #
    # But probability must stay between:
    #
    # 0 and 1
    #
    #
    # Sigmoid converts the raw linear score z
    # into a probability.
    #
    #
    # Large negative z
    #       ↓
    # probability approaches 0
    #
    #
    # z = 0
    #       ↓
    # probability = 0.5
    #
    #
    # Large positive z
    #       ↓
    # probability approaches 1
    #
    #
    # np.exp(-z) calculates e^(-z).

    return 1 / (1 + np.exp(-z))



# ============================================================
# 6. BINARY CROSS ENTROPY LOSS
# ============================================================

def binary_cross_entropy(y, probabilities):

    # --------------------------------------------------------
    # WHY DO WE NEED EPSILON?
    # --------------------------------------------------------
    #
    # BCE contains logarithms.
    #
    # We may need:
    #
    # log(p)
    #
    # or:
    #
    # log(1-p)
    #
    #
    # But log(0) is undefined.
    #
    # If the computer produces probability exactly:
    #
    # 0
    #
    # or:
    #
    # 1
    #
    # BCE could run into numerical problems.
    #
    #
    # So we create an extremely tiny number:
    #
    # 0.000000000000001
    #
    # This doesn't meaningfully change our prediction.
    # It only protects the logarithm.

    epsilon = 1e-15


    # np.clip forces the probabilities to remain between:
    #
    # epsilon
    #
    # and
    #
    # 1 - epsilon
    #
    #
    # Example:
    #
    # probability = 0
    #
    # becomes approximately:
    #
    # 0.000000000000001
    #
    #
    # probability = 1
    #
    # becomes approximately:
    #
    # 0.999999999999999
    #
    #
    # Normal values such as:
    #
    # 0.2
    # 0.6
    # 0.83
    #
    # remain unchanged.

    probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon
    )


    # --------------------------------------------------------
    # CALCULATE BCE
    # --------------------------------------------------------
    #
    # y acts as our switch.
    #
    #
    # CASE 1:
    #
    # Actual y = 1
    #
    # Then:
    #
    # y = 1
    # 1-y = 0
    #
    # So the Class 0 part disappears.
    #
    # We care about:
    #
    # -log(p)
    #
    #
    # CASE 2:
    #
    # Actual y = 0
    #
    # Then:
    #
    # y = 0
    # 1-y = 1
    #
    # So the Class 1 part disappears.
    #
    # We care about:
    #
    # -log(1-p)
    #
    #
    # NumPy performs this calculation for EVERY student.
    #
    # np.mean() then calculates the average loss
    # across all students.
    #
    #
    # So "loss" here is ONE number representing how bad
    # the model currently is across the training dataset.

    loss = -np.mean(
        y * np.log(probabilities)
        +
        (1 - y) * np.log(1 - probabilities)
    )


    # Return the calculated BCE back to the training loop.

    return loss



# ============================================================
# 7. START TRAINING
# ============================================================
#
# THIS IS WHERE THE MODEL ACTUALLY LEARNS.
#
#
# Every epoch performs:
#
#
#                 FORWARD PASS
#
# X
# ↓
# z = Xw + b
# ↓
# sigmoid
# ↓
# probabilities
# ↓
# BCE
#
#
#                 BACKWARD PASS
#
# BCE
# ↓
# gradients
# ↓
# dw and db
#
#
#                 GRADIENT DESCENT
#
# update weights
# update bias
#
#
# Then repeat.
#
# ============================================================

for epoch in range(epochs):


    # ========================================================
    # FORWARD PASS
    # ========================================================


    # --------------------------------------------------------
    # STEP 1: CALCULATE z
    # --------------------------------------------------------
    #
    # Our normal equation is:
    #
    # z = wx + b
    #
    #
    # But we have multiple students.
    #
    # Instead of manually doing:
    #
    # student1_z = x1*w + b
    # student2_z = x2*w + b
    # student3_z = x3*w + b
    #
    # NumPy can calculate ALL of them simultaneously.
    #
    #
    # @ means MATRIX MULTIPLICATION.
    #
    #
    # X shape:
    #
    # (8, 1)
    #
    # weights shape:
    #
    # (1,)
    #
    #
    # X @ weights
    #
    # gives:
    #
    # (8,)
    #
    # meaning:
    #
    # one z value for each student.
    #
    #
    # bias is then added to every z.

    z = X @ weights + bias


    # --------------------------------------------------------
    # STEP 2: CONVERT z INTO PROBABILITY
    # --------------------------------------------------------
    #
    # z is still an unrestricted linear score.
    #
    # Sigmoid converts every z into:
    #
    # P(Y = 1)
    #
    #
    # In our problem:
    #
    # P(student passes)
    #
    #
    # If probabilities becomes:
    #
    # [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    #
    # it means:
    #
    # Student 1 -> 10% probability of passing
    # Student 2 -> 20%
    # ...
    # Student 8 -> 90%

    probabilities = sigmoid(z)


    # --------------------------------------------------------
    # STEP 3: CALCULATE BCE LOSS
    # --------------------------------------------------------
    #
    # Now we have:
    #
    # actual answers:
    #
    # y
    #
    # and:
    #
    # predicted probabilities:
    #
    # probabilities
    #
    #
    # BCE compares them and tells us:
    #
    # "How bad are our current predictions?"
    #
    #
    # This returns ONE average loss value.

    loss = binary_cross_entropy(
        y,
        probabilities
    )



    # ========================================================
    # BACKWARD PASS
    # ========================================================
    #
    # Now we know HOW WRONG the model is.
    #
    # But BCE alone does NOT tell us:
    #
    # "How should I change the weights?"
    #
    #
    # That's why we need gradients.
    #
    #
    # Dependency:
    #
    # weight
    # ↓
    # z
    # ↓
    # probability
    # ↓
    # BCE
    #
    #
    # Using derivatives + chain rule,
    # we determine how changing the parameters
    # changes the final loss.
    #
    # ========================================================


    # --------------------------------------------------------
    # STEP 4: CALCULATE p - y
    # --------------------------------------------------------
    #
    # Earlier we derived:
    #
    # dJ/dz = p - y
    #
    #
    # Here:
    #
    # p = predicted probability
    # y = actual label
    #
    #
    # Suppose:
    #
    # actual y = 0
    # predicted p = 0.9
    #
    # Then:
    #
    # p - y
    #
    # = 0.9 - 0
    #
    # = +0.9
    #
    #
    # Suppose:
    #
    # actual y = 1
    # predicted p = 0.2
    #
    # Then:
    #
    # p - y
    #
    # = 0.2 - 1
    #
    # = -0.8
    #
    #
    # IMPORTANT:
    #
    # We call this variable "error" for convenience.
    #
    # It is NOT our BCE loss.
    #
    # More precisely:
    #
    # error = dJ/dz
    #
    # for each observation.

    error = probabilities - y


    # --------------------------------------------------------
    # STEP 5: CALCULATE WEIGHT GRADIENT
    # --------------------------------------------------------
    #
    # For ONE training example we derived:
    #
    # dJ/dw = (p-y)x
    #
    #
    # That was easy when we had ONE student.
    #
    #
    # But now we have 8 students.
    #
    # Every student contributes something to the
    # final weight gradient.
    #
    #
    # Conceptually we want:
    #
    # student1:
    #
    # x1 * (p1-y1)
    #
    # +
    #
    # student2:
    #
    # x2 * (p2-y2)
    #
    # +
    #
    # student3:
    #
    # x3 * (p3-y3)
    #
    # ...
    #
    #
    # and then average them.
    #
    #
    # NumPy lets us calculate this using:
    #
    # X.T @ error
    #
    #
    # Let's understand the shapes.
    #
    #
    # X originally:
    #
    # shape = (8, 1)
    #
    # Visually:
    #
    # [1]
    # [2]
    # [3]
    # [4]
    # [5]
    # [6]
    # [7]
    # [8]
    #
    #
    # X.T means transpose.
    #
    # Rows become columns.
    #
    # So:
    #
    # X.T
    #
    # becomes approximately:
    #
    # [1 2 3 4 5 6 7 8]
    #
    # shape:
    #
    # (1, 8)
    #
    #
    # error contains:
    #
    # [error1,
    #  error2,
    #  error3,
    #  ...
    #  error8]
    #
    #
    # Matrix multiplication:
    #
    # X.T @ error
    #
    # effectively performs:
    #
    # 1*error1
    # +
    # 2*error2
    # +
    # 3*error3
    # +
    # ...
    # +
    # 8*error8
    #
    #
    # Therefore this is simply the vectorized version of:
    #
    # (p-y)x
    #
    # across the entire dataset.
    #
    #
    # Why divide by n_samples?
    #
    # Because our BCE used np.mean().
    #
    # We are minimizing the AVERAGE loss.
    #
    # Therefore our gradient should also represent
    # the average contribution across observations.

    dw = (X.T @ error) / n_samples


    # --------------------------------------------------------
    # STEP 6: CALCULATE BIAS GRADIENT
    # --------------------------------------------------------
    #
    # Remember:
    #
    # z = wx + b
    #
    #
    # For weight:
    #
    # dz/dw = x
    #
    #
    # That's why x appears in:
    #
    # dJ/dw = (p-y)x
    #
    #
    # But for bias:
    #
    # dz/db = 1
    #
    #
    # Therefore:
    #
    # dJ/db = p-y
    #
    #
    # For multiple observations,
    # we simply calculate the average of all:
    #
    # p-y
    #
    # values.

    db = np.mean(error)



    # ========================================================
    # GRADIENT DESCENT
    # ========================================================
    #
    # We now have:
    #
    # dw = how loss changes with respect to weights
    #
    # db = how loss changes with respect to bias
    #
    #
    # Gradient points toward increasing loss.
    #
    # But we want DECREASING loss.
    #
    # Therefore we move in the opposite direction.
    #
    # ========================================================


    # --------------------------------------------------------
    # STEP 7: UPDATE WEIGHTS
    # --------------------------------------------------------
    #
    # New weight:
    #
    # old weight
    # -
    # learning_rate * gradient
    #
    #
    # If dw is positive:
    #
    # subtracting it decreases the weight.
    #
    #
    # If dw is negative:
    #
    # subtracting a negative number increases the weight.
    #
    #
    # So Gradient Descent automatically determines
    # which direction the weight should move.

    weights = weights - learning_rate * dw


    # --------------------------------------------------------
    # STEP 8: UPDATE BIAS
    # --------------------------------------------------------
    #
    # Same Gradient Descent logic for bias.

    bias = bias - learning_rate * db


    # --------------------------------------------------------
    # STEP 9: DISPLAY TRAINING PROGRESS
    # --------------------------------------------------------
    #
    # We have 5000 epochs.
    #
    # Printing every epoch would create a massive output.
    #
    # epoch % 500 == 0
    #
    # means:
    #
    # print at:
    #
    # epoch 0
    # epoch 500
    # epoch 1000
    # epoch 1500
    # ...
    #
    #
    # Ideally we should see the BCE loss gradually decrease.

    if epoch % 500 == 0:
        print(
            f"Epoch: {epoch:4d} | "
            f"Loss: {loss:.6f}"
        )



# ============================================================
# 8. TRAINING IS COMPLETE
# ============================================================
#
# At this point Gradient Descent has finished.
#
# weights and bias now contain the learned parameters.
#
# We will NOT reset them.
#
# These are what our trained Logistic Regression model
# has learned from the data.
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("Learned weights:", weights)
print("Learned bias:", bias)



# ============================================================
# 9. GET PROBABILITIES FOR TRAINING DATA
# ============================================================
#
# Now let's see what our trained model predicts
# for the original students.
#
#
# Notice:
#
# We use the SAME prediction pipeline:
#
# X
# ↓
# z
# ↓
# sigmoid
# ↓
# probability
#
#
# But we DON'T calculate gradients now.
#
# We are just making predictions.
# ============================================================

z = X @ weights + bias


# Convert the final linear scores into probabilities.

probabilities = sigmoid(z)



# ============================================================
# 10. APPLY DECISION THRESHOLD
# ============================================================
#
# Logistic Regression naturally gives:
#
# probabilities
#
# NOT classes.
#
#
# Example:
#
# 0.12
# 0.31
# 0.48
# 0.67
# 0.91
#
#
# To convert probability into:
#
# FAIL / PASS
#
# we need a decision threshold.
#
#
# We'll use:
#
# threshold = 0.5
#
#
# probability >= 0.5
#
#       ↓
#
# Class 1 / PASS
#
#
# probability < 0.5
#
#       ↓
#
# Class 0 / FAIL

threshold = 0.5


# This comparison produces Boolean values.
#
# Example:
#
# [0.2, 0.7, 0.9] >= 0.5
#
# becomes:
#
# [False, True, True]
#
#
# .astype(int)
#
# converts:
#
# False -> 0
# True  -> 1

predictions = (
    probabilities >= threshold
).astype(int)



# ============================================================
# 11. DISPLAY TRAINING RESULTS
# ============================================================

print("\n========================================")
print("TRAINING DATA RESULTS")
print("========================================")


# X currently looks like:
#
# [[1],
#  [2],
#  [3],
#  ...]
#
#
# X.flatten() converts it into:
#
# [1, 2, 3, 4, 5, 6, 7, 8]
#
#
# zip() lets us iterate through:
#
# hours
# probability
# prediction
# actual answer
#
# together.

for hours, probability, prediction, actual in zip(
    X.flatten(),
    probabilities,
    predictions,
    y
):

    # Convert numerical prediction into readable text.

    predicted_label = (
        "PASS"
        if prediction == 1
        else "FAIL"
    )


    # Convert actual 0/1 into readable text.

    actual_label = (
        "PASS"
        if actual == 1
        else "FAIL"
    )


    print(
        f"Hours: {hours:.1f} | "
        f"Pass Probability: {probability:.4f} | "
        f"Prediction: {predicted_label} | "
        f"Actual: {actual_label}"
    )



# ============================================================
# 12. CREATE NEW UNSEEN DATA
# ============================================================
#
# Now we move from TRAINING to INFERENCE.
#
#
# These students were NOT shown during training.
#
# The model has never seen:
#
# 2.5 hours
# 4.5 hours
# 5.5 hours
# 7.5 hours
#
#
# We want our trained model to make predictions for them.
# ============================================================

new_students = np.array([
    [2.5],
    [4.5],
    [5.5],
    [7.5]
])



# ============================================================
# 13. INFERENCE — CALCULATE z
# ============================================================
#
# We use the LEARNED weights and bias.
#
# Very important:
#
# During inference:
#
# NO BCE
# NO gradient calculation
# NO Gradient Descent
# NO weight update
#
#
# The model has already learned.
#
# We simply ask:
#
# "Using what you learned, what do you predict?"
# ============================================================

z_new = new_students @ weights + bias



# ============================================================
# 14. CONVERT NEW z VALUES INTO PROBABILITIES
# ============================================================
#
# Same sigmoid.
#
# Same prediction pipeline.
#
# Training:
#
# X -> z -> sigmoid -> probability
#
#
# Inference:
#
# New X -> z -> sigmoid -> probability
#
#
# The difference is:
#
# Training continues into:
#
# loss -> gradients -> updates
#
#
# Inference STOPS after prediction.

new_probabilities = sigmoid(z_new)



# ============================================================
# 15. APPLY DECISION THRESHOLD
# ============================================================
#
# Again:
#
# probability >= 0.5 -> PASS
#
# probability < 0.5  -> FAIL

new_predictions = (
    new_probabilities >= threshold
).astype(int)



# ============================================================
# 16. DISPLAY NEW STUDENT PREDICTIONS
# ============================================================

print("\n========================================")
print("NEW STUDENT PREDICTIONS")
print("========================================")


for hours, probability, prediction in zip(
    new_students.flatten(),
    new_probabilities,
    new_predictions
):

    result = (
        "PASS"
        if prediction == 1
        else "FAIL"
    )

    print(
        f"Hours: {hours:.1f} | "
        f"Pass Probability: {probability:.4f} | "
        f"Prediction: {result}"
    )



# ============================================================
# FINAL MENTAL MODEL
# ============================================================
#
#
# TRAINING
# --------
#
# X
# ↓
# z = Xw + b
# ↓
# sigmoid(z)
# ↓
# probabilities
# ↓
# BCE
# ↓
# error = p - y
# ↓
# calculate dw and db
# ↓
# Gradient Descent
# ↓
# update weights and bias
# ↓
# repeat
#
#
#
# INFERENCE
# ---------
#
# New X
# ↓
# z = Xw + b
# ↓
# sigmoid(z)
# ↓
# probability
# ↓
# decision threshold
# ↓
# Class 0 / Class 1
#
#
#
# THE CORE TRAINING CODE IS ACTUALLY JUST:
#
#
# for epoch in range(epochs):
#
#     z = X @ weights + bias
#
#     probabilities = sigmoid(z)
#
#     loss = binary_cross_entropy(
#         y,
#         probabilities
#     )
#
#     error = probabilities - y
#
#     dw = (X.T @ error) / n_samples
#
#     db = np.mean(error)
#
#     weights = weights - learning_rate * dw
#
#     bias = bias - learning_rate * db
#
#
# Everything else in this file is explanation,
# safety, printing and inference.
#
# ============================================================