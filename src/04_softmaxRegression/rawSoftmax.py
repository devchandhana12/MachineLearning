import numpy as np


# ============================================================
# 1. CREATE A SMALL MULTICLASS DATASET
# ============================================================

# We want to classify an animal into ONE of three classes:
#
#   0 = Bird
#   1 = Cat
#   2 = Dog
#
# Each animal has two features:
#
#   X[0] = weight in kg
#   X[1] = height in cm
#
# This is intentionally a tiny/simple dataset because our goal
# is to understand Multiclass Logistic Regression internals,
# NOT build a production-quality animal classifier.


X = np.array([
    # weight, height

    [0.5, 10],     # Bird
    [0.8, 12],     # Bird
    [1.0, 14],     # Bird

    [3.0, 22],     # Cat
    [4.0, 25],     # Cat
    [5.0, 28],     # Cat

    [7.0, 35],     # Dog
    [9.0, 40],     # Dog
    [12.0, 48],    # Dog

], dtype=float)


# Target labels.
#
# IMPORTANT:
#
# These numbers are CLASS INDEXES.
#
# They do NOT mean:
#
# Bird < Cat < Dog
#
# and Dog being "2" does not mean Dog has some numerical
# magnitude of 2.
#
# They are simply identifiers:
#
# index 0 -> Bird
# index 1 -> Cat
# index 2 -> Dog

y = np.array([
    0, 0, 0,       # Birds
    1, 1, 1,       # Cats
    2, 2, 2        # Dogs
])


class_names = np.array([
    "Bird",
    "Cat",
    "Dog"
])


# ============================================================
# 2. UNDERSTAND THE DATASET SHAPE
# ============================================================

n_samples, n_features = X.shape

# We have three possible target classes.
n_classes = 3


print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Number of classes:", n_classes)


# X.shape is:
#
# (9, 2)
#
# meaning:
#
# 9 animals
# 2 features per animal
#
#
# In BINARY Logistic Regression we needed ONE score/logit.
#
# In MULTICLASS Logistic Regression we need ONE score/logit
# PER CLASS.
#
# Since we have:
#
# 2 features
# 3 classes
#
# our weight matrix must contain:
#
#
#                Bird      Cat      Dog
#
# weight_kg       ?         ?        ?
# height_cm       ?         ?        ?
#
#
# Therefore:
#
# weights.shape = (2, 3)
#
# This is one of the biggest differences from our binary
# implementation.


# ============================================================
# 3. INITIALIZE WEIGHTS AND BIASES
# ============================================================

weights = np.zeros(
    (n_features, n_classes)
)


# Each CLASS also gets its own bias.
#
# Therefore:
#
# Bird -> bias
# Cat  -> bias
# Dog  -> bias

bias = np.zeros(n_classes)


print("\nInitial weights:")
print(weights)

print("\nInitial bias:")
print(bias)


# Initially:
#
# weights =
#
# [[0, 0, 0],
#  [0, 0, 0]]
#
#
# bias =
#
# [0, 0, 0]
#
#
# The model knows absolutely nothing yet.
#
# Therefore initially all classes will receive the same score.


# ============================================================
# 4. CONVERT y INTO ONE-HOT ENCODING
# ============================================================

# Current y looks like:
#
# [0, 0, 0, 1, 1, 1, 2, 2, 2]
#
#
# But for understanding Categorical Cross-Entropy,
# we want:
#
# Bird -> [1, 0, 0]
# Cat  -> [0, 1, 0]
# Dog  -> [0, 0, 1]
#
#
# So create a matrix full of zeros:
#
# 9 samples x 3 classes

y_one_hot = np.zeros(
    (n_samples, n_classes)
)


# np.arange(n_samples) produces:
#
# [0,1,2,3,4,5,6,7,8]
#
#
# y contains:
#
# [0,0,0,1,1,1,2,2,2]
#
#
# NumPy therefore performs:
#
# row 0 -> column 0 = 1
# row 1 -> column 0 = 1
# row 2 -> column 0 = 1
#
# row 3 -> column 1 = 1
# row 4 -> column 1 = 1
# row 5 -> column 1 = 1
#
# row 6 -> column 2 = 1
# row 7 -> column 2 = 1
# row 8 -> column 2 = 1

y_one_hot[
    np.arange(n_samples),
    y
] = 1


print("\nOne-hot encoded target:")
print(y_one_hot)


# ============================================================
# 5. SOFTMAX FUNCTION
# ============================================================

def softmax(logits):

    # --------------------------------------------------------
    # logits contains one score PER CLASS PER SAMPLE.
    #
    # Example:
    #
    # [
    #   [1.2, 0.7, 1.8],   <- animal 1
    #   [2.1, 1.4, 0.3],   <- animal 2
    #   ...
    # ]
    #
    # Columns represent:
    #
    # Bird, Cat, Dog
    #
    # These are NOT probabilities yet.
    # --------------------------------------------------------


    # --------------------------------------------------------
    # NUMERICAL STABILITY
    # --------------------------------------------------------
    #
    # Softmax uses exponentials.
    #
    # Something like:
    #
    # e^1000
    #
    # becomes ridiculously large and can overflow.
    #
    # But Softmax only cares about RELATIVE differences
    # between logits.
    #
    # Therefore:
    #
    # [1000, 1001, 999]
    #
    # can safely become:
    #
    # [-1, 0, -2]
    #
    # by subtracting 1001.
    #
    # Their relative differences remain exactly the same.
    #
    # axis=1 means:
    #
    # "Find the maximum horizontally across the classes
    # for EACH SAMPLE."
    #
    # keepdims=True preserves the matrix shape needed
    # for NumPy broadcasting.

    max_logits = np.max(
        logits,
        axis=1,
        keepdims=True
    )


    stable_logits = logits - max_logits


    # --------------------------------------------------------
    # EXPONENTIATE THE LOGITS
    # --------------------------------------------------------
    #
    # Logits can be:
    #
    # -5'
    #  0
    #  4
    #
    # But probability strength must be positive.
    #
    # Exponentiation converts every value into something > 0.
    #
    # It also preserves ranking:
    #
    # if z_dog > z_cat
    #
    # then:
    #
    # e^z_dog > e^z_cat

    exp_logits = np.exp(stable_logits)


    # --------------------------------------------------------
    # FIND TOTAL STRENGTH FOR EACH SAMPLE
    # --------------------------------------------------------
    #
    # Suppose one animal produces:
    #
    # Bird strength = 2
    # Cat strength  = 3
    # Dog strength  = 5
    #
    # Total = 10
    #
    # axis=1 again means:
    #
    # sum horizontally across classes.

    sum_exp = np.sum(
        exp_logits,
        axis=1,
        keepdims=True
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------
    #
    # Each class gets:
    #
    # its strength / total strength
    #
    #
    # Example:
    #
    # Bird = 2 / 10 = .20
    # Cat  = 3 / 10 = .30
    # Dog  = 5 / 10 = .50
    #
    #
    # Therefore probabilities for every sample sum to 1.

    probabilities = exp_logits / sum_exp


    return probabilities


# ============================================================
# 6. CATEGORICAL CROSS-ENTROPY
# ============================================================

def categorical_cross_entropy(
    y_true,
    probabilities
):

    # --------------------------------------------------------
    # probabilities might theoretically become extremely
    # close to zero.
    #
    # log(0) is undefined and approaches -infinity.
    #
    # So we clip the probabilities slightly away from 0.
    # --------------------------------------------------------

    epsilon = 1e-15

    probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon
    )


    # --------------------------------------------------------
    # CCE asks:
    #
    # "How much probability did you assign to the class
    # that ACTUALLY happened?"
    #
    #
    # Example:
    #
    # Actual = Dog
    #
    # y_true:
    #
    # [0, 0, 1]
    #
    #
    # Model:
    #
    # [0.31, 0.13, 0.56]
    #
    #
    # Internally:
    #
    # -[
    #
    #   0 * log(.31)
    #
    #       +
    #
    #   0 * log(.13)
    #
    #       +
    #
    #   1 * log(.56)
    #
    # ]
    #
    #
    # The one-hot target acts like a SWITCH.
    #
    # Only the actual class survives.
    #
    # Therefore:
    #
    # loss = -log(.56)
    # --------------------------------------------------------

    sample_losses = -np.sum(
        y_true * np.log(probabilities),
        axis=1
    )


    # sample_losses now contains ONE loss per animal.
    #
    # Example:
    #
    # [
    #   0.4,
    #   0.7,
    #   0.2,
    #   ...
    # ]
    #
    #
    # We want one overall training objective.
    #
    # So take the average loss.

    average_loss = np.mean(sample_losses)

    return average_loss


# ============================================================
# 7. TRAINING SETTINGS
# ============================================================

learning_rate = 0.01

epochs = 10000


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

for epoch in range(epochs):

    # ========================================================
    # FORWARD PASS
    # ========================================================


    # --------------------------------------------------------
    # STEP 1:
    #
    # Calculate one logit PER CLASS for every sample.
    #
    #
    # X shape:
    #
    # (9, 2)
    #
    #
    # weights shape:
    #
    # (2, 3)
    #
    #
    # Matrix multiplication:
    #
    # (9,2) @ (2,3)
    #
    # gives:
    #
    # (9,3)
    #
    #
    # Therefore each animal receives:
    #
    # [Bird logit, Cat logit, Dog logit]
    #
    #
    # bias shape is:
    #
    # (3,)
    #
    # NumPy broadcasts it across every sample.

    logits = X @ weights + bias


    # --------------------------------------------------------
    # STEP 2:
    #
    # Convert logits into probabilities using Softmax.
    #
    #
    # Example:
    #
    # [1.2, 0.7, 1.8]
    #
    # becomes something like:
    #
    # [.31, .13, .56]

    probabilities = softmax(logits)


    # --------------------------------------------------------
    # STEP 3:
    #
    # Calculate how wrong the model currently is.
    #
    # CCE looks at the probability assigned to the
    # ACTUAL class.

    loss = categorical_cross_entropy(
        y_one_hot,
        probabilities
    )


    # ========================================================
    # BACKWARD PASS
    # ========================================================


    # --------------------------------------------------------
    # THIS IS THE BEAUTIFUL RESULT OF:
    #
    # SOFTMAX + CATEGORICAL CROSS-ENTROPY
    #
    #
    # After doing the calculus, the derivative of the loss
    # with respect to the logits simplifies to:
    #
    # probabilities - y_true
    #
    #
    # This is very similar to what happened with:
    #
    # Sigmoid + BCE
    #
    # where we eventually got:
    #
    # p - y
    #
    #
    # Example:
    #
    # Actual Dog:
    #
    # y_true:
    #
    # [0, 0, 1]
    #
    #
    # prediction:
    #
    # [.31, .13, .56]
    #
    #
    # error:
    #
    # [.31, .13, .56]
    # -
    # [0,   0,   1]
    #
    # =
    #
    # [.31, .13, -.44]
    #
    #
    # Interpretation:
    #
    # Bird:
    # +.31
    #
    # Model assigned too much probability to Bird.
    #
    #
    # Cat:
    # +.13
    #
    # Model assigned too much probability to Cat.
    #
    #
    # Dog:
    # -.44
    #
    # Model did not assign enough probability to the
    # actual Dog class.
    #
    #
    # Gradient Descent will use these errors to adjust
    # the underlying weights.

    error = probabilities - y_one_hot


    # --------------------------------------------------------
    # WEIGHT GRADIENT
    # --------------------------------------------------------
    #
    # In binary Logistic Regression we had:
    #
    # dw = X.T @ error / n_samples
    #
    #
    # SAME IDEA HERE.
    #
    #
    # But now:
    #
    # X.T shape:
    #
    # (2, 9)
    #
    #
    # error shape:
    #
    # (9, 3)
    #
    #
    # Therefore:
    #
    # (2,9) @ (9,3)
    #
    # =
    #
    # (2,3)
    #
    #
    # Exactly the same shape as our weight matrix.
    #
    #
    # The resulting matrix tells us how every feature
    # contributed to the errors for every class:
    #
    #
    #                 Bird     Cat      Dog
    #
    # weight_kg       grad     grad     grad
    #
    # height_cm       grad     grad     grad
    #
    #
    # So Gradient Descent can update ALL class-specific
    # weights simultaneously.

    dw = (
        X.T @ error
    ) / n_samples


    # --------------------------------------------------------
    # BIAS GRADIENT
    # --------------------------------------------------------
    #
    # Bias does not multiply X.
    #
    # So we only need the average error for each class.
    #
    #
    # error shape:
    #
    # (9,3)
    #
    #
    # axis=0 means:
    #
    # average vertically across all samples.
    #
    #
    # Result:
    #
    # [
    #   Bird bias gradient,
    #   Cat bias gradient,
    #   Dog bias gradient
    # ]

    db = np.mean(
        error,
        axis=0
    )


    # ========================================================
    # GRADIENT DESCENT UPDATE
    # ========================================================
    #
    # Remember:
    #
    # Gradient tells us the direction in which loss
    # INCREASES.
    #
    # We want loss to DECREASE.
    #
    # Therefore we move in the opposite direction.
    #
    #
    # new weight =
    #
    # old weight - learning_rate * gradient


    weights = (
        weights
        - learning_rate * dw
    )


    bias = (
        bias
        - learning_rate * db
    )


    # --------------------------------------------------------
    # PRINT TRAINING PROGRESS
    # --------------------------------------------------------

    if epoch % 1000 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"Loss: {loss:.6f}"
        )


# ============================================================
# 9. TRAINING COMPLETE
# ============================================================

print("\nTraining completed.")


print("\nLearned weights:")
print(weights)


print("\nLearned biases:")
print(bias)


# ============================================================
# 10. MAKE PREDICTIONS ON TRAINING DATA
# ============================================================


# Calculate logits using the FINAL learned weights.

logits = X @ weights + bias


# Convert logits into probabilities.

probabilities = softmax(logits)


# ------------------------------------------------------------
# ARGMAX
# ------------------------------------------------------------
#
# Each row contains:
#
# [P(Bird), P(Cat), P(Dog)]
#
#
# Example:
#
# [.02, .91, .07]
#
#
# np.argmax(..., axis=1)
#
# returns the INDEX of the largest probability
# for every sample.
#
#
# Here:
#
# index 1
#
# means:
#
# Cat

predictions = np.argmax(
    probabilities,
    axis=1
)


print("\nPredictions on training data:\n")


for i in range(n_samples):

    actual_index = y[i]

    predicted_index = predictions[i]


    actual_name = class_names[
        actual_index
    ]

    predicted_name = class_names[
        predicted_index
    ]


    print(
        f"Animal {i + 1}"
    )

    print(
        f"Features: "
        f"weight={X[i, 0]} kg, "
        f"height={X[i, 1]} cm"
    )

    print(
        f"Probabilities: "
        f"Bird={probabilities[i, 0]:.4f}, "
        f"Cat={probabilities[i, 1]:.4f}, "
        f"Dog={probabilities[i, 2]:.4f}"
    )

    print(
        f"Actual: {actual_name}"
    )

    print(
        f"Predicted: {predicted_name}"
    )

    print("-" * 50)


# ============================================================
# 11. CALCULATE SIMPLE TRAINING ACCURACY
# ============================================================

accuracy = np.mean(
    predictions == y
)


print(
    f"\nTraining Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 12. INFERENCE ON COMPLETELY NEW ANIMALS
# ============================================================
#
# These animals were NOT part of training.
#
# The model only receives their features.
#
# It does NOT know their y.
#
# This is exactly the distinction we discussed earlier:
#
# During TRAINING:
#
# X + actual y
#     |
#     v
# calculate loss
#     |
#     v
# update weights
#
#
# During INFERENCE:
#
# X only
# |
# v
# logits
# |
# v
# Softmax
# |
# v
# argmax
# |
# v
# prediction
#
#
# There is NO CCE during normal inference because we
# don't know the actual answer yet.


new_animals = np.array([
    [0.7, 11],     # probably Bird-like
    [4.2, 26],     # probably Cat-like
    [10.0, 43]     # probably Dog-like
], dtype=float)


# Calculate logits using the weights learned during training.

new_logits = (
    new_animals @ weights
    + bias
)


# Convert logits into probabilities.

new_probabilities = softmax(
    new_logits
)


# Pick the highest-probability class.

new_predictions = np.argmax(
    new_probabilities,
    axis=1
)


print("\nPredictions for NEW animals:\n")


for i in range(len(new_animals)):

    predicted_index = new_predictions[i]

    predicted_name = class_names[
        predicted_index
    ]


    print(
        f"New Animal {i + 1}"
    )

    print(
        f"Features: "
        f"weight={new_animals[i, 0]} kg, "
        f"height={new_animals[i, 1]} cm"
    )

    print(
        f"Bird probability: "
        f"{new_probabilities[i, 0]:.4f}"
    )

    print(
        f"Cat probability: "
        f"{new_probabilities[i, 1]:.4f}"
    )

    print(
        f"Dog probability: "
        f"{new_probabilities[i, 2]:.4f}"
    )

    print(
        f"Predicted class: "
        f"{predicted_name}"
    )

    print("-" * 50)