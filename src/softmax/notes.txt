# Multiclass Logistic Regression — Softmax & Categorical Cross-Entropy

## 1. Why Multiclass Logistic Regression?

Binary Logistic Regression handles two possible target classes.

Example:

- Churn
- No Churn

The model calculates a linear score (logit) and passes it through Sigmoid to get a probability.

Multiclass classification is different because the target can have more than two mutually exclusive classes.

Example:

- Cat
- Bird
- Dog

Only ONE of these can be the correct answer for a particular sample.

---

## 2. Class Index vs One-Hot Encoding

Assume sklearn or our code stores the classes in this order:

| Index | Class |
|------:|-------|
| 0 | Cat |
| 1 | Bird |
| 2 | Dog |

If the actual animal is Dog:

Actual class index:

y = 2

This DOES NOT mean One-Hot Encoding gives Dog the value 2.

The number 2 is only the position/index representing Dog.

One-Hot Encoding converts it into:

Cat   Bird   Dog
 0      0      1

So:

y = 2

becomes:

y_one_hot = [0, 0, 1]

The `1` identifies the actual class.

---

## 3. Multiclass Logistic Regression Has One Score Per Class

In binary Logistic Regression, we calculate a linear score:

z = wx + b

In multiclass Logistic Regression, every class gets its own weights and bias.

For example:

Input X
   |
   |---- Cat model  ----> Cat logit
   |
   |---- Bird model ----> Bird logit
   |
   |---- Dog model  ----> Dog logit

Suppose the model calculates:

Cat  = 1.2
Bird = 0.7
Dog  = 1.8

So:

logits = [1.2, 0.7, 1.8]

These numbers are NOT probabilities.

A logit can be any value from negative infinity to positive infinity.

We therefore need something that converts these competing scores into probabilities.

That is Softmax.

---

# 4. Softmax

Softmax takes all class logits together and converts them into a probability distribution.

Example:

logits:

[1.2, 0.7, 1.8]

After Softmax:

Cat  = 0.31
Bird = 0.13
Dog  = 0.56

The probabilities sum to approximately:

0.31 + 0.13 + 0.56 = 1

This is required because these classes are mutually exclusive.

The animal cannot simultaneously be:

60% Dog
70% Cat
50% Bird

as independent events.

Instead, the classes compete for the same total probability mass.

---

## 5. How Softmax Works Internally

Softmax performs two important operations.

### Step 1: Exponentiate Every Logit

Exponentiation converts every logit into a positive number.

This is important because logits can be negative, but probabilities cannot.

Conceptually:

logits

[Cat, Bird, Dog]

        |
        v

exponentiate every value

        |
        v

positive relative strengths

Exponentiation also preserves ordering.

If:

Dog logit > Cat logit

then:

Dog's exponentiated score > Cat's exponentiated score

So the model's preference is preserved.

---

### Step 2: Normalize the Scores

After exponentiation, Softmax adds all the positive scores together.

Each class then receives:

its own strength / total strength

This turns each class score into its share of the total.

Conceptually:

Cat strength
----------------
Total strength

Bird strength
----------------
Total strength

Dog strength
----------------
Total strength

Because every class is divided by the same total, all resulting probabilities sum to 1.

The only Softmax formula worth remembering is:

P(class i) = e^(class logit) / sum of e^(all logits)

In plain English:

"Exponentiate this class's score and divide it by the exponentiated scores of all competing classes."

---

# 6. Why Exponentials?

Exponentials are not randomly chosen.

They give us several useful properties:

1. They convert any logit into a positive value.
2. They preserve ordering.
3. Larger differences between logits create stronger differences between probabilities.
4. They naturally connect to the log-odds idea used in Logistic Regression.

A very important property is:

Softmax cares about RELATIVE differences between logits.

For example:

[3, 2, 0]

and:

[103, 102, 100]

produce the same Softmax probability distribution.

Why?

Because we added 100 to every class.

Nobody gained an advantage relative to another class.

This is also why implementations usually subtract the largest logit before exponentiation.

Example:

[1000, 1001, 999]

becomes:

[-1, 0, -2]

The relative differences remain identical, but we avoid extremely large exponentials and numerical overflow.

---

# 7. Relationship Between Sigmoid and Softmax

Binary Logistic Regression:

Input
  |
  v
Linear score
  |
  v
Sigmoid
  |
  v
P(class 1)

The other probability is:

P(class 0) = 1 - P(class 1)

So even binary Logistic Regression represents probabilities that sum to 1.

Multiclass Logistic Regression:

Input
  |
  v
One linear score per class
  |
  v
Multiple logits
  |
  v
Softmax
  |
  v
One probability per class

Softmax can therefore be thought of as the multiclass generalization of the same probability-modeling idea.

With only two competing classes, Softmax can be reduced mathematically to a Sigmoid based on the difference between the two logits.

---

# 8. Making the Final Prediction — Argmax

Suppose Softmax produces:

Cat  = 0.31
Bird = 0.13
Dog  = 0.56

The model needs to choose one final class.

`argmax` returns the INDEX containing the largest value.

Probabilities:

[0.31, 0.13, 0.56]

Indexes:

   0     1     2

Largest probability is at index:

2

And index 2 represents:

Dog

Therefore:

Predicted class = Dog

---

# 9. Argmax Is NOT the Loss Function

This distinction is extremely important.

Argmax answers:

"What class would the model predict?"

Categorical Cross-Entropy answers:

"How good was the probability assigned to the class that actually happened?"

Suppose:

Actual = Dog

Prediction:

Cat  = 0.31
Bird = 0.13
Dog  = 0.56

Argmax says:

Dog

So the classification is correct.

But the model only gave the actual class 56% probability.

That is not a perfect prediction.

Training should still encourage the model to become better.

This is why we need a loss function.

---

# 10. Categorical Cross-Entropy (CCE)

Categorical Cross-Entropy is the multiclass equivalent of Binary Cross-Entropy.

Its main question is extremely simple:

"How much probability did the model assign to the class that actually happened?"

Suppose:

Actual = Dog

One-hot representation:

[0, 0, 1]

Softmax prediction:

[0.31, 0.13, 0.56]

The one-hot encoded target acts like a switch.

Cat:

0 -> ignore it

Bird:

0 -> ignore it

Dog:

1 -> this is the actual class

Therefore Categorical Cross-Entropy only cares about:

Dog probability = 0.56

The essential loss is:

Loss = -log(probability of actual class)

So here:

Loss = -log(0.56)

The important concept is not the formula.

The important concept is:

CCE punishes the model based on how little probability it assigned to what actually happened.

---

# 11. Correct Classification Can Still Have Loss

Consider:

Actual = Dog

Prediction A:

Cat  = 0.33
Bird = 0.33
Dog  = 0.34

Argmax still predicts Dog.

Technically correct.

But the model is extremely uncertain.

Now consider:

Prediction B:

Cat  = 0.005
Bird = 0.005
Dog  = 0.990

Argmax again predicts Dog.

Both predictions are classified correctly.

But Prediction B is much better.

Categorical Cross-Entropy captures this difference.

For the actual class:

High probability -> low loss

Low probability -> high loss

Extremely low probability -> extremely high loss

Therefore, confidently assigning a tiny probability to the actual class is punished heavily.

---

# 12. One-Hot Encoding Acts as a Switch

Suppose:

Classes:

0 = Cat
1 = Bird
2 = Dog

Actual:

Dog

One-hot target:

[0, 0, 1]

Predictions:

[0.31, 0.13, 0.56]

Categorical Cross-Entropy conceptually examines all classes:

Cat  -> target is 0 -> ignore
Bird -> target is 0 -> ignore
Dog  -> target is 1 -> keep

Therefore the loss effectively becomes:

-log(Dog probability)

This is very similar to how `y` and `1-y` acted as switches in Binary Cross-Entropy.

---

# 13. Dataset-Level Loss

For one sample:

Actual = Dog
P(Dog) = 0.56

CCE calculates that sample's loss.

For another sample:

Actual = Cat
P(Cat) = 0.80

CCE calculates that sample's loss.

For another:

Actual = Bird
P(Bird) = 0.90

CCE calculates that sample's loss.

During training, we usually combine/average these losses across the samples.

Gradient Descent then tries to minimize the overall loss.

---

# 14. Complete Training Flow

Suppose the actual animal is Dog.

Class mapping:

0 = Cat
1 = Bird
2 = Dog

Actual class:

Dog -> index 2

One-hot target:

[0, 0, 1]

Now the model receives input features.

                    INPUT X
                       |
                       v
             One linear model
                per class
                       |
                       v
                    LOGITS

             [1.2, 0.7, 1.8]

                       |
                       v
                    SOFTMAX

             [0.31, 0.13, 0.56]

                       |
             +---------+---------+
             |                   |
             v                   v
          ARGMAX                CCE
             |                   |
             v                   v
         index = 2        Compare against
             |             [0, 0, 1]
             v                   |
            DOG                  v
                         Calculate loss
                                 |
                                 v
                         Gradient Descent
                                 |
                                 v
                        Update weights/bias
                                 |
                                 v
                          Next iteration

Argmax is used to determine the predicted class.

Softmax probabilities + actual target are used to calculate the loss.

Gradient Descent uses the loss/gradients to improve the model.

---

# 15. Important: Argmax Is Not Required for Training

This is an important mental distinction.

Training does NOT need to do:

Softmax -> Argmax -> CCE

Instead:

Softmax
   |
   +------> CCE -> Gradient Descent
   |
   +------> Argmax -> Predicted class

CCE needs the probability distribution, not just the final class.

If we converted:

[0.31, 0.13, 0.56]

into only:

Dog

we would lose information about how confident/uncertain the model was.

For training, that probability information is extremely valuable.

---

# 16. Binary vs Multiclass Logistic Regression

## Binary Logistic Regression

Input
  |
  v
Linear score
  |
  v
Sigmoid
  |
  v
Probability
  |
  v
Binary Cross-Entropy
  |
  v
Gradient Descent
  |
  v
Update weights

For final classification:

Probability -> Threshold -> Class


## Multiclass Logistic Regression

Input
  |
  v
One linear score per class
  |
  v
Multiple logits
  |
  v
Softmax
  |
  v
Probability distribution
  |
  v
Categorical Cross-Entropy
  |
  v
Gradient Descent
  |
  v
Update weights

For final classification:

Probabilities -> Argmax -> Class

---

# 17. Softmax vs Independent Sigmoids

Softmax is appropriate when classes are mutually exclusive.

Example:

Cat OR Dog OR Bird

Only one answer can be correct.

Therefore the probabilities compete and sum to 1.

Independent Sigmoids are appropriate for multi-label problems where multiple labels can simultaneously be true.

Example:

An image could contain:

Dog = Yes
Outdoor = Yes
Daytime = Yes

These are not competing classes.

Therefore their probabilities do NOT need to sum to 1.

This distinction is:

MULTICLASS
One correct class
-> Softmax

MULTI-LABEL
Multiple labels can simultaneously be true
-> Independent Sigmoids

---

# Final Mental Model

Multiclass Logistic Regression does not directly predict "Dog".

It first calculates how strongly each class matches the input.

Input
  |
  v
Class-specific linear scores
  |
  v
Logits
  |
  v
Softmax
  |
  v
Probability distribution

Example:

Cat  = 31%
Bird = 13%
Dog  = 56%

Two different things can now happen:

1. Argmax selects Dog as the final prediction.
2. Categorical Cross-Entropy checks how much probability was assigned to the ACTUAL class.

If the actual class is Dog, 56% is correct but not perfect.

The loss therefore tells Gradient Descent that the model can still improve.

Gradient Descent updates the weights and biases so that, across the training data, the model learns to assign higher probability to the correct classes.

The core idea is:

Softmax = "How should probability be distributed among competing classes?"

Categorical Cross-Entropy = "How much probability did you give to what actually happened?"

Argmax = "Which class finally wins?"