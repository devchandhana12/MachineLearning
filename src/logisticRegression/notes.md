LOGISTIC REGRESSION — COMPLETE CONCEPTUAL NOTES

1. WHY DO WE NEED LOGISTIC REGRESSION?

Linear Regression is useful when the target is a continuous numerical value, such as:

- House price
- Sales
- Temperature

But some problems are classification problems.

Examples:

- Student passes or fails
- Customer buys a car or does not buy
- Transaction is fraud or not fraud

In Binary Classification, we have two classes:

Class 0
Class 1

Linear Regression is not suitable for this because its output is unbounded. It can produce values from negative infinity to positive infinity.

For classification, it is more useful to estimate the probability of belonging to a class.

Probability is bounded between 0 and 1.

So we need a way to convert an unbounded linear score into a probability between 0 and 1.


2. FROM LINEAR SCORE TO PROBABILITY

Logistic Regression still starts with a linear score:

z = wx + b

The problem is:

z can range from negative infinity to positive infinity.

Probability can only range from 0 to 1.

To connect these two ranges, we first think about odds.

Odds of an event are:

Probability of the event / Probability of the event not happening

For binary classification, if:

p = probability of Class 1

then:

1 - p = probability of Class 0

Therefore:

Odds = p / (1 - p)

Example:

If p = 0.8,

Odds = 0.8 / 0.2 = 4

This means the odds are 4:1 in favor of Class 1.

Odds range from 0 to positive infinity.

But our linear score z ranges from negative infinity to positive infinity.

So we take the logarithm of the odds.

This gives us log-odds.

Log-odds can range from negative infinity to positive infinity, which matches the range of our linear score.

Therefore Logistic Regression models the log-odds as a linear combination of the features.

If we reverse the log-odds transformation, we get the Sigmoid function.

The Sigmoid function converts the linear score z into a probability between 0 and 1.

Therefore the prediction pipeline becomes:

Features
→ Linear score
→ Sigmoid
→ Probability


3. MAXIMUM LIKELIHOOD ESTIMATION

Now the model can generate probabilities.

But we still need to learn the correct weights and bias.

The question becomes:

"Which values of the weights and bias make the labels we actually observed in the training data most probable?"

This is the idea behind Maximum Likelihood Estimation (MLE).

For every training observation, we look at the probability the model assigned to the actual class.

If the actual class is 1, we use the probability of Class 1.

If the actual class is 0, we use the probability of Class 0.

The likelihood of the complete training dataset is obtained by multiplying these probabilities together.

Our objective is to find the weights and bias that maximize this likelihood.

However, multiplying thousands or millions of probabilities produces extremely small numbers and is computationally inconvenient.

Therefore we take the logarithm.

Logarithms convert multiplication into addition.

Instead of maximizing likelihood, we can maximize log-likelihood.

Because the logarithm is monotonic, maximizing likelihood and maximizing log-likelihood produce the same optimal parameters.


4. FROM LOG-LIKELIHOOD TO BCE

Maximum Likelihood gives us a maximization problem.

However, our optimization framework is usually expressed as minimizing a loss.

Therefore:

Maximize Log-Likelihood

is converted into:

Minimize Negative Log-Likelihood

For binary classification, this gives us Binary Cross Entropy (BCE).

BCE measures how bad the model's predicted probabilities are compared with the actual binary labels.

For an actual Class 1 observation, we care about the probability p.

For an actual Class 0 observation, we care about the probability 1-p.

The actual label y acts like a switch that allows one mathematical expression to handle both cases.

So BCE gives the model a numerical measure of how wrong its predictions are.


5. GRADIENT DESCENT

BCE tells us how wrong the model is.

But knowing the loss alone is not enough.

We also need to know:

"If I change a weight slightly, how will the final loss change?"

The dependency is:

Weight
→ Linear score
→ Probability
→ BCE Loss

The weight affects the linear score.

The linear score affects the probability through Sigmoid.

The probability affects BCE.

Using derivatives and the chain rule, we can trace how changing a weight ultimately changes the loss.

The gradient tells us the direction in which the loss increases the fastest.

Because our goal is to decrease the loss, Gradient Descent moves the parameters in the opposite direction of the gradient.

The learning rate determines how large each update should be.

Training therefore repeatedly performs:

1. Calculate the linear score.
2. Convert it into probabilities using Sigmoid.
3. Calculate BCE.
4. Calculate gradients.
5. Update weights and bias.
6. Repeat.

Eventually, we obtain parameters that produce a lower loss on the training data.


6. DECISION THRESHOLD

Logistic Regression naturally outputs probabilities.

For example:

Probability of Class 1 = 0.73

But sometimes our application requires a final class prediction.

For that, we use a decision threshold.

A common default threshold is 0.5.

If probability >= 0.5:
Predict Class 1.

If probability < 0.5:
Predict Class 0.

However, 0.5 is not a universal rule.

The threshold can be changed depending on the problem.

Lowering the threshold usually predicts more observations as positive.

This generally increases Recall but can also increase False Positives and reduce Precision.

Increasing the threshold makes it harder for an observation to be classified as positive.

Therefore threshold selection should depend on the business cost of False Positives and False Negatives.


7. CONFUSION MATRIX

Classification models can produce four possible outcomes.

True Positive:
Model predicted positive and the actual class was positive.

True Negative:
Model predicted negative and the actual class was negative.

False Positive:
Model predicted positive but the actual class was negative.

False Negative:
Model predicted negative but the actual class was positive.

Easy naming rule:

Positive/Negative = what the model predicted.

True/False = whether that prediction was correct.


8. ACCURACY

Accuracy asks:

"Out of all predictions, how many did the model predict correctly?"

Accuracy can be useful when the classes are reasonably balanced and the costs of different mistakes are similar.

However, accuracy can become misleading with class imbalance.

Example:

1000 transactions:

990 legitimate
10 fraud

Suppose the model predicts every transaction as legitimate.

Accuracy = 99%.

That sounds excellent.

But the model caught:

0 out of 10 fraud cases.

Therefore it is completely useless for detecting fraud despite having 99% accuracy.


9. PRECISION

Precision asks:

"Out of everything the model predicted as positive, how many were actually positive?"

Precision is especially important when False Positives are expensive.

High Precision means:

When the model says "positive", it is usually correct.


10. RECALL

Recall asks:

"Out of all the actual positive cases that existed, how many did the model successfully catch?"

Recall is especially important when False Negatives are expensive.

For example, in disease screening or fraud detection, missing a real positive case can be costly.


11. F1 SCORE

Precision and Recall can trade off against each other.

F1 Score combines Precision and Recall into one metric.

A high F1 Score generally requires both Precision and Recall to be reasonably good.

It is useful when we care about both False Positives and False Negatives, especially with imbalanced datasets.


12. REGULARIZATION

If we optimize only for training BCE, the model may sometimes learn unnecessarily extreme coefficients.

Extreme coefficients can make the model overly confident and sensitive to small changes in the features.

This can contribute to poor generalization.

Regularization changes the objective from:

"Fit the training data as well as possible"

to:

"Fit the training data well without using unnecessarily extreme coefficients."

Therefore the training objective becomes:

BCE + Regularization Penalty

L1 regularization penalizes the absolute magnitude of the coefficients.

L1 can push some coefficients exactly to zero, which can produce sparse models.

L2 regularization penalizes squared coefficient magnitude.

L2 generally shrinks coefficients toward zero without necessarily making them exactly zero.

Regularization strength controls the trade-off between fitting the training data and keeping the model complexity under control.

Important:

A weight is NOT the feature value.

If a feature is "hours studied", a weight of 100 does NOT mean someone studied for 100 hours.

The weight represents how strongly that feature affects the model's log-odds.


13. ASSUMPTION — LINEARITY IN LOG-ODDS

Logistic Regression does NOT assume that probability itself has a linear relationship with the features.

It assumes that the log-odds can be represented as a linear combination of the supplied features.

The probability can still change nonlinearly because the Sigmoid function converts the linear score into probability.

If the true relationship is highly nonlinear and cannot be represented by the supplied features, vanilla Logistic Regression may struggle unless we engineer nonlinear or interaction features.


14. ASSUMPTION — INDEPENDENT OBSERVATIONS

Logistic Regression generally assumes that observations are conditionally independent given the predictors/model.

One row should not simply be treated as completely independent evidence when it is strongly dependent on another row.

This connects directly to Maximum Likelihood because we constructed the dataset likelihood by multiplying the probabilities associated with the observations.


15. MULTICOLLINEARITY

Multicollinearity occurs when multiple features contain highly overlapping information.

For example:

Feature 1 = Salary in rupees
Feature 2 = Salary in lakhs

Both essentially describe the same thing.

When highly correlated predictors contain the same information, the model can struggle to determine how much individual effect belongs to each coefficient.

Predictions may still be reasonable, but individual coefficients can become unstable and difficult to interpret.

Regularization, especially L2, can help stabilize the coefficients.


16. ENOUGH DATA / ENOUGH CLASS EXAMPLES

Logistic Regression needs enough observations to estimate its coefficients reliably.

More importantly, it needs enough useful examples of the different outcomes.

The classes do NOT need to be exactly 50/50.

However, if we have thousands of Class 0 examples and only a tiny number of Class 1 examples, the model may have insufficient evidence to learn the patterns associated with Class 1.


17. NORMAL DISTRIBUTION IS NOT REQUIRED

Logistic Regression does not require input features to follow a normal distribution.

Features can be skewed, binary, continuous, categorical after encoding, etc.

It also does not require normally distributed residuals or homoscedasticity in the same way ordinary Linear Regression does.


18. LIMITATION — LINEAR DECISION BOUNDARY

Vanilla Logistic Regression is fundamentally a linear classifier.

Even though Sigmoid is nonlinear, the underlying decision boundary comes from the linear score.

Therefore Logistic Regression may struggle with complicated nonlinear class boundaries.

Feature engineering, polynomial features, and interaction terms can sometimes help.

Otherwise, nonlinear models may be more appropriate.


19. LIMITATION — INFLUENTIAL OBSERVATIONS

Extreme or unusual observations can sometimes have a strong influence on the fitted coefficients and decision boundary.

However:

Large feature value does NOT automatically mean large weight.

The real concern is whether an observation is influential enough to substantially change the fitted model.


20. LIMITATION — CLASS IMBALANCE

With severe class imbalance, Logistic Regression may perform poorly on the minority class even while showing high overall accuracy.

In such cases, we may need:

- Better evaluation metrics such as Precision, Recall and F1
- Threshold tuning
- Class weighting
- Resampling techniques

The correct approach depends on the actual problem and the cost of different mistakes.


21. LIMITATION — PERFECT SEPARATION

Suppose the training data can be separated perfectly.

For example:

Every student studying below 5 hours fails.

Every student studying above 5 hours passes.

There is absolutely no overlap between the classes.

The model may already correctly predict:

Fail cases with very low probability of passing.

Pass cases with very high probability of passing.

But BCE can still become slightly smaller if the model becomes even more confident.

So the model can keep pushing:

0.90 → 0.99 → 0.999 → 0.9999

and:

0.10 → 0.01 → 0.001 → 0.0001

To create increasingly extreme probabilities, the model can keep increasing the magnitude of its coefficients.

With perfectly separable data, an unregularized Logistic Regression maximum-likelihood solution may therefore fail to have finite coefficients.

This is called Complete Separation.

Regularization helps because increasing coefficient magnitude now carries an additional penalty.

The model must balance:

Reducing BCE

against

Keeping coefficients from becoming unnecessarily extreme.


FINAL MENTAL MODEL

Logistic Regression starts with features.

The features and learned coefficients produce a linear score.

Sigmoid converts that score into a probability.

Maximum Likelihood gives us the principle for choosing good parameters.

Negative Log-Likelihood / BCE gives us a loss to minimize.

Gradient Descent adjusts the parameters to reduce that loss.

The trained model outputs probabilities.

A decision threshold converts those probabilities into classes.

Confusion Matrix, Accuracy, Precision, Recall and F1 help us evaluate those classifications.

Regularization helps control unnecessarily extreme coefficients and improve stability/generalization.

Assumptions and limitations tell us when Logistic Regression is appropriate and when another model or additional feature engineering may be required.