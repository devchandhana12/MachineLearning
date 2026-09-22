# Decision Trees — Classification

## 1. Why Decision Trees?

Logistic Regression learns a linear decision boundary in the supplied feature space.

It works well when the relationship can be represented approximately as:

z = w₁x₁ + w₂x₂ + ... + b

But real-world relationships may be nonlinear or rule-like.

Example:

- If distance is low → customer may buy.
- If distance is high but price is low → customer may still buy.
- If distance is high and price is high → customer may not buy.

A Decision Tree handles this by learning a hierarchy of feature-based questions.

Example:

                    Distance <= 5?
                    /            \
                  Yes             No
                  /                \
             Hungry?             Price <= 200?
              /  \                 /    \
            Yes   No             Yes     No


## 2. What Does a Decision Tree Learn?

Logistic Regression learns parameters:

- weights (w)
- bias (b)

A Decision Tree learns a structure:

- which feature to split on
- what threshold/category to use
- where each split appears
- when splitting should stop
- what each leaf predicts

We do NOT manually write the rules.

The algorithm discovers them from the training data.


# 3. Training Starts at the Root

Suppose our target is:

Bought Biryani = Yes / No

At the beginning, all training samples are inside one node:

                    ROOT

                  6 Yes
                  4 No

The tree now needs to find a question that separates the classes better.

Possible questions:

Distance <= 4.5?
Price <= 250?
Hungry == Yes?

The main training problem is:

"Which question produces the best separation?"


# 4. Generating Questions for Numerical Features

Suppose Distance values in the training data are:

1, 2, 3, 5, 7, 9

There are infinitely many possible real-number thresholds.

But the tree does NOT need to test all of them.

First, sort the observed values:

1   2   3   5   7   9

Consider thresholds between adjacent observed values:

1.5
2.5
4
6
8

Candidate questions become:

Distance <= 1.5?
Distance <= 2.5?
Distance <= 4?
Distance <= 6?
Distance <= 8?


## Why Don't We Test Infinite Thresholds?

Consider:

Distance <= 3.1
Distance <= 3.5
Distance <= 3.9

If the observed values are:

1, 2, 3, 5, 7, 9

all three thresholds produce exactly the same partition:

LEFT:
1, 2, 3

RIGHT:
5, 7, 9

Therefore, testing all those thresholds would be redundant.

We only need meaningful boundaries between observed values.


## Do We Need Both <= 4 and > 4?

No.

Distance <= 4 automatically creates two branches:

TRUE  → Distance <= 4
FALSE → Distance > 4

Testing "Distance > 4" separately would produce the same partition with
the branches reversed.


## What About Thresholds Like -4 or 5000?

If all observed distances are positive:

Distance <= -4

would produce:

LEFT  = empty
RIGHT = everyone

Similarly:

Distance <= 5000

might produce:

LEFT  = everyone
RIGHT = empty

These do not meaningfully partition the training samples.


# 5. What Makes a Split Good?

Suppose we have:

                    ROOT

                  3 Yes
                  3 No

Consider two questions.

Split A:

LEFT                RIGHT

3 Yes               0 Yes
0 No                3 No

This produces very pure groups.

Split B:

LEFT                RIGHT

2 Yes               1 Yes
1 No                2 No

These groups are still mixed.

Humans can visually recognize that Split A is better.

A computer needs a numerical measure.

This creates the need for an impurity metric.


# 6. Gini Impurity

Gini impurity measures how mixed the class labels are inside ONE node.

Formula:

Gini = 1 - Σ(pᵢ²)

where pᵢ is the proportion of class i inside the node.


## Pure Node

10 Yes
0 No

p(Yes) = 1
p(No)  = 0

Gini = 1 - (1² + 0²)
      = 0

A pure node has Gini = 0.


## Completely Mixed Binary Node

5 Yes
5 No

p(Yes) = 0.5
p(No)  = 0.5

Gini = 1 - (0.5² + 0.5²)
      = 0.5

For binary classification, 50/50 gives the maximum Gini of 0.5.


## Example

8 Yes
2 No

p(Yes) = 0.8
p(No)  = 0.2

Gini = 1 - (0.8² + 0.2²)
      = 1 - (0.64 + 0.04)
      = 0.32


Therefore:

10Y / 0N → Gini = 0
9Y  / 1N → Gini = 0.18
8Y  / 2N → Gini = 0.32
5Y  / 5N → Gini = 0.50

Lower Gini = purer node.
Higher Gini = more mixed node.


## Important

Gini measures ONE NODE.

A split creates multiple child nodes, so each child gets its own Gini.


# 7. Evaluating a Candidate Split

Suppose the parent contains:

6 Yes
4 No

Parent Gini:

p(Yes) = 0.6
p(No)  = 0.4

Gini(parent)
= 1 - (0.6² + 0.4²)
= 0.48


Suppose we test:

Distance <= 5.5?

and obtain:

LEFT:

4 Yes
1 No

Gini(left) = 0.32


RIGHT:

2 Yes
3 No

Gini(right) = 0.48


We now have two child Ginis.

We need ONE score representing the entire split.


# 8. Weighted Child Impurity

We cannot simply average child Ginis because the children may contain
different numbers of samples.

Example:

LEFT:
1 customer
Gini = 0

RIGHT:
99 customers
Gini = 0.5

A simple average would give:

(0 + 0.5) / 2 = 0.25

This gives the 1-customer node the same importance as the 99-customer node,
which is misleading.

Therefore, child impurity is weighted by the number of samples.

Weighted Child Impurity:

(N_left / N) × Gini_left
+
(N_right / N) × Gini_right


Example:

Parent = 10 customers

LEFT:
5 customers
Gini = 0.32

RIGHT:
5 customers
Gini = 0.48

Weighted impurity:

(5/10)(0.32) + (5/10)(0.48)

= 0.16 + 0.24

= 0.40


# 9. Impurity Reduction

Before the split:

Parent Gini = 0.48

After the split:

Weighted Child Impurity = 0.40

Therefore:

Impurity Reduction
= Parent Impurity - Weighted Child Impurity

= 0.48 - 0.40

= 0.08


A candidate that produces a larger impurity reduction is preferred.


# 10. How the Best Split Is Selected

At a node, the tree conceptually evaluates candidate questions such as:

Distance <= 1.5
Distance <= 2.5
Distance <= 3.5
...

Price <= 175
Price <= 225
Price <= 275
...

Hungry == Yes

For every candidate:

1. Partition the training rows.
2. Calculate Gini for the left child.
3. Calculate Gini for the right child.
4. Calculate weighted child impurity.
5. Calculate impurity reduction.

Conceptually:

Candidate Question
       ↓
Split Rows
       ↓
Child Ginis
       ↓
Weighted Child Impurity
       ↓
Parent Impurity - Child Impurity
       ↓
Impurity Reduction

The candidate with the best impurity reduction is selected.

Only then does the tree commit to that split.


# 11. Recursive Tree Growth

Suppose the root chooses:

                    Distance <= 5.5?
                    /              \
                 LEFT              RIGHT

                4Y / 1N           2Y / 3N

Neither child is pure.

The root's job is now finished.

The LEFT child becomes its own smaller Decision Tree problem.

Using only the rows that reached LEFT:

1. Calculate current impurity.
2. Generate candidate questions.
3. Evaluate their impurity reductions.
4. Choose the best split.
5. Create children.

The RIGHT child independently does the same thing.

Therefore, Decision Tree training is recursive.


Conceptually:

BUILD_TREE(rows):

    inspect current node

    if stopping condition:
        create leaf

    otherwise:
        generate candidate splits
        evaluate candidate splits
        choose best split

        BUILD_TREE(left_rows)
        BUILD_TREE(right_rows)


# 12. Features Can Be Reused

Using Distance at the root does NOT mean Distance is unavailable later.

Example:

                    Distance <= 8?
                    /
              Distance <= 3?

The same numerical feature can be used multiple times.

Each split isolates a different region of the feature space.

When processing a child node, candidate thresholds are generated using
the samples that reached that node.


# 13. Trees Do Not Need Balanced Levels

Decision Trees should be thought of node-by-node, not level-by-level.

Example:

                         ROOT
                       /      \
                    LEAF      split
                              /   \
                           LEAF   split
                                  /   \
                               LEAF   LEAF

One branch may become a leaf early while another continues growing.

All leaves do NOT need to have the same depth.


# 14. Leaf Prediction

A leaf does not have to be pure.

Suppose training stops at:

8 Yes
2 No

The predicted class is normally the majority class:

Predict → YES

Estimated probabilities:

P(Yes) = 8/10 = 0.8
P(No)  = 2/10 = 0.2

This is the basic idea behind a classification tree's predict_proba().

Logistic Regression probability:

w·x + b
   ↓
sigmoid
   ↓
probability

Decision Tree probability:

new sample
   ↓
traverse tree
   ↓
reach leaf
   ↓
class distribution of training samples in that leaf
   ↓
probability


# 15. Why Decision Trees Overfit

A tree can continue splitting as long as it finds ways to reduce training
impurity.

Eventually it may create highly specific rules such as:

Age <= 31.5
    ↓
Distance > 3.72
    ↓
Price <= 287
    ↓
Age > 27.5
    ↓
1 training customer

That leaf may have:

1 Yes
0 No

Gini = 0

Training fit improved.

But the tree may simply have memorized one unusual training example.

The real goal is NOT:

"Make training impurity as small as possible."

The real goal is:

"Generalize well to unseen data."


As tree complexity increases:

Training error generally decreases.

Validation performance may initially improve, then worsen.

This is overfitting.


# 16. Pre-Pruning

Pre-pruning controls tree growth DURING training.

Instead of allowing the tree to grow freely, we impose stopping rules.


## max_depth

Limits how deep the tree may grow.

Example:

max_depth = 3

means no path may grow beyond depth 3.

It is a maximum, not a required depth.

A branch can stop earlier if appropriate.


## min_samples_split

Minimum number of samples required in a node before the algorithm is
allowed to attempt another split.

Example:

min_samples_split = 10

Node has 50 samples → may attempt splitting.
Node has 6 samples  → cannot split.


## min_samples_leaf

Controls the minimum number of samples allowed in resulting leaves.

Suppose a node has 100 samples.

A candidate split creates:

LEFT  = 1 sample
RIGHT = 99 samples

Even though the parent was large enough to split, this can create a
tiny region that memorizes training data.

min_samples_leaf prevents such splits.


Difference:

min_samples_split:
"Is the parent large enough to attempt splitting?"

min_samples_leaf:
"Will the resulting leaves be large enough?"


## min_impurity_decrease

Suppose:

Parent impurity = 0.4200
Child impurity  = 0.4199

Improvement = 0.0001

Technically the split improves training impurity, but the improvement
may be too small to justify additional complexity.

min_impurity_decrease allows us to require a meaningful improvement
before creating another split.


## max_leaf_nodes

Limits the total number of final leaf nodes.

More leaves mean more separate prediction regions and therefore greater
model flexibility.

This is another way of controlling model complexity.


# 17. Why Pure Nodes Stop

Suppose a node contains:

20 Yes
0 No

Gini = 0

There is no impurity left to reduce.

Further splitting cannot improve class purity.

Therefore, the node can become a leaf.

However, nodes do NOT need to become pure before stopping.

Pre-pruning constraints may intentionally stop at mixed nodes to reduce
overfitting.


# 18. Post-Pruning

Pre-pruning says:

"Don't grow unnecessary complexity."

Post-pruning takes another approach:

"Grow a larger tree first, then remove branches whose benefit does not
justify their complexity."


Example:

Before pruning:

                 NODE
                   |
                 split
                /     \
             split    leaf
             /   \
           leaf  leaf


After pruning:

                 NODE
                   ↓
                 LEAF


The simpler tree may fit the training data slightly worse but generalize
better.


# 19. Cost-Complexity Pruning

Cost-complexity pruning balances:

1. How well the tree fits the training data.
2. How complex the tree is.

Conceptually:

Tree Cost
=
Training Fit Cost
+
alpha × Tree Complexity


Common notation:

R_alpha(T) = R(T) + alpha × |T|

where:

T       = tree
R(T)    = training fit/impurity cost
|T|     = number of terminal leaves
alpha   = penalty applied to complexity


If alpha is very small:

Complexity is cheap.
Larger trees are tolerated.

If alpha increases:

Complexity becomes more expensive.
More branches must justify their existence.

Therefore:

alpha ↑ → generally more pruning
alpha ↓ → generally less pruning

In sklearn:

ccp_alpha


# 20. Weakest-Link Pruning

Not every branch contributes equally.

Some subtrees produce large improvements in fit.

Others produce tiny improvements while adding significant complexity.

Cost-complexity pruning can progressively remove weak subtrees.

Conceptually:

Large Tree
    ↓
remove weakest subtree
    ↓
Smaller Tree
    ↓
remove next weakest subtree
    ↓
Smaller Tree
    ↓
...


This creates a sequence of candidate tree complexities.


# 21. Choosing the Pruned Tree

We should not choose the final tree only by training performance.

Example:

Tree A:
Train Accuracy = 100%
Validation Accuracy = 76%

Tree B:
Train Accuracy = 94%
Validation Accuracy = 85%

Tree B may generalize better even though its training accuracy is lower.

Validation data or cross-validation can be used to select an appropriate
tree complexity / ccp_alpha.

The goal is not:

"Find the smallest tree."

The goal is:

"Find a tree complex enough to learn useful patterns but not so complex
that it memorizes training noise."


# 22. Pre-Pruning vs Post-Pruning

PRE-PRUNING

Control complexity while the tree is growing.

Examples:

- max_depth
- min_samples_split
- min_samples_leaf
- min_impurity_decrease
- max_leaf_nodes


POST-PRUNING

Allow a larger tree to grow and then remove weak/unnecessary subtrees.

Main concept covered:

- Cost-Complexity Pruning
- ccp_alpha


Both approaches attempt to improve generalization by controlling model
complexity.


# 23. Complete Mental Model So Far

At every node:

Training rows arrive
        ↓
Calculate current impurity
        ↓
Check stopping conditions
        ↓
Generate candidate feature questions
        ↓
For numerical features:
sort observed values and generate meaningful thresholds
        ↓
For each candidate:
partition rows
        ↓
calculate child Ginis
        ↓
calculate weighted child impurity
        ↓
calculate impurity reduction
        ↓
choose the best candidate
        ↓
commit the split
       /     \
      /       \
 left rows   right rows
     ↓           ↓
repeat        repeat


During growth:

Pre-pruning controls complexity.


After growth:

Optional post-pruning can remove weak subtrees.


During inference:

New sample
    ↓
Start at root
    ↓
Answer learned questions
    ↓
Follow corresponding branches
    ↓
Reach leaf
    ↓
Return leaf prediction / class probabilities


# Key Takeaway

A Decision Tree is not trained by gradient descent.

There are:

- no learned weights like Logistic Regression
- no learning rate
- no epochs
- no backpropagation

Instead, training is a greedy recursive search:

"At this node, which available question gives me the best improvement
in class separation?"

The tree chooses that question, partitions the data, and repeats the same
process independently inside the resulting child nodes.

Its power comes from flexible nonlinear partitioning.

Its major weakness is that this flexibility can easily lead to
overfitting, which is why controlling tree complexity is essential.