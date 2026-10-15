# Decision Tree Classification — Think Like the Algorithm

# 1. My Job

I am a Decision Tree classifier.

I receive:

X = input features

Example:
- Distance
- Price
- Hungry

y = target

Example:
- Bought = Yes
- Bought = No


My job is to learn questions such as:

                Distance <= 5?
                /            \
              Yes             No
              /                \
         Hungry?            Price <= 200?
          /   \               /       \
        Yes    No           Yes        No


Unlike Logistic Regression, I do NOT learn:

w1, w2, w3, b

I learn:

- which feature to ask about
- what threshold to use
- where that question should appear
- when to stop asking questions
- what to predict at the leaves


--------------------------------------------------
2. WHY DO I EXIST?
--------------------------------------------------

Logistic Regression essentially learns:

z = w1*x1 + w2*x2 + ... + b

and creates a linear decision boundary in the supplied feature space.

But relationships are not always linear.

The real pattern might be:

IF Distance <= 5:
    likely BUY

ELSE:
    IF Price <= 200:
        likely BUY
    ELSE:
        likely NOT BUY


This is nonlinear and interaction-based.

Instead of fitting one global linear equation, I divide the feature
space into regions using questions.


--------------------------------------------------
3. I START WITH EVERYONE IN ONE NODE
--------------------------------------------------

Suppose my training data contains:

10 customers

6 YES
4 NO


Initially:

                ROOT
               6Y / 4N


Now I need to ask:

"What question should I ask first?"


Possible questions:

Distance <= 3?
Distance <= 5?
Price <= 250?
Price <= 350?
Hungry = Yes?


But I need a mathematical way to determine which question is best.


--------------------------------------------------
4. HOW I CREATE QUESTIONS FOR NUMERICAL FEATURES
--------------------------------------------------

Suppose Distance contains:

1, 2, 3, 5, 7, 9


First I sort the observed values:

1   2   3   5   7   9


I don't need to test every real number.

For example:

Distance <= 3.1
Distance <= 3.5
Distance <= 3.9

all produce:

LEFT:
1, 2, 3

RIGHT:
5, 7, 9


They create exactly the same partition.

So conceptually I only need meaningful boundaries between observed
values.

Possible representative thresholds:

1.5
2.5
4
6
8


Then I can test:

Distance <= 1.5?
Distance <= 2.5?
Distance <= 4?
Distance <= 6?
Distance <= 8?


With n unique values, there are roughly at most n-1 meaningful
boundaries.


--------------------------------------------------
5. WHY I DON'T TEST BOTH <= AND >
--------------------------------------------------

Suppose I ask:

Distance <= 4?


That automatically gives me:

TRUE:
Distance <= 4

FALSE:
Distance > 4


Testing:

Distance > 4?

would produce the same partition with the branch labels reversed.

So there is no need to treat it as a fundamentally different split.


--------------------------------------------------
6. WHAT ABOUT USELESS THRESHOLDS?
--------------------------------------------------

Suppose all observed distances are between 1 and 10.

Question:

Distance <= -500?

would produce:

LEFT  = nobody
RIGHT = everybody


Similarly:

Distance <= 5000?

might produce:

LEFT  = everybody
RIGHT = nobody


I care about questions that actually partition my training samples.


--------------------------------------------------
7. NOW I HAVE MANY POSSIBLE QUESTIONS
--------------------------------------------------

Maybe I have:

Distance <= 1.5?
Distance <= 2.5?
Distance <= 4?
...

Price <= 175?
Price <= 225?
Price <= 275?
...

Hungry = Yes?


Now I need to determine:

WHICH QUESTION CREATES BETTER CLASS SEPARATION?


To answer that, I need to measure how mixed a node is.


--------------------------------------------------
8. GINI IMPURITY
--------------------------------------------------

Suppose a node contains:

10 YES
0 NO


This node is perfectly pure.


Another node:

5 YES
5 NO


This node is highly mixed.


I need one number representing that mixedness.

One possible measure is Gini impurity:

Gini = 1 - sum(p_k^2)

where:

p_k = proportion belonging to class k


Example:

8 YES
2 NO


p(YES) = 0.8
p(NO)  = 0.2


Gini:

1 - (0.8^2 + 0.2^2)

= 1 - (0.64 + 0.04)

= 0.32


Some useful binary examples:

10Y / 0N -> Gini = 0

9Y / 1N -> Gini = 0.18

8Y / 2N -> Gini = 0.32

5Y / 5N -> Gini = 0.50


Therefore:

LOW GINI
=
PURE


HIGH GINI
=
MIXED


Important:

GINI DESCRIBES ONE NODE.

It does not directly describe an entire tree level.


--------------------------------------------------
9. I EVALUATE A CANDIDATE QUESTION
--------------------------------------------------

Suppose my parent node contains:

6 YES
4 NO


Parent Gini:

1 - (0.6^2 + 0.4^2)

= 0.48


Now I test:

Distance <= 5.5?


It creates:


LEFT:

4 YES
1 NO

Gini = 0.32


RIGHT:

2 YES
3 NO

Gini = 0.48


Now I have two child impurities.

But I need ONE score representing the entire split.


--------------------------------------------------
10. WHY I WEIGHT CHILD IMPURITY
--------------------------------------------------

Suppose:

LEFT:
1 customer
Gini = 0


RIGHT:
99 customers
Gini = 0.5


I should NOT calculate:

(0 + 0.5) / 2


That treats the 1-customer group as equally important as the
99-customer group.


Instead:

Weighted Child Impurity

=

(N_left / N) * Gini_left

+

(N_right / N) * Gini_right


For our earlier 5/5 example:

LEFT:
5 samples
Gini = 0.32

RIGHT:
5 samples
Gini = 0.48


Weighted impurity:

(5/10)(0.32) + (5/10)(0.48)

= 0.40


--------------------------------------------------
11. DID THE QUESTION ACTUALLY HELP?
--------------------------------------------------

Before asking the question:

Parent impurity = 0.48


After asking the question:

Weighted child impurity = 0.40


Therefore:

Impurity Reduction

=

Parent Impurity
-
Weighted Child Impurity


= 0.48 - 0.40

= 0.08


So this question improved class separation by 0.08.


--------------------------------------------------
12. I DO THIS FOR EVERY CANDIDATE
--------------------------------------------------

For each candidate question:

                Candidate
                    |
                    v
               Split rows
                    |
                    v
            Calculate child
                 Ginis
                    |
                    v
             Weight children
             by sample count
                    |
                    v
        Weighted Child Impurity
                    |
                    v
 Parent Impurity - Child Impurity
                    |
                    v
          Impurity Reduction


Suppose:

Distance <= 4   -> improvement 0.07

Distance <= 6   -> improvement 0.12

Price <= 250    -> improvement 0.21

Hungry = Yes    -> improvement 0.09


I choose:

Price <= 250


because:

0.21

is the largest improvement.


--------------------------------------------------
13. IMPORTANT: I AM GREEDY
--------------------------------------------------

I choose the best split AVAILABLE RIGHT NOW.

I do not normally search every possible future tree and find the
globally perfect tree.

I ask:

"Which split looks best at this node right now?"


Then I commit to it.


That is why Decision Tree training is called GREEDY.


--------------------------------------------------
14. AFTER CHOOSING THE ROOT, I REPEAT
--------------------------------------------------

Suppose I choose:

                    Price <= 250?
                    /           \
                 LEFT           RIGHT

                4Y/1N           2Y/3N


Now the root's job is finished.


I take LEFT:

4Y / 1N


and treat it as another Decision Tree problem.


I again:

- generate candidate questions
- calculate impurities
- calculate impurity reductions
- choose the best split


Separately, RIGHT does the same thing.


Therefore my growth is RECURSIVE.


Conceptually:

BUILD_TREE(rows):

    if I should stop:
        create leaf

    otherwise:

        generate candidate splits

        evaluate them

        choose best split

        BUILD_TREE(left_rows)

        BUILD_TREE(right_rows)


--------------------------------------------------
15. I CAN REUSE FEATURES
--------------------------------------------------

Suppose I already used:

Distance <= 8?

at the root.


Later I can still ask:

Distance <= 3?


Example:

                    Distance <= 8?
                    /
              Distance <= 3?


Features are NOT consumed after being used once.

Different thresholds can isolate different regions.


--------------------------------------------------
16. MY TREE DOES NOT NEED TO BE BALANCED
--------------------------------------------------

I might grow like this:

                         ROOT
                        /    \
                     LEAF    NODE
                            /    \
                         LEAF    NODE
                                /    \
                             LEAF    LEAF


One branch may stop very early.

Another may continue deeper.


I grow NODE BY NODE.

I do not need every branch to have the same depth.


--------------------------------------------------
17. WHEN DO I CREATE A LEAF?
--------------------------------------------------

One obvious case:

20 YES
0 NO


Gini = 0


There is no impurity left to reduce.

So I can stop.


But a leaf does NOT have to be pure.


I might stop at:

8 YES
2 NO


because:

- max depth was reached
- too few samples remain
- improvement is too small
- pruning/regularization constraints stop me


--------------------------------------------------
18. WHAT DOES A LEAF PREDICT?
--------------------------------------------------

Suppose a leaf contains:

8 YES
2 NO


For classification:

Predicted class:

YES


because YES is the majority class.


The class distribution also gives:

P(YES) = 8/10 = 0.8

P(NO) = 2/10 = 0.2


Conceptually:

NEW CUSTOMER
      |
      v
Start at root
      |
      v
Answer questions
      |
      v
Follow branches
      |
      v
Reach leaf
      |
      v
Use leaf's class distribution


This is fundamentally different from Logistic Regression.


LOGISTIC REGRESSION:

x
|
v
w*x + b
|
v
sigmoid
|
v
probability


DECISION TREE:

x
|
v
follow learned questions
|
v
reach leaf
|
v
leaf class distribution


--------------------------------------------------
19. I DON'T USE GRADIENT DESCENT
--------------------------------------------------

I do NOT have:

- learning rate
- epochs
- backpropagation
- gradient descent
- learned weights/bias like Logistic Regression


My learning mechanism is:

SEARCH FOR SPLITS.


At every node:

"Which available question produces the best impurity reduction?"


--------------------------------------------------
20. MY BIG PROBLEM: I CAN OVERFIT LIKE CRAZY
--------------------------------------------------

Suppose I keep growing.


Eventually I might learn:

Age <= 31.5
    |
Distance > 3.72
    |
Price <= 287
    |
Age > 27.5
    |
one specific customer


Maybe:

1 YES
0 NO


Beautiful.

Gini = 0.


But I may simply have memorized that customer.


My training accuracy can become:

99-100%


while validation accuracy is much worse.


The problem:

I am extremely flexible.

If you allow me to keep splitting, I can create increasingly tiny
regions around training observations.


So we need to control my complexity.


==================================================
PART 2 — PRE-PRUNING
==================================================

Pre-pruning means:

CONTROL ME WHILE I AM GROWING.


--------------------------------------------------
21. max_depth
--------------------------------------------------

You tell me:

max_depth = 3


I interpret:

"No path may grow beyond depth 3."


Even if I discover another potentially useful split at depth 3,
I must stop.


Important:

max_depth = 3

does NOT mean every branch must reach depth 3.


It means:

maximum allowed depth = 3.


--------------------------------------------------
22. min_samples_split
--------------------------------------------------

Suppose:

min_samples_split = 10


I arrive at a node containing:

8 samples


I think:

"I am not even allowed to attempt another split."

So I create a leaf.


This parameter asks:

IS THE PARENT LARGE ENOUGH TO SPLIT?


--------------------------------------------------
23. min_samples_leaf
--------------------------------------------------

Suppose my parent contains:

100 samples.


A candidate split produces:

LEFT  = 1 sample
RIGHT = 99 samples


Technically, this might improve training impurity.

But that 1-sample region smells like memorization.


If:

min_samples_leaf = 10


this split is illegal.


So remember:


min_samples_split

asks:

"Is my PARENT large enough to attempt splitting?"


min_samples_leaf

asks:

"Would my resulting LEAVES contain enough samples?"


--------------------------------------------------
24. min_impurity_decrease
--------------------------------------------------

Suppose:

Parent impurity = 0.4200

Best child impurity = 0.4199


Improvement:

0.0001


Technically:

YES, I improved.


But did I improve enough to justify another branch?


If:

min_impurity_decrease = 0.01


then:

0.0001 < 0.01


I stop.


--------------------------------------------------
25. max_leaf_nodes
--------------------------------------------------

Instead of limiting my depth, you can limit the total number of
prediction regions I create.


Example:

max_leaf_nodes = 4


I may create at most four terminal leaves.


These two trees can both have four leaves:


Balanced-ish:

                ROOT
               /    \
             NODE   NODE
             / \    / \
            L   L  L   L


Unbalanced:

                ROOT
               /    \
              L     NODE
                    /  \
                   L   NODE
                       / \
                      L   L


Same number of leaves.

Different depth/shape.


So:

max_depth

controls PATH DEPTH.


max_leaf_nodes

controls TOTAL FINAL REGIONS.


--------------------------------------------------
26. THE WEAKNESS OF PRE-PRUNING
--------------------------------------------------

Now an important question:

"If pre-pruning exists, why would I need post-pruning?"


Because pre-pruning makes decisions EARLY.


Suppose I see:

                    NODE
                     |
                   Split A
                  /       \
              30Y/20N    25Y/25N


Split A itself gives only modest improvement.


A strict pre-pruning rule might tell me:

"Not good enough. STOP."


But suppose I had been allowed to continue:


                    Split A
                   /       \
              30Y/20N      ...
                 |
              Split B
              /     \
           28Y/2N   2Y/18N


Now we discovered very useful deeper structure.


Pre-pruning could have prevented me from ever discovering it.


So:

PRE-PRUNING:

"Should I allow this structure to grow?"


POST-PRUNING:

"I already saw what this structure became.
Was it actually worth keeping?"


==================================================
PART 3 — POST-PRUNING
==================================================

--------------------------------------------------
27. LET ME GROW FIRST
--------------------------------------------------

Suppose I already built:

                     NODE B
                     4Y / 2N
                        |
                  Hungry = Yes?
                  /          \
               4Y/0N        0Y/2N


Training fit is excellent.

Both children are pure.


Post-pruning asks:

"What if I remove Hungry = Yes?"


Then:

                     NODE B
                     4Y / 2N
                        |
                        v
                       LEAF


I lose some training fit.

But I also become simpler.


This is the fundamental tradeoff:

BETTER TRAINING FIT

vs

LOWER MODEL COMPLEXITY


--------------------------------------------------
28. COST-COMPLEXITY PRUNING
--------------------------------------------------

I need an objective that cares about BOTH.


Let:

R(T)

represent my training fit cost / impurity cost.


Let:

|T|

represent my complexity, commonly expressed through the number of
terminal leaves.


If I only minimize:

R(T)


then larger trees usually win because they can fit training data
better.


So we introduce a complexity penalty.


Cost-complexity objective:

R_alpha(T)

=

R(T) + alpha * |T|


Read it as:

TOTAL COST

=

TRAINING FIT COST

+

PRICE OF COMPLEXITY


alpha determines how expensive complexity is.


--------------------------------------------------
29. I AM THE ALGORITHM — SHOULD I KEEP THIS BRANCH?
--------------------------------------------------

Suppose KEEPING a subtree gives:

Training cost = 0.10

Leaves = 2


Let:

alpha = 0.05


Then:

Total cost

= 0.10 + (0.05)(2)

= 0.20


Now suppose I PRUNE it.


Training cost becomes:

0.13


because I lost useful training structure.


But now:

Leaves = 1


Total:

0.13 + (0.05)(1)

= 0.18


Compare:

KEEP  = 0.20

PRUNE = 0.18


Lower wins.


Therefore:

PRUNE.


My reasoning:

"Keeping this branch improves training fit by only 0.03,
but the additional complexity costs me 0.05.

Bad deal."


--------------------------------------------------
30. WHAT DOES alpha MEAN?
--------------------------------------------------

alpha is essentially:

HOW EXPENSIVE DO WE CONSIDER TREE COMPLEXITY?


If:

alpha = 0


then:

R_alpha(T) = R(T)


Complexity has no penalty in this objective.


Training fit dominates.


Small alpha:

complexity is cheap

-> larger tree tolerated


Large alpha:

complexity is expensive

-> branches need to provide significant benefit

-> more pruning


Therefore:

alpha increases
        |
        v
stronger complexity penalty
        |
        v
generally more pruning


--------------------------------------------------
31. SAME BRANCH, DIFFERENT alpha
--------------------------------------------------

Suppose:

KEEP:

Training cost = 0.10
Leaves = 2


PRUNE:

Training cost = 0.13
Leaves = 1


Now:

alpha = 0.01


KEEP:

0.10 + 0.01(2)

= 0.12


PRUNE:

0.13 + 0.01(1)

= 0.14


Now:

KEEP = 0.12

PRUNE = 0.14


KEEP wins.


Why?


The extra complexity costs only:

0.01


while it improves training fit by:

0.03


Good deal.


So the same subtree may be:

worth keeping for small alpha

but

not worth keeping for larger alpha.


--------------------------------------------------
32. BUT I HAVE MANY SUBTREES — WHICH ONE GOES FIRST?
--------------------------------------------------

Suppose I have:

                         ROOT
                       /      \
                      A        B
                    /  \      / \
                   L    C    D   L
                       / \  / \
                      L  L L   E
                             / \
                            L   L


I could potentially prune:

C
D
E


I should not randomly choose one.


I need to ask:

"Which subtree gives me the least useful fit improvement relative to
the complexity it adds?"


This leads to weakest-link pruning.


--------------------------------------------------
33. WEAKEST-LINK PRUNING
--------------------------------------------------

Consider subtree C.


If pruning C causes:

training cost:

0.20 -> 0.22


then fit lost:

0.02


Suppose pruning removes:

1 leaf worth of extra complexity.


Loss per complexity removed:

0.02 / 1

= 0.02


---------------------


Now subtree D.


Training cost:

0.20 -> 0.24


Fit lost:

0.04


But pruning D removes:

2 units of leaf complexity.


Therefore:

0.04 / 2

= 0.02


---------------------


Now subtree E.


Training cost:

0.20 -> 0.205


Fit lost:

0.005


Complexity removed:

1


Therefore:

0.005 / 1

= 0.005


Compare:

C -> 0.020

D -> 0.020

E -> 0.005


E gives me the LEAST training-fit benefit per unit of complexity.


Therefore E is my weakest link.


I prune E first.


--------------------------------------------------
34. WHAT WEAKEST-LINK REALLY MEANS
--------------------------------------------------

Conceptually, for each subtree I ask:

FIT LOST IF I REMOVE IT
------------------------
COMPLEXITY REMOVED


Small value means:

"I can simplify myself quite cheaply.

I lose very little useful training fit."


Large value means:

"Careful.

This subtree is providing significant training fit for its complexity."


So the weakest subtree gets removed first.


--------------------------------------------------
35. CONNECTION BETWEEN WEAKEST-LINK AND alpha
--------------------------------------------------

Suppose a subtree provides approximately:

0.005

fit benefit per extra unit of complexity.


If:

alpha = 0.001


complexity costs less than the benefit.


KEEP.


But if:

alpha = 0.01


I'm paying:

0.01


for something giving approximately:

0.005


Not worth it.


PRUNE.


Therefore weakest-link pruning effectively identifies when different
subtrees stop being worth their complexity as the complexity penalty
increases.


--------------------------------------------------
36. THE PRUNING PATH
--------------------------------------------------

Start with a large tree.


As complexity becomes more expensive:

                LARGE TREE
                     |
                     v
            prune weakest link
                     |
                     v
              smaller tree
                     |
                     v
            prune next weak link
                     |
                     v
              smaller tree
                     |
                     v
                    ...


Eventually we can obtain a sequence of progressively simpler trees.


This is the cost-complexity pruning path.


In sklearn, the relevant parameter is:

ccp_alpha


--------------------------------------------------
37. WHICH PRUNED TREE DO WE ACTUALLY WANT?
--------------------------------------------------

The largest tree is not automatically best.

The smallest tree is not automatically best either.


Suppose:

alpha      Leaves     Train Acc     Validation Acc

0          90         100%          75%

0.002      45          97%          81%

0.008      18          93%          85%

0.020       7          86%          82%


Initially:

too complex
-> overfitting


Then:

remove noisy complexity
-> validation improves


Eventually:

too much pruning
-> useful patterns disappear
-> underfitting


Therefore the goal is NOT:

"Prune as much as possible."


The goal is:

FIND THE COMPLEXITY THAT GENERALIZES WELL.


Validation data or cross-validation can help us choose the appropriate
tree complexity / ccp_alpha.


==================================================
PART 4 — THE ENTIRE ALGORITHM IN MY HEAD
==================================================

I receive:

X, y


I place all training rows at the root.


At the current node:

        |
        v

How mixed are my labels?

        |
        v

Calculate impurity.

        |
        v

Should I stop?

If yes:

CREATE LEAF.


If no:

        |
        v

Generate candidate questions from available features.

        |
        v

For every candidate:

split rows
        |
        v
calculate child impurities
        |
        v
weight by child sizes
        |
        v
calculate impurity reduction

        |
        v

Choose the best available split.

        |
        v

Commit to it.

        |
        +----------------+
        |                |
        v                v
    LEFT CHILD       RIGHT CHILD

        |                |
        v                v

repeat recursively


While growing:

pre-pruning constraints may stop me.


After growing:

post-pruning may remove complexity whose training benefit does not
justify keeping it.


At inference:

new sample
    |
    v
root
    |
    v
answer question
    |
    v
follow branch
    |
    v
answer next question
    |
    v
...
    |
    v
leaf
    |
    v
class prediction / probability


==================================================
PART 5 — IMPORTANT CHARACTERISTICS TO REMEMBER
==================================================

1. I AM NONLINEAR

Repeated splits create nonlinear, piecewise decision regions.


2. I AUTOMATICALLY MODEL FEATURE INTERACTIONS

I can learn:

IF Hungry = Yes:
    THEN check Distance

ELSE:
    THEN check Price


The effect of one feature can therefore depend on previous questions.


3. I AM GREEDY

I choose the best LOCAL split at each node.

I do not normally search every possible complete tree.


4. I DO NOT USE GRADIENT DESCENT

No:

- learning rate
- epochs
- backpropagation


5. I CAN OVERFIT VERY EASILY

Deep trees can memorize tiny training regions.


6. MY LEAVES DO NOT NEED TO BE PURE

Stopping rules may intentionally leave mixed leaves.


7. FEATURES CAN APPEAR MULTIPLE TIMES

Using Distance once does not consume it.


8. MY BRANCHES CAN HAVE DIFFERENT DEPTHS

Trees do not need to be balanced.


9. MY PREDICTION COMES FROM THE REACHED LEAF

For classification, majority class gives the class prediction and the
leaf's class proportions provide probability estimates.


==================================================
PART 6 — FINAL MENTAL MODEL
==================================================

If I forget everything about Decision Trees, reconstruct it from this:

"I am standing at a node containing training samples.

My labels are mixed.

Can I ask a feature-based question that separates these labels into
cleaner groups?

I generate possible questions.

For each question, I calculate how impure the resulting groups are.

I account for how many samples each child contains.

I choose the question giving the largest useful impurity reduction.

Then I repeat the exact same process independently inside each child.

If you let me continue indefinitely, I can memorize the training data.

So you control my complexity while I grow using pre-pruning, or allow
me to grow further and later remove branches whose improvement in
training fit is not worth their complexity using post-pruning.

Once training is finished, a new sample simply follows my learned
questions until it reaches a leaf."

That is a Decision Tree classifier.