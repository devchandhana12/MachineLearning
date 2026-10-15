# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 2 — VECTOR ADDITION AND SUBTRACTION

---

# 1. RECAP: WHAT DID WE LEARN?

In Chapter 1, we understood three important mathematical objects.

### Scalar

A single numerical value.

Example:

80

### Vector

An ordered collection of numerical values.

Example:

Rahul = [80, 70, 90]

These numbers represent Rahul's marks in:

- Maths = 80
- Physics = 70
- Chemistry = 90

### Matrix

A rectangular arrangement of numbers.

Our school dataset:

| Student | Maths | Physics | Chemistry |
| ------- | ----: | ------: | --------: |
| Rahul   |    80 |      70 |        90 |
| Priya   |    60 |      85 |        75 |
| Arjun   |    90 |      80 |        85 |
| Sneha   |    50 |      65 |        70 |
| Kiran   |    75 |      90 |        80 |

Each student has a three-dimensional vector.

For example:

R = [80, 70, 90]

P = [60, 85, 75]

Now comes our next question:

Can we perform mathematical operations on these vectors?

Yes.

Let's start with addition.

---

# 2. WHAT IS VECTOR ADDITION?

Imagine the school announces that Rahul and Priya will participate together in an academic competition.

We want to calculate their combined marks in each subject.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

How would we calculate their combined Maths marks?

80 + 60 = 140

Combined Physics marks?

70 + 85 = 155

Combined Chemistry marks?

90 + 75 = 165

Instead of performing three separate calculations, we can represent this entire operation mathematically as:

R + P

Let's calculate:

R + P = [80, 70, 90] + [60, 85, 75]

Add corresponding positions:

First component:

80 + 60 = 140

Second component:

70 + 85 = 155

Third component:

90 + 75 = 165

Final answer:

R + P = [140, 155, 165]

This is vector addition.

### Definition

Vector addition is the operation of adding corresponding components of two vectors.

It produces another vector.

### Important observation

We did not add Rahul's Maths marks to Priya's Physics marks.

We added:

Maths + Maths

Physics + Physics

Chemistry + Chemistry

**Vector addition is performed component by component.**

---

# 3. WHY DO WE ADD CORRESPONDING COMPONENTS?

Consider:

R = [80, 70, 90]

P = [60, 85, 75]

We know the order is:

[Maths, Physics, Chemistry]

Now imagine adding:

Rahul's Maths + Priya's Chemistry

80 + 75 = 155

Mathematically, that addition is possible.

But does it represent their combined marks in the same subject?

No.

It combines values that have different meanings.

When adding vectors representing student marks, we must preserve the meaning of each position.

Position 1 must correspond to Maths in both vectors.

Position 2 must correspond to Physics in both vectors.

Position 3 must correspond to Chemistry in both vectors.

This is why the order we discussed in Chapter 1 matters.

### Important distinction

The mathematical rule requires matching dimensions.

The real-world interpretation additionally requires matching the meaning and units of corresponding components.

For example, adding a vector of marks to a vector of student ages would generally not be meaningful, even if both contained three numbers.

---

# 4. GENERAL FORMULA FOR VECTOR ADDITION

Let's introduce mathematical notation.

Suppose we have two vectors:

A = [a₁, a₂, a₃]

B = [b₁, b₂, b₃]

Their sum is:

A + B = [a₁+b₁, a₂+b₂, a₃+b₃]

This looks complicated only because we're replacing numbers with letters.

Let's substitute our students.

a₁ = 80

a₂ = 70

a₃ = 90

b₁ = 60

b₂ = 85

b₃ = 75

Therefore:

A + B = [80+60, 70+85, 90+75]

A + B = [140, 155, 165]

Exactly the same result.

### Formula for n dimensions

If two vectors contain n components, their sum is:

A + B = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]

The three dots mean that the same operation continues for the remaining components.

No matter how many components exist, the fundamental operation stays the same.

---

# 5. CAN WE ADD VECTORS OF DIFFERENT DIMENSIONS?

Suppose Rahul has marks in three subjects:

R = [80, 70, 90]

But Priya has marks recorded in only two subjects:

P = [60, 85]

Can we perform ordinary vector addition?

R + P = ?

Let's try.

First components:

80 + 60 = 140

Second components:

70 + 85 = 155

Third components:

90 + ?

Priya's third component is missing.

We cannot perform ordinary vector addition because these vectors have different dimensions.

Rahul's vector has three components.

Priya's vector has two components.

### Rule

**Two vectors must have the same dimension to be added using ordinary vector addition.**

Examples:

[2, 3] + [4, 5] is valid.

[2, 3, 4] + [5, 6, 7] is valid.

[2, 3] + [4, 5, 6] is not defined as ordinary vector addition.

We'll encounter special array operations called broadcasting much later. They do not change this basic mathematical rule.

---

# 6. WHAT IS VECTOR SUBTRACTION?

Now let's consider a different problem.

We want to compare Rahul's and Priya's subject marks.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Instead of combining their marks, we want to know how their marks differ.

Let's calculate:

R - P

### Step 1: Maths

Rahul = 80

Priya = 60

Difference:

80 - 60 = 20

Rahul scored 20 marks more than Priya.

### Step 2: Physics

Rahul = 70

Priya = 85

Difference:

70 - 85 = -15

Rahul scored 15 marks fewer than Priya.

### Step 3: Chemistry

Rahul = 90

Priya = 75

Difference:

90 - 75 = 15

Rahul scored 15 marks more than Priya.

Therefore:

R - P = [20, -15, 15]

This result is called a difference vector.

---

# 7. UNDERSTANDING NEGATIVE VALUES IN VECTOR SUBTRACTION

Let's examine our result:

R - P = [20, -15, 15]

What does the positive 20 mean?

Rahul scored 20 marks more than Priya in Maths.

What does the negative 15 mean?

Rahul scored 15 marks fewer than Priya in Physics.

What does the positive 15 mean?

Rahul scored 15 marks more than Priya in Chemistry.

### Important

Negative values in a difference vector are not errors.

They communicate direction of difference.

In our context:

Positive = First student's marks are higher.

Negative = First student's marks are lower.

Zero = Both students scored the same marks.

Example:

A = [80, 70, 90]

B = [80, 85, 90]

A - B = [0, -15, 0]

This tells us:

Both students scored the same Maths marks.

Student A scored 15 fewer marks in Physics.

Both students scored the same Chemistry marks.

---

# 8. WHY DOES SUBTRACTION ORDER MATTER?

We calculated:

R - P = [20, -15, 15]

What happens if we reverse the order?

P - R

= [60-80, 85-70, 75-90]

= [-20, 15, -15]

Compare both results.

R - P = [20, -15, 15]

P - R = [-20, 15, -15]

The signs are reversed.

Why?

Because we're now asking the opposite comparison.

R - P asks:

How much higher or lower are Rahul's marks than Priya's?

P - R asks:

How much higher or lower are Priya's marks than Rahul's?

### Mathematical relationship

R - P = -(P - R)

Or:

P - R = -(R - P)

This is always true for two vectors of the same dimension.

### Key takeaway

**Vector subtraction is not commutative.**

In general:

A - B ≠ B - A

This is different from vector addition.

---

# 9. VECTOR ADDITION VS VECTOR SUBTRACTION

Let's compare the two operations.

We have:

R = [80, 70, 90]

P = [60, 85, 75]

### Addition

R + P = [140, 155, 165]

Interpretation:

Combined subject marks of both students.

### Subtraction

R - P = [20, -15, 15]

Interpretation:

How Rahul's marks differ from Priya's in each subject.

### Main difference

Vector addition combines component values.

Vector subtraction calculates component-wise differences.

Both operations produce vectors.

Both require matching dimensions.

---

# 10. WHAT IS THE ZERO VECTOR?

Suppose we compare Rahul's marks with his own marks.

R = [80, 70, 90]

Calculate:

R - R

= [80-80, 70-70, 90-90]

= [0, 0, 0]

This is called the zero vector.

### Definition

A zero vector is a vector whose components are all zero.

Examples:

Two-dimensional zero vector:

[0, 0]

Three-dimensional zero vector:

[0, 0, 0]

Four-dimensional zero vector:

[0, 0, 0, 0]

### Important

The zero vector is different from the scalar zero.

Scalar:

0

Three-dimensional zero vector:

[0, 0, 0]

The first is one number.

The second is a vector with three components.

---

# 11. WHY IS THE ZERO VECTOR IMPORTANT?

Consider:

R = [80, 70, 90]

And the zero vector:

Z = [0, 0, 0]

Add them:

R + Z

= [80+0, 70+0, 90+0]

= [80, 70, 90]

Nothing changes.

Therefore:

R + Z = R

This is called the additive identity property.

### Definition

An additive identity is an element that leaves another element unchanged when added.

For ordinary numbers:

5 + 0 = 5

For vectors:

[80, 70, 90] + [0, 0, 0] = [80, 70, 90]

The same principle applies.

---

# 12. WHAT IS THE NEGATIVE OF A VECTOR?

Suppose:

R = [80, 70, 90]

The negative of this vector is:

-R = [-80, -70, -90]

Every component has its sign reversed.

Now calculate:

R + (-R)

= [80-80, 70-70, 90-90]

= [0, 0, 0]

Therefore:

R + (-R) = 0 vector

The negative vector is also called the additive inverse.

### Why?

Because it cancels the original vector when added.

This also helps explain why subtraction can be written as addition:

R - P = R + (-P)

Let's verify.

P = [60, 85, 75]

-P = [-60, -85, -75]

Then:

R + (-P)

= [80, 70, 90] + [-60, -85, -75]

= [20, -15, 15]

That's exactly R - P.

---

# 13. PROPERTIES OF VECTOR ADDITION

These properties are not arbitrary rules. They follow from ordinary number arithmetic applied component by component.

## 13.1 Commutative Property

For vectors:

A + B = B + A

Example:

R + P

= [140, 155, 165]

P + R

= [140, 155, 165]

Both produce the same result.

Addition order does not matter.

## 13.2 Associative Property

Suppose we combine the marks of Rahul, Priya, and Arjun.

R = [80, 70, 90]

P = [60, 85, 75]

A = [90, 80, 85]

We could first combine Rahul and Priya, then add Arjun.

(R + P) + A

Or first combine Priya and Arjun, then add Rahul.

R + (P + A)

Let's calculate the first method.

R + P = [140, 155, 165]

Add Arjun:

[140, 155, 165] + [90, 80, 85]

= [230, 235, 250]

Second method:

P + A

= [150, 165, 160]

Now add Rahul:

[80, 70, 90] + [150, 165, 160]

= [230, 235, 250]

Both produce the same result.

Therefore:

(R + P) + A = R + (P + A)

This is the associative property.

### Meaning

When adding multiple vectors, changing the grouping does not change the mathematical result.

## 13.3 Additive Identity

A + 0 = A

Adding a zero vector leaves the vector unchanged.

## 13.4 Additive Inverse

A + (-A) = 0

Adding a vector to its negative produces a zero vector.

---

# 14. UNDERSTANDING VECTORS GEOMETRICALLY

So far, we've treated vectors as collections of subject marks.

Now let's understand that the same vectors can also be viewed geometrically.

Our students have three subjects, so their vectors are three-dimensional.

Visualizing three dimensions is possible, but it introduces unnecessary complexity at this stage.

Instead, let's temporarily use only two subjects:

Maths and Physics.

Rahul:

R = [80, 70]

Priya:

P = [60, 85]

Each student can be represented as a point on a graph.

Horizontal axis = Maths marks.

Vertical axis = Physics marks.

Rahul is at:

(80, 70)

Priya is at:

(60, 85)

This gives us a new way to interpret vectors.

A vector can describe a position relative to an origin.

It can also describe a displacement.

Let's understand displacement first.

---

# 15. VECTORS AS MOVEMENT

Imagine standing at position:

(0, 0)

Now suppose you move:

3 steps right.

2 steps up.

Your final position becomes:

(3, 2)

We can represent the movement as:

A = [3, 2]

The first component tells us the horizontal movement.

The second component tells us the vertical movement.

This is a geometric vector.

### Important distinction

A point identifies a position.

A displacement vector describes a movement from one position to another.

The coordinates of a point can be used as a position vector from the origin.

For now, we will represent movements using arrows.

---

# 16. GEOMETRIC VECTOR ADDITION

Suppose:

A = [3, 2]

B = [2, 1]

First follow movement A:

Move 3 right and 2 up.

Then follow movement B:

Move another 2 right and 1 up.

Your final movement from the origin is:

Horizontal = 3 + 2 = 5

Vertical = 2 + 1 = 3

Therefore:

A + B = [5, 3]

### Head-to-tail rule

To add vectors geometrically:

1. Draw the first vector.
2. Start the second vector where the first ends.
3. Draw an arrow from the original starting position to the final endpoint.

That final arrow represents their sum.

This is called the head-to-tail method.

### What should you understand?

Vector addition doesn't simply combine numbers randomly.

It combines movements along corresponding coordinate directions.

The same mathematical operation works with student marks because each component represents a different coordinate of the student's numerical record.

---

# 17. GEOMETRIC VECTOR SUBTRACTION

Suppose we have two positions:

A = [3, 2]

B = [5, 4]

We want to understand the movement required to travel from A to B.

Calculate:

B - A

= [5-3, 4-2]

= [2, 2]

Interpretation:

From A, move 2 units right and 2 units up to reach B.

Now reverse the subtraction:

A - B = [-2, -2]

That represents the movement from B to A.

### General rule

Displacement from A to B:

B - A

Displacement from B to A:

A - B

This is why subtraction order matters geometrically.

---

# 18. HOW DOES THIS RELATE TO OUR STUDENTS?

Let's return to our school dataset.

Rahul:

R = [80, 70]

Priya:

P = [60, 85]

Calculate:

P - R

= [60-80, 85-70]

= [-20, 15]

This tells us how to move from Rahul's point to Priya's point on our marks graph.

Move 20 units left along the Maths axis.

Move 15 units up along the Physics axis.

So the difference vector is:

[-20, 15]

This is the same result we obtain from ordinary subtraction.

Now we have two ways to understand the same operation:

### Numerical meaning

Priya scored 20 fewer Maths marks and 15 more Physics marks.

### Geometric meaning

The vector [-20, 15] describes the displacement from Rahul's position to Priya's position in the two-dimensional subject-mark space.

This geometric interpretation becomes very useful when we study distance.

---

# 19. WHY DOES SUBTRACTION MATTER FOR MEASURING DISTANCE?

Suppose we want to know how similar Rahul and Priya are based on their marks.

Their two-subject vectors are:

R = [80, 70]

P = [60, 85]

First calculate their difference:

R - P = [20, -15]

This tells us how the marks differ.

But we still don't have one number describing how far apart the two students are.

We have two numbers:

20 and -15.

Our next objective will be to convert this difference vector into a single nonnegative distance.

That requires another concept:

**Vector magnitude.**

We'll study magnitude and distance in Chapter 3.

For now, understand that subtraction gives us the displacement vector needed for that calculation.

---

# 20. VECTOR ADDITION IS NOT THE SAME AS ADDING ALL COMPONENTS

This is an important distinction.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

### Operation A: Add the two vectors

R + P = [140, 155, 165]

The result is a vector with three components.

### Operation B: Calculate Rahul's total marks

80 + 70 + 90 = 240

The result is one number.

### Operation C: Calculate Priya's total marks

60 + 85 + 75 = 220

Again, the result is one number.

These are different operations.

Vector addition combines corresponding components of multiple vectors.

Summing the components of a single vector produces one scalar.

We will explore such operations more deeply later.

---

# 21. WHEN IS VECTOR ADDITION MEANINGFUL?

Consider two examples.

### Example A: Marks from two students

R = [80, 70, 90]

P = [60, 85, 75]

Adding the vectors produces combined subject marks.

That has a clear interpretation.

### Example B: Marks and ages

Marks vector:

A = [80, 70, 90]

Age vector:

B = [15, 16, 17]

Mathematically:

A + B = [95, 86, 107]

The operation is defined because both vectors contain three real numbers.

But does the result have a meaningful interpretation?

Usually not.

We have added marks to ages.

Their units and meanings differ.

### Key lesson

**An operation can be mathematically valid without being meaningful for a particular real-world problem.**

When using mathematics to represent data, always ask what the numbers and operations actually represent.

This habit will become important in machine learning.

---

# 22. IMPORTANT FORMULAS

### Vector Addition

A + B = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]

### Vector Subtraction

A - B = [a₁-b₁, a₂-b₂, ..., aₙ-bₙ]

### Additive Identity

A + 0 = A

### Additive Inverse

A + (-A) = 0

### Commutative Addition

A + B = B + A

### Associative Addition

(A+B)+C = A+(B+C)

### Subtraction Reversal

A - B = -(B - A)

### Displacement

Displacement from point A to point B:

B - A

---

# 23. INTERVIEW-LEVEL UNDERSTANDING

Although we're learning from scratch, let's understand what you should eventually be able to explain confidently.

### Q1. What is vector addition?

Vector addition combines vectors of the same dimension by adding corresponding components.

### Q2. What is vector subtraction?

Vector subtraction calculates the difference between corresponding components of two vectors.

### Q3. Why must dimensions match?

Because each component must have a corresponding component in the other vector for ordinary addition or subtraction.

### Q4. Is vector addition commutative?

Yes.

A + B = B + A

### Q5. Is vector subtraction commutative?

No.

Generally:

A - B ≠ B - A

### Q6. What is a zero vector?

A vector whose components are all zero.

It acts as the additive identity.

### Q7. What is the additive inverse of a vector?

A vector obtained by reversing the sign of every component.

For example:

-[2, -3] = [-2, 3]

### Q8. What does B - A represent geometrically?

The displacement vector from point A to point B.

### Q9. Does vector addition always have a meaningful real-world interpretation?

No.

The mathematical operation can be valid even if corresponding components represent incompatible quantities.

### Q10. Is adding two vectors the same as calculating the sum of one vector's components?

No.

Vector addition produces another vector.

Summing all components of a vector produces a scalar.

---

# 24. COMPLETE CHAPTER SUMMARY

In this chapter, we learned:

1. Vector addition combines corresponding components.
2. Vector subtraction measures component-wise differences.
3. Both operations require matching dimensions.
4. Vector addition is commutative.
5. Vector addition is associative.
6. Vector subtraction is generally not commutative.
7. A zero vector acts as the additive identity.
8. The negative of a vector is its additive inverse.
9. Geometrically, vectors can represent displacements.
10. Vector addition combines displacements.
11. Vector subtraction finds displacement between positions.
12. The result of vector addition or subtraction is another vector.
13. Adding two vectors is different from summing the components of one vector.
14. Real-world interpretations must respect component meanings and units.

### THE BIG PICTURE

We started with ordinary marks:

80, 70, 90

We organized them into vectors.

We learned to add vectors.

We learned to subtract vectors.

We then discovered that the same operations can describe movements in geometric space.

These ideas are the foundation for understanding vector length, distance, angles, and other important concepts.

---

END OF CHAPTER 2

NEXT: CHAPTER 3 — SCALAR MULTIPLICATION, VECTOR MAGNITUDE, AND NORMS
