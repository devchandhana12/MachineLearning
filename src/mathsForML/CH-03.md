# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 3 — SCALAR MULTIPLICATION, VECTOR MAGNITUDE & NORMS

---

# 1. RECAP OF OUR EXAMPLE

We're working with the same five students.

| Student | Maths | Physics | Chemistry |
| ------- | ----: | ------: | --------: |
| Rahul   |    80 |      70 |        90 |
| Priya   |    60 |      85 |        75 |
| Arjun   |    90 |      80 |        85 |
| Sneha   |    50 |      65 |        70 |
| Kiran   |    75 |      90 |        80 |

Each student can be represented by a vector.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Arjun:

A = [90, 80, 85]

In Chapter 2, we learned:

Vector addition:

R + P = [140, 155, 165]

Vector subtraction:

R - P = [20, -15, 15]

Remember:

- Addition combines corresponding values.
- Subtraction calculates corresponding differences.
- Both operations produce another vector.

Now let's learn a new operation.

---

# 2. WHAT IS SCALAR MULTIPLICATION?

## 2.1 Let's start with an ordinary situation

Suppose Rahul scored:

Maths = 80

Physics = 70

Chemistry = 90

His marks vector is:

R = [80, 70, 90]

Imagine we want to calculate half of his marks in every subject.

We could calculate them separately.

Maths:

80 × 0.5 = 40

Physics:

70 × 0.5 = 35

Chemistry:

90 × 0.5 = 45

Therefore, the result is:

[40, 35, 45]

But we don't need three separate expressions.

We can write:

0.5 × R

Or simply:

0.5R

Since:

R = [80, 70, 90]

We get:

0.5R = [40, 35, 45]

This operation is called scalar multiplication.

## 2.2 Definition

Scalar multiplication is the operation of multiplying every component of a vector by a single number.

Remember Chapter 1?

A scalar is one number.

A vector is an ordered collection of numbers.

In scalar multiplication, we multiply one scalar by one vector.

The result is another vector.

---

# 3. UNDERSTANDING SCALAR MULTIPLICATION STEP BY STEP

Let's multiply Rahul's marks by 2.

R = [80, 70, 90]

Scalar = 2

Calculate:

2R = 2[80, 70, 90]

Multiply each component by 2.

First component:

2 × 80 = 160

Second component:

2 × 70 = 140

Third component:

2 × 90 = 180

Therefore:

2R = [160, 140, 180]

Notice something.

We did not change the number of components.

Before multiplication:

Three components.

After multiplication:

Three components.

Scalar multiplication changes component values but not the vector's dimension.

The marks exceed the normal 100-mark scale here, but that's fine for demonstrating a mathematical operation.

---

# 4. GENERAL FORMULA FOR SCALAR MULTIPLICATION

Suppose we have a vector:

V = [v₁, v₂, v₃]

And a scalar:

k

Then:

kV = [kv₁, kv₂, kv₃]

For a vector containing n components:

V = [v₁, v₂, ..., vₙ]

The general formula is:

kV = [kv₁, kv₂, ..., kvₙ]

The letter k represents any real number.

### Example

V = [10, 20, 30]

k = 3

Then:

3V = [30, 60, 90]

It's the same operation regardless of how many components the vector contains.

---

# 5. WHAT HAPPENS WHEN THE SCALAR IS A FRACTION?

Let's calculate:

0.5R

We already know:

R = [80, 70, 90]

Then:

0.5R = [40, 35, 45]

Multiplying by 0.5 means taking half.

What about multiplying by 0.25?

0.25R = [20, 17.5, 22.5]

Multiplying by 0.25 means taking one-quarter.

### Important idea

If a scalar is between 0 and 1, multiplying by it makes each nonzero component smaller in absolute value.

Examples:

0.5 × 80 = 40

0.25 × 80 = 20

0.1 × 80 = 8

We'll later see what this means geometrically.

---

# 6. WHAT HAPPENS WHEN THE SCALAR IS NEGATIVE?

Suppose:

R = [80, 70, 90]

Multiply by -1.

-R = [-80, -70, -90]

All component signs reverse.

Now multiply by -2.

-2R = [-160, -140, -180]

### What does this mean?

Mathematically, we're scaling every component and reversing its sign.

In the school example, negative marks might not have a direct practical interpretation.

But negative vector components are perfectly valid mathematically.

The geometric meaning is more useful.

A negative scalar reverses the direction of a nonzero vector.

We'll visualize that shortly.

---

# 7. WHAT HAPPENS WHEN WE MULTIPLY BY ZERO?

Suppose:

R = [80, 70, 90]

Multiply by zero.

0R = [0, 0, 0]

The result is the zero vector.

Remember Chapter 2:

A zero vector contains zeros in every position.

### Mathematical rule

For any vector V:

0V = 0

Here, the zero on the right represents the zero vector of the appropriate dimension.

### Important distinction

Multiplying a three-dimensional vector by zero produces a three-dimensional zero vector.

The dimension doesn't disappear.

For example:

0 × [80, 70, 90] = [0, 0, 0]

---

# 8. GEOMETRIC MEANING OF SCALAR MULTIPLICATION

Until now, we've treated vectors as collections of marks.

But vectors can also be represented as arrows.

To visualize this, let's temporarily use only Maths and Physics.

Suppose we have:

V = [3, 2]

Interpretation:

Move three units right and two units up from the origin.

Now multiply by 2.

2V = [6, 4]

The new vector moves six units right and four units up.

It points in the same direction.

But it is twice as long.

Now multiply by 0.5.

0.5V = [1.5, 1]

The vector still points in the same direction.

But its length is half the original.

Now multiply by -1.

-V = [-3, -2]

This points in the opposite direction.

### Three important cases

**Positive scalar:**

The direction remains the same, and the length is multiplied by the scalar.

**Negative scalar:**

The direction reverses, and the length is multiplied by the scalar's absolute value.

**Zero scalar:**

The result is a zero vector, which has no defined direction.

This geometric understanding is essential for later chapters.

---

# 9. PROPERTIES OF SCALAR MULTIPLICATION

Let's learn four properties.

## Property 1: Multiplying by 1

For any vector V:

1V = V

Example:

1[80, 70, 90] = [80, 70, 90]

Nothing changes.

## Property 2: Multiplying by 0

For any vector V:

0V = 0 vector

Example:

0[80, 70, 90] = [0, 0, 0]

## Property 3: Distributive property

For vectors A and B:

k(A+B) = kA + kB

Let's verify with two students.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Take k = 2.

First method:

R + P = [140, 155, 165]

Multiply by 2:

2(R+P) = [280, 310, 330]

Second method:

2R = [160, 140, 180]

2P = [120, 170, 150]

Add them:

2R + 2P = [280, 310, 330]

Both results match.

Therefore:

2(R+P) = 2R + 2P

## Property 4: Associativity

For scalars a and b:

a(bV) = (ab)V

Example:

V = [2, 3]

a = 2

b = 3

First:

2(3V)

3V = [6, 9]

2[6, 9] = [12, 18]

Second:

(2×3)V = 6V

6[2, 3] = [12, 18]

Both produce the same result.

---

# 10. PRACTICAL APPLICATION: CALCULATING AVERAGE MARKS

Now let's use scalar multiplication for something meaningful.

Suppose Rahul and Priya scored:

R = [80, 70, 90]

P = [60, 85, 75]

We want to calculate their average marks in each subject.

### Step 1: Add the vectors

R + P = [140, 155, 165]

### Step 2: Divide every component by 2

We could write:

(R + P) / 2

But dividing by 2 is equivalent to multiplying by 0.5.

Therefore:

Average = 0.5(R + P)

= [70, 77.5, 82.5]

### Interpretation

Average Maths marks = 70

Average Physics marks = 77.5

Average Chemistry marks = 82.5

We used:

1. Vector addition.
2. Scalar multiplication.

And obtained a new vector representing the subject-wise averages of two students.

---

# 11. AVERAGE MARKS OF ALL FIVE STUDENTS

Let's calculate the average marks for the entire class.

Our vectors:

R = [80, 70, 90]

P = [60, 85, 75]

A = [90, 80, 85]

S = [50, 65, 70]

K = [75, 90, 80]

### Step 1: Add all vectors

R + P + A + S + K

Calculate Maths total:

80 + 60 + 90 + 50 + 75 = 355

Physics total:

70 + 85 + 80 + 65 + 90 = 390

Chemistry total:

90 + 75 + 85 + 70 + 80 = 400

Combined vector:

[355, 390, 400]

### Step 2: Divide by the number of students

There are five students.

Average vector:

(1/5)[355, 390, 400]

= [71, 78, 80]

### Final answer

The class averages are:

Maths = 71

Physics = 78

Chemistry = 80

### Why is this important?

Instead of calculating each subject's average separately, we can express the entire calculation with vector operations.

This is one reason vectors are useful for working with numerical datasets.

---

# 12. WHAT IS VECTOR MAGNITUDE?

So far, we've learned how to:

- Add vectors.
- Subtract vectors.
- Multiply vectors by scalars.

Now comes a new question.

**How do we calculate the length of a vector?**

Imagine a two-dimensional vector:

V = [3, 4]

It represents movement:

3 units right.

4 units up.

If we draw an arrow from the origin to (3,4), how long is the arrow?

That arrow's length is called the magnitude of the vector.

### Definition

The magnitude of a vector is its geometric length.

For ordinary Euclidean vectors, this is calculated using the Pythagorean theorem.

---

# 13. RECAP: PYTHAGOREAN THEOREM

Before calculating magnitude, let's revisit basic geometry.

Imagine a right-angled triangle.

It has:

- One horizontal side.
- One vertical side.
- One slanted side called the hypotenuse.

The Pythagorean theorem says:

Hypotenuse² = Horizontal side² + Vertical side²

If we call:

Horizontal side = a

Vertical side = b

Hypotenuse = c

Then:

c² = a² + b²

Taking the square root:

c = √(a² + b²)

### Example

a = 3

b = 4

Then:

c = √(3² + 4²)

= √(9 + 16)

= √25

= 5

So a right triangle with sides 3 and 4 has a hypotenuse of length 5.

This is exactly the mathematics we need for vector magnitude.

---

# 14. MAGNITUDE OF A TWO-DIMENSIONAL VECTOR

Suppose:

V = [3, 4]

Its horizontal component is 3.

Its vertical component is 4.

The vector forms the hypotenuse of a right triangle.

Therefore:

Magnitude = √(3² + 4²)

= √(9 + 16)

= √25

= 5

We represent magnitude using double vertical bars:

||V||

Read:

"Norm of V."

For Euclidean magnitude:

||V|| = √(3² + 4²)

||V|| = 5

### Important

The original vector is:

V = [3, 4]

Its magnitude is:

||V|| = 5

The vector contains two components.

The magnitude is a single nonnegative number.

Therefore, the magnitude is a scalar.

---

# 15. GENERAL FORMULA FOR VECTOR MAGNITUDE

Suppose:

V = [x, y]

Then:

||V|| = √(x² + y²)

For a three-dimensional vector:

V = [x, y, z]

The formula becomes:

||V|| = √(x² + y² + z²)

For an n-dimensional vector:

V = [v₁, v₂, ..., vₙ]

Then:

||V||₂ = √(v₁² + v₂² + ... + vₙ²)

The subscript 2 identifies the Euclidean norm, also called the L2 norm.

The same principle extends to any finite number of dimensions.

---

# 16. MAGNITUDE OF RAHUL'S MARKS VECTOR

Now let's use our school example.

Rahul:

R = [80, 70, 90]

Calculate:

||R||₂ = √(80² + 70² + 90²)

Step 1: Square every component.

80² = 6400

70² = 4900

90² = 8100

Step 2: Add the squared values.

6400 + 4900 + 8100 = 19400

Step 3: Take the square root.

||R||₂ = √19400

||R||₂ ≈ 139.28

### What does 139.28 mean?

It is the length of Rahul's marks vector in the three-dimensional coordinate space defined by Maths, Physics, and Chemistry.

It is NOT:

- His total marks.
- His average marks.
- His percentage.

Let's compare.

Total marks:

80 + 70 + 90 = 240

Average marks:

240 / 3 = 80

Euclidean magnitude:

√(80² + 70² + 90²) ≈ 139.28

These are three different calculations.

**A vector's magnitude is not the same as the sum or average of its components.**

---

# 17. WHY DO WE SQUARE COMPONENTS?

Let's consider:

V = [-3, 4]

Suppose we simply add the components.

-3 + 4 = 1

But does that mean the arrow has length 1?

No.

The vector moves:

3 units left.

4 units up.

Its geometric length is still 5.

Calculate using magnitude:

||V||₂ = √((-3)² + 4²)

= √(9 + 16)

= 5

Squaring the coordinates avoids cancellation between negative and positive components and follows from the Pythagorean theorem.

### Important distinction

We don't square merely because negative numbers are inconvenient.

We square because Euclidean length is derived from geometry.

Other ways of measuring vector size also exist.

We'll examine those next.

---

# 18. WHAT IS A NORM?

We've just calculated vector magnitude.

But you may encounter another term:

**Norm.**

A norm is a mathematical function that measures the size or length of a vector according to certain rules.

The familiar Euclidean length is one type of norm.

Different norms measure vector size differently.

The three norms we'll study are:

1. L1 norm.
2. L2 norm.
3. L-infinity norm.

Let's understand them using the same vector.

V = [3, -4]

---

# 19. L1 NORM — MANHATTAN NORM

## 19.1 What is the L1 norm?

The L1 norm is the sum of the absolute values of the vector's components.

Formula:

||V||₁ = |v₁| + |v₂| + ... + |vₙ|

The vertical bars |x| mean absolute value.

Absolute value measures how far a number is from zero on the number line.

Examples:

|5| = 5

|-5| = 5

|0| = 0

### Example

V = [3, -4]

L1 norm:

||V||₁ = |3| + |-4|

= 3 + 4

= 7

### Interpretation

The L1 norm adds the absolute movement along each coordinate axis.

It is sometimes called the Manhattan norm because it resembles travelling along a rectangular street grid.

For example:

Move 3 blocks horizontally.

Move 4 blocks vertically.

Total movement along the grid = 7 blocks.

This is different from travelling diagonally in a straight line.

---

# 20. L2 NORM — EUCLIDEAN NORM

The L2 norm is the familiar geometric length.

Formula:

||V||₂ = √(v₁² + v₂² + ... + vₙ²)

Example:

V = [3, -4]

Calculate:

||V||₂ = √(3² + (-4)²)

= √(9 + 16)

= 5

### Interpretation

The L2 norm measures the straight-line length from the origin to the endpoint of the vector.

It comes from Euclidean geometry.

### Important

When someone says "vector magnitude" in introductory linear algebra, they usually mean the Euclidean or L2 norm.

---

# 21. L-INFINITY NORM

The L-infinity norm is different.

Instead of adding components or combining squares, it finds the largest absolute component.

Formula:

||V||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

Example:

V = [3, -4]

Absolute values:

[3, 4]

Largest value = 4

Therefore:

||V||∞ = 4

### Another example

V = [10, -25, 7]

Absolute values:

[10, 25, 7]

Maximum = 25

Therefore:

||V||∞ = 25

### Interpretation

The L-infinity norm measures the largest absolute component of a vector.

---

# 22. COMPARING L1, L2, AND L-INFINITY

Let's use:

V = [3, -4]

L1:

||V||₁ = 3 + 4 = 7

L2:

||V||₂ = √(9+16) = 5

L-infinity:

||V||∞ = max(3,4) = 4

Same vector.

Three different measurements.

| Norm | Calculation | Result |
| ---- | ----------- | -----: |
| L1   | 3 + 4       |      7 |
| L2   | √(9+16)     |      5 |
| L∞   | max(3,4)    |      4 |

### Which one is correct?

All three are correct according to their definitions.

They measure vector size using different rules.

### Which is most commonly used?

The L2 norm is the familiar straight-line length in Euclidean geometry.

But L1 and L-infinity norms are also important in mathematics and machine learning.

---

# 23. PROPERTIES OF NORMS

A valid norm must satisfy certain mathematical properties.

Let's understand them intuitively.

## Property 1: Non-negativity

A norm can never be negative.

||V|| ≥ 0

Why?

Because a length or size cannot be negative.

Example:

V = [-3, 4]

Its L2 norm is:

5

Not -5.

## Property 2: Zero condition

The norm of a vector is zero if and only if the vector is the zero vector.

||V|| = 0 exactly when V = 0

Example:

V = [0, 0, 0]

Its norm is zero.

But:

V = [0, 0, 2]

has a positive norm.

## Property 3: Scaling

For a scalar k:

||kV|| = |k| ||V||

Let's verify.

V = [3, 4]

||V||₂ = 5

Now multiply by 2.

2V = [6, 8]

||2V||₂ = √(36+64)

= √100

= 10

And:

2 × ||V||₂ = 2 × 5 = 10

Both results match.

What if the scalar is -2?

-2V = [-6, -8]

Its norm is still 10.

That's why the formula uses |k|.

## Property 4: Triangle inequality

For vectors A and B:

||A+B|| ≤ ||A|| + ||B||

This says that the length of the combined displacement cannot exceed the sum of the individual lengths.

Imagine walking from one location to another.

The direct distance between two points cannot be greater than the length of a route that connects them through an intermediate point.

This principle is called the triangle inequality.

---

# 24. WHAT IS DISTANCE BETWEEN TWO VECTORS?

Now let's answer the question we introduced in Chapter 2.

How do we measure how far apart two students are based on their marks?

Let's temporarily use only Maths and Physics.

Rahul:

R = [80, 70]

Priya:

P = [60, 85]

Step 1: Subtract the vectors.

R - P

= [80-60, 70-85]

= [20, -15]

Step 2: Calculate the magnitude of this difference.

Distance(R,P) = ||R-P||₂

= √(20² + (-15)²)

= √(400 + 225)

= √625

= 25

### Final answer

Euclidean distance between Rahul and Priya = 25.

### What does 25 mean?

It means their two-dimensional marks vectors are separated by 25 units in the Maths-Physics coordinate space.

It does not mean Rahul scored 25 marks more than Priya.

Their individual differences were:

Maths = +20

Physics = -15

The Euclidean distance combines these differences into one nonnegative number.

---

# 25. EUCLIDEAN DISTANCE FORMULA

Suppose:

A = [a₁, a₂]

B = [b₁, b₂]

Then:

Distance(A,B) = √((a₁-b₁)² + (a₂-b₂)²)

For three-dimensional vectors:

Distance(A,B) = √((a₁-b₁)² + (a₂-b₂)² + (a₃-b₃)²)

For n-dimensional vectors:

Distance(A,B) = √(Σᵢ₌₁ⁿ (aᵢ-bᵢ)²)

Don't worry about the summation symbol.

It simply means:

Calculate each squared difference, add all the results, and take the square root.

---

# 26. DISTANCE USING ALL THREE SUBJECTS

Now let's calculate the full distance between Rahul and Priya.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

### Step 1: Subtract

R - P

= [20, -15, 15]

### Step 2: Square the differences

20² = 400

(-15)² = 225

15² = 225

### Step 3: Add

400 + 225 + 225 = 850

### Step 4: Square root

Distance(R,P) = √850

≈ 29.15

### Result

Rahul and Priya are approximately 29.15 units apart in our three-dimensional marks space.

---

# 27. MANHATTAN DISTANCE BETWEEN STUDENTS

Now let's use the L1 norm to calculate a different type of distance.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Difference:

R - P = [20, -15, 15]

Manhattan distance:

d₁(R,P) = ||R-P||₁

= |20| + |-15| + |15|

= 20 + 15 + 15

= 50

Therefore:

Manhattan distance = 50

Compare:

Euclidean distance ≈ 29.15

Manhattan distance = 50

Both describe differences between the same two students, but according to different distance rules.

Neither value is inherently more correct.

---

# 28. L-INFINITY DISTANCE BETWEEN STUDENTS

Again:

R - P = [20, -15, 15]

Calculate absolute differences:

[20, 15, 15]

Take the largest:

max(20,15,15) = 20

Therefore:

L-infinity distance = 20

### What does this mean?

The largest subject-wise difference between Rahul and Priya is 20 marks.

That's their Maths difference.

This gives the L-infinity distance a very intuitive interpretation in our example.

---

# 29. WHY DO DIFFERENT DISTANCE MEASURES EXIST?

Consider three students represented by two-subject vectors.

Student A:

[0, 0]

Student B:

[3, 4]

Student C:

[5, 0]

These are simplified score coordinates.

Let's compare distances from A.

### Euclidean distance

A to B:

√(3²+4²) = 5

A to C:

√(5²+0²) = 5

Both are equally distant using Euclidean distance.

### Manhattan distance

A to B:

3+4 = 7

A to C:

5+0 = 5

Under Manhattan distance, C is closer.

### What does this demonstrate?

Different distance measures can produce different notions of similarity or closeness.

This matters because not every real-world problem should use the same measurement.

The appropriate distance depends on what differences we consider important.

---

# 30. A SUBTLE BUT IMPORTANT POINT: FEATURE SCALES

Suppose we compare students using:

- Maths marks out of 100.
- Attendance percentage out of 100.

These are numerically similar scales.

But what if we instead record attendance as a fraction between 0 and 1?

Then:

Student A = [80, 0.90]

Student B = [70, 0.80]

Difference:

[10, 0.10]

When calculating Euclidean distance, the Maths difference of 10 will dominate the attendance difference of 0.10.

That doesn't automatically mean Maths is more important.

It may simply mean the variables use different numerical scales.

### Why mention this now?

Because distance depends on the coordinates and their units.

When we eventually study machine learning, we'll learn ways to put features on comparable scales.

For now, just understand this principle:

**Distances between numerical records can be strongly influenced by how those numbers are measured.**

---

# 31. COMMON BEGINNER MISTAKES

## Mistake 1: Scalar multiplication changes vector dimension

Incorrect.

Multiplying a three-dimensional vector by a scalar produces another three-dimensional vector.

## Mistake 2: Magnitude equals sum of components

Incorrect.

For:

V = [3,4]

Sum of components = 7

L2 magnitude = 5

## Mistake 3: Magnitude can be negative

Incorrect.

A norm is always nonnegative.

## Mistake 4: L1 and L2 norms are identical

Incorrect.

They measure vector size differently.

## Mistake 5: Vector subtraction directly produces a scalar distance

Incorrect.

Subtraction produces another vector.

Applying a norm to the difference produces a scalar distance.

## Mistake 6: Euclidean distance tells us which student performed better

Incorrect.

Euclidean distance tells us how far apart their numerical records are.

It does not directly determine which student's performance is better.

## Mistake 7: A large norm always means a better student

Incorrect.

The norm combines components according to a mathematical rule.

Whether that corresponds to better performance depends on what the components represent and how the problem is defined.

---

# 32. INTERVIEW QUESTIONS AND ANSWERS

### Q1. What is scalar multiplication?

Scalar multiplication multiplies every component of a vector by a single scalar.

### Q2. Does scalar multiplication change the dimension of a vector?

No. It changes component values but preserves the vector's dimension.

### Q3. What happens when a vector is multiplied by a negative scalar?

Its direction reverses, and its magnitude is multiplied by the absolute value of the scalar.

### Q4. What is vector magnitude?

Magnitude is the geometric length of a vector, usually measured using the Euclidean norm.

### Q5. What is the L1 norm?

The sum of the absolute values of a vector's components.

### Q6. What is the L2 norm?

The square root of the sum of squared components.

### Q7. What is the L-infinity norm?

The maximum absolute component of a vector.

### Q8. What is the difference between a vector and its norm?

A vector contains ordered numerical components.

Its norm is a single nonnegative scalar measuring its size.

### Q9. What is Euclidean distance?

The L2 norm of the difference between two vectors.

### Q10. Can two vectors have the same Euclidean magnitude but different directions?

Yes.

For example:

A = [3,4]

B = [-3,4]

Both have magnitude 5 but point in different directions.

### Q11. Can two observations have zero Euclidean distance?

Yes, when their corresponding numerical components are all identical.

### Q12. Why can changing the scale of a feature affect distance?

Because the numerical differences in that feature change, altering its contribution to the distance calculation.

---

# 33. IMPORTANT FORMULAS

### Scalar Multiplication

kV = [kv₁, kv₂, ..., kvₙ]

### L1 Norm

||V||₁ = Σ|vᵢ|

### L2 Norm

||V||₂ = √(Σvᵢ²)

### L-infinity Norm

||V||∞ = max(|vᵢ|)

### Euclidean Distance

d₂(A,B) = ||A-B||₂

### Manhattan Distance

d₁(A,B) = ||A-B||₁

### L-infinity Distance

d∞(A,B) = ||A-B||∞

### Norm Scaling Property

||kV|| = |k| ||V||

### Triangle Inequality

||A+B|| ≤ ||A|| + ||B||

---

# 34. COMPLETE CHAPTER SUMMARY

We started with a simple task: multiplying marks by a number.

That taught us scalar multiplication.

Then we asked how long a vector is.

That introduced magnitude.

We learned that vector magnitude can be measured using a norm.

We studied:

- L1 norm.
- L2 norm.
- L-infinity norm.

Next, we applied these norms to the difference between two students' marks.

That gave us different distance measures.

Our most important discovery was:

**Subtracting vectors gives a difference vector. Taking the norm of that difference gives a distance.**

For Rahul and Priya:

R = [80,70,90]

P = [60,85,75]

Difference:

R-P = [20,-15,15]

Euclidean distance:

√850 ≈ 29.15

Manhattan distance:

50

L-infinity distance:

20

We also learned that different distance measures can produce different comparisons.

These concepts will help us understand geometric relationships between vectors.

---

END OF CHAPTER 3

NEXT CHAPTER: DOT PRODUCT AND ANGLES BETWEEN VECTORS
