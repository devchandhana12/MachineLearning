# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 4 — DOT PRODUCT AND ANGLES BETWEEN VECTORS

---

# 1. RECAP OF OUR SCHOOL DATASET

We're continuing with the same five students.

| Student | Maths | Physics | Chemistry |
| ------- | ----: | ------: | --------: |
| Rahul   |    80 |      70 |        90 |
| Priya   |    60 |      85 |        75 |
| Arjun   |    90 |      80 |        85 |
| Sneha   |    50 |      65 |        70 |
| Kiran   |    75 |      90 |        80 |

Each student is represented by a vector.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Arjun:

A = [90, 80, 85]

In Chapter 3, we learned:

**Vector magnitude**

For V = [3, 4]:

||V||₂ = √(3² + 4²) = 5

**Euclidean distance**

For two vectors A and B:

d(A,B) = ||A-B||₂

Remember the distinction:

- A vector contains components.
- Its magnitude is one nonnegative number.
- Distance measures separation between two vectors.

We'll use all three ideas in this chapter.

---

# 2. WHY DO WE NEED THE DOT PRODUCT?

Imagine our school introduces an academic competition.

The competition gives different importance to the three subjects.

For every Maths mark, a student receives 2 competition points.

For every Physics mark, a student receives 3 points.

For every Chemistry mark, a student receives 1 point.

We want to calculate Rahul's competition score.

His marks are:

R = [80, 70, 90]

The points per mark are:

C = [2, 3, 1]

Let's calculate the score manually.

Maths contribution:

80 × 2 = 160

Physics contribution:

70 × 3 = 210

Chemistry contribution:

90 × 1 = 90

Total competition score:

160 + 210 + 90 = 460

We just performed three multiplications and added the results.

What if there were 50 subjects?

We would have to repeat the same pattern 50 times.

Linear algebra gives us a compact operation for this calculation.

That operation is called the **dot product**.

---

# 3. WHAT IS A DOT PRODUCT?

A dot product combines two vectors into a single number.

Let's use our example.

Rahul's marks:

R = [80, 70, 90]

Competition points per mark:

C = [2, 3, 1]

We write:

R · C

The dot symbol means dot product.

To calculate it:

Step 1: Multiply corresponding components.

80 × 2 = 160

70 × 3 = 210

90 × 1 = 90

Step 2: Add the products.

160 + 210 + 90 = 460

Therefore:

R · C = 460

### Definition

The dot product of two real-valued vectors of the same dimension is the sum of the products of their corresponding components.

### Important difference from vector addition

Vector addition:

[2, 3] + [4, 5] = [6, 8]

The result is another vector.

Dot product:

[2, 3] · [4, 5]

= 2×4 + 3×5

= 8 + 15

= 23

The result is a scalar.

**Dot product: Vector × Vector → Scalar**

This is our first major distinction.

---

# 4. GENERAL FORMULA FOR THE DOT PRODUCT

Suppose we have two vectors:

A = [a₁, a₂, a₃]

B = [b₁, b₂, b₃]

Their dot product is:

A · B = a₁b₁ + a₂b₂ + a₃b₃

For vectors containing n components:

A · B = a₁b₁ + a₂b₂ + ... + aₙbₙ

You may also see this written using summation:

A · B = Σᵢ₌₁ⁿ aᵢbᵢ

Let's understand the symbols.

Σ means add the terms.

i represents the current component position.

n represents the number of components.

aᵢbᵢ means multiply the corresponding components at position i.

### Example

A = [1, 2, 3]

B = [4, 5, 6]

Then:

A · B = (1×4) + (2×5) + (3×6)

= 4 + 10 + 18

= 32

That's all the dot product calculation requires.

---

# 5. WHAT IF THE COMPONENTS ARE NEGATIVE?

The calculation doesn't change.

Suppose:

A = [3, -2]

B = [4, 5]

Calculate:

A · B = (3×4) + (-2×5)

= 12 - 10

= 2

Now consider:

A = [3, -4]

B = [-2, 5]

Dot product:

A · B = (3×-2) + (-4×5)

= -6 - 20

= -26

### Important observation

A dot product can be:

- Positive
- Negative
- Zero

Unlike magnitude, it is not always nonnegative.

Later, we'll discover that the sign of the dot product tells us something about the angle between vectors.

---

# 6. DOT PRODUCT VS ELEMENT-WISE MULTIPLICATION

This is a common confusion.

Suppose:

A = [2, 3]

B = [4, 5]

### Element-wise multiplication

Multiply corresponding components but don't add them.

Result:

[2×4, 3×5]

= [8, 15]

The result is a vector.

### Dot product

Multiply corresponding components and then add.

A · B = 8 + 15

= 23

The result is a scalar.

### Comparison

| Operation                   | Result  |
| --------------------------- | ------- |
| Vector addition             | [6, 8]  |
| Element-wise multiplication | [8, 15] |
| Dot product                 | 23      |

These are three different operations.

---

# 7. CALCULATING COMPETITION SCORES USING DOT PRODUCTS

Now let's return to our five students.

Competition scoring vector:

C = [2, 3, 1]

This means:

Maths: 2 points per mark

Physics: 3 points per mark

Chemistry: 1 point per mark

### Rahul

R = [80, 70, 90]

R · C = 80×2 + 70×3 + 90×1

= 160 + 210 + 90

= 460

### Priya

P = [60, 85, 75]

P · C = 60×2 + 85×3 + 75×1

= 120 + 255 + 75

= 450

### Arjun

A = [90, 80, 85]

A · C = 90×2 + 80×3 + 85×1

= 180 + 240 + 85

= 505

### Sneha

S = [50, 65, 70]

S · C = 50×2 + 65×3 + 70×1

= 100 + 195 + 70

= 365

### Kiran

K = [75, 90, 80]

K · C = 75×2 + 90×3 + 80×1

= 150 + 270 + 80

= 500

Final results:

| Student | Competition Score |
| ------- | ----------------: |
| Rahul   |               460 |
| Priya   |               450 |
| Arjun   |               505 |
| Sneha   |               365 |
| Kiran   |               500 |

Arjun has the highest competition score.

### What did the dot product accomplish?

It converted three subject marks into one combined score using the competition's scoring rules.

We supplied:

One marks vector.

One points-per-mark vector.

The dot product returned one numerical score.

We'll eventually see that this type of calculation appears repeatedly in mathematics and machine learning.

---

# 8. IMPORTANT PROPERTIES OF THE DOT PRODUCT

Let's understand four important properties.

## 8.1 Commutative property

A · B = B · A

Example:

A = [2, 3]

B = [4, 5]

A · B = 2×4 + 3×5 = 23

B · A = 4×2 + 5×3 = 23

Both results match.

For real vectors, the dot product is commutative.

## 8.2 Distributive property

A · (B+C) = A·B + A·C

Let's verify.

A = [1, 2]

B = [3, 4]

C = [5, 6]

First method:

B+C = [8, 10]

A · (B+C) = 1×8 + 2×10

= 28

Second method:

A·B = 1×3 + 2×4 = 11

A·C = 1×5 + 2×6 = 17

Add:

11+17 = 28

Both methods produce 28.

## 8.3 Scalar multiplication property

For a scalar k:

(kA)·B = k(A·B)

Example:

A = [2, 3]

B = [4, 5]

k = 2

First calculate:

A·B = 23

Then:

2(A·B) = 46

Alternatively:

2A = [4, 6]

(2A)·B = 4×4 + 6×5

= 16 + 30

= 46

The results match.

## 8.4 Dot product of a vector with itself

For any real vector A:

A·A = a₁² + a₂² + ... + aₙ²

Example:

A = [3, 4]

A·A = 3×3 + 4×4

= 9+16

= 25

Notice something.

In Chapter 3:

||A||₂ = √(3²+4²) = 5

Therefore:

A·A = ||A||₂²

Or:

||A||₂ = √(A·A)

This is a very important relationship.

**We can calculate the Euclidean magnitude of a vector using its dot product with itself.**

---

# 9. WHY DOES THE DOT PRODUCT RELATE TO VECTOR LENGTH?

Let's derive this carefully.

Suppose:

A = [a₁, a₂]

We know the Euclidean magnitude is:

||A||₂ = √(a₁²+a₂²)

Now calculate its dot product with itself:

A·A = a₁a₁ + a₂a₂

= a₁² + a₂²

Take the square root:

√(A·A) = √(a₁²+a₂²)

But that is exactly the formula for magnitude.

Therefore:

||A||₂ = √(A·A)

This works for real vectors of any finite dimension.

### Example with Rahul

R = [80, 70, 90]

R·R = 80² + 70² + 90²

= 6400 + 4900 + 8100

= 19400

Therefore:

||R||₂ = √19400 ≈ 139.28

Exactly what we obtained in Chapter 3.

---

# 10. WHY DO WE NEED ANGLES BETWEEN VECTORS?

We've learned how to calculate a dot product.

Now let's understand its geometric meaning.

Imagine two students represented only by their Maths and Physics marks.

Rahul:

R = [80, 70]

Priya:

P = [60, 85]

We know we can represent them as points in a two-dimensional coordinate system.

We can also draw arrows from the origin to those points.

Now ask:

How similar are the directions of these arrows?

This is different from asking:

How far apart are the endpoints?

Distance and direction are related, but they measure different things.

### Consider another simple example

Vector A = [2, 2]

Vector B = [4, 4]

B is twice A.

They have different lengths.

But both point in exactly the same direction.

Their angle is zero degrees.

Now consider:

A = [2, 2]

C = [-2, -2]

C points in exactly the opposite direction.

The angle between them is 180 degrees.

We need a mathematical way to calculate these angles.

Surprisingly, the dot product gives us that ability.

---

# 11. UNDERSTANDING ANGLES

A full rotation is 360 degrees.

A straight angle is 180 degrees.

A right angle is 90 degrees.

When comparing two nonzero vectors, we normally use the smaller angle between them, ranging from 0° to 180°.

### Case 1: Same direction

Angle = 0°

### Case 2: Perpendicular

Angle = 90°

### Case 3: Opposite directions

Angle = 180°

Angles between these values describe intermediate directional relationships.

For two vectors drawn from the same origin, this angle tells us how their directions differ.

---

# 12. INTRODUCTION TO COSINE

To calculate angles using the dot product, we need one basic trigonometric concept.

Cosine.

Cosine is a function of an angle.

In a right-angled triangle:

cos(θ) = Adjacent side / Hypotenuse

For now, it's enough to understand that cosine maps an angle to a number.

Some important values:

| Angle |  Cosine |
| ----- | ------: |
| 0°    |       1 |
| 30°   | ≈ 0.866 |
| 45°   | ≈ 0.707 |
| 60°   |     0.5 |
| 90°   |       0 |
| 120°  |    -0.5 |
| 180°  |      -1 |

Notice the pattern.

As the angle increases from 0° to 180°, cosine decreases from 1 to -1.

This makes cosine useful for comparing vector directions.

---

# 13. GEOMETRIC FORMULA FOR THE DOT PRODUCT

We already know the numerical formula:

A·B = a₁b₁ + a₂b₂ + ... + aₙbₙ

There is another equivalent formula:

A·B = ||A||₂ ||B||₂ cos(θ)

Where:

A·B = Dot product.

||A||₂ = Magnitude of A.

||B||₂ = Magnitude of B.

θ = Angle between the two vectors.

cos(θ) = Cosine of that angle.

This formula connects numerical multiplication with geometric direction.

### What does it mean?

The dot product depends on:

1. The length of the first vector.
2. The length of the second vector.
3. The angle between them.

Two long vectors can have a large dot product.

But the angle also matters.

If the vectors are perpendicular, their dot product becomes zero regardless of their nonzero lengths.

Let's understand why.

---

# 14. DERIVING THE ANGLE FORMULA

We have:

A·B = ||A||₂ ||B||₂ cos(θ)

We want to calculate θ.

First divide both sides by the magnitudes:

cos(θ) = (A·B) / (||A||₂ ||B||₂)

Then apply inverse cosine:

θ = cos⁻¹((A·B)/(||A||₂ ||B||₂))

Here, cos⁻¹ means inverse cosine, also called arccos.

It does not mean 1/cos.

### Important condition

This formula requires both vectors to be nonzero.

Why?

The magnitude of a zero vector is zero.

We cannot divide by zero.

The zero vector also has no defined direction.

Therefore, the angle between a zero vector and another vector is undefined.

---

# 15. EXAMPLE: CALCULATE THE ANGLE BETWEEN TWO VECTORS

Let's use simple numbers first.

A = [1, 0]

B = [0, 1]

### Step 1: Dot product

A·B = 1×0 + 0×1

= 0

### Step 2: Magnitude of A

||A||₂ = √(1²+0²) = 1

### Step 3: Magnitude of B

||B||₂ = √(0²+1²) = 1

### Step 4: Calculate cosine

cos(θ) = 0/(1×1)

= 0

### Step 5: Find the angle

θ = cos⁻¹(0)

θ = 90°

Therefore, the vectors are perpendicular.

This introduces an important concept:

**Orthogonality.**

---

# 16. WHAT DOES ORTHOGONAL MEAN?

Orthogonal means perpendicular.

Two nonzero vectors are perpendicular if the angle between them is 90 degrees.

For real Euclidean vectors:

A·B = 0

means the vectors are perpendicular, provided both are nonzero.

More generally, vectors with zero dot product are called orthogonal, including the zero vector by mathematical convention.

### Example

A = [3, 0]

B = [0, 4]

Dot product:

A·B = 3×0 + 0×4

= 0

The vectors are orthogonal.

### Another example

A = [2, 3]

B = [3, -2]

Dot product:

A·B = 2×3 + 3×(-2)

= 6-6

= 0

Therefore, these vectors are also perpendicular.

Notice something important.

Perpendicular vectors do not have to align with the horizontal and vertical axes.

Any two nonzero vectors with zero dot product are perpendicular.

---

# 17. WHAT DOES A POSITIVE DOT PRODUCT MEAN?

Consider:

A = [1, 0]

B = [1, 1]

Calculate:

A·B = 1×1 + 0×1

= 1

The result is positive.

Calculate magnitudes:

||A||₂ = 1

||B||₂ = √2

Therefore:

cos(θ) = 1/√2

≈ 0.707

So:

θ ≈ 45°

The angle is less than 90 degrees.

### General principle

For nonzero real vectors:

If A·B > 0, the angle between them is acute.

An acute angle is smaller than 90 degrees.

This tells us that the vectors point in broadly similar directions.

---

# 18. WHAT DOES A NEGATIVE DOT PRODUCT MEAN?

Consider:

A = [1, 0]

B = [-1, 1]

Dot product:

A·B = 1×(-1) + 0×1

= -1

Magnitudes:

||A||₂ = 1

||B||₂ = √2

Therefore:

cos(θ) = -1/√2

≈ -0.707

Angle:

θ ≈ 135°

This is greater than 90 degrees.

### General principle

For nonzero real vectors:

If A·B < 0, the angle between them is obtuse.

An obtuse angle lies between 90° and 180°.

The vectors point in more opposing directions.

---

# 19. THREE IMPORTANT DOT PRODUCT CASES

For two nonzero vectors:

### Positive dot product

A·B > 0

Angle < 90°

Vectors point in broadly similar directions.

### Zero dot product

A·B = 0

Angle = 90°

Vectors are perpendicular.

### Negative dot product

A·B < 0

Angle > 90°

Vectors point in broadly opposing directions.

### Important warning

The sign alone doesn't tell us the exact angle.

For example, both 20° and 80° produce positive dot products.

To calculate the angle, we must also consider the vector magnitudes.

---

# 20. COSINE SIMILARITY

We learned:

cos(θ) = (A·B) / (||A||₂ ||B||₂)

When used to compare vectors, this quantity is called cosine similarity.

Formula:

Cosine Similarity(A,B) = (A·B) / (||A||₂ ||B||₂)

It measures how closely the directions of two nonzero vectors align.

### Range

Cosine similarity lies between:

-1 and +1

### Interpretation

+1 = Same direction.

0 = Perpendicular directions.

-1 = Opposite directions.

Values between these extremes indicate intermediate directional relationships.

### Important

Cosine similarity does not primarily measure the difference between vector lengths.

It compares direction after accounting for magnitude.

---

# 21. EXAMPLE: SAME DIRECTION, DIFFERENT MAGNITUDES

Suppose:

A = [2, 4]

B = [4, 8]

Clearly:

B = 2A

They point in the same direction.

Let's verify using cosine similarity.

### Step 1: Dot product

A·B = 2×4 + 4×8

= 8+32

= 40

### Step 2: Magnitude of A

||A||₂ = √(2²+4²)

= √20

≈ 4.472

### Step 3: Magnitude of B

||B||₂ = √(4²+8²)

= √80

≈ 8.944

### Step 4: Cosine similarity

cos(θ) = 40/(√20 × √80)

= 40/40

= 1

Therefore:

Cosine similarity = 1

Even though B is twice as long as A, they have perfect directional alignment.

---

# 22. COSINE SIMILARITY USING STUDENT MARKS

Now let's return to Rahul and Priya.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

We want to compare the directions of their marks vectors.

### Step 1: Dot product

R·P = (80×60) + (70×85) + (90×75)

= 4800 + 5950 + 6750

= 17500

### Step 2: Magnitude of Rahul

||R||₂ = √(80²+70²+90²)

= √19400

≈ 139.28

### Step 3: Magnitude of Priya

||P||₂ = √(60²+85²+75²)

= √(3600+7225+5625)

= √16450

≈ 128.26

### Step 4: Cosine similarity

cos(θ) = 17500/(√19400 × √16450)

≈ 0.9798

### Result

Cosine similarity ≈ 0.980

This is close to +1.

Their mark vectors point in similar directions.

### What does that mean here?

Both students have relatively high positive marks in all three subjects, and their overall subject profiles are directionally similar.

But be careful.

It does not mean their marks are identical.

They still have differences in individual subjects.

Cosine similarity and Euclidean distance answer different questions.

---

# 23. EUCLIDEAN DISTANCE VS COSINE SIMILARITY

Let's compare them.

Consider:

A = [20, 40]

B = [40, 80]

### Euclidean distance

A-B = [-20, -40]

Distance = √(20²+40²)

= √2000

≈ 44.72

The distance is not zero.

### Cosine similarity

B = 2A

Therefore:

Cosine similarity = 1

The vectors point in exactly the same direction.

### What did we discover?

Euclidean distance considers absolute coordinate differences.

Cosine similarity measures directional alignment.

Two vectors can have:

- Perfect cosine similarity.
- Nonzero Euclidean distance.

Neither metric is universally better.

The correct choice depends on what we want to compare.

---

# 24. WHY DO WE DIVIDE BY MAGNITUDES IN COSINE SIMILARITY?

This is an important conceptual question.

Suppose:

A = [1, 1]

B = [2, 2]

Dot product:

A·B = 4

Now multiply B by 10:

C = [20, 20]

Dot product:

A·C = 40

The dot product increased tenfold.

But did the direction change?

No.

B and C point in exactly the same direction.

Their lengths are different.

Therefore, the raw dot product depends on magnitude as well as direction.

If our goal is to compare direction alone, we need to remove the influence of length.

That's why cosine similarity divides the dot product by the product of the vector magnitudes.

This normalization makes cosine similarity unchanged when either vector is multiplied by a positive scalar.

### Important exception

If a vector is multiplied by a negative scalar, its direction reverses, so the sign of its cosine similarity reverses.

---

# 25. DOT PRODUCT WITH THE ZERO VECTOR

Suppose:

A = [3, 4]

Z = [0, 0]

Dot product:

A·Z = 3×0 + 4×0

= 0

This is perfectly valid.

But can we calculate cosine similarity?

Cosine similarity requires:

(A·Z) / (||A||₂ ||Z||₂)

Since:

||Z||₂ = 0

The denominator becomes zero.

Division by zero is undefined.

Therefore:

Cosine similarity with a zero vector is mathematically undefined.

### Important distinction

Dot product with a zero vector = 0.

Cosine similarity involving a zero vector = undefined.

These are not the same statement.

---

# 26. DOT PRODUCT AND ROW/COLUMN VECTORS

Remember Chapter 1?

A column vector is arranged vertically.

Suppose:

A =

[2]
[3]

And:

B =

[4]
[5]

We can transpose A into a row vector:

Aᵀ = [2, 3]

Now multiply:

AᵀB

= [2, 3] ×

[4]
[5]

= 2×4 + 3×5

= 23

Therefore:

A·B = AᵀB

This is the matrix-notation form of the dot product for real column vectors.

We'll study matrix multiplication in detail in a later chapter.

For now, remember that dot products can be expressed using transposes.

---

# 27. IMPORTANT MATHEMATICAL FACT: CAUCHY-SCHWARZ INEQUALITY

Here's a result that explains why cosine similarity always lies between -1 and +1.

For any two real vectors:

|A·B| ≤ ||A||₂ ||B||₂

This is called the Cauchy-Schwarz inequality.

We don't need to prove it yet.

Let's understand its consequence.

Divide both sides by the positive product of magnitudes, assuming both vectors are nonzero.

We obtain:

|(A·B)/(||A||₂||B||₂)| ≤ 1

But the expression inside the absolute value is cosine similarity.

Therefore:

-1 ≤ Cosine Similarity ≤ 1

This is not an arbitrary restriction.

It follows from a fundamental mathematical property of dot products.

---

# 28. DERIVING THE GEOMETRIC DOT PRODUCT FORMULA

Let's connect the numerical and geometric definitions.

Suppose two vectors A and B start at the origin.

The angle between them is θ.

The vector joining their endpoints is:

A-B

Using the Euclidean norm:

||A-B||₂² = (A-B)·(A-B)

Expand the dot product:

= A·A - A·B - B·A + B·B

Since:

A·B = B·A

We get:

||A-B||₂² = ||A||₂² + ||B||₂² - 2(A·B)

Now recall the law of cosines from geometry.

For a triangle formed by A, B, and A-B:

||A-B||₂² = ||A||₂² + ||B||₂² - 2||A||₂||B||₂cos(θ)

Both expressions describe the same squared length.

Compare them:

-2(A·B) = -2||A||₂||B||₂cos(θ)

Divide by -2:

A·B = ||A||₂||B||₂cos(θ)

That's our geometric dot product formula.

### Why is this useful?

We didn't invent a separate rule for angles.

The dot product formula emerges naturally from the geometry of vectors.

You don't have to memorize the proof immediately, but understand the connection:

**Vector subtraction → Distance → Geometry → Dot product → Angle**

All the concepts we've learned are related.

---

# 29. COMMON BEGINNER MISTAKES

## Mistake 1: Dot product produces another vector

Incorrect.

The dot product of two real coordinate vectors produces a scalar.

## Mistake 2: Dot product and element-wise multiplication are identical

Incorrect.

Element-wise multiplication preserves the individual products.

Dot product adds those products together.

## Mistake 3: Dot product is always positive

Incorrect.

It can be positive, negative, or zero.

## Mistake 4: Zero dot product always means two vectors have a 90-degree angle

Not quite.

For two nonzero real vectors, zero dot product means they are perpendicular.

But the zero vector has zero dot product with every vector and no defined direction or angle.

## Mistake 5: Cosine similarity measures absolute distance

Incorrect.

It measures directional alignment.

## Mistake 6: Perfect cosine similarity means two vectors are identical

Incorrect.

A = [2, 4]

B = [4, 8]

They are not identical, but their cosine similarity is 1.

## Mistake 7: Cosine similarity can always be calculated

Incorrect.

It is undefined if either vector has zero magnitude.

## Mistake 8: Large dot product means vectors have similar directions

Not necessarily.

Large magnitudes can produce large dot products even when directional alignment is not especially close.

Cosine similarity removes the influence of positive scaling.

---

# 30. INTERVIEW QUESTIONS AND ANSWERS

### Q1. What is the dot product?

The dot product is the sum of products of corresponding components of two real vectors of the same dimension.

### Q2. What does the dot product return?

A scalar.

### Q3. Is the dot product commutative?

Yes, for real vectors:

A·B = B·A

### Q4. How is the dot product related to magnitude?

A·A = ||A||₂²

### Q5. What does a zero dot product mean?

The vectors are orthogonal. If both are nonzero, their angle is 90 degrees.

### Q6. What is cosine similarity?

The dot product divided by the product of the Euclidean magnitudes of the two nonzero vectors.

### Q7. What is the range of cosine similarity?

From -1 to +1.

### Q8. What is the difference between dot product and cosine similarity?

The dot product depends on both vector magnitude and directional alignment.

Cosine similarity normalizes by magnitudes to compare direction.

### Q9. Can cosine similarity be 1 for different vectors?

Yes, if one nonzero vector is a positive scalar multiple of the other.

### Q10. Can Euclidean distance be large while cosine similarity is 1?

Yes.

For example:

A = [1, 1]

B = [100, 100]

They point in the same direction, but their endpoints are far apart.

### Q11. What happens when cosine similarity is -1?

The nonzero vectors point in exactly opposite directions.

### Q12. Why is cosine similarity undefined for zero vectors?

Because its formula divides by the product of vector magnitudes, and the zero vector has magnitude zero.

### Q13. What is orthogonality?

Two vectors are orthogonal when their dot product is zero. For nonzero Euclidean vectors, this means they are perpendicular.

### Q14. Can a dot product be calculated between vectors of different dimensions?

Not using the ordinary coordinate dot product.

Both vectors must have matching dimensions.

### Q15. What mathematical theorem restricts cosine similarity to [-1,1]?

The Cauchy-Schwarz inequality.

---

# 31. IMPORTANT FORMULAS

### Dot Product

A·B = Σᵢ₌₁ⁿ aᵢbᵢ

### Magnitude from Dot Product

||A||₂ = √(A·A)

### Geometric Dot Product

A·B = ||A||₂||B||₂cos(θ)

### Cosine Similarity

cos_sim(A,B) = (A·B)/(||A||₂||B||₂)

### Angle Between Vectors

θ = cos⁻¹((A·B)/(||A||₂||B||₂))

### Orthogonality

A·B = 0

### Cauchy-Schwarz Inequality

|A·B| ≤ ||A||₂||B||₂

### Matrix Notation

A·B = AᵀB

For real column vectors of the same dimension.

---

# 32. COMPLETE CHAPTER SUMMARY

We started with our school's academic competition.

Each subject contributed a different number of competition points.

We learned that multiplying corresponding components and adding the products produces a single score.

That operation is the dot product.

Then we discovered that the dot product also has a geometric interpretation.

It depends on:

1. The magnitude of the first vector.
2. The magnitude of the second vector.
3. The angle between them.

This led us to cosine similarity.

Cosine similarity compares directions while removing the influence of positive vector scaling.

We also learned:

- Positive dot products correspond to acute angles for nonzero vectors.
- Zero dot products correspond to perpendicular vectors.
- Negative dot products correspond to obtuse angles.
- Cosine similarity ranges from -1 to +1.
- The zero vector has no defined cosine similarity.
- Dot products can calculate vector magnitude.
- Euclidean distance and cosine similarity measure different relationships.

### THE BIG PICTURE

Chapter 1: We learned how to represent numbers using vectors.

Chapter 2: We learned to add and subtract vectors.

Chapter 3: We learned scalar multiplication, magnitude, norms, and distance.

Chapter 4: We learned dot products, angles, orthogonality, and cosine similarity.

Every chapter builds on the previous one.

---

END OF CHAPTER 4

NEXT CHAPTER: LINEAR COMBINATIONS, SPAN, AND LINEAR INDEPENDENCE
