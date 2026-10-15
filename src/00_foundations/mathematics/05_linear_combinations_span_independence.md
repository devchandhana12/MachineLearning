# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 5 — LINEAR COMBINATIONS, SPAN & LINEAR INDEPENDENCE

---

# 1. RECAP: WHAT DO WE ALREADY KNOW?

In previous chapters, we learned:

Chapter 1:

- Scalars
- Vectors
- Dimensions
- Matrices

Chapter 2:

- Vector addition
- Vector subtraction
- Zero vectors

Chapter 3:

- Scalar multiplication
- Vector magnitude
- Norms
- Distance

Chapter 4:

- Dot products
- Angles between vectors
- Orthogonality
- Cosine similarity

We can now perform the following operations.

Vector addition:

[2, 3] + [4, 5] = [6, 8]

Scalar multiplication:

3[2, 3] = [6, 9]

Our next topic combines these two operations.

---

# 2. WHAT IS A LINEAR COMBINATION?

Let's begin with our school example.

Rahul scored:

R = [80, 70, 90]

Priya scored:

P = [60, 85, 75]

Our component order is always:

[Maths, Physics, Chemistry]

Imagine we want to construct a new vector by combining their subject marks.

We decide to take:

- Twice Rahul's marks vector.
- Once Priya's marks vector.

Mathematically:

2R + P

Let's calculate it.

## Step 1: Multiply Rahul's vector by 2

R = [80, 70, 90]

2R = [160, 140, 180]

## Step 2: Take Priya's vector

P = [60, 85, 75]

## Step 3: Add both vectors

2R + P

= [160, 140, 180] + [60, 85, 75]

= [220, 225, 255]

Therefore:

2R + P = [220, 225, 255]

We created a new vector using two existing vectors.

The operation involved:

1. Scalar multiplication.
2. Vector addition.

**This is called a linear combination.**

### Definition

A linear combination is a vector formed by multiplying existing vectors by scalars and adding the results.

The resulting values do not have to represent valid exam marks out of 100. We're studying mathematical combinations, not creating an actual student's report card.

---

# 3. GENERAL FORMULA FOR A LINEAR COMBINATION

Suppose we have two vectors:

A and B

We choose two scalars:

a and b

A linear combination is:

V = aA + bB

Where:

- A and B are vectors.
- a and b are scalars.
- V is the resulting vector.

For three vectors:

V = aA + bB + cC

For n vectors:

V = a₁V₁ + a₂V₂ + ... + aₙVₙ

Here, each coefficient can be any real number.

It can be:

- Positive.
- Negative.
- Zero.
- A fraction.
- A decimal.

### Important

A linear combination doesn't require the coefficients to add up to 1.

For example:

2A + 3B

is a linear combination.

So is:

0.5A + 0.5B

And:

A - 2B

And:

0A + 0B

The last one produces the zero vector.

---

# 4. WHY ARE NEGATIVE COEFFICIENTS ALLOWED?

Let's use Rahul and Priya again.

R = [80, 70, 90]

P = [60, 85, 75]

Consider:

R - P

We already know:

R - P = [20, -15, 15]

But subtraction can be rewritten as addition:

R + (-1)P

Therefore:

R - P = 1R + (-1)P

This is also a linear combination.

### Important conclusion

Vector subtraction is a special case of linear combination.

The coefficients are simply:

1 and -1.

This tells us something significant:

Linear combinations are built entirely from the vector operations we already learned.

---

# 5. LINEAR COMBINATIONS USING SIMPLE VECTORS

Our student vectors contain large numbers.

To understand the geometry, let's temporarily use just two subject coordinates and smaller values.

Consider:

A = [1, 0]

B = [0, 1]

Imagine these represent simplified Maths and Physics coordinates.

Let's calculate:

3A + 2B

First:

3A = [3, 0]

Then:

2B = [0, 2]

Add:

3A + 2B = [3, 2]

We have constructed:

V = [3, 2]

Now try:

5A + 4B

= [5, 0] + [0, 4]

= [5, 4]

Another vector.

Try:

-2A + 3B

= [-2, 0] + [0, 3]

= [-2, 3]

Another vector.

### What do you notice?

By changing the scalars, we can create many different vectors.

In fact, using A = [1,0] and B = [0,1], we can create every vector in two-dimensional real space.

We'll understand why in the next section.

---

# 6. WHAT IS SPAN?

This is where linear combinations become especially interesting.

Suppose we have:

A = [1, 0]

B = [0, 1]

We can form:

1A + 1B = [1,1]

2A + 3B = [2,3]

-4A + 5B = [-4,5]

0A + 7B = [0,7]

What if we want:

[100, 200]?

We simply choose:

100A + 200B

= [100,200]

What if we want:

[-5, -8]?

Choose:

-5A - 8B

= [-5,-8]

Because the coefficients can be any real numbers, we can generate every possible two-dimensional vector.

The set of all possible vectors we can generate is called the **span**.

### Definition

The span of a set of vectors is the set of all their possible linear combinations.

Mathematically:

Span(A,B) = {aA+bB | a,b ∈ ℝ}

Read this as:

"The span of A and B contains every vector we can produce by choosing real-valued coefficients a and b."

For our example:

Span([1,0], [0,1]) = ℝ²

This means the two vectors span the entire two-dimensional plane.

---

# 7. UNDERSTANDING SPAN GEOMETRICALLY

Let's start with just one vector.

A = [2, 1]

What can we create using only A?

We can multiply A by any scalar.

For example:

1A = [2,1]

2A = [4,2]

3A = [6,3]

-1A = [-2,-1]

0.5A = [1,0.5]

Every result points along the same straight line through the origin.

Why?

Because scalar multiplication can change a vector's length and reverse its direction.

But it cannot rotate a nonzero vector into a different direction.

Therefore:

**The span of one nonzero vector in ℝ² is a line through the origin.**

### Important exception

The span of the zero vector contains only the zero vector.

Span([0,0]) = {[0,0]}

It doesn't generate a line.

---

# 8. WHAT IF WE HAVE TWO VECTORS?

Suppose:

A = [1,0]

B = [0,1]

A points horizontally.

B points vertically.

Using linear combinations, we can move along both coordinate directions.

For example:

3A + 2B = [3,2]

We can reach any point on the plane.

Therefore, their span is the entire plane.

Now consider a different pair.

A = [1,2]

B = [2,4]

Notice:

B = 2A

Both vectors point in the same direction.

Even though we have two vectors, they lie along the same line.

Let's calculate their linear combination.

aA + bB

= a[1,2] + b[2,4]

= [a,2a] + [2b,4b]

= [a+2b, 2a+4b]

Factor the result:

= (a+2b)[1,2]

No matter what values a and b take, the result is always a scalar multiple of [1,2].

That means every result lies on the same line.

Therefore:

Span([1,2], [2,4]) is only a line, not the entire plane.

### Key insight

Having two vectors does not automatically mean we can span a two-dimensional plane.

What matters is whether the vectors provide genuinely different directions.

---

# 9. CAN WE CREATE ANY TARGET VECTOR?

Let's make span more practical.

Suppose:

A = [1,0]

B = [0,1]

We want to create:

T = [7,5]

Can we find scalars a and b such that:

aA + bB = T?

Substitute:

a[1,0] + b[0,1] = [7,5]

Calculate the left side:

[a,b] = [7,5]

Compare components.

First component:

a = 7

Second component:

b = 5

Therefore:

T = 7A + 5B

Yes, the target belongs to the span.

### Another example

A = [1,2]

B = [2,4]

Target:

T = [3,5]

Can we create T?

Set up:

aA + bB = T

a[1,2] + b[2,4] = [3,5]

Compare components:

a + 2b = 3

2a + 4b = 5

But look at the equations.

If the first equation is true:

a + 2b = 3

Multiplying it by 2 gives:

2a + 4b = 6

However, our second equation requires:

2a + 4b = 5

Both cannot be true simultaneously.

Therefore, no values of a and b can create [3,5].

The target vector is not in the span.

### What did we learn?

Checking whether a target vector belongs to a span means asking:

"Can we find coefficients that produce the target as a linear combination?"

This leads naturally to solving systems of linear equations.

---

# 10. WHAT IS LINEAR DEPENDENCE?

Let's return to:

A = [1,2]

B = [2,4]

Notice:

B = 2A

Vector B can be created using vector A.

Therefore, B doesn't provide a genuinely new direction.

This is called linear dependence.

### Intuitive definition

A set of vectors is linearly dependent if at least one vector can be expressed as a linear combination of the other vectors.

For our pair:

B = 2A

Therefore, A and B are linearly dependent.

### What does dependence mean?

It doesn't mean the vectors are equal.

It means their information is mathematically redundant in terms of generating directions.

Both vectors lie along the same line.

Adding B to our collection doesn't expand the span beyond what A already provides.

---

# 11. WHAT IS LINEAR INDEPENDENCE?

Now consider:

A = [1,0]

B = [0,1]

Can we create B using only multiples of A?

Let's try:

kA = B

k[1,0] = [0,1]

This gives:

[k,0] = [0,1]

Compare components:

k = 0

But the second component would require:

0 = 1

Impossible.

Therefore, B cannot be created using only A.

Similarly, A cannot be created using only B.

These vectors are linearly independent.

### Intuitive definition

Vectors are linearly independent when none can be formed as a linear combination of the others.

Each vector contributes a direction not already generated by the remaining vectors.

---

# 12. FORMAL TEST FOR LINEAR INDEPENDENCE

Let's learn the mathematical rule.

Suppose we have vectors:

V₁, V₂, ..., Vₙ

Consider the equation:

a₁V₁ + a₂V₂ + ... + aₙVₙ = 0

Here, 0 is the zero vector.

The question is:

Can we produce the zero vector using coefficients that are not all zero?

### Case 1: Only all-zero coefficients work

If the equation is possible only when:

a₁ = a₂ = ... = aₙ = 0

Then the vectors are linearly independent.

### Case 2: Some nonzero coefficients work

If there is a solution where at least one coefficient is nonzero, the vectors are linearly dependent.

### Why?

Let's examine dependent vectors:

A = [1,2]

B = [2,4]

We know:

B = 2A

Rearrange:

2A - B = 0

Substitute:

2[1,2] - [2,4]

= [2,4] - [2,4]

= [0,0]

We used coefficients:

a = 2

b = -1

These are not both zero.

Therefore, A and B are linearly dependent.

This is called a nontrivial solution.

---

# 13. TRIVIAL VS NONTRIVIAL SOLUTIONS

The equation:

aA + bB = 0

always has one obvious solution:

a = 0

b = 0

Because:

0A + 0B = 0

This is called the trivial solution.

But we're interested in whether another solution exists.

### Independent vectors

Only the trivial solution exists.

### Dependent vectors

At least one nontrivial solution exists.

Example:

A = [1,0]

B = [0,1]

Equation:

a[1,0] + b[0,1] = [0,0]

Calculate:

[a,b] = [0,0]

Therefore:

a = 0

b = 0

There is no other solution.

So the vectors are independent.

### Important

The test is not whether we can create the zero vector.

We always can, using all-zero coefficients.

The test is whether we can create it using a nontrivial combination.

---

# 14. A THREE-VECTOR EXAMPLE

Let's use three-dimensional vectors, just like our student marks.

Suppose:

A = [1,0,0]

B = [0,1,0]

C = [0,0,1]

We want to know whether these three vectors are independent.

Consider:

aA + bB + cC = [0,0,0]

Substitute:

a[1,0,0] + b[0,1,0] + c[0,0,1] = [0,0,0]

Calculate:

[a,b,c] = [0,0,0]

Therefore:

a = 0

b = 0

c = 0

Only the trivial solution exists.

So the three vectors are linearly independent.

### What is their span?

Any target vector:

T = [x,y,z]

can be written:

T = xA + yB + zC

Therefore:

Span(A,B,C) = ℝ³

These three vectors can generate every three-dimensional vector.

---

# 15. CAN THREE VECTORS BE DEPENDENT?

Absolutely.

Consider:

A = [1,0,0]

B = [0,1,0]

C = [1,1,0]

Notice:

C = A + B

Because:

[1,0,0] + [0,1,0] = [1,1,0]

So C can be created using A and B.

Therefore, the three vectors are linearly dependent.

### Formal verification

A + B - C = 0

Calculate:

[1,0,0] + [0,1,0] - [1,1,0]

= [0,0,0]

We used coefficients:

1, 1, -1

These are not all zero.

Therefore, the vectors are dependent.

### Does C expand the span?

No.

Anything we can construct using A, B, and C can already be constructed using just A and B.

This is because C itself is A+B.

---

# 16. NUMBER OF VECTORS VS DIMENSION

This is an important principle.

In a two-dimensional space, at most two vectors can be linearly independent.

In a three-dimensional space, at most three vectors can be linearly independent.

More generally:

In ℝⁿ, at most n vectors can be linearly independent.

### Why?

Because an n-dimensional vector space has only n independent coordinate directions.

If you have more than n vectors, at least one must be expressible as a combination of others.

For example:

In ℝ²:

Three vectors must be dependent.

In ℝ³:

Four vectors must be dependent.

### Important distinction

Having n vectors in ℝⁿ does not guarantee independence.

For example:

[1,2] and [2,4]

are two vectors in ℝ², but they are dependent.

The number of vectors provides an upper limit on independence, not a guarantee.

---

# 17. WHAT IS A BASIS?

We've learned:

- Linear combinations create vectors.
- Span tells us what vectors can be created.
- Independence tells us whether any vector is redundant.

Now combine these ideas.

A basis is a set of vectors that:

1. Spans the entire vector space.
2. Is linearly independent.

Let's understand.

In ℝ²:

A = [1,0]

B = [0,1]

These vectors are linearly independent.

They also span the entire plane.

Therefore, they form a basis for ℝ².

### Why is a basis useful?

Because it gives us a minimal collection of independent directions needed to represent every vector in the space.

For example:

T = [7,5]

We can write:

T = 7A + 5B

The coefficients 7 and 5 are the coordinates of T relative to this basis.

---

# 18. IS THERE ONLY ONE BASIS?

No.

This is a subtle but important concept.

Consider:

A = [1,0]

B = [0,1]

They form a basis for ℝ².

But so do:

C = [1,1]

D = [1,-1]

Why?

Because C and D are independent and span ℝ².

Let's demonstrate by creating:

T = [4,2]

We want:

aC + bD = T

Substitute:

a[1,1] + b[1,-1] = [4,2]

Compare components:

a+b = 4

a-b = 2

Add both equations:

2a = 6

a = 3

Substitute:

3+b = 4

b = 1

Therefore:

T = 3C + 1D

= 3[1,1] + [1,-1]

= [3,3] + [1,-1]

= [4,2]

So both sets can represent every vector in ℝ².

### Important conclusion

A vector space can have many different bases.

But every basis of a finite-dimensional vector space contains the same number of vectors.

That number is the dimension of the space.

---

# 19. CONNECTING THIS TO OUR STUDENT EXAMPLE

Now let's return to our school dataset.

Each student has three marks:

[Maths, Physics, Chemistry]

Therefore, each student's marks vector belongs to ℝ³.

A simple basis for this space is:

E₁ = [1,0,0]

E₂ = [0,1,0]

E₃ = [0,0,1]

What do these represent in our school example?

E₁ corresponds to one unit along the Maths coordinate.

E₂ corresponds to one unit along the Physics coordinate.

E₃ corresponds to one unit along the Chemistry coordinate.

Now take Rahul:

R = [80,70,90]

We can write:

R = 80E₁ + 70E₂ + 90E₃

Substitute:

R = 80[1,0,0] + 70[0,1,0] + 90[0,0,1]

Calculate:

= [80,0,0] + [0,70,0] + [0,0,90]

= [80,70,90]

That's Rahul's original vector.

### What does this mean?

Every student's three-subject marks vector can be represented as a linear combination of these three basis vectors.

The subject marks act as coefficients.

This is the coordinate interpretation of a basis.

---

# 20. ARE OUR FIVE STUDENTS' VECTORS INDEPENDENT?

We have five students:

R, P, A, S, K

Each vector contains three components.

Therefore, all five vectors belong to ℝ³.

We learned:

In ℝ³, at most three vectors can be independent.

We have five vectors.

Therefore:

**The collection of all five students' marks vectors must be linearly dependent.**

Notice that we didn't need to calculate anything.

The number of vectors is greater than the dimension of the space.

That alone guarantees dependence.

### Does this mean two students have identical marks?

No.

Linear dependence does not necessarily mean two vectors are identical.

It means at least one vector can be expressed as a linear combination of the others.

The coefficients may be positive, negative, or fractional.

---

# 21. DOES LINEAR DEPENDENCE MEAN THE SAME THING AS SIMILARITY?

No.

This distinction is important.

Suppose:

A = [1,2]

B = [2,4]

These vectors are dependent because:

B = 2A

They point in the same direction.

Now consider:

A = [1,2]

B = [1.001,2]

These vectors are numerically very similar.

But they are not exact scalar multiples.

Therefore, they are linearly independent in ℝ².

### Key takeaway

Similarity and linear dependence are different concepts.

Similarity can be approximate.

Linear dependence is an exact mathematical relationship.

In practical numerical calculations, we may also study near-dependence because floating-point computations have limited precision.

That becomes important in advanced chapters.

---

# 22. WHY DO WE CARE ABOUT REDUNDANT DIRECTIONS?

Imagine recording three values for each student:

1. Maths marks.
2. Physics marks.
3. Twice the Maths marks.

Example:

Rahul:

[80,70,160]

Priya:

[60,85,120]

Arjun:

[90,80,180]

Notice something.

The third value is always twice the first.

It doesn't introduce new information.

If we already know Maths marks, we can calculate the third value immediately.

This is a simple example of redundant numerical information.

### Important distinction

We're discussing dependence between feature columns across students here.

Earlier, we discussed dependence among vectors representing students.

Both use linear algebra, but they're different ways of organizing the problem.

This becomes clearer when we study matrices and matrix rank.

---

# 23. COMMON BEGINNER MISTAKES

## Mistake 1: A linear combination requires coefficients to sum to 1

Incorrect.

Any real coefficients are allowed.

## Mistake 2: Two vectors always span a plane

Incorrect.

If they are dependent and nonzero, their span may be only a line.

## Mistake 3: Dependent vectors must be identical

Incorrect.

One may be a scalar multiple or combination of others.

## Mistake 4: Independent vectors must be perpendicular

Incorrect.

Vectors can be independent without being perpendicular.

For example:

[1,0] and [1,1]

are independent but not perpendicular.

## Mistake 5: More vectors always mean a larger span

Incorrect.

Adding a dependent vector doesn't expand the span.

## Mistake 6: Any three vectors in ℝ³ form a basis

Incorrect.

They must be linearly independent and span ℝ³.

## Mistake 7: Similarity and linear dependence are the same

Incorrect.

Two vectors can be nearly identical yet linearly independent.

## Mistake 8: A basis must use the standard coordinate vectors

Incorrect.

Many different sets can form a basis for the same vector space.

---

# 24. INTERVIEW QUESTIONS AND ANSWERS

### Q1. What is a linear combination?

A linear combination is a sum of vectors multiplied by scalar coefficients.

### Q2. What is span?

Span is the collection of all possible linear combinations of a given set of vectors.

### Q3. What is linear dependence?

A set of vectors is dependent if there exists a nontrivial combination of them that produces the zero vector.

### Q4. What is linear independence?

A set of vectors is independent if the only combination producing the zero vector uses all-zero coefficients.

### Q5. Can two independent vectors span ℝ²?

Yes, any two linearly independent vectors in ℝ² span the entire plane.

### Q6. Can three vectors in ℝ² be independent?

No.

The maximum number of linearly independent vectors in ℝ² is two.

### Q7. What is a basis?

A linearly independent set of vectors that spans the entire vector space.

### Q8. Can a vector space have multiple bases?

Yes.

Different sets of independent vectors can span the same space.

### Q9. How many vectors are in a basis for ℝ³?

Exactly three.

### Q10. Can a linearly dependent set span ℝ³?

Yes.

For example, four vectors can span ℝ³ while being dependent.

### Q11. What happens when we add a vector already in the span?

The span does not expand.

### Q12. Are orthogonal nonzero vectors independent?

Yes.

Any collection of mutually orthogonal nonzero vectors is linearly independent.

### Q13. Why is the zero vector problematic for independence?

Any set containing the zero vector is linearly dependent.

We can assign the zero vector a nonzero coefficient and still obtain the zero vector.

### Q14. Can two nonzero vectors be dependent without being equal?

Yes.

For example:

[1,2] and [3,6]

are dependent because the second is three times the first.

### Q15. What is the difference between dimension and number of vectors?

Dimension describes the number of vectors in a basis for the space.

A collection may contain more vectors than the dimension, but those vectors cannot all be independent.

---

# 25. IMPORTANT FORMULAS

### Linear Combination

V = a₁V₁ + a₂V₂ + ... + aₙVₙ

### Span

Span(V₁,...,Vₙ)

= {a₁V₁ + ... + aₙVₙ | aᵢ ∈ ℝ}

### Independence Test

a₁V₁ + ... + aₙVₙ = 0

Only the trivial solution:

a₁ = ... = aₙ = 0

means independence.

### Dependence Test

A nontrivial solution to:

a₁V₁ + ... + aₙVₙ = 0

means dependence.

### Basis Conditions

A set is a basis when it is:

1. Linearly independent.
2. Spanning the entire vector space.

---

# 26. COMPLETE CHAPTER SUMMARY

We started with the vectors representing our students' marks.

First, we learned that we could multiply vectors by scalars and add the results.

That is a linear combination.

Then we asked:

"What are all the possible vectors we can create?"

That introduced span.

Next, we discovered that some vectors can already be created using other vectors.

That introduced linear dependence.

We also learned the opposite:

Linear independence means no vector is redundant within the collection.

Finally, we combined span and independence to understand basis.

### The complete relationship

**Linear combination:** Creates a vector.

**Span:** Describes every vector we can create.

**Linear independence:** Determines whether vectors contribute distinct directions.

**Basis:** A collection of independent vectors sufficient to represent the entire space.

### Most important takeaway

The number of vectors we possess is not always the number of independent directions they provide.

A set can contain many vectors but still span only a line or plane.

Understanding this distinction is essential for studying higher-dimensional data.

---

END OF CHAPTER 5

NEXT CHAPTER: MATRICES, MATRIX SHAPES, AND MATRIX OPERATIONS
