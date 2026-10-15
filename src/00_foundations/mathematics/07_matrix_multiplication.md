# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 7 — MATRIX MULTIPLICATION

---

# 1. RECAP: WHAT WE ALREADY KNOW

From Chapter 6, we learned that a matrix is a rectangular arrangement of numbers.

Our school dataset is:

| Student | Maths | Physics | Chemistry |
| ------- | ----: | ------: | --------: |
| Rahul   |    80 |      70 |        90 |
| Priya   |    60 |      85 |        75 |
| Arjun   |    90 |      80 |        85 |
| Sneha   |    50 |      65 |        70 |
| Kiran   |    75 |      90 |        80 |

We represent it as:

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

Shape = 5 × 3

We already know:

- Matrix addition
- Matrix subtraction
- Scalar multiplication
- Transpose
- Matrix shapes

Today, we are learning how to multiply matrices.

But matrix multiplication is different from ordinary multiplication.

Let's understand why.

---

# 2. WHY DO WE NEED MATRIX MULTIPLICATION?

Imagine our school introduces a competition.

Every student receives competition points based on their subject marks.

The scoring rules are:

Maths: 2 points per mark.

Physics: 3 points per mark.

Chemistry: 1 point per mark.

Let's calculate Rahul's competition score.

Rahul:

R = [80, 70, 90]

Scoring values:

S = [2, 3, 1]

We calculate:

80 × 2 = 160

70 × 3 = 210

90 × 1 = 90

Add:

160 + 210 + 90 = 460

Rahul's competition score = 460.

Remember Chapter 4?

We learned the dot product.

The same calculation can be expressed as:

R · S = 460

But here's our next problem.

We have five students.

Should we calculate five separate dot products manually?

We could.

But what if we had 10,000 students?

Or 100,000 students?

We need a way to perform all those calculations together.

That's one purpose of matrix multiplication.

---

# 3. FROM DOT PRODUCT TO MATRIX MULTIPLICATION

Let's start with just two students.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Arrange them into a matrix.

A =

[80 70 90]
[60 85 75]

Shape = 2 × 3

Our competition scoring vector is:

S =

[2]
[3]
[1]

Shape = 3 × 1

Now calculate:

AS

This means:

A × S

The result is:

[80×2 + 70×3 + 90×1]
[60×2 + 85×3 + 75×1]

Calculate each row.

Rahul:

160 + 210 + 90 = 460

Priya:

120 + 255 + 75 = 450

Therefore:

AS =

[460]
[450]

Shape = 2 × 1

We calculated both students' competition scores in one matrix operation.

### What happened?

Each row of A was multiplied with the column of S using a dot product.

This is the fundamental idea behind matrix multiplication.

---

# 4. THE ROW-BY-COLUMN RULE

Let's state the rule clearly.

**In standard matrix multiplication, each result element is the dot product of one row from the first matrix and one column from the second matrix.**

Let's understand using smaller numbers.

Suppose:

A =

[1 2]
[3 4]

B =

[5 6]
[7 8]

We want:

C = AB

To calculate the first element of C:

Take row 1 of A:

[1, 2]

Take column 1 of B:

[5, 7]

Calculate their dot product:

1×5 + 2×7

= 5 + 14

= 19

Therefore:

C₁₁ = 19

### Next element

Take row 1 of A:

[1, 2]

Take column 2 of B:

[6, 8]

Dot product:

1×6 + 2×8

= 6 + 16

= 22

Therefore:

C₁₂ = 22

### Third element

Take row 2 of A:

[3, 4]

Take column 1 of B:

[5, 7]

Dot product:

3×5 + 4×7

= 15 + 28

= 43

Therefore:

C₂₁ = 43

### Fourth element

Take row 2 of A:

[3, 4]

Take column 2 of B:

[6, 8]

Dot product:

3×6 + 4×8

= 18 + 32

= 50

Therefore:

C₂₂ = 50

### Final result

AB =

[19 22]
[43 50]

This is standard matrix multiplication.

---

# 5. WHY DON'T WE MULTIPLY CORRESPONDING ELEMENTS?

You might wonder:

Why not calculate:

[1×5 2×6]
[3×7 4×8]

That would produce:

[5 12]
[21 32]

This operation exists.

It's called element-wise multiplication, or the Hadamard product.

But it is not standard matrix multiplication.

### Difference

Element-wise multiplication:

Multiply corresponding entries.

Matrix multiplication:

Take row-column dot products.

### Why does standard matrix multiplication exist?

Because matrices can represent relationships between inputs and outputs, and transformations of vectors.

Standard matrix multiplication lets us combine these relationships.

For example, one matrix may describe how subject marks are converted into competition scores.

Another matrix may describe how those scores are converted into a new set of values.

Multiplying the matrices lets us combine the two transformations.

We'll demonstrate this later in the chapter.

---

# 6. THE MOST IMPORTANT RULE: MATRIX SHAPES

This is where many beginners get confused.

Suppose:

A has shape 2 × 3.

B has shape 3 × 4.

Can we multiply A × B?

Yes.

Why?

The number of columns in A equals the number of rows in B.

A has 3 columns.

B has 3 rows.

The inner dimensions match.

### General rule

If:

A has shape m × n

B has shape n × p

Then:

AB has shape m × p

The result contains:

m rows

p columns

### Easy way to remember

(2 × 3) × (3 × 4)

The inner numbers are both 3.

Therefore, multiplication is valid.

The resulting shape comes from the outer numbers:

2 × 4

### Another example

(5 × 3) × (3 × 1)

Inner dimensions:

3 and 3

They match.

Result:

5 × 1

That's exactly what happens when we multiply our five-student matrix by a three-subject scoring vector.

---

# 7. WHY MUST THE INNER DIMENSIONS MATCH?

Let's understand rather than memorize.

Suppose:

A has shape 2 × 3.

Each row of A contains three numbers.

Now suppose:

B has shape 3 × 2.

Each column of B contains three numbers.

We can calculate a dot product between:

A row containing three numbers.

B column containing three numbers.

That works.

But suppose B has shape 4 × 2.

Each column of B now contains four numbers.

We would be trying to calculate a dot product between:

A row containing three numbers.

B column containing four numbers.

Ordinary dot product requires equal dimensions.

Therefore, matrix multiplication is undefined.

### Key takeaway

Matrix multiplication depends on dot products.

The inner dimensions must match because the two vectors participating in every dot product must contain the same number of components.

---

# 8. PRACTICE: MATRIX SHAPES

### Example 1

A = 4 × 3

B = 3 × 2

Multiplication:

AB is valid.

Result shape:

4 × 2

### Example 2

A = 5 × 2

B = 2 × 7

Valid.

Result:

5 × 7

### Example 3

A = 3 × 4

B = 2 × 3

Invalid.

Why?

A has 4 columns.

B has 2 rows.

The inner dimensions do not match.

### Example 4

A = 10 × 5

B = 5 × 1

Valid.

Result:

10 × 1

### Example 5

A = 3 × 3

B = 3 × 3

Valid.

Result:

3 × 3

### Important

Never start multiplying individual values before checking the shapes.

Shape checking is always the first step.

---

# 9. RETURNING TO OUR FIVE STUDENTS

Our complete matrix:

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

Shape = 5 × 3

Competition scoring vector:

S =

[2]
[3]
[1]

Shape = 3 × 1

We calculate:

MS

The output shape will be:

5 × 1

Why?

Because:

(5 × 3)(3 × 1) = 5 × 1

### Calculate each student's score

Rahul:

80×2 + 70×3 + 90×1 = 460

Priya:

60×2 + 85×3 + 75×1 = 450

Arjun:

90×2 + 80×3 + 85×1 = 505

Sneha:

50×2 + 65×3 + 70×1 = 365

Kiran:

75×2 + 90×3 + 80×1 = 500

Final result:

MS =

[460]
[450]
[505]
[365]
[500]

### What does this mean?

We started with a matrix containing five students and three subject marks.

We applied one scoring rule.

We obtained one score for each student.

The result contains five rows because there are five students.

It contains one column because we calculated one competition score per student.

---

# 10. WHAT IF THE SCHOOL HAS TWO COMPETITIONS?

Now let's expand the same example.

Suppose there are two competitions.

### Competition A

Maths: 2 points per mark.

Physics: 3 points per mark.

Chemistry: 1 point per mark.

### Competition B

Maths: 1 point per mark.

Physics: 1 point per mark.

Chemistry: 2 points per mark.

Can we calculate both competition scores for all five students at once?

Yes.

We need a scoring matrix.

Let's construct it.

S =

[2 1]
[3 1]
[1 2]

Look at the columns.

First column:

[2,3,1]

Represents Competition A.

Second column:

[1,1,2]

Represents Competition B.

Shape of S:

3 × 2

Our marks matrix M:

5 × 3

Therefore:

MS has shape:

(5 × 3)(3 × 2)

= 5 × 2

The result will have:

Five rows for students.

Two columns for competition scores.

---

# 11. MULTIPLYING THE FULL MATRICES

Let's calculate:

M × S

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

S =

[2 1]
[3 1]
[1 2]

### Rahul — Competition A

80×2 + 70×3 + 90×1

= 460

### Rahul — Competition B

80×1 + 70×1 + 90×2

= 80 + 70 + 180

= 330

Therefore, Rahul's output row is:

[460, 330]

### Priya

Competition A:

60×2 + 85×3 + 75×1

= 450

Competition B:

60×1 + 85×1 + 75×2

= 295

Output:

[450, 295]

### Arjun

Competition A:

90×2 + 80×3 + 85×1

= 505

Competition B:

90×1 + 80×1 + 85×2

= 340

Output:

[505, 340]

### Sneha

Competition A:

50×2 + 65×3 + 70×1

= 365

Competition B:

50×1 + 65×1 + 70×2

= 255

Output:

[365, 255]

### Kiran

Competition A:

75×2 + 90×3 + 80×1

= 500

Competition B:

75×1 + 90×1 + 80×2

= 325

Output:

[500, 325]

### Final matrix

MS =

[460 330]
[450 295]
[505 340]
[365 255]
[500 325]

### Interpretation

| Student | Competition A | Competition B |
| ------- | ------------: | ------------: |
| Rahul   |           460 |           330 |
| Priya   |           450 |           295 |
| Arjun   |           505 |           340 |
| Sneha   |           365 |           255 |
| Kiran   |           500 |           325 |

We calculated ten competition scores using one matrix multiplication.

This demonstrates why matrix multiplication is useful for combining multiple numerical records with multiple scoring rules.

---

# 12. GENERAL FORMULA FOR MATRIX MULTIPLICATION

Suppose:

A has shape m × n.

B has shape n × p.

Their product is:

C = AB

Where:

C has shape m × p.

Each result element is:

Cᵢⱼ = Σₖ₌₁ⁿ Aᵢₖ Bₖⱼ

Let's understand the notation.

i = Row of the result.

j = Column of the result.

k = Position used while multiplying corresponding components.

n = Number of components in each row-column dot product.

Σ means add all the products.

### Example

Suppose we want C₂₁.

Take:

Row 2 of A.

Column 1 of B.

Multiply corresponding entries.

Add the results.

That gives the element in row 2, column 1 of C.

### Important

The formula is simply the row-by-column rule written mathematically.

It isn't a new operation.

---

# 13. WHY MATRIX MULTIPLICATION IS NOT COMMUTATIVE

Ordinary numbers satisfy:

2 × 3 = 3 × 2

Both equal 6.

But matrices behave differently.

In general:

AB ≠ BA

Let's understand with an example.

A =

[1 2]
[3 4]

B =

[5 6]
[7 8]

We already calculated:

AB =

[19 22]
[43 50]

Now let's calculate BA.

### First element

Row 1 of B:

[5,6]

Column 1 of A:

[1,3]

Dot product:

5×1 + 6×3 = 23

### Second element

5×2 + 6×4 = 34

### Third element

7×1 + 8×3 = 31

### Fourth element

7×2 + 8×4 = 46

Therefore:

BA =

[23 34]
[31 46]

Compare:

AB =

[19 22]
[43 50]

BA =

[23 34]
[31 46]

They are different.

Therefore:

AB ≠ BA

### Why does order matter?

Matrix multiplication can represent a sequence of transformations.

Changing the order of those transformations can change the final result.

Also, for some matrix shapes, AB may be valid while BA is not even defined.

### Important rule

**Matrix multiplication is generally not commutative.**

Some special matrices do commute, but we cannot assume that for arbitrary matrices.

---

# 14. CAN AB BE VALID WHILE BA IS INVALID?

Yes.

Suppose:

A has shape 2 × 3.

B has shape 3 × 4.

AB:

(2 × 3)(3 × 4)

Valid.

Result:

2 × 4

Now reverse the order.

BA:

(3 × 4)(2 × 3)

The inner dimensions are:

4 and 2

They do not match.

Therefore, BA is undefined.

### Important lesson

Reversing the multiplication order may:

1. Produce a different result.
2. Produce a result with a different shape.
3. Make the multiplication invalid.

Always preserve multiplication order unless a valid mathematical rule allows you to change it.

---

# 15. ASSOCIATIVE PROPERTY OF MATRIX MULTIPLICATION

Matrix multiplication is not generally commutative.

But it is associative.

For compatible matrices:

(AB)C = A(BC)

Let's understand the difference.

### Commutative property

Changes order:

AB versus BA.

Not generally allowed.

### Associative property

Changes grouping:

(AB)C versus A(BC).

Allowed when the products are defined.

### Why is this useful?

Suppose we have three matrices representing three consecutive calculations.

We can choose which pair to multiply first without changing the final mathematical result.

This can make calculations much more efficient.

For example:

A shape = 100 × 10

B shape = 10 × 5

C shape = 5 × 2

We could calculate:

(AB)C

Or:

A(BC)

Both produce a 100 × 2 matrix.

But their computational costs differ.

Let's calculate the approximate scalar multiplication counts.

### Method 1: (AB)C

First AB:

100 × 10 × 5 = 5000 multiplications.

Result shape = 100 × 5.

Then multiply by C:

100 × 5 × 2 = 1000 multiplications.

Total = 6000 multiplications.

### Method 2: A(BC)

First BC:

10 × 5 × 2 = 100 multiplications.

Result shape = 10 × 2.

Then multiply by A:

100 × 10 × 2 = 2000 multiplications.

Total = 2100 multiplications.

### Result

Both methods give the same mathematical result.

But the second method uses fewer scalar multiplications.

This illustrates why matrix multiplication order and grouping can affect computational efficiency.

---

# 16. DISTRIBUTIVE PROPERTY

Matrix multiplication distributes over addition.

Suppose A, B, and C have compatible shapes.

Then:

A(B+C) = AB + AC

Similarly:

(A+B)C = AC + BC

Let's verify with small matrices.

A =

[1 2]

B =

[3]
[4]

C =

[5]
[6]

### First method

B+C =

[8]
[10]

A(B+C):

1×8 + 2×10

= 28

### Second method

AB:

1×3 + 2×4 = 11

AC:

1×5 + 2×6 = 17

Add:

11 + 17 = 28

Both results match.

Therefore:

A(B+C) = AB+AC

---

# 17. IDENTITY MATRIX IN MULTIPLICATION

Remember Chapter 6?

We introduced the identity matrix.

For two dimensions:

I =

[1 0]
[0 1]

Let's multiply:

A =

[2 3]
[4 5]

Calculate:

AI

First row, first column:

2×1 + 3×0 = 2

First row, second column:

2×0 + 3×1 = 3

Second row, first column:

4×1 + 5×0 = 4

Second row, second column:

4×0 + 5×1 = 5

Therefore:

AI =

[2 3]
[4 5]

That's the original matrix.

### Rule

AI = A

IA = A

Using identity matrices of the appropriate dimensions.

### Important

The identity matrix plays a role similar to the scalar number 1.

But it must have the correct shape for multiplication.

---

# 18. MULTIPLICATION BY A ZERO MATRIX

Suppose:

A =

[2 3]
[4 5]

And:

Z =

[0 0]
[0 0]

Calculate:

AZ

Every row-column dot product contains only multiplication by zeros.

Therefore:

AZ =

[0 0]
[0 0]

### Rule

Multiplying a matrix by a compatible zero matrix produces a zero matrix.

The resulting shape follows the ordinary matrix multiplication shape rule.

---

# 19. TRANSPOSE OF A MATRIX PRODUCT

This is another important property.

For compatible matrices:

(AB)ᵀ = BᵀAᵀ

Notice the order reverses.

Let's understand why.

Suppose:

A has shape 2 × 3.

B has shape 3 × 4.

Then:

AB has shape 2 × 4.

Its transpose:

(AB)ᵀ has shape 4 × 2.

Now consider:

Bᵀ has shape 4 × 3.

Aᵀ has shape 3 × 2.

Their product:

BᵀAᵀ has shape 4 × 2.

The shapes match.

### Important

The identity is:

(AB)ᵀ = BᵀAᵀ

Not generally:

(AB)ᵀ = AᵀBᵀ

The reversal is essential.

---

# 20. MATRIX-VECTOR MULTIPLICATION

Matrix-vector multiplication is a special case of matrix multiplication.

Suppose:

A =

[2 3]
[4 5]

And:

x =

[10]
[20]

Calculate:

Ax

First row:

2×10 + 3×20 = 80

Second row:

4×10 + 5×20 = 140

Result:

Ax =

[80]
[140]

### Shape

A = 2 × 2

x = 2 × 1

Result = 2 × 1

### Why is this important?

A matrix can transform one vector into another vector.

Here, we started with a two-dimensional input vector.

After multiplying by A, we obtained a new two-dimensional output vector.

We'll explore this geometric interpretation more deeply in the chapter on linear transformations.

---

# 21. MATRIX MULTIPLICATION AS COMBINING SCORING RULES

Let's return to our school example.

Suppose we first convert subject marks into two competition scores.

We already did that:

M × S = C

Where:

M has shape 5 × 3.

S has shape 3 × 2.

C has shape 5 × 2.

Now suppose the school combines the two competition scores into one final score.

Competition A contributes 60%.

Competition B contributes 40%.

We can represent this using:

W =

[0.6]
[0.4]

Shape = 2 × 1

Calculate:

CW

Result shape:

(5 × 2)(2 × 1)

= 5 × 1

For Rahul:

Competition A = 460

Competition B = 330

Final score:

460×0.6 + 330×0.4

= 276 + 132

= 408

### Here's the interesting part

We have:

C = MS

And:

Final = CW

Therefore:

Final = (MS)W

By associativity:

Final = M(SW)

We could combine the scoring rules first and then apply them to the original marks.

Let's calculate SW.

S =

[2 1]
[3 1]
[1 2]

W =

[0.6]
[0.4]

First row:

2×0.6 + 1×0.4 = 1.6

Second row:

3×0.6 + 1×0.4 = 2.2

Third row:

1×0.6 + 2×0.4 = 1.4

Therefore:

SW =

[1.6]
[2.2]
[1.4]

Now calculate Rahul's final score directly:

80×1.6 + 70×2.2 + 90×1.4

= 128 + 154 + 126

= 408

Exactly the same result.

### What did we discover?

Matrix multiplication lets us combine multiple stages of linear calculations into a single transformation.

This is one of its most powerful mathematical properties.

---

# 22. MATRIX MULTIPLICATION AND LINEAR COMBINATIONS

Remember Chapter 5?

We learned linear combinations.

Let's connect them to matrix multiplication.

Suppose:

A =

[2 4]
[3 5]

And:

x =

[2]
[3]

Calculate:

Ax

We get:

First row:

2×2 + 4×3 = 16

Second row:

3×2 + 5×3 = 21

Therefore:

Ax =

[16]
[21]

Now look at the columns of A.

Column 1:

c₁ = [2,3]

Column 2:

c₂ = [4,5]

We can rewrite:

Ax = 2c₁ + 3c₂

Calculate:

2[2,3] + 3[4,5]

= [4,6] + [12,15]

= [16,21]

Same result.

### Important conclusion

Matrix-vector multiplication can be interpreted as a linear combination of the matrix's columns.

The vector x provides the coefficients.

This connects:

- Matrix multiplication.
- Linear combinations.
- Span.
- Linear transformations.

We'll use this connection later when studying systems of linear equations.

---

# 23. COMMON BEGINNER MISTAKES

## Mistake 1: Multiplying corresponding elements

That's element-wise multiplication, not standard matrix multiplication.

## Mistake 2: Checking outer dimensions for validity

The inner dimensions must match.

## Mistake 3: Assuming AB = BA

Matrix multiplication is generally not commutative.

## Mistake 4: Assuming multiplication preserves matrix shape

The output shape depends on the outer dimensions.

## Mistake 5: Confusing matrix-vector multiplication with dot product

A matrix-vector product can return a vector.

A dot product of two vectors returns a scalar.

## Mistake 6: Assuming AB valid means BA valid

Incorrect.

One direction can be valid while the reverse is undefined.

## Mistake 7: Reversing transpose incorrectly

Correct:

(AB)ᵀ = BᵀAᵀ

## Mistake 8: Believing matrix multiplication is only a numerical shortcut

It also describes linear transformations and the composition of transformations.

---

# 24. INTERVIEW QUESTIONS AND ANSWERS

### Q1. What is matrix multiplication?

An operation where each output element is calculated as the dot product of a row of the first matrix and a column of the second.

### Q2. What is the shape rule?

If A is m × n and B is n × p, then AB is m × p.

### Q3. Why must inner dimensions match?

Because each row-column dot product requires equal numbers of components.

### Q4. Is matrix multiplication commutative?

Generally no.

AB ≠ BA.

### Q5. Is matrix multiplication associative?

Yes.

(AB)C = A(BC), for compatible matrices.

### Q6. What is the difference between matrix multiplication and element-wise multiplication?

Matrix multiplication combines rows and columns through dot products.

Element-wise multiplication multiplies corresponding entries.

### Q7. What is matrix-vector multiplication?

Multiplication of a matrix by a vector, producing another vector when dimensions are compatible.

### Q8. How does matrix-vector multiplication relate to linear combinations?

Ax is a linear combination of the columns of A, with coefficients supplied by x.

### Q9. What is the effect of multiplying by an identity matrix?

The original matrix remains unchanged.

### Q10. What is the transpose of a matrix product?

(AB)ᵀ = BᵀAᵀ.

### Q11. Why does matrix multiplication order matter?

Because the order of transformations can change the result, and reversing factors can even make multiplication undefined.

### Q12. Can two different parenthesizations of three matrices produce the same result?

Yes.

Matrix multiplication is associative.

### Q13. Does matrix multiplication always produce a square matrix?

No.

The output can be rectangular, square, or a single-column or single-row matrix.

### Q14. What is the computational cost of ordinary dense matrix multiplication?

For an m × n matrix multiplied by an n × p matrix, the straightforward algorithm requires approximately O(mnp) arithmetic operations.

### Q15. Why is matrix multiplication important in machine learning?

It provides a compact way to express weighted calculations, linear transformations, and operations on many numerical observations simultaneously.

---

# 25. IMPORTANT FORMULAS

### Matrix Multiplication

C = AB

### Element Formula

Cᵢⱼ = Σₖ₌₁ⁿ AᵢₖBₖⱼ

### Shape Rule

(m × n)(n × p) = m × p

### Associativity

(AB)C = A(BC)

### Distributivity

A(B+C) = AB+AC

### Identity

AI = IA = A

With appropriately sized identity matrices.

### Transpose of Product

(AB)ᵀ = BᵀAᵀ

### Matrix-Vector Product

Ax = x₁c₁ + x₂c₂ + ... + xₙcₙ

Where c₁, c₂, ..., cₙ are the columns of A.

---

# 26. COMPLETE CHAPTER SUMMARY

We began with our school competition.

We learned how to calculate one student's competition score using a dot product.

Then we organized multiple students into a matrix and calculated their scores together.

We learned:

1. Matrix multiplication follows the row-by-column dot product rule.
2. Inner dimensions must match.
3. Outer dimensions determine the output shape.
4. Matrix multiplication differs from element-wise multiplication.
5. Matrix multiplication is generally not commutative.
6. Matrix multiplication is associative.
7. Multiplication distributes over addition.
8. Identity matrices leave compatible matrices unchanged.
9. Transposing a product reverses the multiplication order.
10. Matrix-vector multiplication produces linear combinations of matrix columns.
11. Multiple linear scoring stages can be combined using matrix multiplication.
12. Multiplication grouping can affect computational efficiency.

### THE BIG PICTURE

Matrix multiplication connects several concepts we've already studied.

Dot product:

Combines two vectors into one scalar.

Linear combination:

Creates a vector by scaling and adding vectors.

Matrix multiplication:

Uses dot products to calculate outputs and can represent multiple linear combinations simultaneously.

These ideas are different views of the same underlying mathematics.

---

END OF CHAPTER 7

NEXT CHAPTER: LINEAR TRANSFORMATIONS — HOW MATRICES CHANGE VECTORS GEOMETRICALLY
