# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 6 — MATRICES, SHAPES & BASIC OPERATIONS

---

# 1. WHY DO WE NEED MATRICES?

Imagine you're a teacher managing marks for five students.

Our familiar dataset:

| Student | Maths | Physics | Chemistry |
| ------- | ----: | ------: | --------: |
| Rahul   |    80 |      70 |        90 |
| Priya   |    60 |      85 |        75 |
| Arjun   |    90 |      80 |        85 |
| Sneha   |    50 |      65 |        70 |
| Kiran   |    75 |      90 |        80 |

So far, we've represented individual students using vectors.

Rahul:

R = [80, 70, 90]

Priya:

P = [60, 85, 75]

Arjun:

A = [90, 80, 85]

These vectors work well when we're dealing with individual students.

But suppose we need to perform calculations for the entire class.

Examples:

- Calculate the total marks in each subject.
- Add bonus marks to every student.
- Compare two examination results.
- Calculate class averages.
- Work with thousands of students simultaneously.

We need a structure that organizes all the numerical information together.

That structure is called a matrix.

---

# 2. WHAT IS A MATRIX?

A matrix is a rectangular arrangement of numbers organized into rows and columns.

Let's represent our complete dataset.

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

This is a matrix.

Each horizontal line is a row.

Each vertical line is a column.

### What do the rows represent?

Row 1: Rahul

Row 2: Priya

Row 3: Arjun

Row 4: Sneha

Row 5: Kiran

### What do the columns represent?

Column 1: Maths

Column 2: Physics

Column 3: Chemistry

### Important observation

A matrix allows us to represent multiple students and multiple subjects in one mathematical object.

This makes calculations on large collections of numbers easier to express.

---

# 3. WHAT IS MATRIX SHAPE?

The shape of a matrix tells us how many rows and columns it contains.

Our matrix has:

5 rows

3 columns

Therefore:

Shape = 5 × 3

Read this as:

"Five by three."

### General notation

If a matrix contains:

m rows

n columns

Its shape is:

m × n

We often write:

A ∈ ℝ^(m×n)

This means A is a real-valued matrix with m rows and n columns.

### Example 1

A =

[2 3]
[4 5]

Rows = 2

Columns = 2

Shape = 2 × 2

### Example 2

B =

[1 2 3]
[4 5 6]

Rows = 2

Columns = 3

Shape = 2 × 3

### Example 3

C =

[10]
[20]
[30]

Rows = 3

Columns = 1

Shape = 3 × 1

### Important rule

The first number always represents rows.

The second number always represents columns.

---

# 4. MATRIX ELEMENTS AND INDEXING

Suppose:

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

How do we identify one particular value?

We use two indices.

Notation:

Mᵢⱼ

Where:

i = Row number

j = Column number

### Example 1

Find M₂₃.

First index = 2.

Go to the second row.

[60, 85, 75]

Second index = 3.

Take the third element.

Answer:

M₂₃ = 75

This is Priya's Chemistry mark.

### Example 2

Find M₄₁.

Fourth row:

[50, 65, 70]

First column:

50

Therefore:

M₄₁ = 50

### Example 3

Find M₅₂.

Fifth row:

[75, 90, 80]

Second column:

90

Therefore:

M₅₂ = 90

### Important

Mathematics generally uses indexing starting from 1.

Many programming languages use indexing starting from 0.

We'll focus on mathematical notation for now.

---

# 5. TYPES OF MATRICES

Before learning operations, let's understand different matrix structures.

## 5.1 Row Matrix

A matrix containing exactly one row.

Example:

A = [80 70 90]

Shape = 1 × 3

This can also represent a row vector.

## 5.2 Column Matrix

A matrix containing exactly one column.

Example:

B =

[80]
[70]
[90]

Shape = 3 × 1

This can represent a column vector.

## 5.3 Rectangular Matrix

A matrix where the number of rows differs from the number of columns.

Our school dataset is a rectangular matrix.

Shape = 5 × 3

Other examples:

2 × 4

3 × 7

10 × 5

## 5.4 Square Matrix

A matrix with the same number of rows and columns.

Example:

A =

[1 2]
[3 4]

Shape = 2 × 2

Another example:

B =

[1 2 3]
[4 5 6]
[7 8 9]

Shape = 3 × 3

### Why are square matrices important?

Certain operations and concepts apply specifically to square matrices.

Examples include determinants, eigenvalues, and ordinary matrix inverses.

We'll study these separately.

## 5.5 Zero Matrix

A matrix whose elements are all zero.

Example:

Z =

[0 0 0]
[0 0 0]

Shape = 2 × 3

Zero matrices can have different shapes.

### Important

The zero matrix is different from the scalar zero.

A zero matrix has a defined number of rows and columns.

---

# 6. DIAGONAL AND IDENTITY MATRICES

These two types are important enough to introduce now.

## 6.1 Main Diagonal

Consider:

A =

[2 5 7]
[4 3 8]
[6 9 1]

The main diagonal runs from the top-left corner to the bottom-right corner.

Its elements are:

2, 3, 1

Their positions are:

A₁₁

A₂₂

A₃₃

## 6.2 Diagonal Matrix

A diagonal matrix is a square matrix whose elements outside the main diagonal are zero.

Example:

D =

[2 0 0]
[0 3 0]
[0 0 5]

The diagonal elements can have different values.

Some can even be zero.

## 6.3 Identity Matrix

An identity matrix is a special diagonal matrix.

Every diagonal element equals 1.

Every off-diagonal element equals 0.

Example:

I =

[1 0 0]
[0 1 0]
[0 0 1]

This is the 3 × 3 identity matrix.

### Why is it called an identity matrix?

Because multiplying a compatible matrix by an identity matrix leaves it unchanged.

This is similar to multiplying an ordinary number by 1.

For example:

5 × 1 = 5

For matrices:

AI = A

We will calculate this properly when we study matrix multiplication.

---

# 7. MATRIX EQUALITY

Two matrices are equal when:

1. They have the same shape.
2. Their corresponding elements are equal.

Example:

A =

[1 2]
[3 4]

B =

[1 2]
[3 4]

Therefore:

A = B

Now consider:

C =

[1 3]
[2 4]

Although C contains the same numbers, their positions differ.

Therefore:

A ≠ C

### Important

The order and position of matrix elements matter, just as they do for vectors.

---

# 8. MATRIX ADDITION

Now let's perform our first matrix operation.

Imagine our school conducts two tests.

Test 1 marks:

A =

[40 35 45]
[30 40 35]

Test 2 marks:

B =

[35 30 40]
[25 35 30]

The first row represents Rahul.

The second row represents Priya.

The columns represent Maths, Physics, and Chemistry.

We want their combined marks across both tests, subject by subject.

### Step 1: Add corresponding elements

A + B =

[40+35 35+30 45+40]
[30+25 40+35 35+30]

### Step 2: Calculate

A + B =

[75 65 85]
[55 75 65]

### Interpretation

Rahul's combined marks:

[Maths=75, Physics=65, Chemistry=85]

Priya's combined marks:

[Maths=55, Physics=75, Chemistry=65]

This is matrix addition.

### Definition

Matrix addition adds corresponding elements of two matrices of the same shape.

The result is another matrix with that same shape.

---

# 9. GENERAL FORMULA FOR MATRIX ADDITION

Suppose:

A =

[a b]
[c d]

B =

[e f]
[g h]

Then:

A+B =

[a+e b+f]
[c+g d+h]

More generally:

Cᵢⱼ = Aᵢⱼ + Bᵢⱼ

Where C is the resulting matrix.

Every element is added to the element at the same row and column position.

---

# 10. CAN WE ADD MATRICES OF DIFFERENT SHAPES?

Consider:

A =

[1 2 3]
[4 5 6]

Shape = 2 × 3

And:

B =

[7 8]
[9 10]

Shape = 2 × 2

Can we calculate A+B?

Not using ordinary matrix addition.

Why?

A has three columns.

B has two columns.

The elements do not have a complete one-to-one correspondence.

### Rule

For ordinary matrix addition:

Both matrices must have exactly the same shape.

Examples:

2 × 3 + 2 × 3: Valid.

3 × 3 + 3 × 3: Valid.

2 × 3 + 3 × 2: Invalid.

5 × 3 + 5 × 4: Invalid.

### Important distinction

Having the same number of elements is not enough.

For example:

A has shape 2 × 3.

B has shape 3 × 2.

Both contain six values.

But their shapes differ.

Therefore, ordinary addition is not defined.

---

# 11. MATRIX SUBTRACTION

Matrix subtraction follows the same component-wise approach.

Let's use our original school dataset.

Suppose Rahul and Priya take another examination.

Their original marks:

A =

[80 70 90]
[60 85 75]

Their new marks:

B =

[85 75 92]
[70 80 80]

We want to calculate how their marks changed.

Use:

B - A

### Step 1: Subtract corresponding elements

B-A =

[85-80 75-70 92-90]
[70-60 80-85 80-75]

### Step 2: Calculate

B-A =

[5 5 2]
[10 -5 5]

### Interpretation

Rahul:

Maths improved by 5.

Physics improved by 5.

Chemistry improved by 2.

Priya:

Maths improved by 10.

Physics decreased by 5.

Chemistry improved by 5.

### Important

Positive values represent increases.

Negative values represent decreases.

Zero represents no change.

This interpretation depends on subtracting the old marks from the new marks.

---

# 12. GENERAL FORMULA FOR MATRIX SUBTRACTION

Suppose A and B have the same shape.

Then:

C = A-B

Each element is:

Cᵢⱼ = Aᵢⱼ - Bᵢⱼ

Just like vector subtraction, matrix subtraction is generally not commutative.

A-B ≠ B-A

In fact:

A-B = -(B-A)

---

# 13. SCALAR MULTIPLICATION OF MATRICES

Remember Chapter 3?

We multiplied vectors by scalars.

For example:

2[3,4] = [6,8]

We can do the same with matrices.

Suppose:

A =

[40 35 45]
[30 40 35]

Multiply the matrix by 2.

2A =

[2×40 2×35 2×45]
[2×30 2×40 2×35]

Result:

2A =

[80 70 90]
[60 80 70]

### Definition

Scalar multiplication of a matrix means multiplying every element by the same scalar.

### General formula

If:

B = kA

Then:

Bᵢⱼ = kAᵢⱼ

### Important

Scalar multiplication does not change the matrix shape.

If A is 5 × 3, then 2A is also 5 × 3.

---

# 14. MULTIPLYING MATRICES BY NEGATIVE SCALARS

Suppose:

A =

[2 -3]
[4 5]

Multiply by -2.

-2A =

[-4 6]
[-8 -10]

Notice:

Positive elements become negative.

Negative elements become positive.

This is exactly the same rule we learned for vectors.

---

# 15. CALCULATING CLASS AVERAGES USING MATRIX OPERATIONS

Let's use our original five-student matrix.

M =

[80 70 90]
[60 85 75]
[90 80 85]
[50 65 70]
[75 90 80]

Suppose we want average subject marks across all five students.

### Step 1: Add the rows

[Maths total, Physics total, Chemistry total]

Maths:

80+60+90+50+75 = 355

Physics:

70+85+80+65+90 = 390

Chemistry:

90+75+85+70+80 = 400

We get the vector:

T = [355,390,400]

### Step 2: Divide by five

Average = (1/5)T

= [71,78,80]

Therefore:

Maths average = 71.

Physics average = 78.

Chemistry average = 80.

### Important observation

We used the matrix to organize the dataset and then used vector addition and scalar multiplication to calculate the averages.

Later, matrix multiplication will allow us to express related calculations more compactly.

---

# 16. MATRIX TRANSPOSE

Now let's introduce a different operation.

Suppose we have:

A =

[80 70 90]
[60 85 75]

This matrix contains:

2 rows.

3 columns.

Shape = 2 × 3

Now imagine we want to rearrange it so that subjects become rows and students become columns.

The result is:

Aᵀ =

[80 60]
[70 85]
[90 75]

This is called the transpose.

### Definition

The transpose of a matrix is obtained by exchanging rows and columns.

### What happened?

Original A:

2 × 3

Transpose Aᵀ:

3 × 2

Rows became columns.

Columns became rows.

---

# 17. UNDERSTANDING TRANSPOSE STEP BY STEP

Let's take a smaller matrix.

A =

[1 2 3]
[4 5 6]

Step 1:

First row of A:

[1,2,3]

Becomes the first column of Aᵀ.

Step 2:

Second row of A:

[4,5,6]

Becomes the second column of Aᵀ.

Result:

Aᵀ =

[1 4]
[2 5]
[3 6]

### General rule

If:

A has shape m × n

Then:

Aᵀ has shape n × m

### Element notation

(Aᵀ)ᵢⱼ = Aⱼᵢ

The row and column indices exchange positions.

---

# 18. PROPERTIES OF TRANSPOSE

## Property 1: Transposing twice

(Aᵀ)ᵀ = A

Let's verify.

A =

[1 2]
[3 4]

Transpose:

Aᵀ =

[1 3]
[2 4]

Transpose again:

(Aᵀ)ᵀ =

[1 2]
[3 4]

We get the original matrix.

## Property 2: Transpose of a sum

(A+B)ᵀ = Aᵀ+Bᵀ

Transposing a sum gives the same result as transposing the individual matrices and adding them.

## Property 3: Transpose of scalar multiplication

(kA)ᵀ = kAᵀ

Scaling and transposing can be performed in either order.

---

# 19. SYMMETRIC MATRICES

Now that we understand transpose, we can introduce symmetry.

Consider:

A =

[1 2]
[2 3]

Transpose it:

Aᵀ =

[1 2]
[2 3]

Nothing changes.

Therefore:

Aᵀ = A

This is called a symmetric matrix.

### Definition

A real square matrix is symmetric if it equals its transpose.

Example:

B =

[1 4 5]
[4 2 6]
[5 6 3]

Notice:

The element in row 1, column 2 equals the element in row 2, column 1.

The element in row 1, column 3 equals the element in row 3, column 1.

The element in row 2, column 3 equals the element in row 3, column 2.

This mirror-like structure defines symmetry.

### Why are symmetric matrices important?

They appear in many mathematical problems, including covariance matrices and optimization.

We'll study their deeper properties later.

---

# 20. MATRIX TRACE

The trace is a simple operation defined for square matrices.

It is the sum of the elements on the main diagonal.

Suppose:

A =

[2 5 7]
[4 3 8]
[6 9 1]

Main diagonal:

2, 3, 1

Trace:

tr(A) = 2+3+1

= 6

### Definition

The trace of a square matrix is the sum of its main diagonal elements.

### General formula

tr(A) = A₁₁ + A₂₂ + ... + Aₙₙ

We'll encounter trace again when studying more advanced matrix properties.

---

# 21. MATRIX ADDITION PROPERTIES

Suppose A, B, and C are matrices of the same shape.

### Commutative property

A+B = B+A

### Associative property

(A+B)+C = A+(B+C)

### Additive identity

A+0 = A

Here, 0 is the zero matrix of the same shape.

### Additive inverse

A+(-A) = 0

### Distributive scalar multiplication

k(A+B) = kA+kB

These are extensions of the arithmetic properties we've already learned for vectors.

---

# 22. IMPORTANT DISTINCTION: MATRIX ADDITION VS MULTIPLICATION

So far, we've studied matrix addition and subtraction.

Both operate element by element.

Example:

A =

[1 2]
[3 4]

B =

[5 6]
[7 8]

Addition:

A+B =

[6 8]
[10 12]

But standard matrix multiplication does not simply multiply corresponding elements.

It follows a different rule involving rows, columns, and dot products.

We are deliberately not covering that operation here.

It deserves a dedicated chapter.

---

# 23. COMMON BEGINNER MISTAKES

## Mistake 1: Confusing rows and columns

A 5 × 3 matrix contains five rows and three columns.

Not the other way around.

## Mistake 2: Assuming matrices with the same number of elements can be added

Incorrect.

Their shapes must match.

## Mistake 3: Thinking transpose changes the values

Transpose rearranges element positions.

It does not change the underlying numerical values.

## Mistake 4: Assuming every matrix is square

Incorrect.

Many matrices are rectangular.

## Mistake 5: Confusing zero matrix and scalar zero

A zero matrix contains multiple zero elements organized into rows and columns.

## Mistake 6: Assuming scalar multiplication changes shape

Incorrect.

It changes values but preserves shape.

## Mistake 7: Assuming matrix subtraction is commutative

Incorrect.

A-B generally differs from B-A.

## Mistake 8: Thinking trace is defined for all rectangular matrices

In standard linear algebra, trace is defined for square matrices.

---

# 24. INTERVIEW QUESTIONS AND ANSWERS

### Q1. What is a matrix?

A rectangular arrangement of numbers organized into rows and columns.

### Q2. What does matrix shape represent?

The number of rows and columns.

### Q3. What is the shape of a matrix containing 100 students and five subject marks each?

100 × 5

### Q4. When can two matrices be added?

When they have the same shape.

### Q5. Does scalar multiplication change matrix dimensions?

No.

### Q6. What is a transpose?

The operation of exchanging rows and columns.

### Q7. What happens to an m × n matrix after transposition?

It becomes n × m.

### Q8. What is a symmetric matrix?

A square matrix equal to its transpose.

### Q9. What is an identity matrix?

A square matrix with ones on the main diagonal and zeros elsewhere.

### Q10. What is a zero matrix?

A matrix whose elements are all zero.

### Q11. What is matrix trace?

The sum of the main diagonal elements of a square matrix.

### Q12. Is matrix addition commutative?

Yes, for matrices of matching shapes.

### Q13. Is matrix subtraction commutative?

Generally no.

### Q14. Can a matrix represent a collection of vectors?

Yes.

For example, each row of our school matrix represents one student's marks vector.

### Q15. Why are matrix shapes important?

Because the validity and result of matrix operations depend on their dimensions.

---

# 25. IMPORTANT FORMULAS

### Matrix Shape

A ∈ ℝ^(m×n)

### Matrix Addition

(A+B)ᵢⱼ = Aᵢⱼ+Bᵢⱼ

### Matrix Subtraction

(A-B)ᵢⱼ = Aᵢⱼ-Bᵢⱼ

### Scalar Multiplication

(kA)ᵢⱼ = kAᵢⱼ

### Matrix Transpose

(Aᵀ)ᵢⱼ = Aⱼᵢ

### Double Transpose

(Aᵀ)ᵀ = A

### Symmetric Matrix

Aᵀ = A

### Trace

tr(A) = Σᵢ₌₁ⁿ Aᵢᵢ

---

# 26. COMPLETE CHAPTER SUMMARY

We started with our familiar five-student dataset.

We learned how to organize all their marks into one matrix.

Then we studied:

1. Matrix rows and columns.
2. Matrix shapes.
3. Matrix element indexing.
4. Row and column matrices.
5. Rectangular and square matrices.
6. Zero matrices.
7. Diagonal matrices.
8. Identity matrices.
9. Matrix equality.
10. Matrix addition.
11. Matrix subtraction.
12. Scalar multiplication.
13. Matrix transpose.
14. Symmetric matrices.
15. Matrix trace.

The key idea is that matrices let us organize numerical values and apply mathematical operations to an entire rectangular structure.

### Remember

Matrix addition and subtraction work component by component.

Scalar multiplication scales every component.

Transpose exchanges rows and columns.

Matrix shape determines whether operations are valid.

---

END OF CHAPTER 6

NEXT CHAPTER: MATRIX MULTIPLICATION — FROM BASIC INTUITION TO STEP-BY-STEP CALCULATIONS
