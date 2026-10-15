# MATHEMATICS FOR MACHINE LEARNING

## PART 1: LINEAR ALGEBRA

### CHAPTER 1 — SCALARS, VECTORS, AND DIMENSIONS

---

# 1. WHAT IS LINEAR ALGEBRA?

Before learning any formulas, let's understand the purpose of linear algebra.

Imagine you're a schoolteacher.

You have five students, and each student has marks in three subjects.

You want to:

- Store their marks.
- Calculate their total marks.
- Compare two students.
- Find which students have similar marks.
- Calculate average marks.
- Understand patterns across subjects.

You could do these calculations one number at a time.

But what happens if the school has 10,000 students and 20 subjects?

Doing everything manually would become difficult.

We need a systematic way to organize numbers and perform calculations on many numbers together.

This is where linear algebra helps.

**Linear algebra is a branch of mathematics that studies vectors, matrices, and the operations and relationships involving them.**

For now, think of it as mathematics that helps us organize and work with collections of numbers.

This is a starting intuition, not its complete mathematical definition.

---

# 2. WHAT IS A SCALAR?

Let's start with the simplest possible idea.

Rahul scored 80 marks in Maths.

The number 80 is a scalar.

A scalar is simply one number.

Examples from our school:

Rahul's Maths marks = 80

Priya's Physics marks = 85

Arjun's Chemistry marks = 85

Each of these individual values is a scalar.

## 2.1 Can scalars be negative?

Yes.

Examples:

10

-5

0

3.5

-0.25

All are scalars.

A scalar doesn't have to be a positive whole number.

## 2.2 Scalar operations

We already know how to perform mathematical operations on ordinary numbers.

Addition:

80 + 10 = 90

Subtraction:

80 - 10 = 70

Multiplication:

80 × 2 = 160

Division:

80 ÷ 2 = 40

Nothing new so far.

## 2.3 Mathematical notation

Instead of writing:

Rahul's Maths marks = 80

We could write:

a = 80

Here, a is a symbol representing a numerical value.

We could also write:

b = 70

c = 90

These letters are variables holding scalar values.

### Important takeaway

A scalar is one number.

That's all you need to understand for now.

---

# 3. WHAT IS A VECTOR?

Now imagine that instead of looking at just Rahul's Maths marks, we want to look at all three subjects together.

Rahul scored:

Maths = 80

Physics = 70

Chemistry = 90

We could write:

80, 70, 90

But mathematics provides a more structured way:

**[80, 70, 90]**

This is a vector.

A vector is an ordered collection of numbers.

We can write:

Rahul = [80, 70, 90]

Each number represents one subject.

The first value represents Maths.

The second represents Physics.

The third represents Chemistry.

## 3.1 Why do we need a vector?

Because we want to represent multiple related numbers as one mathematical object.

Think about a student report card.

A report card groups subject marks together for one student.

Similarly, a vector groups numbers together in a specific order.

Instead of handling three separate values every time, we can work with the complete collection.

## 3.2 Another example

Priya scored:

Maths = 60

Physics = 85

Chemistry = 75

Her vector is:

Priya = [60, 85, 75]

Arjun's vector:

Arjun = [90, 80, 85]

Each student has a vector containing three numbers.

## 3.3 Are the square brackets important?

They help indicate that we're treating the numbers as a collection.

For now:

[80, 70, 90]

means a vector containing three components.

Later, we will distinguish row vectors and column vectors.

---

# 4. WHAT ARE VECTOR COMPONENTS?

Let's look at Rahul's vector.

R = [80, 70, 90]

This vector contains three individual numbers.

Each number is called a component or element.

First component = 80

Second component = 70

Third component = 90

We can label these components mathematically.

R₁ = 80

R₂ = 70

R₃ = 90

Read:

R₁ as "R subscript one."

R₂ as "R subscript two."

R₃ as "R subscript three."

The subscript tells us the position of the component.

It does not mean multiplication or an exponent.

For example:

R₂ does not mean R squared.

It means the second component of vector R.

## 4.1 What does the position represent?

In our example:

Position 1 = Maths

Position 2 = Physics

Position 3 = Chemistry

Therefore:

R₁ = Maths marks

R₂ = Physics marks

R₃ = Chemistry marks

This gives every position a clear meaning.

---

# 5. WHY DOES ORDER MATTER?

Suppose Rahul scored:

Maths = 80

Physics = 70

Chemistry = 90

Correct vector:

R = [80, 70, 90]

Now suppose someone writes:

[90, 70, 80]

Do these two vectors mean the same thing?

No.

Because we agreed on this order:

[Maths, Physics, Chemistry]

The second vector would mean:

Maths = 90

Physics = 70

Chemistry = 80

That's a different report card.

Even though the numbers are the same, their positions have changed.

**Important rule: The order of components matters in a vector.**

This becomes extremely important when we use vectors to represent real-world information.

---

# 6. WHAT IS A DIMENSION?

Now let's introduce a word that sounds complicated but is simple at this stage.

Dimension.

Rahul's vector:

R = [80, 70, 90]

How many components does this vector contain?

Three.

Therefore, Rahul's vector is a three-dimensional vector.

That's what dimension means here.

**The dimension of a coordinate vector is the number of components it contains.**

## 6.1 One-dimensional vector

Suppose we only record Maths marks.

R = [80]

Number of components = 1

Dimension = 1

## 6.2 Two-dimensional vector

Suppose we record Maths and Physics.

R = [80, 70]

Number of components = 2

Dimension = 2

## 6.3 Three-dimensional vector

Now record all three subjects.

R = [80, 70, 90]

Number of components = 3

Dimension = 3

## 6.4 What if we have 10 subjects?

Imagine Rahul studies 10 subjects.

His marks can be represented by a vector containing 10 numbers.

That would be a 10-dimensional vector.

What if there are 100 subjects?

A vector containing 100 components is 100-dimensional.

We don't need to visualize 100 physical directions.

Dimensions can represent abstract information, such as subjects, not just physical space.

---

# 7. UNDERSTANDING VECTOR NOTATION

In mathematics, you'll frequently encounter notation like:

R ∈ ℝ³

Let's understand every symbol.

R = The vector representing Rahul's marks.

∈ = "Belongs to" or "is an element of."

ℝ = The set of real numbers.

³ = Three components.

Therefore:

R ∈ ℝ³

means:

"Rahul's vector is a three-dimensional real-valued vector."

## 7.1 What are real numbers?

Real numbers include values such as:

5

-3

0

3.14

1/2

√2

They can appear on the ordinary number line.

For our school example, marks such as 80 and 70 are real numbers.

## 7.2 More examples

A vector containing two real numbers:

A ∈ ℝ²

A vector containing five real numbers:

B ∈ ℝ⁵

A vector containing ten real numbers:

C ∈ ℝ¹⁰

There is no need to memorize these as separate formulas.

The superscript simply indicates how many real-number components the vector contains.

---

# 8. WHAT IS A ROW VECTOR?

Let's return to Rahul.

R = [80, 70, 90]

Notice that we wrote the values horizontally.

That arrangement is called a row vector.

It has one row and three columns.

We say its shape is:

1 × 3

Read:

"One by three."

Why?

Because there is:

1 row

3 columns

## Example

[60, 85, 75]

This is also a row vector with shape 1 × 3.

---

# 9. WHAT IS A COLUMN VECTOR?

We can write the same numbers vertically.

R =

[80]
[70]
[90]

This is a column vector.

It contains:

3 rows

1 column

Therefore, its shape is:

3 × 1

## 9.1 Are row and column vectors the same?

They contain the same numerical components in the same sequence.

But their mathematical orientations differ.

Row vector:

[80, 70, 90]

Shape = 1 × 3

Column vector:

[80]
[70]
[90]

Shape = 3 × 1

This distinction matters when performing certain mathematical operations.

For now, remember:

**Dimension tells us how many components a vector contains.**

**Shape tells us how those components are arranged as rows and columns.**

A three-dimensional vector can be represented as either a 1 × 3 row or a 3 × 1 column.

---

# 10. WHAT IS A MATRIX?

Now imagine we want to represent all five students together.

Their marks are:

Rahul = [80, 70, 90]

Priya = [60, 85, 75]

Arjun = [90, 80, 85]

Sneha = [50, 65, 70]

Kiran = [75, 90, 80]

Instead of keeping five separate vectors, we can arrange them into one rectangular structure.

M =

[80, 70, 90]
[60, 85, 75]
[90, 80, 85]
[50, 65, 70]
[75, 90, 80]

This is called a matrix.

A matrix is a rectangular arrangement of numbers organized into rows and columns.

## 10.1 What does each row mean?

Row 1 = Rahul's marks

Row 2 = Priya's marks

Row 3 = Arjun's marks

Row 4 = Sneha's marks

Row 5 = Kiran's marks

## 10.2 What does each column mean?

Column 1 = Maths marks of all students

Column 2 = Physics marks of all students

Column 3 = Chemistry marks of all students

Therefore, the same matrix allows us to organize information both student-wise and subject-wise.

## 10.3 What is the shape of this matrix?

Count the rows.

5 students = 5 rows

Count the columns.

3 subjects = 3 columns

Therefore:

Matrix shape = 5 × 3

This is a five-by-three matrix.

## 10.4 What if we add more students?

Suppose the school has 100 students studying the same three subjects.

The matrix would have:

100 rows

3 columns

Shape = 100 × 3

What if the school has 100 students and 10 subjects?

Shape = 100 × 10

The same mathematical representation works regardless of the dataset's size.

---

# 11. UNDERSTANDING MATRIX ELEMENTS

A matrix contains individual elements.

We can identify any element using its row number and column number.

The notation is:

Mᵢⱼ

Here:

i = Row number

j = Column number

Example:

M₂₃

means:

The element in the second row and third column.

Look at our matrix:

Row 2 = Priya

Column 3 = Chemistry

Priya scored 75 in Chemistry.

Therefore:

M₂₃ = 75

Another example:

M₄₁

Row 4 = Sneha

Column 1 = Maths

Therefore:

M₄₁ = 50

This is how we refer to individual elements mathematically.

---

# 12. VECTOR VS MATRIX

Let's compare them.

A scalar:

80

One number.

A vector:

[80, 70, 90]

One ordered collection of numbers.

A matrix:

[80, 70, 90]
[60, 85, 75]
[90, 80, 85]

A rectangular arrangement of numbers.

### Important distinction

A vector is not simply "one student," and a matrix is not simply "many students."

Those descriptions are specific to our example.

Mathematically:

A vector is an element of a vector space.

A matrix is a rectangular array that can also represent a linear transformation.

We'll develop these deeper definitions gradually.

For now, our school dataset is a useful concrete way to understand both.

---

# 13. WHY WILL THIS MATTER WHEN LEARNING MACHINE LEARNING?

We are not studying algorithms yet.

But here's a simple connection.

Imagine we eventually want a computer to study our school dataset and identify patterns.

For example:

Do students with high Maths marks also tend to score well in Physics?

Which students have similar performance patterns?

Are some subjects strongly related?

To investigate such questions, the computer needs a numerical representation of the students.

Our vectors and matrices provide that representation.

Later, we'll learn mathematical operations that help us compare and transform these numbers.

That is how linear algebra connects to machine learning.

You don't need to know how ML algorithms work yet.

---

# 14. COMMON BEGINNER CONFUSIONS

## Q1. Is 80 a vector?

No, when written simply as the number 80, it is a scalar.

But [80] can represent a one-dimensional vector.

## Q2. Is [80, 70, 90] a vector?

Yes.

It is a vector with three components.

## Q3. Does a 3D vector always represent physical space?

No.

In our example, the three dimensions represent Maths, Physics, and Chemistry.

## Q4. Is a 3D vector always a 3 × 1 column?

No.

It may be represented as a row vector or a column vector.

The dimension is three in either case.

## Q5. Are dimensions and observations the same thing?

No.

In our school dataset:

Observations = 5 students

Components per student's vector = 3 subjects

So we have five observations, each represented by a three-dimensional vector.

## Q6. Is a matrix just a collection of vectors?

We can construct a matrix by stacking equally sized row vectors or column vectors.

But matrices also have deeper mathematical roles, including representing linear transformations.

We'll explore those later.

---

# 15. CHAPTER SUMMARY

We have covered the following concepts:

**Scalar:** A single number.

Example:

80

**Vector:** An ordered collection of numbers.

Example:

[80, 70, 90]

**Component:** One individual value within a vector.

Example:

The second component is 70.

**Dimension:** The number of components in a vector.

Example:

[80, 70, 90] has three dimensions.

**Row Vector:** A vector arranged horizontally.

Example:

[80, 70, 90]

Shape = 1 × 3

**Column Vector:** A vector arranged vertically.

Shape = 3 × 1

**Matrix:** A rectangular arrangement of numbers.

Our complete school dataset is a 5 × 3 matrix.

**Matrix Element:** An individual value identified by its row and column positions.

Example:

M₂₃ = 75

**Most important takeaway:**

Scalars represent individual numbers.

Vectors organize numerical components.

Matrices organize numerical values into rows and columns.

These ideas form the starting foundation for studying linear algebra and machine learning.

---

END OF CHAPTER 1
