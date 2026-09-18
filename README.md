# Machine Learning & Applied AI 🚀

A hands-on engineering repository documenting my work across **Machine Learning, Transformers, RAG, AI Agents, Fine-Tuning, and AI System Design**.

I come from a software engineering background, and this repository represents my transition deeper into **Machine Learning and Applied AI Engineering**.

Rather than treating ML algorithms or LLMs as black boxes, my goal is to understand:

* **Why** an approach works
* What happens mathematically during training
* How to implement the underlying concepts
* How libraries abstract those concepts
* How models behave during inference
* How everything fits into a real production system

---

# 🎯 Current Direction

My learning currently runs across two parallel tracks:

```text
                Machine Learning
                       │
        Supervised Learning Foundations
                       │
              Decision Trees  ← Current
                       │
       Ensembles / Model Evaluation
                       │
              Deeper ML Concepts
                       │
                       ▼
                Deep Learning
                       │
                       │
                       ▼
            Modern AI / Transformers
                       ▲
                       │
            Transformer Architecture
                       │
                  RAG Systems
                       │
                   AI Agents
                       │
            Fine-Tuning / LoRA / QLoRA
```

These are not completely separate tracks.

Whenever something on the AI side depends on a Machine Learning or Deep Learning concept, I go back and understand that prerequisite before continuing.

---

# 🧠 Machine Learning

## ✅ Data Preparation & Preprocessing

Before training models, I started by understanding what happens to real-world data before it ever reaches an algorithm.

Topics explored include:

* Dataset grain / unit of observation
* Dataset and schema understanding
* Numerical vs categorical variables
* Continuous, discrete and ordinal features
* Target variables
* Target leakage
* Duplicate records
* Missing values
* MCAR / MAR / MNAR
* Structural vs actual missing values
* Inconsistent categories
* Conditional missing-value handling
* Categorical encoding
* One-Hot Encoding
* Data validation
* Feature preprocessing

### Practical Work

One of the main datasets used for this is the **Kaggle House Prices / Ames Housing dataset**, where I'm treating preprocessing as a real data-understanding problem rather than simply running `fillna()` across columns.

Examples include understanding relationships such as:

```text
GarageType = NULL
        │
        ├── GarageArea = 0
        │       ↓
        │   No Garage
        │
        └── GarageArea > 0
                ↓
         Potential Data Issue
```

The goal is to distinguish between:

```text
Missing because information is unavailable
                    vs
Missing because the feature does not exist
```

---

# 📈 Supervised Learning

## ✅ Linear Regression

Covered from intuition through implementation.

### Concepts

* Regression intuition
* Linear relationships
* Prediction equation
* Features and weights
* Bias / intercept
* Loss functions
* Mean Squared Error
* Gradient Descent
* Learning rate
* Weight updates
* Convergence intuition
* Underfitting
* Overfitting
* Training vs inference

The emphasis was understanding the full flow:

```text
Features
   ↓
Prediction
   ↓
Compare with Actual Value
   ↓
Calculate Loss
   ↓
Calculate Gradients
   ↓
Update Weights
   ↓
Repeat
```

And importantly:

```text
Training
   ↓
Prediction → Loss → Gradient → Weight Update

Inference
   ↓
Prediction only
```

---

## ✅ Logistic Regression

Studied binary classification from the linear model all the way through the optimization process.

### Binary Classification

Covered:

* Classification intuition
* Linear decision function
* Logits
* Sigmoid
* Probability interpretation
* Classification thresholds
* Decision boundaries
* Binary Cross Entropy
* Log loss intuition
* Gradient Descent
* Weight updates
* Multiple features
* Categorical variables
* One-Hot Encoding
* Training vs inference
* Underfitting and overfitting

### Multiclass Classification

Extended the same ideas into multiclass problems.

Covered:

* Binary vs multiclass classification
* Multiple class logits
* Softmax
* Why Softmax probabilities sum to `1`
* Probability distributions across classes
* One-Hot encoded targets
* Argmax
* Categorical Cross-Entropy
* Correct-class probability

Relationship between binary and multiclass classification:

```text
Binary Classification
        ↓
Sigmoid
        ↓
Binary Cross Entropy
```

```text
Multiclass Classification
        ↓
Softmax
        ↓
Categorical Cross Entropy
```

Example:

```text
Classes:
[Cat, Horse, Dog]

Model logits
      ↓
Softmax
      ↓
[0.12, 0.32, 0.56]
                  ↑
             Highest Probability

Actual:
[0, 0, 1]

Prediction:
Dog ✅

Categorical Cross Entropy
        ↓
Measures how much probability
the model assigned to the
correct class.
```

Important concepts are also implemented with **plain Python and NumPy** before relying completely on higher-level ML libraries.

---

# 🌳 Decision Trees

### 🚧 Currently Studying

Current Machine Learning topic.

The goal is to understand:

* Why Decision Trees are needed
* How a tree chooses a split
* Decision boundaries
* Entropy
* Gini Impurity
* Information Gain
* Recursive splitting
* Leaf nodes
* Stopping conditions
* Overfitting in trees
* Tree depth
* Feature importance

This will naturally lead into:

```text
Decision Tree
      ↓
Random Forest
      ↓
Boosting
      ↓
Gradient Boosting / XGBoost
```

---

# 📊 Model Evaluation

Model evaluation is studied alongside algorithms where it becomes relevant rather than as an isolated collection of formulas.

Topics include:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC
* AUC
* Classification thresholds
* Class imbalance
* Train / validation / test splits
* Cross-validation
* Bias vs variance

---

# 🗺️ ML Roadmap

My current priority is to build depth across important supervised learning algorithms before expanding further into classical ML.

```text
Preprocessing ✅
      ↓
Linear Regression ✅
      ↓
Logistic Regression ✅
      ↓
Softmax / Multiclass Classification ✅
      ↓
Decision Trees 🚧
      ↓
Random Forest
      ↓
Boosting / XGBoost
      ↓
KNN / Naive Bayes / SVM
      ↓
Model Selection & Cross-Validation
      ↓
Deep Learning Foundations
```

Unsupervised learning topics such as **K-Means and PCA** remain part of the broader roadmap, but my current priority is supervised ML, Deep Learning fundamentals, and Applied AI.

---

# 🤖 Transformer & Modern AI Track

In parallel with classical Machine Learning, I'm studying modern AI systems starting from the architecture underneath Large Language Models.

The goal is not merely:

```python
model.generate(...)
```

but understanding what happens before that call works.

---

# 🧠 Transformer Architecture

The Transformer track starts from raw text and follows the information through the model.

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
Positional Information
   ↓
Self-Attention
   ↓
Multi-Head Attention
   ↓
Feed Forward Network
   ↓
Residual Connections
   ↓
Layer Normalization
   ↓
Transformer Blocks
   ↓
Output Representation
   ↓
Language Model Head
   ↓
Next Token Probabilities
```

Topics being explored include:

* Tokenization
* Vocabulary
* Token IDs
* Different tokenization strategies
* Embeddings
* Positional information
* Attention
* Self-Attention
* Query, Key and Value
* Scaled Dot-Product Attention
* Multi-Head Attention
* Feed Forward Networks
* Residual connections
* Layer normalization
* Encoder architecture
* Decoder architecture
* Decoder-only Transformers
* GPT-style architectures
* Training vs inference
* Next-token prediction

Whenever concepts such as neural networks, activation functions, optimization or backpropagation become necessary, I connect them back to the corresponding Deep Learning fundamentals.

---

# 🔎 Retrieval-Augmented Generation

I've already worked with RAG systems and am revisiting them with more emphasis on **retrieval quality and system design**.

Areas of focus include:

* Document ingestion
* Chunking strategies
* Embeddings
* Vector databases
* Semantic search
* Dense retrieval
* Sparse retrieval
* Hybrid search
* Reranking
* Context construction
* Prompt construction
* Retrieval evaluation
* Caching
* Latency
* Production architecture

Typical architecture:

```text
Documents
    ↓
Chunking
    ↓
Embedding Model
    ↓
Vector Database
    │
    │
User Query
    ↓
Query Embedding
    ↓
Retrieval
    ↓
Reranking
    ↓
Relevant Context
    ↓
Prompt Construction
    ↓
LLM
    ↓
Response
```

---

# 🤖 AI Agents

The next major Applied AI implementation focus is moving beyond static LLM workflows into systems capable of deciding when and how to use tools.

Topics include:

* Tool calling
* Structured outputs
* Agent state
* Tool selection
* Workflow orchestration
* LangGraph
* Memory
* Error handling
* Retry strategies
* Human-in-the-loop workflows
* Multi-step execution
* Agent observability

Typical flow:

```text
User
 ↓
LLM / Agent
 ↓
Understand Task
 ↓
Select Tool
 ↓
Execute Tool
 ↓
Observe Result
 ↓
Decide Next Action
 ↓
Final Response
```

The goal is to build agents as **software systems**, not just prompts wrapped around an LLM.

---

# 🔧 Fine-Tuning

After strengthening RAG and agent implementations, the next major area is model fine-tuning.

Topics planned include:

* When to fine-tune
* Fine-tuning vs RAG
* Dataset preparation
* Instruction datasets
* Supervised Fine-Tuning
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* LoRA
* QLoRA
* Quantization
* Training configuration
* Evaluation
* Model serving
* GPU memory considerations

A major focus will be understanding the engineering trade-off:

```text
Prompt Engineering
        vs
RAG
        vs
Fine-Tuning
```

rather than automatically choosing one technique.

---

# 🏗️ ML & AI System Design

System design is not something I'm postponing until after learning the algorithms.

For every major topic, I try to ask:

* Where does this model live in a production architecture?
* Where does preprocessing happen?
* What happens during training?
* What happens during inference?
* Where is the model stored?
* How is it deployed?
* How do clients communicate with it?
* What needs to be cached?
* What happens when traffic increases?
* How is the system monitored?
* What should happen when the model fails?
* How do we evaluate output quality?
* What are the latency / accuracy / cost trade-offs?

This eventually connects concepts across:

```text
Data
 ↓
Training Pipeline
 ↓
Model Artifact
 ↓
Model Registry
 ↓
Serving Layer
 ↓
API
 ↓
Application
 ↓
Monitoring
 ↓
Feedback
```

and modern AI architectures such as:

```text
Client
 ↓
API Gateway
 ↓
AI Service
 ├── LLM
 ├── Retriever
 ├── Vector DB
 ├── Agent
 ├── Tools
 └── Cache
 ↓
Observability
```

---

# 🧪 How I Learn

My preferred learning loop is:

```text
Why does this exist?
        ↓
Build intuition
        ↓
Understand the concept
        ↓
Understand the necessary math
        ↓
Implement it manually
        ↓
Implement with libraries
        ↓
Experiment with real data
        ↓
Understand failure cases
        ↓
Think about production/system design
```

I don't aim to derive every equation from first principles before moving forward.

The goal is to understand each concept deeply enough to **reason about it, implement it, debug it, explain it, and use it correctly** — then deepen specific areas when necessary.

---

# 🛠️ Tech Stack

### Machine Learning & Data

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter
* Kaggle

### Deep Learning & LLMs

* PyTorch
* Hugging Face
* Transformers
* PEFT
* Hugging Face Datasets

### Applied AI

* LangChain
* LangGraph
* FastAPI
* Qdrant
* Pinecone
* ChromaDB
* Redis

---

# 🧩 Projects

This repository will gradually contain both focused experiments and complete end-to-end projects.

Areas include:

* Data preprocessing pipelines
* Regression
* Classification
* Structured-data ML
* Transformer experiments
* RAG systems
* AI Agents
* Fine-tuning
* ML APIs
* AI system design

For larger projects, the aim is to document not just:

> **What did I build?**

but also:

> **Why was it designed this way?**

Including architecture, trade-offs, failure cases, and possible improvements.

---

# 📍 Current Focus

### Machine Learning

```text
Logistic Regression ✅
        ↓
Softmax & Multiclass Classification ✅
        ↓
Decision Trees 🚧
        ↓
Random Forest
        ↓
Boosting / XGBoost
        ↓
Broader Supervised ML
```

### Applied AI

```text
Transformer Fundamentals 🚧
        ↓
RAG Review + System Design
        ↓
Build an AI Agent
        ↓
Fine-Tuning
        ↓
LoRA / QLoRA
```

### Longer Term

```text
Machine Learning
       +
Deep Learning
       +
Transformers
       +
Production AI
       ↓
Applied ML / AI Engineering
```

---

# 🌱 Why This Repository Exists

This repository is my engineering notebook for Machine Learning and AI.

Instead of only consuming courses or tutorials, I'm using it to document:

* Concepts
* Questions
* Implementations
* Experiments
* Mistakes
* Debugging
* Projects
* Architecture decisions
* System design

The goal is not to collect as many algorithms as possible.

The goal is to become capable of looking at an ML or AI problem and reasoning through:

```text
What problem are we solving?

Why should we use this approach?

How does it work?

How should we evaluate it?

How should we implement it?

How should we deploy it?

What breaks at scale?

What would I change in production?
```

> **Understand enough of the fundamentals to build, debug and design real Machine Learning and AI systems — without treating the model as magic.**

---

### 🚧 Work in Progress

This repository is continuously evolving as I learn and build.

```text
Understand
   ↓
Implement
   ↓
Break
   ↓
Debug
   ↓
Build
   ↓
Design
   ↓
Improve
```
