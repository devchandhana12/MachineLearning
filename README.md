# Machine Learning & Applied AI 🚀

A hands-on repository documenting my journey through **Machine Learning, Deep Learning, Transformers, and Applied AI**.

The goal of this repository is not just to collect code, but to build a strong understanding of **why algorithms work, how they behave, how to implement them, and how they fit into real-world AI systems**.

---

## 🎯 What I'm Working Towards

My learning path runs in parallel across two tracks:

### 🧠 Core Machine Learning

Building strong foundations in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Supervised learning
* Unsupervised learning
* Model evaluation
* Optimization
* Deep learning fundamentals
* MLOps and model deployment

### 🤖 Modern AI

Understanding and building systems around:

* Transformer architecture
* NLP
* Embeddings
* Retrieval-Augmented Generation (RAG)
* Vector databases
* AI agents
* Tool calling
* Fine-tuning
* LoRA / QLoRA
* AI system design

The idea is to eventually connect both tracks — understanding modern AI systems without treating the underlying ML concepts as a black box.

---

# 📚 Current Progress

## Data Preparation & Preprocessing

Working with real datasets to understand how data should be inspected, cleaned, validated, and transformed before training a model.

Topics covered include:

* Dataset grain / unit of observation
* Schema understanding
* Numerical vs categorical features
* Target variables
* Target leakage
* Missing values
* MCAR / MAR / MNAR
* Structural vs actual missing values
* Duplicate records
* Inconsistent categories
* Data validation
* Categorical encoding
* Feature preprocessing

Practical work includes datasets such as:

* Kaggle House Prices
* Customer Churn datasets
* Other datasets used for preprocessing and experimentation

---

## Machine Learning Algorithms

### ✅ Linear Regression

Covered:

* Regression intuition
* Linear relationships
* Prediction equation
* Loss functions
* Mean Squared Error
* Gradient Descent
* Learning rate
* Weight updates
* Underfitting and overfitting
* Training vs inference

---

### ✅ Logistic Regression

Covered:

* Binary classification
* Linear decision function
* Sigmoid function
* Probabilities
* Decision boundaries
* Binary Cross Entropy
* Log loss intuition
* Gradient Descent
* Weight updates
* Training vs inference
* Multiple input features
* Categorical features
* One-Hot Encoding

Currently expanding this into more realistic classification problems.

---

## 🔄 Coming Next

Machine Learning:

* Evaluation metrics

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
  * Confusion Matrix
* Regularization
* Decision Trees
* Random Forest
* Ensemble methods
* KNN
* Naive Bayes
* Support Vector Machines
* Clustering
* Feature engineering
* Model selection

---

# 🧠 Transformer & AI Track

Alongside traditional ML, I'm revisiting Transformer architecture from the ground up.

The goal here is to understand what actually happens inside modern language models rather than only learning how to call an API.

Topics include:

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Token Embeddings
 ↓
Positional Information
 ↓
Self-Attention
 ↓
Multi-Head Attention
 ↓
Feed Forward Networks
 ↓
Transformer Blocks
 ↓
Language Model
```

This track will gradually cover:

* Tokenization
* Embeddings
* Attention
* Self-Attention
* Query, Key and Value
* Multi-Head Attention
* Feed Forward Networks
* Residual connections
* Layer normalization
* Encoder / Decoder architecture
* GPT-style decoder-only models
* Training objectives
* Inference
* Fine-tuning

Wherever a Transformer concept depends on a Machine Learning or Deep Learning prerequisite, I study that prerequisite alongside it.

---

# 🏗️ AI System Design

System design is part of the learning process from the beginning rather than something reserved for later.

For each major topic, I try to understand questions such as:

* Where does this component sit in a production system?
* What happens during training vs inference?
* How does data flow through the system?
* What are the scalability bottlenecks?
* What should be cached?
* Where should preprocessing happen?
* How should models be served?
* How are embeddings stored?
* How do different AI services communicate?
* How do latency, accuracy and cost trade-offs affect architecture?

Eventually this will expand into architectures for:

* ML prediction systems
* RAG systems
* AI agents
* LLM applications
* Fine-tuned model serving

---

# 🧪 Learning Approach

I try to follow the same pattern for every major concept:

```text
Problem
   ↓
Why do we need it?
   ↓
Intuition
   ↓
Concept
   ↓
Math
   ↓
Implementation
   ↓
Experiment
   ↓
Real-world system design
```

I prefer understanding **why something works before using the abstraction provided by a library**.

Libraries like Scikit-learn, PyTorch, Hugging Face and LangChain are incredibly useful — but the goal is to understand what is happening underneath them as well.

---

# 🛠️ Tech Stack

### Machine Learning & Data

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter

### Deep Learning & AI

* PyTorch
* Hugging Face
* Transformers
* LangChain
* LangGraph

### Applied AI

* FastAPI
* Qdrant
* Pinecone
* ChromaDB
* Redis

More tools will be added as the repository grows.

---

# 📂 Repository Structure

The repository will gradually follow a structure similar to:

```text
machine-learning/
│
├── data-preprocessing/
│   ├── notes/
│   ├── notebooks/
│   └── projects/
│
├── supervised-learning/
│   ├── linear-regression/
│   ├── logistic-regression/
│   └── ...
│
├── unsupervised-learning/
│
├── model-evaluation/
│
├── feature-engineering/
│
├── deep-learning/
│
├── transformers/
│   ├── tokenization/
│   ├── embeddings/
│   ├── attention/
│   └── ...
│
├── applied-ai/
│   ├── rag/
│   ├── agents/
│   └── fine-tuning/
│
├── projects/
│
└── README.md
```

The structure may evolve as the learning journey progresses.

---

# 🧩 Projects

This repository will contain small experiments as well as complete end-to-end projects.

Projects will focus on areas such as:

* Data preprocessing
* Classification
* Regression
* Customer churn prediction
* Feature engineering
* Model evaluation
* RAG
* AI agents
* Fine-tuning
* ML / AI system design

The intention is to move beyond isolated notebooks and gradually build **production-oriented ML and AI systems**.

---

# 📈 Current Focus

Right now my main focus is:

**Machine Learning**

```text
Data Preprocessing
        ↓
Linear Regression ✅
        ↓
Logistic Regression ✅
        ↓
Evaluation Metrics
        ↓
More Supervised ML Algorithms
```

**Modern AI**

```text
Transformer Architecture
        ↓
RAG
        ↓
AI Agents
        ↓
Fine-Tuning
        ↓
LoRA / QLoRA
```

Both tracks will eventually converge as I move deeper into **Machine Learning and AI Engineering**.

---

## 🌱 Why This Repository Exists

This repository is primarily my engineering notebook.

Instead of only watching courses or reading theory, I'm documenting concepts, writing implementations, breaking things, experimenting with datasets, and gradually connecting the pieces together.

The goal is simple:

> **Understand the fundamentals deeply enough to build useful ML and AI systems without treating the model as a black box.**

---

### 🚧 Work in Progress

This repository is actively being updated as I learn, experiment, and build.

Expect plenty of:

**notes → experiments → mistakes → fixes → projects → better systems.**
