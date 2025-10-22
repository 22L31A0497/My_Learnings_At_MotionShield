# 🧠 Mastering PyTorch Tokenization :

  
---

## 📌 Table of Contents
1. [What Is Tokenization?](#what-is-tokenization)
2. [Why Tokenization Matters in NLP](#why-tokenization-matters-in-nlp)
3. [Types of Tokens](#types-of-tokens)
4. [Visualizing Tokenization with OpenAI Tokenizer](#visualizing-tokenization-with-openai-tokenizer)
5. [Setting Up PyTorch Tokenization](#setting-up-pytorch-tokenization)
6. [Installing torchtext](#installing-torchtext)
7. [Using `get_tokenizer` from torchtext](#using-get_tokenizer-from-torchtext)
8. [Tokenizing a Sample Sentence](#tokenizing-a-sample-sentence)
9. [Printing and Understanding Token Output](#printing-and-understanding-token-output)
10. [Why Tokenization Is Crucial for Language Models](#why-tokenization-is-crucial-for-language-models)
11. [Next Steps: Embeddings](#next-steps-embeddings)

---

## 🧩 What Is Tokenization?

Tokenization is the process of breaking down a sequence of text into smaller units called **tokens**. These tokens can be:
- Words
- Subwords
- Characters
- Phrases

### Example:
```text
Sentence: "PyTorch tokenization becomes easy"
Tokens: ["PyTorch", "tokenization", "becomes", "easy"]
```

---

## 🧠 Why Tokenization Matters in NLP

Tokenization is the **first and most essential step** in any NLP pipeline. Before a model can understand or generate text, it must convert raw sentences into structured tokens.

### Key Reasons:
- Enables numerical representation of text
- Helps models learn patterns and relationships
- Prepares data for embeddings and training

---

## 🧱 Types of Tokens

Depending on the strategy, tokens can vary:
- **Word-level**: Each word is a token
- **Subword-level**: Words are split into smaller parts (e.g., "tokenization" → "token", "ization")
- **Character-level**: Each character is a token

### Strategy Choice:
- Depends on language complexity
- Influences model performance and vocabulary size

---

## 🔍 Visualizing Tokenization with OpenAI Tokenizer

The video demonstrates tokenization using OpenAI’s tokenizer interface.

### Example Input:
```text
"With PyTorch, tokenization becomes easy"
```

### Output:
- **Token count**: 8
- **Character count**: 38
- Each word is split into smaller chunks based on encoding rules

### Adding More Words:
```text
"With PyTorch, tokenization becomes easy and everyone can do it"
```
- **Token count**: 14
- **Character count**: 64

This shows how token count increases with sentence length.

---

## ⚙️ Setting Up PyTorch Tokenization

To implement tokenization in PyTorch, the video uses the `torchtext` library.

### Required Tool:
- `torchtext` — a PyTorch NLP toolkit

---

## 📦 Installing torchtext

Use pip to install:
```bash
pip install torchtext
```

This library includes tokenizers, datasets, and preprocessing tools for NLP tasks.

---

## 🧰 Using `get_tokenizer` from torchtext

Import the tokenizer:
```python
from torchtext.data.utils import get_tokenizer
```

### Available Tokenizers:
- `"basic_english"` — splits text using simple English rules
- Others include `"spacy"`, `"moses"`, `"toktok"`

---

## ✂️ Tokenizing a Sample Sentence

Define your input text:
```python
text = "With PyTorch, tokenization becomes easy and everyone can do it"
```

Initialize tokenizer:
```python
tokenizer = get_tokenizer("basic_english")
```

Apply tokenizer:
```python
tokens = tokenizer(text)
```

---

## 🖨️ Printing and Understanding Token Output

Print the tokens:
```python
print(tokens)
```

### Output:
```text
["with", "pytorch", "tokenization", "becomes", "easy", "and", "everyone", "can", "do", "it"]
```

Each word is now a separate token, ready for embedding or model input.

---

## 🧬 Why Tokenization Is Crucial for Language Models

Before training any NLP model — especially large language models (LLMs) — tokenization is a must.

### Real-World Use:
- GPT, BERT, and other LLMs tokenize billions of sentences
- Tokenization enables models to learn grammar, semantics, and context

### Without Tokenization:
- Models cannot process raw text
- No way to convert words into numerical form

---

Absolutely, Jagan! Here's a full-length, beginner-friendly Markdown breakdown of the current video you're watching — ["Word Embeddings NLP Tutorial with PyTorch and GloVe"](https://www.youtube.com/watch?v=lXD5yzMBRaQ). This version is structured for GitHub documentation or printable study sheets, with expanded explanations, layered headings, and practical code insights.

---

# 🧠 Word Embeddings with PyTorch and GloVe

---

## 📌 Table of Contents
1. [What Are Word Embeddings?](#what-are-word-embeddings)
2. [Why Word Embeddings Matter in NLP](#why-word-embeddings-matter-in-nlp)
3. [Understanding GloVe](#understanding-glove)
4. [Installing Required Libraries](#installing-required-libraries)
5. [Importing PyTorch and torchtext Modules](#importing-pytorch-and-torchtext-modules)
6. [Tokenizing Text](#tokenizing-text)
7. [Creating GloVe Embedding Object](#creating-glove-embedding-object)
8. [Converting Tokens to Indexes](#converting-tokens-to-indexes)
9. [Generating Word Embeddings](#generating-word-embeddings)
10. [Printing Embeddings and Shape](#printing-embeddings-and-shape)
11. [Final Thoughts](#final-thoughts)

---

## 🧩 What Are Word Embeddings?

Word embeddings are dense vector representations of words in a high-dimensional space. They allow machines to understand relationships between words based on their meaning and usage.

### Key Concepts:
- Each word is mapped to a vector of real numbers.
- Similar words have similar vectors.
- Embeddings capture **semantic** (meaning) and **syntactic** (structure) relationships.

---

## 🧠 Why Word Embeddings Matter in NLP

Embeddings are the foundation of modern NLP models. They transform raw text into numerical form, enabling deep learning models to process language.

### Benefits:
- Capture word similarity and context.
- Reduce dimensionality compared to one-hot encoding.
- Enable transfer learning via pre-trained models.

---

## 📦 Understanding GloVe

**GloVe** stands for **Global Vectors for Word Representation**. It’s an unsupervised learning algorithm that learns word embeddings from word co-occurrence statistics in a corpus.

### Features:
- Trained on massive datasets (Wikipedia, Common Crawl).
- Embeddings available in various dimensions (50, 100, 200, 300).
- Captures global context better than Word2Vec.

---

## ⚙️ Installing Required Libraries

You’ll need:
- `torch`
- `torchtext`

### Installation:
```bash
pip install torch torchtext
```

These libraries provide tools for tokenization, vocabulary management, and embedding layers.

---

## 📥 Importing PyTorch and torchtext Modules

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
import torch
```

These imports allow access to:
- Tokenizer functions
- Pre-trained GloVe embeddings
- Tensor operations

---

## ✂️ Tokenizing Text

Tokenization splits text into individual words or subwords.

### Sample Code:
```python
text = "This is my first word embedding project"
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)
print(tokens)
```

### Output:
```text
['this', 'is', 'my', 'first', 'word', 'embedding', 'project']
```

Each word is now a token, ready for embedding.

---

## 🧰 Creating GloVe Embedding Object

Initialize GloVe with desired dimension:
```python
glove = GloVe(name='6B', dim=100)
```

### Parameters:
- `name='6B'`: Refers to the 6 billion token corpus.
- `dim=100`: Each word will be represented by a 100-dimensional vector.

---

## 🔢 Converting Tokens to Indexes

Use GloVe’s `stoi` (string-to-index) method:
```python
indexes = [glove.stoi[token] for token in tokens]
print(indexes)
```

### Note:
- Misspelled or unknown words may raise a KeyError.
- Ensure correct spelling before embedding.

---

## 🧬 Generating Word Embeddings

Convert indexes to vectors:
```python
embeddings = glove.vectors[indexes]
print(embeddings)
```

Convert to PyTorch tensor:
```python
embeddings = torch.tensor(embeddings)
```

Now each word is represented as a dense vector.

---

## 📐 Printing Embeddings and Shape

Check the shape of the embedding matrix:
```python
print(embeddings.shape)
```

### Example Output:
```text
torch.Size([7, 100])
```

This means:
- 7 tokens
- Each represented by a 100-dimensional vector

---

## ✅ Final Thoughts

This tutorial walks through the full pipeline:
1. Raw text → Tokenization
2. Tokens → Indexes
3. Indexes → Embeddings

### Why It Matters:
- Embeddings are the input to NLP models.
- Pre-trained vectors like GloVe save time and improve accuracy.
- PyTorch + torchtext make implementation simple and modular.

---
# 🧠 Transformers Explained 

> A comprehensive walkthrough of the Transformer architecture powering models like GPT and BERT. Includes intuitive analogies, vector math, embeddings, attention mechanisms, and architecture breakdown.

---

## 📌 Table of Contents
1. [Introduction to Transformers](#introduction-to-transformers)
2. [Language Models & Word Prediction](#language-models--word-prediction)
3. [Word Embeddings](#word-embeddings)
4. [Static vs Contextual Embeddings](#static-vs-contextual-embeddings)
5. [Transformer Architecture Overview](#transformer-architecture-overview)
6. [Encoder & Decoder Roles](#encoder--decoder-roles)
7. [Attention Mechanism](#attention-mechanism)
8. [Query, Key, Value Analogy](#query-key-value-analogy)
9. [Multi-Head Attention](#multi-head-attention)
10. [Feedforward Network](#feedforward-network)
11. [Training & Backpropagation](#training--backpropagation)
12. [Final Summary](#final-summary)

---

## 🧭 Introduction to Transformers

Transformers are a type of deep learning architecture introduced in the 2017 paper *“Attention is All You Need”* by Google researchers. They revolutionized natural language processing (NLP) by replacing older models like RNNs and LSTMs with a parallelizable, attention-based system.

### Key Features:
- **Parallel processing**: Unlike RNNs, Transformers process entire sequences simultaneously.
- **Contextual understanding**: They capture long-range dependencies between words.
- **Scalability**: Transformers scale well with data and compute, enabling massive models like GPT-4.

---

## ✍️ Language Models & Word Prediction

A language model is a system trained to predict the next word in a sentence. This simple task underpins powerful applications like autocomplete, translation, and chatbots.

### Example:
- Input: “I made a sweet Indian rice dish…”
- Output: “Kheer”, “Pongal”, “Biryani” — depending on context.

### GPT vs BERT:
- **GPT (Generative Pre-trained Transformer)**: Predicts next word (causal language modeling).
- **BERT (Bidirectional Encoder Representations from Transformers)**: Understands context from both directions (masked language modeling).

---

## 🔢 Word Embeddings

Words must be converted into numbers for machines to process them. This conversion is called **embedding** — a vector representation of a word.

### Static Embedding Example:
- “King” → [Authority: 1, Rich: 1, Gender: -1, Tail: 0]
- “Queen” → [Authority: 1, Rich: 1, Gender: 1, Tail: 0]

### Vector Math:
```text
King - Man + Woman ≈ Queen
```
This shows how semantic relationships can be encoded in vector space.

---

## 🧱 Static vs Contextual Embeddings

### Static Embeddings:
- Fixed meaning for each word.
- Models: Word2Vec, GloVe.
- Limitation: Cannot adapt to sentence context.

### Contextual Embeddings:
- Meaning changes based on surrounding words.
- Example:
  - “Track” in “train track” ≠ “track my package”
  - “Dish” in “rice dish” ≠ “cheese dish”

Transformers generate contextual embeddings using attention mechanisms.

---

## 🏗️ Transformer Architecture Overview

Transformers consist of two main components:

### 1. Encoder
- Processes input sentence.
- Generates contextual embeddings for each word.

### 2. Decoder
- Uses encoder output + previous words.
- Predicts next word or translates sentence.

Each component contains:
- Multi-head attention layers.
- Feedforward neural networks.
- Positional encoding.
- Layer normalization and residual connections.

---

## 🔄 Encoder & Decoder Roles

### Encoder:
- Converts input tokens into rich contextual embeddings.
- Captures relationships between all words in the input.

### Decoder:
- Uses encoder output and previously generated tokens.
- Predicts next token step-by-step.
- Used in tasks like translation and text generation.

### Architecture Variants:
- **BERT**: Encoder-only.
- **GPT**: Decoder-only.
- **T5, BART**: Encoder-decoder.

---

## 🎯 Attention Mechanism

Attention allows each word to focus on other relevant words in the sentence.

### Example:
- Sentence: “I made a sweet Indian rice dish.”
- Word: “Dish”
- Relevant modifiers: “Sweet”, “Indian”, “Rice”

Each modifier contributes a weighted influence to the meaning of “Dish”.

### Attention Scores:
- “Sweet” → 36%
- “Indian” → 14%
- “Rice” → 18%
- “I” → 2%

These scores are computed using dot products between query and key vectors.

---

## 📚 Query, Key, Value Analogy

### Library Analogy:
- **Query**: “I want a book on quantum computing.”
- **Key**: Labels on shelves (e.g., “Science”, “History”).
- **Value**: Actual book content.

### Transformer Analogy:
- Each word has:
  - **Query vector**: What it wants to know.
  - **Key vector**: What it offers.
  - **Value vector**: Its contribution.

Attention = dot product between query and key → weighted sum of values.

---

## 🧠 Multi-Head Attention

Instead of one attention mechanism, Transformers use multiple heads.

### Why?
- Each head focuses on different aspects:
  - Head 1 → Adjectives
  - Head 2 → Verbs
  - Head 3 → Pronouns
  - Head 4 → Cultural context

### Benefit:
- Captures diverse relationships in parallel.
- Improves model’s ability to understand complex language patterns.

---

## 🔄 Feedforward Network

After attention, each word’s embedding is passed through a feedforward neural network.

### Purpose:
- Adds non-linear transformations.
- Refines and enriches the contextual embedding.

### Structure:
- Input layer → Hidden layer → Output layer
- Same dimensionality as input embedding (e.g., 768 for BERT)

This step helps model learn higher-order features beyond attention.

---

## 🏋️ Training & Backpropagation

Transformers are trained on massive datasets using self-supervised learning.

### Training Steps:
1. Input sentence → Predict next word.
2. Compare prediction with actual word → Compute error.
3. Backpropagate error → Update weights (WQ, WK, WV).
4. Repeat over millions of sentences.

### Vocabulary:
- GPT: ~50,000 tokens
- BERT: ~30,000 tokens

Each token has a static embedding vector stored in a lookup matrix.

---

## ✅ Final Summary

Transformers are powerful architectures that:
- Convert words into rich embeddings.
- Use attention to understand context.
- Predict or generate text using encoder-decoder structure.

### Key Takeaways:
- GPT uses decoder-only for generation.
- BERT uses encoder-only for understanding.
- Multi-head attention and feedforward layers enrich token representations.
- Training involves massive data and backpropagation.


---

# 📐 Cosine Similarity & Cosine Distance — Beginner-Friendly Notes

> A visual and intuitive explanation of cosine similarity and cosine distance, with real-world examples and Python implementation using `scikit-learn`. Perfect for data science, NLP, and machine learning learners.

---

## 📌 Table of Contents
1. [Introduction to Document Similarity](#introduction-to-document-similarity)
2. [Real-World Motivation](#real-world-motivation)
3. [Word Count as Vectors](#word-count-as-vectors)
4. [Understanding Vector Angles](#understanding-vector-angles)
5. [Cosine Similarity Explained](#cosine-similarity-explained)
6. [Cosine Distance Explained](#cosine-distance-explained)
7. [Python Implementation with scikit-learn](#python-implementation-with-scikit-learn)
8. [Working with Real Documents](#working-with-real-documents)
9. [Conclusion](#conclusion)

---

## 🧠 Introduction to Document Similarity

In data science and NLP, we often need to compare documents to see how similar they are. This is useful for:
- Auto-tagging documents
- Clustering similar texts
- Search engines and recommendation systems

Cosine similarity is a popular metric for this task.

---

## 🏢 Real-World Motivation

Imagine you're a data scientist at a financial company. You receive documents without metadata, and you want to identify which company each document is about.

### Example:
- Document mentions “iPhone” 3 times and “Galaxy” once → Likely about Apple.
- Another document mentions “Galaxy” 4 times and “iPhone” once → Likely about Samsung.

You can use word frequency ratios to infer document topics.

---

## 📊 Word Count as Vectors

We can represent word counts as vectors in 2D space:
- X-axis: iPhone count
- Y-axis: Galaxy count

### Example Vectors:
- Apple doc: (3, 1)
- Samsung doc: (1, 4)
- New doc: (6, 2)

Each document becomes a point in vector space.

---

## 📐 Understanding Vector Angles

The angle between two vectors tells us how similar they are:
- **Smaller angle** → More similar
- **Larger angle** → Less similar

### Key Insight:
- If two documents have similar word ratios, their vectors point in similar directions.

---

## 📈 Cosine Similarity Explained

Cosine similarity measures the cosine of the angle between two vectors:
\[
\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \cdot ||B||}
\]

### Properties:
- Range: 0 to 1
- 1 → Identical direction (very similar)
- 0 → Orthogonal (very different)

### Example:
- Angle = 17° → Cosine similarity ≈ 0.95 → High similarity

---

## 📉 Cosine Distance Explained

Cosine distance is derived from cosine similarity:
\[
\text{Cosine Distance} = 1 - \text{Cosine Similarity}
\]

### Properties:
- Range: 0 to 1
- 0 → Very similar
- 1 → Very different

Used when you want to treat similarity as a “distance” metric.

---

## 🐍 Python Implementation with scikit-learn

Use `scikit-learn` to compute cosine similarity and distance.

### Setup:
```python
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
```

### Example Vectors:
```python
A = [[3, 1]]
B = [[6, 2]]
```

### Cosine Similarity:
```python
similarity = cosine_similarity(A, B)
print(similarity)  # Output: [[1.]]
```

### Cosine Distance:
```python
distance = cosine_distances(A, B)
print(distance)  # Output: [[0.]]
```

---

## 📄 Working with Real Documents

Create a DataFrame with word counts from actual documents.

### Sample Code:
```python
import pandas as pd

data = {
    'iPhone': [3, 3, 1, 1],
    'Galaxy': [1, 2, 3, 4]
}
df = pd.DataFrame(data, index=['doc1', 'doc2', 'doc3', 'doc4'])
```

### Compare Documents:
```python
similarity = cosine_similarity([df.loc['doc1']], [df.loc['doc2']])
print(similarity)  # Output: [[0.94]]
```

### Interpretation:
- doc1 and doc2 → Similar (Apple docs)
- doc1 and doc3 → Less similar
- doc3 and doc4 → Very similar (Samsung docs)

---

## ✅ Conclusion

Cosine similarity and distance are powerful tools for comparing documents based on word frequency vectors.

### Summary:
- Use cosine similarity to measure directional closeness.
- Use cosine distance when you need a distance metric.
- Python’s `scikit-learn` makes implementation easy.
- Ideal for NLP, clustering, and recommendation systems.

---
