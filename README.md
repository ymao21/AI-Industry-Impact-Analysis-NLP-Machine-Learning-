# AI-Industry-Impact-Analysis-NLP-Machine-Learning-


AI Industry Analysis using NLP & Transformer Models
Overview

This project applies natural language processing and machine learning to ~200K news articles on artificial intelligence to extract structured insights from unstructured text.

The focus is on building an end-to-end pipeline combining:

topic modeling
entity extraction
transformer-based sentiment classification
Dataset
~200,000 AI-related news articles
Fields: title, text, date, domain
Source: public parquet dataset
Pipeline Architecture
1. Data Cleaning & Preprocessing
Removed duplicates, null values, and short documents
Normalized text (URLs, symbols, whitespace)
Different preprocessing strategies per model:
Task	Processing
LDA	tokenization, stopword removal, lemmatization
BERTopic	minimal cleaning (preserve context)
Sentiment	title + truncated body

Final dataset: ~196K documents

2. Topic Modeling
LDA (Baseline)
Bag-of-words representation
Trained on 40K sample
Provides interpretable keyword distributions
BERTopic (Final Model)
Transformer-based embeddings
Sentence embeddings + UMAP + HDBSCAN
Trained on 25K sample

Why BERTopic

Captures semantic relationships
Produces cleaner and more distinct topic clusters
Better suited for long-form text
3. Entity Extraction (NER)
Model: spaCy (en_core_web_sm)
Extracted:
organizations (companies)
technology-related terms (keyword matching)
Input: truncated article text (title + body)
Post-processing:
noise filtering
frequency thresholding (≥50 mentions)
4. Sentiment Analysis (Transformer Fine-Tuning)
Model
Base: distilbert-base-uncased
Fine-tuned for 3-class classification:
positive / neutral / negative
Training Data
Financial PhraseBank (~2.7K labeled samples)
High-quality “all-agree” subset
Training Setup
4 epochs
AdamW optimizer
Learning rate: 2e-5
Stratified train/val/test split
Performance
Accuracy: 96%
Macro F1: 0.95
Inference
Applied to full dataset (~196K articles)
Input: title + first 400 characters
Output:
sentiment label
probability scores
compound score (pos − neg)
5. Data Integration & Aggregation
Joined outputs from:
BERTopic (topics)
NER (entities)
sentiment model
Final dataset:
article-level topic
industry mapping
extracted entities
sentiment scores
Key Technical Components
Transformer embeddings for semantic topic modeling
Dimensionality reduction (UMAP) + clustering (HDBSCAN)
Custom fine-tuned transformer classifier for sentiment
NER pipeline with post-processing for noise reduction
Large-scale inference (~200K documents)
Tech Stack
Python
Pandas, NumPy
spaCy (NER)
BERTopic
Hugging Face Transformers (DistilBERT)
PyTorch
UMAP, HDBSCAN
Matplotlib / Seaborn
Limitations
Sentiment model trained on financial data (domain mismatch)
Topic modeling performed on sample due to compute constraints
NER relies on general-purpose model + rules
Future Improvements
Train domain-specific sentiment dataset
Use custom NER for AI-specific entities
Scale BERTopic to full dataset
Replace keyword-based tech extraction with learned model
