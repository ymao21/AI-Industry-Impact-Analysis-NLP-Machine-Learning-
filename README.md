# AI Industry Analysis using NLP & Transformer Models

## Overview
This project applies natural language processing and machine learning to ~200K news articles on artificial intelligence to extract structured insights from unstructured text.

The pipeline combines:
- Topic modeling  
- Entity extraction (NER)  
- Transformer-based sentiment classification  

---

## Dataset
- ~200,000 AI-related news articles  
- Fields: title, text, date, domain  
- Source: public parquet dataset  

---

## Pipeline Architecture

### 1. Data Cleaning & Preprocessing
- Removed duplicates, null values, and short documents  
- Normalized text (URLs, symbols, whitespace)  

Different preprocessing strategies were used depending on the model:

| Task        | Processing                                      |
|-------------|------------------------------------------------|
| LDA         | Tokenization, stopword removal, lemmatization  |
| BERTopic    | Minimal cleaning (preserve context)            |
| Sentiment   | Title + truncated body                         |

Final dataset: ~196K documents  

---

### 2. Topic Modeling

#### LDA (Baseline)
- Bag-of-words representation  
- Trained on 40K sample  
- Interpretable keyword-based topics  

#### BERTopic (Final Model)
- Transformer-based embeddings  
- Uses sentence embeddings + UMAP + HDBSCAN  
- Trained on 25K sample  

**Why BERTopic**
- Captures semantic meaning better than LDA  
- Produces cleaner and more distinct clusters  
- Better suited for long-form text  

---

### 3. Entity Extraction (NER)
- Model: spaCy (`en_core_web_sm`)  
- Extracted:
  - Organizations (companies)  
  - Technology-related terms (via keyword matching)  

- Input: title + truncated body text  
- Post-processing:
  - Noise filtering  
  - Frequency thresholding (≥50 mentions)  

---

### 4. Sentiment Analysis (Transformer Fine-Tuning)

#### Model
- Base: `distilbert-base-uncased`  
- Fine-tuned for 3-class classification:
  - Positive / Neutral / Negative  

#### Training Data
- Financial PhraseBank (~2.7K labeled samples)  
- High-quality "all-agree" subset  

#### Training Setup
- 4 epochs  
- AdamW optimizer  
- Learning rate: 2e-5  
- Stratified train/val/test split  

#### Performance
- Accuracy: 96%  
- Macro F1: 0.95  

#### Inference
- Applied to full dataset (~196K articles)  
- Input: title + first 400 characters  
- Output:
  - Sentiment label  
  - Probability scores  
  - Compound score (pos − neg)  

---

### 5. Data Integration & Aggregation
Combined outputs from:
- BERTopic (topics)  
- NER (entities)  
- Sentiment model  

Final dataset includes:
- Article-level topic  
- Industry mapping  
- Extracted entities  
- Sentiment scores  

---

## Key Technical Components
- Transformer embeddings for semantic topic modeling  
- UMAP (dimensionality reduction) + HDBSCAN (clustering)  
- Fine-tuned transformer (DistilBERT) for sentiment classification  
- NER pipeline with post-processing  
- Large-scale inference (~200K documents)  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- spaCy (NER)  
- BERTopic  
- Hugging Face Transformers (DistilBERT)  
- PyTorch  
- UMAP, HDBSCAN  
- Matplotlib / Seaborn  

---

## Limitations
- Sentiment model trained on financial data (domain mismatch)  
- Topic modeling performed on sample due to compute constraints  
- NER relies on general-purpose model + rules  

---

## Future Improvements
- Train domain-specific sentiment dataset  
- Use custom NER for AI-specific entities  
- Scale BERTopic to full dataset  
- Replace keyword-based tech extraction with learned models  

---

## Author
Yining Mao
