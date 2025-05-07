# Not all Reviews are Equal

Online product reviews are a cornerstone of ecommerce platforms, influencing buyer behavior
and product visibility. However, not all reviews
are perceived as equally helpful. This project addresses the following question: 

*Does the sentiment
expressed in a review causally affect its helpfulness, as measured by helpful votes?*

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Steps](#pipeline-steps)
3. [Quick Start](#quick-start)
4. [Command‑line toggles](#command-line-toggles)
5. [File Layout](#file-layout)
6. [Performance Hints](#performance-hints)
7. [Dependencies](#dependencies)

---

## Overview

1. **Samples** an equal number of raw reviews per Amazon category from the [McAuley‑Lab/Amazon‑Reviews‑2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) HuggingFace dataset.
2. **Cleans** the text (lower‑cases, trims, drops very short reviews, counts tokens).
3. **Scores sentiment** with two approaches

   * **VADER** (`nltk` lexicon)
   * **Multilingual BERT** (`nlptown/bert-base-multilingual-uncased-sentiment`)
4. Writes each intermediate dataset to CSV so you can inspect / reuse any stage.
5. Casual Inference on several datasets

By changing two flags you can benchmark VADER vs BERT latency on your own machine.

---

## Dataset

The [McAuley‑Lab/Amazon‑Reviews‑2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) is a large-scale Amazon Reviews dataset.
This dataset of product reviews covers
a wide variety of product categories and provides
insights into customer opinions and behaviors on
Amazon.
Below is an aggregated summary of the dataset:

- Total Number of Reviews - 571.4M
- Total Number of Users - 240.8M
- Average Review Length (tokens) - 52.7
- Total categories - 34 (33 named categories and an "Unknown" category)

Since the entire dataset is pretty large, we randomly sample it.
The `dataset.py` defines function to generate and clean dataset
of different sizes. 

### Generating a sample of the dataset 
From each category the script selects a random sample equal to that specified by the `--category-size` argument.
It defaults to `25000`

### Cleaning the Reviews
The cleaning process is surmised below:

- Keeps only string reviews  
- Lower‑cases & strips
- Drops reviews ≤ 5 words<br>
- Adds `review_length`(total words in review) & `token_count` (total tokens in review)


```shell
python main.py --category-size=25000
```

## Pipeline Steps

| Function                                             | What it does                                                                                                                  | Key I/O                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **`generate_dataset(batch_size, file_path)`**        | Streams *N = batch\_size* reviews from **each** of 33 categories into a single CSV.                                           | ➜ `reviews-{N}.csv`                |
| **`clean_dataset(file_path, clean_path)`**           |  • Keeps only string reviews<br>• Lower‑cases & strips<br>• Drops reviews ≤ 5 words<br>• Adds `review_length` & `token_count` | ➜ `reviews-{N}-cleaned.csv`        |
| **`vader_sentiment_analysis(clean_path, out_path)`** | Adds a `sentiment` column using NLTK’s VADER.                                                                                 | ➜ `reviews-{N}-analysis-vader.csv` |
| **`bert_sentiment_analysis(clean_path, out_path)`**  | Adds a `sentiment` column using the multilingual BERT star‑rating model, mapped to *negative / neutral / positive*.           | ➜ `reviews-{N}-analysis-bert.csv`  |

---

## Quick Start

```bash
# 1. Set up a fresh venv (optional)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install deps
pip install -r requirements.txt
#   (transformers, datasets, nltk, pandas, tqdm, torch, etc.)

# 3. Run the pipeline with default (500 reviews / category)
python main.py
```

The script prints elapsed time for the BERT step so you can gauge throughput.

---

## Command‑line toggles

Inside `main()`:

```python
batch_size            = 500          # samples per category
generate_dataset(...) # uncomment to rebuild raw CSV
clean_dataset(...)    # uncomment to rebuild cleaned CSV
vader_sentiment_analysis(...)  # toggle VADER
bert_sentiment_analysis(...)   # toggle BERT
```

---

## File Layout (default `batch_size = 500`)

```
reviews-500.csv                     # raw sample, all 33 categories
reviews-500-cleaned.csv             # lowercase, >5 words, token counts
reviews-500-analysis-vader.csv      # + VADER sentiment
reviews-500-analysis-bert.csv       # + BERT sentiment
```

---

## Dependencies

```
pandas
datasets>=2.0
transformers>=4.34
torch          # CUDA build recommended for GPU
tqdm
nltk
```

```python
# one‑time download for VADER
import nltk; nltk.download("vader_lexicon")
```

---

### License

MIT — do whatever you want, but cite the *McAuley‑Lab/Amazon‑Reviews‑2023* dataset if you publish results.

Happy sentiment mining!
