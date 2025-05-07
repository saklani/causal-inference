import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from utils import columns_sentiment, csv_chunksize


def vader_extract_score(model, text):
    score = model.polarity_scores(text)
    compound = score['compound']

    sentiment = 'neutral'
    if compound >= 0.05:
        sentiment = "positive"

    elif compound <= -0.05:
        sentiment = "negative"

    return sentiment


def vader_sentiment_analysis(cleaned_path: str, analysis_path: str):
    nltk.download("vader_lexicon")
    model = SentimentIntensityAnalyzer()

    df = pd.DataFrame(columns=columns_sentiment)
    df.to_csv(analysis_path, index=False)
    for df in pd.read_csv(cleaned_path, chunksize=csv_chunksize):
        df["sentiment"] = df["text"].apply(
            lambda x: vader_extract_score(model, x))
        df.to_csv(
            analysis_path,
            index=False,
            header=False,
            mode="a",
            escapechar='\\')


LABEL2SENTIMENT = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


def bert_extract_score(model, text: str) -> list[str]:
    """
    Run the Hugging Face pipeline on a single text and map
    the star‑rating label to 'negative' / 'neutral' / 'positive'.
    """
    raws = model(text, truncation=True, max_length=512)
    # Fallback to 'neutral' if the label is unexpected
    return [LABEL2SENTIMENT.get(raw["label"], "neutral") for raw in raws]


def bert_sentiment_analysis(cleaned_path: str, analysis_path: str):
    model = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

    df = pd.DataFrame(columns=columns_sentiment)
    df.to_csv(analysis_path, index=False)
    for i, chunk in enumerate(pd.read_csv(
            cleaned_path, chunksize=csv_chunksize)):
        print("Chunk", i)
        chunk["sentiment"] = bert_extract_score(model, chunk["text"].tolist())
        chunk.to_csv(
            analysis_path,
            index=False,
            header=False,
            mode="a",
            escapechar='\\')


def distilbert_extract_score(model, text: str) -> list[str]:
    raws = model(text, truncation=True, max_length=512)
    return [raw["label"] if 0.4 < raw["score"]
            or raw["score"] > 0.6 else "neutral" for raw in raws]


def distilbert_sentiment_analysis(cleaned_path: str, analysis_path: str):
    model = pipeline(
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    df = pd.DataFrame(columns=columns_sentiment)
    df.to_csv(analysis_path, index=False)
    for i, chunk in enumerate(pd.read_csv(
            cleaned_path, chunksize=csv_chunksize)):
        print("Chunk", i)
        chunk["sentiment"] = distilbert_extract_score(
            model, chunk["text"].tolist())
        chunk.to_csv(
            analysis_path,
            index=False,
            header=False,
            mode="a",
            escapechar='\\')
