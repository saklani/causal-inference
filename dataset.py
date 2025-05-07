import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from utils import columns, categories, columns_cleaned, csv_chunksize


def generate_dataset(batch_size: int, file_path: str):
    res = pd.DataFrame(columns=columns)
    res.to_csv(file_path, index=False)
    print("Generating dataset...")
    for category in tqdm(categories):
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name="raw_review_" + category,
            split="full",
            trust_remote_code=True,
            streaming=True
        )
        for batch in dataset.take(batch_size).iter(batch_size=batch_size):
            res = pd.DataFrame(batch, columns=columns)
            res["category"] = category
            res.to_csv(
                file_path,
                index=False,
                header=False,
                mode="a",
                escapechar='\\')
    print("Dataset Path:", file_path)


def clean_dataset(file_path: str, clean_path: str):
    print("Cleaning dataset...")
    df = pd.DataFrame(columns=columns_cleaned)
    df.to_csv(clean_path, index=False)
    for df in pd.read_csv(file_path, chunksize=csv_chunksize):
        df = df[df["text"].apply(type) == str]
        df["text"] = df["text"].apply(lambda x: x.strip().lower())
        df["review_length"] = df["text"].apply(lambda x: len(x.split()))
        df = df[df["review_length"] > 5]
        df["token_count"] = df["text"].apply(lambda x: len(x))
        df = df[df["token_count"] <= 256]
        df.to_csv(
            clean_path,
            index=False,
            header=False,
            mode="a",
            escapechar='\\')
    print("Cleaned Dataset Path:", clean_path)
