import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse


def disagreement(df1, df2):
    print(f"df1: {df1.shape}")
    print(f"df2: {df2.shape}")
    # 1. frequency table
    freq = pd.crosstab(df1['sentiment'], columns='VADER')\
        .join(pd.crosstab(df2['sentiment'], columns='BERT'))
    print(freq / len(df1))

    # 2. disagreement matrix
    cm = pd.crosstab(df1['sentiment'], df2['sentiment'],
                     rownames=['VADER'], colnames=['BERT'])
    print(cm)

    # 3. Cohen’s kappa
    kappa = cohen_kappa_score(df1['sentiment'], df2['sentiment'])
    print("Cohen κ =", kappa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Disagreement")
    parser.add_argument(
        "--vader",
        help="VADER sentiment analysis file",
        default='reviews-100000-analysis-vader.csv')
    parser.add_argument(
        "--bert",
        help="BERT sentiment analysis file",
        default='reviews-100000-analysis-bert.csv')
    args = parser.parse_args()

    df1 = pd.read_csv(args.vader)
    df2 = pd.read_csv(args.bert)
    disagreement(df1, df2)
