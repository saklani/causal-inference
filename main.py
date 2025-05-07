from sentiment_analysis import bert_sentiment_analysis, vader_sentiment_analysis
from dataset import generate_dataset, clean_dataset
import argparse

def main(category_size):
    print("Current Category Size:", category_size)

    file_path = f"reviews-{category_size}.csv"
    clean_path = f"reviews-{category_size}-cleaned.csv"
    vader_analysis_path = f"reviews-{category_size}-analysis-vader.csv"
    bert_analysis_path = f"reviews-{category_size}-analysis-bert.csv"

    generate_dataset(category_size, file_path)
    clean_dataset(file_path, clean_path)
    vader_sentiment_analysis(clean_path, vader_analysis_path)
    bert_sentiment_analysis(clean_path, bert_analysis_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Review Sentiment Analysis")
    parser.add_argument("--category_size", help="number of reviews per category to sample", default=25000, type=int)
    args = parser.parse_args()
    main(args.category_size)
