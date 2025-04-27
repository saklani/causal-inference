from datasets import load_dataset
import pandas as pd
import sys

from examples.vgg7 import sample_size

categories = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown"
]
columns = ['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']

def generate_dataset(seed: int, res: pd.DataFrame, sample_size: int):
    for category in categories:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="raw_review_" + category, split="full",
                               trust_remote_code=True, streaming=True)
        shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=sample_size)
        for row in shuffled_dataset.take(sample_size):
            res.loc[len(res)] = row
        res.to_csv("reviews-small.csv", index=False, mode="a", header=False)


def main():
    mode = sys.argv[1]
    seed = int(sys.argv[2])
    if not seed:
        seed = 42

    if mode == 'large':
        sample_size = 500000
    elif mode == "small":
        sample_size = 10000
    else:
        sample_size = 1

    res = pd.DataFrame(columns=columns)
    res.to_csv(f"reviews-{mode}.csv")
    generate_dataset(seed, res, sample_size)
