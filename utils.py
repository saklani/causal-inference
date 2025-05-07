csv_chunksize = 10 ** 5  # Controls the chunksize of the csv loaded in memory
categories = [
    "Electronics",
    "Home_and_Kitchen",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Beauty_and_Personal_Care",
    "Health_and_Household",
    "Toys_and_Games",
    "Sports_and_Outdoors",
    "Office_Products",
    "Tools_and_Home_Improvement"
]
columns = [
    'rating',
    'title',
    'text',
    'images',
    'asin',
    'parent_asin',
    'user_id',
    'timestamp',
    'helpful_vote',
    'verified_purchase',
    'category',

]
columns_cleaned = columns + ["review_length", "token_count"]
columns_sentiment = columns_cleaned + ["sentiment"]
