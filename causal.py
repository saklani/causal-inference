import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df):
    """Preprocess the data"""

    # Prepare confounders
    review_length = df["review_length"].values.reshape(-1, 1)
    year = pd.to_datetime(df["timestamp"], unit='ms').dt.year.values.reshape(-1, 1)
    verified_purchase = df["verified_purchase"].values.reshape(-1, 1)
    rating = df["rating"].values.reshape(-1, 1)


    # ASIN Frequency is a proxy for product popularity

    df["asin_freq"] = df.groupby("asin")["asin"].transform("size")
    asin_freq = df["asin_freq"].values.reshape(-1, 1)

    encoder = OneHotEncoder()
    category = encoder.fit_transform(df["category"].values.reshape(-1, 1)).toarray()
    confounders = np.hstack((review_length, year, category, verified_purchase, rating, asin_freq))
    
    scaler = StandardScaler()
    scaled_confounders = scaler.fit_transform(confounders)

    # Prepare treatment
    treatment = (df["sentiment"] == "positive").astype(int)
    
    # Prepare outcome
    outcome = df['helpful_vote']

    return scaled_confounders, treatment, outcome

def compute_ipw(T, propensity_score):
    """Compute the inverse propensity weight"""
    return T / propensity_score + (1 - T) / (1 - propensity_score)

def main():
    df = pd.read_csv("reviews-25000-analysis-bert.csv")
    X, T, y = prepare_data(df)
    
    logreg = LogisticRegression()
    logreg.fit(X, T)

    propensity_score = logreg.predict_proba(X)[:, 1]
    ipw = compute_ipw(T, propensity_score)
    ipw = np.clip(ipw, 0, 20)

    X_reg = sm.add_constant(T)

    model = sm.GLM(y, X_reg, family=sm.families.NegativeBinomial(), freq_weights=ipw)
    result = model.fit()

    print(result.summary())

if __name__ == "__main__":
    main()


