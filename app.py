import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd 

df = pd.read_csv("product_wm_1000.csv")

X_text = df["product_name"].values

cv = make_pipeline(
    CountVectorizer(
            ngram_range=(3, 7),
            analyzer="char"
        ),
    Normalizer()
)

cv.fit(X_text)

X = cv.transform(X_text)

def search(term):
    X_term = cv.transform([term])

    simularities = cosine_similarity(X_term, X)

    idxmax = np.argmax(simularities[0])

    return df.loc[idxmax]

if __name__ == "__main__":
    term = "choclate cooies mnt"
    print(search(term))