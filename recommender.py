import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
anime = pd.read_csv("data/anime.csv")

# Keep important columns
anime = anime[["name", "genre", "type", "rating"]]

# Remove missing values
anime = anime.dropna()

# Combine features
anime["combined"] = (
    anime["genre"].astype(str) + " " +
    anime["type"].astype(str)
)

# Convert text to vectors
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(anime["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)


def recommend(title, top_n=5):

    title = title.lower()

    matches = anime[anime["name"].str.lower() == title]

    if len(matches) == 0:
        return ["Anime/Manga not found"]

    idx = matches.index[0]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:top_n+1]

    recommendations = []

    for item in scores:

        manga = anime.iloc[item[0]]

        recommendations.append({
            "title": manga["name"],
            "genre": manga["genre"],
            "rating": manga["rating"]
        })

    return recommendations