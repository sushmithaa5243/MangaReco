import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Load Dataset
# ==========================================

anime = pd.read_csv("data/anime.csv")

anime = anime[['anime_id', 'name', 'genre', 'rating']]

anime = anime.dropna(subset=['genre'])

anime['rating'] = anime['rating'].fillna(0)

anime = anime.drop_duplicates(subset='name')

anime = anime.reset_index(drop=True)

anime['genre'] = anime['genre'].str.lower().str.strip()

anime['search_name'] = anime['name'].str.lower().str.strip()

# ==========================================
# TF-IDF
# ==========================================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(anime["genre"])

# ==========================================
# Cosine Similarity
# ==========================================

cosine_sim = cosine_similarity(tfidf_matrix)

# ==========================================
# Index Mapping
# ==========================================

indices = pd.Series(
    anime.index,
    index=anime["search_name"]
).drop_duplicates()

# ==========================================
# Recommendation Function
# ==========================================

def recommend(title, top_n=10):

    title = title.lower().strip()

    if title not in indices:

        matches = anime[
            anime["search_name"].str.contains(
                title,
                case=False,
                na=False
            )
        ]

        if matches.empty:
            return None

        return matches[["name"]].head(10)

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    anime_indices = [
        i[0]
        for i in sim_scores
    ]

    recommendations = anime.iloc[
        anime_indices
    ][
        [
            "anime_id",
            "name",
            "genre",
            "rating"
        ]
    ].copy()

    return recommendations.reset_index(drop=True)

# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("Content-Based Recommendation Test")
    print("=" * 60)

    title = input("Enter Anime Name: ")

    result = recommend(title)

    if result is None:
        print("No anime found.")

    else:
        print(result.to_string(index=False))