import joblib
import pandas as pd

# ==========================================
# Load Saved Model
# ==========================================

model = joblib.load("models/svd_model.pkl")

# ==========================================
# Load Datasets
# ==========================================

ratings = pd.read_csv("data/rating.csv")
ratings = ratings[ratings["rating"] != -1]

anime = pd.read_csv("data/anime.csv")
anime = anime[["anime_id", "name", "genre"]]
anime = anime.dropna(subset=["genre"])

# ==========================================
# Predict Rating
# ==========================================

def predict_rating(user_id, anime_id):
    """
    Predict rating of a user for an anime.
    """
    return model.predict(user_id, anime_id).est

# ==========================================
# Recommend For User
# ==========================================

def recommend_for_user(user_id, top_n=10):

    watched = ratings[
        ratings["user_id"] == user_id
    ]["anime_id"].unique()

    unseen = anime[
        ~anime["anime_id"].isin(watched)
    ]

    predictions = []

    for anime_id in unseen["anime_id"]:

        score = model.predict(
            user_id,
            anime_id
        ).est

        predictions.append(
            (
                anime_id,
                score
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_predictions = predictions[:top_n]

    result = []

    for anime_id, score in top_predictions:

        row = anime[
            anime["anime_id"] == anime_id
        ].iloc[0]

        result.append({
            "anime_id": anime_id,
            "name": row["name"],
            "genre": row["genre"],
            "predicted_rating": round(score, 2)
        })

    return pd.DataFrame(result)

# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print("\nPrediction Example")
    print(predict_rating(994, 1562))

    print("\nRecommendations\n")

    print(
        recommend_for_user(
            994
        ).to_string(index=False)
    )