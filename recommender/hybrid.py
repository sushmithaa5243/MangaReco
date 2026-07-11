import pandas as pd

from recommender.content_based import recommend
from recommender.collaborative import predict_rating


def hybrid_recommend(user_id, anime_title, top_n=10):
    """
    Hybrid Recommendation System

    Step 1 : Get content-based recommendations.
    Step 2 : Predict user's rating for each recommendation using SVD.
    Step 3 : Rank recommendations by predicted rating.
    """

    # Get content-based recommendations
    content_results = recommend(anime_title, top_n=50)

    if content_results is None:
        return None

    predictions = []

    for _, row in content_results.iterrows():

        anime_id = row["anime_id"]

        score = predict_rating(user_id, anime_id)

        predictions.append({
            "Anime": row["name"],
            "Genre": row["genre"],
            "Community Rating": row["rating"],
            "Predicted Rating": round(score, 2)
        })

    results = pd.DataFrame(predictions)

    results = results.sort_values(
        by="Predicted Rating",
        ascending=False
    )

    return results.head(top_n).reset_index(drop=True)


if __name__ == "__main__":

    print("=" * 60)
    print("Hybrid Recommendation System")
    print("=" * 60)

    user_id = int(input("Enter User ID : "))
    anime = input("Enter Anime Name : ")

    recommendations = hybrid_recommend(
        user_id=user_id,
        anime_title=anime
    )

    if recommendations is None:
        print("\nAnime not found.")

    else:
        print("\nTop Recommendations\n")
        print(recommendations.to_string(index=False))