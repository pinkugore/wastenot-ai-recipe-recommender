"""
Core recommendation logic for WasteNot.

Idea: represent the user's available ingredients as one "document",
and each recipe's ingredient list as another "document". Use TF-IDF
to turn these into vectors, then use cosine similarity to see which
recipes are the closest match to what the user has.

On top of that, we add an "expiry score": recipes that use more of
the user's soon-to-expire ingredients get ranked higher, so nothing
goes to waste.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_recipes(csv_path="recipes.csv"):
    """Load the recipe dataset from a CSV file."""
    return pd.read_csv(csv_path)


def recommend_recipes(available_ingredients, expiring_ingredients, recipes_df, top_n=5):
    """
    Recommend recipes based on what the user has in their kitchen.

    available_ingredients : list[str]  - everything the user currently has
    expiring_ingredients  : list[str]  - subset that's about to expire
    recipes_df            : DataFrame with columns
                             ['recipe_name', 'ingredients', 'cuisine', 'prep_time_mins']
    top_n                 : how many recipes to return

    Returns a DataFrame sorted by relevance with extra columns:
    similarity_score, expiry_score, final_score,
    matched_ingredients, missing_ingredients
    """
    user_ingredients_text = " ".join(available_ingredients).lower()
    recipe_texts = recipes_df["ingredients"].str.lower()

    # TF-IDF converts ingredient lists into numeric vectors we can compare
    vectorizer = TfidfVectorizer()
    all_texts = list(recipe_texts) + [user_ingredients_text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Last row = the user's pantry vector; compare it against every recipe
    user_vector = tfidf_matrix[-1]
    recipe_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(user_vector, recipe_vectors).flatten()

    results = recipes_df.copy()
    results["similarity_score"] = similarity_scores

    expiring_set = {i.strip().lower() for i in expiring_ingredients}
    available_set = {i.strip().lower() for i in available_ingredients}

    def expiry_score(ingredients_str):
        recipe_ings = {i.strip().lower() for i in ingredients_str.split(",")}
        if not recipe_ings:
            return 0.0
        return len(recipe_ings & expiring_set) / len(recipe_ings)

    def matched_and_missing(ingredients_str):
        recipe_ings = [i.strip() for i in ingredients_str.split(",")]
        matched = [i for i in recipe_ings if i.lower() in available_set]
        missing = [i for i in recipe_ings if i.lower() not in available_set]
        return pd.Series([", ".join(matched), ", ".join(missing)])

    results["expiry_score"] = results["ingredients"].apply(expiry_score)
    results[["matched_ingredients", "missing_ingredients"]] = results["ingredients"].apply(matched_and_missing)

    # Weighted score: reward overall match, but weight expiring-ingredient use higher
    results["final_score"] = (0.6 * results["similarity_score"]) + (0.4 * results["expiry_score"])

    ranked = results.sort_values("final_score", ascending=False)
    relevant = ranked[ranked["final_score"] > 0]
    return relevant.head(top_n)