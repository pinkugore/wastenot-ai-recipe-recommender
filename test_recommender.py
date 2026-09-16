from recommender import load_recipes, recommend_recipes


def test_basic_recommendation():
    recipes_df = load_recipes("recipes.csv")
    available = ["onion", "tomato", "potato", "rice"]
    expiring = ["tomato"]

    results = recommend_recipes(available, expiring, recipes_df, top_n=3)

    assert len(results) > 0, "Should return at least one recipe"
    assert "final_score" in results.columns
    assert results.iloc[0]["final_score"] >= results.iloc[-1]["final_score"], "Should be sorted highest first"

    print("Test passed. Top recommendation:", results.iloc[0]["recipe_name"])


def test_expiry_priority():
    recipes_df = load_recipes("recipes.csv")
    # Both recipes need tomato; only one recipe set marks tomato as expiring
    available = ["tomato", "onion", "garlic", "pepper", "butter"]
    expiring = ["tomato"]

    results = recommend_recipes(available, expiring, recipes_df, top_n=5)
    top_recipe_ingredients = results.iloc[0]["ingredients"].lower()

    assert "tomato" in top_recipe_ingredients, "Top recipe should use the expiring ingredient"
    print("Test passed. Expiry-aware ranking works.")


if __name__ == "__main__":
    test_basic_recommendation()
    test_expiry_priority()
    print("\nAll tests passed!")