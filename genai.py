"""
Generative AI layer for WasteNot.

The classic ML recommender (recommender.py) uses TF-IDF + cosine similarity
to RETRIEVE the best-matching recipe from the dataset based on what
ingredients the user has.

This file takes that retrieved recipe and uses a local LLM (via Ollama +
LangChain) to GENERATE a personalized, natural-language write-up: a short
note on why the recipe is a good choice right now, plus simple step-by-step
cooking instructions.

Retrieve, then generate — this is a small-scale version of the same
Retrieval-Augmented Generation (RAG) pattern used in larger AI systems.

Requires Ollama running locally (https://ollama.com) with a small model
pulled first, e.g.:
    ollama pull llama3.2:1b
"""

from langchain_ollama import OllamaLLM

MODEL_NAME = "llama3.2:1b"  # small and fast; works fine on a CPU-only laptop


def build_prompt(recipe_row, available_ingredients, expiring_ingredients):
    """Turn a retrieved recipe + the user's pantry into a prompt for the LLM."""
    expiring_text = ", ".join(expiring_ingredients) if expiring_ingredients else "none"

    return f"""You are a friendly home cooking assistant.

Recipe: {recipe_row['recipe_name']}
Cuisine: {recipe_row['cuisine']}
Ingredients needed: {recipe_row['ingredients']}
Ingredients the user has that are expiring soon: {expiring_text}

Write:
1. A short, friendly one-line note on why this recipe is a good choice right
   now (mention the expiring ingredients if relevant).
2. Simple, numbered step-by-step cooking instructions for a beginner cook.

Keep it concise and practical.
"""


def generate_recipe_writeup(recipe_row, available_ingredients, expiring_ingredients):
    """
    Call the local LLM to generate a personalized write-up for a recipe.
    Returns the generated text, or a friendly error message if Ollama
    isn't installed or running.
    """
    prompt = build_prompt(recipe_row, available_ingredients, expiring_ingredients)
    try:
        llm = OllamaLLM(model=MODEL_NAME)
        return llm.invoke(prompt)
    except Exception as exc:
        return (
            "Couldn't reach the local AI model. Make sure Ollama is installed "
            "and running, and that you've pulled the model with:\n\n"
            f"    ollama pull {MODEL_NAME}\n\n"
            f"(Technical detail: {exc})"
        )
