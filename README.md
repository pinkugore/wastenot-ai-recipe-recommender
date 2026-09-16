# WasteNot — AI Recipe Recommender to Reduce Food Waste

## Problem
Households frequently throw away food because they forget what's about to
expire, or don't know what to cook with the ingredients they already have.
This leads to unnecessary food waste and grocery spend.

## What I Built
WasteNot is a hybrid AI system: a classical ML recommender retrieves the
best-matching recipes from a dataset based on the user's available
ingredients, and a local generative AI layer turns the top match into
personalized, natural-language cooking instructions — explaining why the
recipe is a good choice right now (especially for ingredients about to
expire).

## Architecture (Retrieve, then Generate)
1. **Retrieval (classical ML)** — `recommender.py`
   - Ingredient lists (user's pantry vs. each recipe) are converted into
     TF-IDF vectors.
   - **Cosine similarity** measures how closely a recipe matches what the
     user has.
   - An **expiry score** boosts recipes that use ingredients about to
     expire.
   - The top-ranked recipes are the "retrieved" candidates.

2. **Generation (GenAI)** — `genai.py`
   - The top retrieved recipe is passed, along with the user's expiring
     ingredients, into a prompt for a local LLM (via **Ollama** +
     **LangChain**).
   - The LLM generates a short explanation of why the recipe fits, plus
     beginner-friendly step-by-step cooking instructions.
   - This retrieve-then-generate pattern is a simplified version of
     **Retrieval-Augmented Generation (RAG)** — the same architecture used
     in larger production AI systems.

3. **Interface** — `app.py`
   - A Streamlit web app lets the user pick ingredients, see ranked
     recipes, and click a button to get AI-generated cooking instructions
     for any recipe.

## Tech Stack
- **Python** — core logic
- **Pandas** — data handling
- **Scikit-learn** — TF-IDF vectorization + cosine similarity (retrieval)
- **LangChain + Ollama** — local LLM for generative cooking instructions
- **Streamlit** — interactive web UI
- Includes **unit tests** (`test_recommender.py`) for the deterministic
  retrieval logic. The generative layer is verified manually since LLM
  output is non-deterministic by nature.

## How to Run Locally
1. Install [Ollama](https://ollama.com/download) and pull a small model:
   ```bash
   ollama pull llama3.2:1b
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python -m streamlit run app.py

   ```

## Running Tests
```bash
python test_recommender.py
```

## Future Scope
- Detect ingredients automatically from a fridge/pantry photo using a
  pretrained computer vision model
- Add real expiry-date tracking with reminder notifications
- Personalize recommendations over time using a user's cooking history
  (collaborative filtering)
- Swap the local LLM for a hosted one and add proper RAG over a larger
  recipe corpus using a vector database (e.g. ChromaDB)
- Convert the backend to FastAPI + Docker and deploy on AWS for scale

## Why This Project
This project demonstrates both classical ML (content-based recommendation
using TF-IDF and cosine similarity) and generative AI (local LLM via
LangChain/Ollama) working together in a retrieve-then-generate pipeline —
built to solve a real, everyday problem, with clean modular code and
tested core logic.
