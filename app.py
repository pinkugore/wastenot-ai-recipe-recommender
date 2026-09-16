import streamlit as st
from recommender import load_recipes, recommend_recipes
from genai import generate_recipe_writeup

st.set_page_config(page_title="WasteNot - AI Recipe Recommender", page_icon="🥗")

st.title("🥗 WasteNot — AI Recipe Recommender")
st.write(
    "Tell me what's in your kitchen, and I'll suggest recipes — "
    "prioritizing ingredients that are about to expire, so nothing goes to waste."
)

recipes_df = load_recipes("recipes.csv")
all_ingredients = sorted(
    {ing.strip() for row in recipes_df["ingredients"] for ing in row.split(",")}
)

available = st.multiselect("✅ Ingredients you currently have:", all_ingredients)
expiring = st.multiselect("⏰ Which of these are expiring soon?", options=available)

if st.button("Find recipes"):
    if not available:
        st.warning("Please select at least one ingredient.")
    else:
        st.session_state["results"] = recommend_recipes(available, expiring, recipes_df, top_n=5)
        st.session_state["available"] = available
        st.session_state["expiring"] = expiring

if "results" in st.session_state:
    if st.session_state["results"].empty:
        st.info("No close matches found for these ingredients — try adding a few more.")
    else:
        st.subheader("Recommended recipes")
    for idx, row in st.session_state["results"].iterrows():
        header = f"🍽️ {row['recipe_name']}  ({row['cuisine']}, {row['prep_time_mins']} mins)"
        with st.expander(header):
            st.write(f"**Uses ingredients you have:** {row['matched_ingredients']}")
            if row["missing_ingredients"]:
                st.write(f"**You'll also need:** {row['missing_ingredients']}")
            st.progress(min(row["final_score"], 1.0))
            st.caption(f"Match score: {row['final_score']:.2f}")

            if st.button("✨ Get AI cooking instructions", key=f"ai_{idx}"):
                with st.spinner("Asking the local AI model for instructions..."):
                    writeup = generate_recipe_writeup(
                        row, st.session_state["available"], st.session_state["expiring"]
                    )
                st.markdown(writeup)