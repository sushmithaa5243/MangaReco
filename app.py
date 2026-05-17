import streamlit as st
from recommender import recommend, anime

st.set_page_config(
    page_title="MangaReco",
    page_icon="📚",
    layout="wide"
)

st.title("📚 MangaReco")
st.subheader("AI-powered manga, manhwa and anime recommendation system")

st.write(
    "Select multiple titles you have already watched/read, "
    "and get personalized recommendations."
)

# Genre explanation section
with st.expander("📖 New reader? Learn common genres"):

    st.markdown("""
    ### Isekai
    Main character gets transported or reborn into another world.

    ### Shounen
    Action/adventure stories aimed at younger audiences.

    ### Seinen
    Mature stories with darker and deeper themes.

    ### Slice of Life
    Calm stories based on everyday life.

    ### Murim
    Martial arts world common in manhwa.

    ### Regression
    Character returns to the past to change events.

    ### Psychological
    Focuses on mind games, emotions and strategy.
    """)

# Title selection
title_list = sorted(anime["name"].dropna().unique())

selected_titles = st.multiselect(
    "Choose titles you have read:",
    title_list
)

# Number of recommendations
top_n = st.slider(
    "Number of recommendations",
    3,
    15,
    5
)

if st.button("Recommend"):

    if not selected_titles:
        st.warning("Please select at least one title.")

    else:

        all_results = []

        # Collect recommendations
        for title in selected_titles:

            try:
                recs = recommend(title, top_n)

                for r in recs:
                    if isinstance(r, dict):
                        all_results.append(r)

            except:
                pass

        # Remove duplicates
        seen = set()
        final_results = []

        for item in all_results:

            if (
                item["title"] not in seen
                and item["title"] not in selected_titles
            ):

                seen.add(item["title"])
                final_results.append(item)

        # Limit final recommendations
        final_results = final_results[:top_n]

        st.subheader("🔥 Recommended For You")

        if not final_results:
            st.error("No recommendations found.")

        else:

            for item in final_results:

                with st.container(border=True):

                    st.markdown(f"## {item['title']}")

                    st.write(f"**Genre:** {item['genre']}")

                    st.write(f"**Community Rating:** {item['rating']}")

                    st.write(
                        "**Where to read/watch:** "
                        "Check official sources like "
                        "WEBTOON, Manga Plus, Tapas, Crunchyroll, "
                        "Kodansha or publisher websites."
                    )