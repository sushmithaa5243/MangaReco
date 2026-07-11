import streamlit as st
from recommender.hybrid import hybrid_recommend
from recommender.content_based import anime

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MangaReco",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container{
    padding-top:2rem;
}

.big-title{
    font-size:52px;
    font-weight:800;
    color:white;
}

.sub-title{
    font-size:20px;
    color:#9ca3af;
}

.card{
    background-color:#161B22;
    padding:20px;
    border-radius:15px;
    margin-bottom:18px;
    border:1px solid #30363d;
}

.metric{
    text-align:center;
    padding:15px;
    border-radius:12px;
    background:#161B22;
    border:1px solid #30363d;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 MangaReco")

    st.markdown("---")

    st.markdown("## Hybrid AI System")

    st.success("✔ Content-Based Filtering")

    st.success("✔ Collaborative Filtering")

    st.success("✔ Hybrid Recommendation")

    st.markdown("---")

    st.metric("Anime", "12,294")

    st.metric("Ratings", "6.3 Million")

    st.metric("AI Model", "Hybrid")

    st.markdown("---")

    st.markdown("""
Built using

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Surprise (SVD)
""")

# ==========================================
# HERO SECTION
# ==========================================

st.markdown(
    "<div class='big-title'>📚 MangaReco</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Hybrid AI Anime Recommendation System</div>",
    unsafe_allow_html=True
)

st.write("")

st.info("""
This recommendation engine combines

✅ TF-IDF Content-Based Filtering

✅ Cosine Similarity

✅ SVD Collaborative Filtering

to generate personalized anime recommendations.
""")

# ==========================================
# METRICS
# ==========================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Anime", "12,294")

with c2:
    st.metric("Ratings", "6.3 Million")

with c3:
    st.metric("Recommendation Model", "Hybrid AI")

st.divider()

# ==========================================
# INPUTS
# ==========================================

left, right = st.columns([2,1])

with left:

    anime_list = sorted(anime["name"].dropna().unique())

    selected = st.selectbox(
        "🎬 Select an Anime",
        anime_list
    )

with right:

    user_id = st.number_input(
        "👤 User ID",
        min_value=1,
        value=994
    )

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

st.write("")

# ==========================================
# BUTTON
# ==========================================

if st.button("🚀 Generate Recommendations", use_container_width=True):

    with st.spinner("Generating AI Recommendations..."):

        recommendations = hybrid_recommend(
            user_id=user_id,
            anime_title=selected,
            top_n=top_n
        )

    if recommendations is None or recommendations.empty:

        st.error("No recommendations found.")

    else:

        st.success("Recommendations Generated!")

        st.write("")

        for _, row in recommendations.iterrows():

            with st.container():

                st.markdown(f"""
<div class="card">

# 🎬 {row['Anime']}

⭐ **Community Rating:** {row['Community Rating']}

🤖 **Predicted Rating:** {row['Predicted Rating']}

🎭 **Genre:** {row['Genre']}

</div>
""", unsafe_allow_html=True)

# ==========================================
# PIPELINE
# ==========================================

st.divider()

st.subheader("⚙ Recommendation Pipeline")

st.code("""
User Input
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Cosine Similarity
      │
      ▼
Top 50 Similar Anime
      │
      ▼
SVD Collaborative Filtering
      │
      ▼
Personalized Ranking
      │
      ▼
Top Recommendations
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class='footer'>

Built with ❤️ using Python • Streamlit • Pandas • Scikit-Learn • Surprise (SVD)

</div>
""", unsafe_allow_html=True)