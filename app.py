"""
🎮 Game Recommendation System
Streamlit App - Deployment Interface
Dataset: vgsales.csv (Kaggle Video Game Sales)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Tambahkan path project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.recommender import GameRecommender
from utils.data_loader import generate_sample_data, validate_dataset

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎮 Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #6C63FF;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .rec-card {
        background: #1a1a2e;
        border: 1px solid #6C63FF;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stButton button {
        background: linear-gradient(135deg, #6C63FF, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #6C63FF;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & CACHE MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model(use_sample: bool = False, uploaded_file=None):
    recommender = GameRecommender()

    if uploaded_file is not None:
        # Gunakan file yang diupload user
        df = pd.read_csv(uploaded_file)
        df.to_csv("data/vgsales.csv", index=False)
        recommender.fit("data/vgsales.csv")
    elif os.path.exists("data/vgsales.csv") and not use_sample:
        recommender.fit("data/vgsales.csv")
    else:
        # Gunakan sample data untuk demo
        st.info("⚠️ Menggunakan sample data untuk demo. Upload vgsales.csv untuk hasil penuh.")
        df = generate_sample_data(1000)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/vgsales.csv", index=False)
        recommender.fit("data/vgsales.csv")

    return recommender


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # Upload dataset
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload vgsales.csv dari Kaggle",
        type=["csv"],
        help="Download dari: https://www.kaggle.com/code/upadorprofzs/eda-video-game-sales/input"
    )

    use_sample = st.checkbox("Gunakan Sample Data (Demo)", value=True if uploaded_file is None else False)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    **Source**: Kaggle Video Game Sales  
    **Records**: ~16.500 games  
    **Fitur**: Platform, Genre, Publisher, Sales Region
    """)



# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🎮 Game Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Content-Based Filter | vgsales Dataset</div>', unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("🔄 Memuat model rekomendasi..."):
    try:
        model = load_model(use_sample=use_sample, uploaded_file=uploaded_file)
        stats = model.get_stats()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎮 Total Games", f"{stats['total_games']:,}")
with col2:
    st.metric("🎭 Genre Tersedia", len(stats['genres']))
with col3:
    st.metric("🖥️ Platform Tersedia", len(stats['platforms']))
with col4:
    st.metric("📅 Rentang Tahun", f"{stats['year_range'][0]} – {stats['year_range'][1]}")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab5 = st.tabs([
    "🔍 Cari by Judul",
    "🎭 Cari by Genre",
    "⚙️ Filter Preferensi",
    "📊 EDA & Visualisasi"
])


# ── TAB 1: By Title ──────────────────────────
with tab1:
    st.subheader("🔍 Rekomendasi Berdasarkan Judul Game")
    st.caption("Masukkan judul game yang kamu suka dan sistem akan merekomendasikan game serupa.")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        game_title = st.text_input(
            "Nama Game",
            placeholder="Contoh: Grand Theft Auto V, FIFA, Mario...",
            label_visibility="collapsed"
        )
    with col_btn:
        top_n_title = st.number_input("Top N", min_value=5, max_value=20, value=10, label_visibility="collapsed")

    # Autocomplete suggestion
    if game_title and len(game_title) >= 3:
        suggestions = [
            name for name in model.df_processed["Name"].tolist()
            if game_title.lower() in name.lower()
        ][:5]
        if suggestions:
            st.caption(f"💡 Saran: {' | '.join(suggestions)}")

    if st.button("🚀 Cari Rekomendasi", key="btn_title"):
        if not game_title:
            st.warning("⚠️ Masukkan judul game terlebih dahulu!")
        else:
            with st.spinner("Mencari rekomendasi..."):
                # Tampilkan info game yang dicari
                game_info = model.get_game_info(game_title)
                if game_info:
                    with st.expander("📋 Info Game yang Dicari", expanded=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("🎮 Platform", game_info.get("Platform", "-"))
                        c2.metric("🎭 Genre", game_info.get("Genre", "-"))
                        c3.metric("📅 Tahun", int(game_info.get("Year", 0)))
                        c4.metric("💰 Global Sales", f"{game_info.get('Global_Sales', 0):.2f}M")

                result = model.recommend_by_title(game_title, top_n=int(top_n_title))

                if "Error" in result.columns:
                    st.error(result["Error"][0])
                else:
                    st.success(f"✅ Ditemukan {len(result)} rekomendasi untuk **'{game_title}'**")
                    display_result = result.drop(columns=["Similarity_Score"], errors="ignore")
                    st.dataframe(
                        display_result.style.background_gradient(subset=["Global_Sales"], cmap="Purples"),
                        use_container_width=True,
                        hide_index=True
                    )


# ── TAB 2: By Genre ──────────────────────────
with tab2:
    st.subheader("🎭 Rekomendasi Berdasarkan Genre")
    st.caption("Pilih genre favoritmu dan lihat top game berdasarkan penjualan global.")

    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        selected_genre = st.selectbox("Pilih Genre", stats["genres"])
    with col_g2:
        top_n_genre = st.number_input("Jumlah Top Games", min_value=5, max_value=20, value=10, key="top_genre")

    if st.button("🎯 Tampilkan Top Games", key="btn_genre"):
        with st.spinner("Memuat..."):
            result = model.recommend_by_genre(selected_genre, top_n=int(top_n_genre))

            if "Error" in result.columns:
                st.error(result["Error"][0])
            else:
                st.success(f"✅ Top {len(result)} game genre **{selected_genre}**")

                # Pie chart platform distribution
                col_table, col_chart = st.columns([3, 2])
                with col_table:
                    st.dataframe(
                        result.style.background_gradient(subset=["Global_Sales"], cmap="Greens"),
                        use_container_width=True,
                        hide_index=True
                    )
                with col_chart:
                    platform_counts = result["Platform"].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.pie(platform_counts.values, labels=platform_counts.index,
                            autopct="%1.0f%%", startangle=90,
                            colors=sns.color_palette("Set2", len(platform_counts)))
                    ax2.set_title(f"Distribusi Platform\nGenre {selected_genre}", color="white")
                    fig2.patch.set_facecolor("#0e1117")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()


# ── TAB 3: Filter Preferensi ─────────────────
with tab3:
    st.subheader("⚙️ Rekomendasi Berdasarkan Preferensi")
    st.caption("Filter game sesuai preferensimu: genre, platform, dan rentang tahun rilis.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pref_genre = st.selectbox("Genre (opsional)", ["Semua"] + stats["genres"], key="pref_genre")
        pref_platform = st.selectbox("Platform (opsional)", ["Semua"] + stats["platforms"])
    with col_p2:
        year_range = st.slider(
            "Rentang Tahun Rilis",
            min_value=stats["year_range"][0],
            max_value=stats["year_range"][1],
            value=(2000, stats["year_range"][1])
        )
        top_n_pref = st.number_input("Jumlah Rekomendasi", min_value=5, max_value=30, value=10, key="top_pref")

    if st.button("🔍 Cari Game", key="btn_pref"):
        with st.spinner("Mencari..."):
            result = model.recommend_by_preferences(
                genre=None if pref_genre == "Semua" else pref_genre,
                platform=None if pref_platform == "Semua" else pref_platform,
                year_min=year_range[0],
                year_max=year_range[1],
                top_n=int(top_n_pref)
            )

            if "Error" in result.columns:
                st.warning(result["Error"][0])
            else:
                st.success(f"✅ Ditemukan **{len(result)}** game sesuai preferensi")
                st.dataframe(
                    result.style.background_gradient(subset=["Global_Sales"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True
                )

                # Timeline chart
                if len(result) > 3:
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    ax3.scatter(result["Year"], result["Global_Sales"],
                                c=result["Global_Sales"], cmap="plasma", s=100, alpha=0.8)
                    for _, row in result.iterrows():
                        ax3.annotate(row["Name"][:15], (row["Year"], row["Global_Sales"]),
                                     fontsize=7, color="#ccc", xytext=(3, 3), textcoords="offset points")
                    ax3.set_xlabel("Tahun Rilis", color="white")
                    ax3.set_ylabel("Global Sales (Juta)", color="white")
                    ax3.set_title("Distribusi Sales per Tahun", color="white")
                    ax3.set_facecolor("#0e1117")
                    fig3.patch.set_facecolor("#0e1117")
                    ax3.tick_params(colors="white")
                    ax3.spines["bottom"].set_color("#444")
                    ax3.spines["left"].set_color("#444")
                    ax3.spines["top"].set_visible(False)
                    ax3.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()

# ── TAB 5: EDA ──────────────────────────────
with tab5:
    st.subheader("📊 Exploratory Data Analysis")

    df_viz = model.df_processed

    col_v1, col_v2 = st.columns(2)

    # 1. Top 10 games by global sales
    with col_v1:
        st.markdown("**🏆 Top 10 Game by Total Global Sales**")
        top10 = df_viz.nlargest(10, "Global_Sales")
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        colors5 = sns.color_palette("rocket", 10)
        ax5.barh(top10["Name"][::-1], top10["Global_Sales"][::-1], color=colors5[::-1])
        ax5.set_xlabel("Global Sales (Juta)", color="white")
        ax5.set_facecolor("#0e1117")
        fig5.patch.set_facecolor("#0e1117")
        ax5.tick_params(colors="white")
        ax5.spines["bottom"].set_color("#444")
        ax5.spines["left"].set_color("#444")
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    # 2. Sales by genre
    with col_v2:
        st.markdown("**🎭 Total Sales per Genre**")
        genre_sales = df_viz.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        sns.barplot(x=genre_sales.values, y=genre_sales.index, ax=ax6, palette="mako")
        ax6.set_xlabel("Total Global Sales (Juta)", color="white")
        ax6.set_facecolor("#0e1117")
        fig6.patch.set_facecolor("#0e1117")
        ax6.tick_params(colors="white")
        ax6.spines["bottom"].set_color("#444")
        ax6.spines["left"].set_color("#444")
        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    col_v3, col_v4 = st.columns(2)

    # 3. Sales trend per year
    with col_v3:
        st.markdown("**📅 Total Sales per Tahun untuk Semua Games**")
        year_sales = df_viz.groupby("Year")["Global_Sales"].sum()
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        ax7.plot(year_sales.index, year_sales.values, color="#6C63FF", linewidth=2.5)
        ax7.fill_between(year_sales.index, year_sales.values, alpha=0.2, color="#6C63FF")
        ax7.set_xlabel("Tahun", color="white")
        ax7.set_ylabel("Global Sales (Juta)", color="white")
        ax7.set_facecolor("#0e1117")
        fig7.patch.set_facecolor("#0e1117")
        ax7.tick_params(colors="white")
        ax7.spines["bottom"].set_color("#444")
        ax7.spines["left"].set_color("#444")
        ax7.spines["top"].set_visible(False)
        ax7.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    # 4. Top publishers
    with col_v4:
        st.markdown("**🏢 Top 10 Publisher by Sales**")
        pub_sales = df_viz.groupby("Publisher")["Global_Sales"].sum().nlargest(10)
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        colors8 = sns.color_palette("flare", 10)
        ax8.barh(pub_sales.index[::-1], pub_sales.values[::-1], color=colors8[::-1])
        ax8.set_xlabel("Total Sales (Juta)", color="white")
        ax8.set_facecolor("#0e1117")
        fig8.patch.set_facecolor("#0e1117")
        ax8.tick_params(colors="white")
        ax8.spines["bottom"].set_color("#444")
        ax8.spines["left"].set_color("#444")
        ax8.spines["top"].set_visible(False)
        ax8.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()

    # 5. Regional sales comparison
    st.markdown("**🌍 Perbandingan Penjualan per Region**")
    regions = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    region_totals = df_viz[regions].sum()
    region_labels = ["North America", "Europe", "Japan", "Other"]
    fig9, ax9 = plt.subplots(figsize=(8, 4))
    colors9 = ["#6C63FF", "#F59E0B", "#EF4444", "#10B981"]
    bars9 = ax9.bar(region_labels, region_totals.values, color=colors9, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars9, region_totals.values):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.0f} Jt", ha="center", va="bottom", color="white", fontsize=10)
    ax9.set_ylabel("Total Sales (Juta)", color="white")
    ax9.set_title("Total Penjualan per Region", color="white")
    ax9.set_facecolor("#0e1117")
    fig9.patch.set_facecolor("#0e1117")
    ax9.tick_params(colors="white")
    ax9.spines["bottom"].set_color("#444")
    ax9.spines["left"].set_color("#444")
    ax9.spines["top"].set_visible(False)
    ax9.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    # Raw data sample
    with st.expander("🗂️ Lihat Sample Data"):
        st.dataframe(df_viz[["Name", "Platform", "Genre", "Publisher", "Year",
                               "NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales"]].head(50),
                     use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    🎮 Game Recommendation System <br>
</div>
""", unsafe_allow_html=True)
