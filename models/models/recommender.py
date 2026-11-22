"""
Game Recommendation System
Menggunakan Content-Based Filtering + KMeans Clustering
Dataset: vgsales.csv dari Kaggle
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")


class GameRecommender:
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.feature_matrix = None
        self.cosine_sim = None
        self.kmeans = None
        self.knn = None
        self.scaler = MinMaxScaler()
        self.le_platform = LabelEncoder()
        self.le_genre = LabelEncoder()
        self.le_publisher = LabelEncoder()
        self.indices = None
        self.is_fitted = False

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dan preprocess dataset vgsales.csv"""
        df = pd.read_csv(filepath)

        # Drop rows dengan missing values penting
        df = df.dropna(subset=["Name", "Genre", "Platform", "Publisher"])
        df["Year"] = df["Year"].fillna(df["Year"].median())
        df["Year"] = df["Year"].astype(int)

        # Drop duplikat nama game (ambil yang Global_Sales tertinggi)
        df = df.sort_values("Global_Sales", ascending=False)
        df = df.drop_duplicates(subset=["Name"], keep="first")
        df = df.reset_index(drop=True)

        self.df = df
        return df

    def build_features(self):
        """Bangun feature matrix untuk content-based filtering"""
        df = self.df.copy()

        # Encode categorical
        df["Platform_enc"] = self.le_platform.fit_transform(df["Platform"])
        df["Genre_enc"] = self.le_genre.fit_transform(df["Genre"])
        df["Publisher_enc"] = self.le_publisher.fit_transform(df["Publisher"])

        # Fitur numerik
        features = [
            "Platform_enc",
            "Genre_enc",
            "Publisher_enc",
            "Year",
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales",
            "Global_Sales",
        ]

        feature_matrix = df[features].values
        feature_matrix = self.scaler.fit_transform(feature_matrix)

        self.feature_matrix = feature_matrix
        self.df_processed = df

        # Index berdasarkan nama game
        self.indices = pd.Series(df.index, index=df["Name"].str.lower())
        return feature_matrix

    def train_cosine_similarity(self):
        """Hitung cosine similarity antar game"""
        self.cosine_sim = cosine_similarity(self.feature_matrix, self.feature_matrix)
        return self.cosine_sim

    def train_kmeans(self, n_clusters: int = 20):
        """Clustering game menggunakan KMeans"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_processed["cluster"] = self.kmeans.fit_predict(self.feature_matrix)
        return self.kmeans

    def train_knn(self, n_neighbors: int = 11):
        """KNN untuk fallback rekomendasi"""
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(self.feature_matrix)
        return self.knn

    def fit(self, filepath: str):
        """Pipeline lengkap training"""
        print("📂 Loading data...")
        self.load_data(filepath)
        print(f"✅ Loaded {len(self.df)} games")

        print("🔧 Building features...")
        self.build_features()

        print("📐 Training Cosine Similarity...")
        self.train_cosine_similarity()

        print("🔵 Training KMeans Clustering...")
        self.train_kmeans()

        print("🔍 Training KNN...")
        self.train_knn()

        self.is_fitted = True
        print("✅ Model siap digunakan!")
        return self

    def recommend_by_title(self, game_title: str, top_n: int = 10) -> pd.DataFrame:
        """Rekomendasi berdasarkan judul game (Content-Based)"""
        game_lower = game_title.lower().strip()

        # Cari game yang mirip jika tidak exact match
        if game_lower not in self.indices:
            matches = [idx for name, idx in self.indices.items() if game_lower in name]
            if not matches:
                return pd.DataFrame({"Error": [f"Game '{game_title}' tidak ditemukan."]})
            idx = matches[0]
        else:
            idx = self.indices[game_lower]

        # Ambil similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: top_n + 1]  # exclude diri sendiri

        game_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 4) for i in sim_scores]

        result = self.df_processed.iloc[game_indices][
            ["Name", "Platform", "Genre", "Publisher", "Year", "Global_Sales"]
        ].copy()
        result["Similarity_Score"] = scores
        result = result.reset_index(drop=True)
        return result

    def recommend_by_genre(self, genre: str, top_n: int = 10) -> pd.DataFrame:
        """Rekomendasi berdasarkan genre, diurutkan Global_Sales"""
        genre_lower = genre.lower()
        filtered = self.df_processed[
            self.df_processed["Genre"].str.lower() == genre_lower
        ]
        if filtered.empty:
            return pd.DataFrame({"Error": [f"Genre '{genre}' tidak ditemukan."]})

        result = filtered.nlargest(top_n, "Global_Sales")[
            ["Name", "Platform", "Genre", "Publisher", "Year", "Global_Sales"]
        ].reset_index(drop=True)
        return result

    def recommend_by_cluster(self, game_title: str, top_n: int = 10) -> pd.DataFrame:
        """Rekomendasi berdasarkan cluster KMeans"""
        game_lower = game_title.lower().strip()

        if game_lower not in self.indices:
            matches = [idx for name, idx in self.indices.items() if game_lower in name]
            if not matches:
                return pd.DataFrame({"Error": [f"Game '{game_title}' tidak ditemukan."]})
            idx = matches[0]
        else:
            idx = self.indices[game_lower]

        cluster_id = self.df_processed.iloc[idx]["cluster"]
        cluster_games = self.df_processed[
            (self.df_processed["cluster"] == cluster_id)
            & (self.df_processed.index != idx)
        ]
        result = cluster_games.nlargest(top_n, "Global_Sales")[
            ["Name", "Platform", "Genre", "Publisher", "Year", "Global_Sales", "cluster"]
        ].reset_index(drop=True)
        return result

    def recommend_by_preferences(
        self,
        genre: str = None,
        platform: str = None,
        year_min: int = None,
        year_max: int = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Rekomendasi berdasarkan preferensi user"""
        filtered = self.df_processed.copy()

        if genre:
            filtered = filtered[filtered["Genre"].str.lower() == genre.lower()]
        if platform:
            filtered = filtered[filtered["Platform"].str.lower() == platform.lower()]
        if year_min:
            filtered = filtered[filtered["Year"] >= year_min]
        if year_max:
            filtered = filtered[filtered["Year"] <= year_max]

        if filtered.empty:
            return pd.DataFrame({"Error": ["Tidak ada game yang cocok dengan preferensi."]})

        result = filtered.nlargest(top_n, "Global_Sales")[
            ["Name", "Platform", "Genre", "Publisher", "Year", "Global_Sales"]
        ].reset_index(drop=True)
        return result

    def get_game_info(self, game_title: str) -> dict:
        """Ambil info detail suatu game"""
        game_lower = game_title.lower().strip()
        if game_lower not in self.indices:
            matches = [idx for name, idx in self.indices.items() if game_lower in name]
            if not matches:
                return {}
            idx = matches[0]
        else:
            idx = self.indices[game_lower]

        row = self.df_processed.iloc[idx]
        return row.to_dict()

    def get_stats(self) -> dict:
        """Statistik dataset"""
        return {
            "total_games": len(self.df_processed),
            "genres": sorted(self.df_processed["Genre"].unique().tolist()),
            "platforms": sorted(self.df_processed["Platform"].unique().tolist()),
            "year_range": (int(self.df_processed["Year"].min()), int(self.df_processed["Year"].max())),
            "top_publishers": self.df_processed.groupby("Publisher")["Global_Sales"]
            .sum()
            .nlargest(10)
            .index.tolist(),
        }
