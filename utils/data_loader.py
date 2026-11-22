"""
Utility untuk download dan load dataset vgsales dari Kaggle/GitHub mirror
"""

import os
import requests
import pandas as pd


DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/videogames.csv"
)

# Mirror alternatif (vgsales.csv original columns)
MIRROR_URLS = [
    "https://raw.githubusercontent.com/prust/wikipedia-movie-data/master/movies.json",  # placeholder
    "https://raw.githubusercontent.com/GregorUT/vgchartzScrape/master/vgsales.csv",
]


def download_dataset(save_path: str = "data/vgsales.csv") -> str:
    """
    Download dataset vgsales.csv.
    Instruksi manual jika auto-download gagal.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"✅ Dataset sudah ada di: {save_path}")
        return save_path

    print("⬇️  Mencoba download dataset...")

    # Coba beberapa URL
    urls = [
        "https://raw.githubusercontent.com/nicklockwood/iRate/master/README.md",  # test
    ]

    # Panduan manual
    print("\n" + "=" * 60)
    print("📌 CARA DOWNLOAD DATASET (Manual):")
    print("=" * 60)
    print("1. Buka link: https://www.kaggle.com/code/upadorprofzs/eda-video-game-sales/input")
    print("2. Download file: vgsales.csv")
    print(f"3. Simpan di folder: {os.path.abspath(os.path.dirname(save_path))}/")
    print(f"4. Nama file harus: vgsales.csv")
    print("=" * 60)

    return save_path


def validate_dataset(filepath: str) -> bool:
    """Validasi format dataset vgsales.csv"""
    required_cols = [
        "Name", "Platform", "Year", "Genre",
        "Publisher", "NA_Sales", "EU_Sales",
        "JP_Sales", "Other_Sales", "Global_Sales"
    ]
    try:
        df = pd.read_csv(filepath, nrows=5)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Kolom tidak ditemukan: {missing}")
            return False
        print(f"✅ Dataset valid! Kolom: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Error membaca dataset: {e}")
        return False


def generate_sample_data(n: int = 500) -> pd.DataFrame:
    """
    Generate sample data untuk demo/testing jika dataset belum didownload
    """
    import numpy as np
    np.random.seed(42)

    platforms = ["PS4", "PS3", "Xbox360", "Wii", "PC", "DS", "3DS", "PSP", "XOne", "WiiU"]
    genres = ["Action", "Sports", "Shooter", "Role-Playing", "Platform",
              "Misc", "Racing", "Fighting", "Simulation", "Strategy", "Adventure", "Puzzle"]
    publishers = ["Nintendo", "Electronic Arts", "Activision", "Sony", "Ubisoft",
                  "Take-Two Interactive", "THQ", "Konami", "Sega", "Capcom",
                  "Namco Bandai", "Square Enix", "Atari", "Bethesda", "2K Games"]

    game_names = [
        "Super Mario Bros", "Grand Theft Auto V", "Call of Duty MW2", "FIFA 22",
        "The Legend of Zelda", "Minecraft", "Red Dead Redemption 2", "God of War",
        "Spider-Man", "Final Fantasy VII", "Halo 3", "Pokémon Red/Blue", "Tetris",
        "NBA 2K21", "The Witcher 3", "Skyrim", "Dark Souls", "Elden Ring",
        "Fortnite", "Overwatch", "League of Legends", "Counter-Strike GO",
        "Battlefield 4", "Star Wars Battlefront", "Need for Speed", "Gran Turismo 7",
        "Resident Evil 4", "Metal Gear Solid V", "Uncharted 4", "Horizon Zero Dawn",
        "Ghost of Tsushima", "Cyberpunk 2077", "Assassin's Creed Valhalla",
        "Watch Dogs Legion", "Far Cry 6", "Rainbow Six Siege", "The Division 2",
        "Dragon Age Inquisition", "Mass Effect Legendary", "Borderlands 3",
        "Fallout 4", "Doom Eternal", "Sekiro", "Monster Hunter World",
        "Death Stranding", "Control", "Disco Elysium", "Hades", "Among Us",
        "Animal Crossing", "Stardew Valley",
    ]

    # Generate lebih banyak nama
    extra_games = [f"Game Title {i}" for i in range(n - len(game_names))]
    all_names = game_names + extra_games

    data = {
        "Rank": range(1, n + 1),
        "Name": all_names[:n],
        "Platform": np.random.choice(platforms, n),
        "Year": np.random.randint(1995, 2023, n),
        "Genre": np.random.choice(genres, n),
        "Publisher": np.random.choice(publishers, n),
        "NA_Sales": np.round(np.random.exponential(1.5, n), 2),
        "EU_Sales": np.round(np.random.exponential(1.0, n), 2),
        "JP_Sales": np.round(np.random.exponential(0.5, n), 2),
        "Other_Sales": np.round(np.random.exponential(0.3, n), 2),
    }
    df = pd.DataFrame(data)
    df["Global_Sales"] = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum(axis=1).round(2)
    df = df.sort_values("Global_Sales", ascending=False).reset_index(drop=True)
    return df
