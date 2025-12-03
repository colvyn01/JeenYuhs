import os
import pandas as pd

album_tracklist = pd.read_csv(
    os.path.join("..", "data", "raw", "kanye_album_tracklist.csv")
)
lyrics_data = pd.read_csv(os.path.join("..", "data", "raw", "kanye_lyrics.csv"))


def clean_and_merge():
    print("Cleaning and merging datasets...")
    print("Cleaning album tracklist data...")

    album_clean = album_tracklist.copy()

    album_clean["Featured Artists"] = album_clean["Featured Artists"].fillna("")

    album_clean["Release Date"] = pd.to_datetime(album_clean["Release Date"])

    print("Cleaning lyrics data...")
    lyrics_clean = lyrics_data.copy()

    if "Lyrics" in lyrics_data.columns:
        lyrics_clean = lyrics_clean.dropna(subset=["Lyrics"])
        lyrics_clean["Word Count"] = lyrics_clean["Lyrics"].str.split().str.len()

    print(lyrics_clean.head())

    print("Merging datasets...")

    merged = pd.merge(
        album_clean, lyrics_clean, on=["Track Name", "Album"], how="inner"
    )
    print(f"Merged dataset has {merged.shape[0]} rows and {merged.shape[1]} columns.")
    merged.to_csv(
        os.path.join("..", "data", "processed", "kanye_merged.csv"), index=False
    )


if __name__ == "__main__":
    clean_and_merge()
