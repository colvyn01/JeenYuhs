"""
Merges and cleans album tracklist with lyrics data.
"""

import os
import pandas as pd


def clean_and_merge(album_path, lyrics_path, output_path):
    album_tracklist = pd.read_csv(album_path)
    lyrics_data = pd.read_csv(lyrics_path)

    album_clean = album_tracklist.copy()
    album_clean["Featured Artists"] = album_clean["Featured Artists"].fillna("")
    album_clean["Release Date"] = pd.to_datetime(album_clean["Release Date"])

    lyrics_clean = lyrics_data.copy()
    if "Lyrics" in lyrics_data.columns:
        lyrics_clean = lyrics_clean.dropna(subset=["Lyrics"])
        lyrics_clean["Word Count"] = lyrics_clean["Lyrics"].str.split().str.len()

    merged = pd.merge(
        album_clean, lyrics_clean, on=["Track Name", "Album"], how="inner"
    )
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
