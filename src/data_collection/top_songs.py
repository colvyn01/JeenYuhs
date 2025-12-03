"""
Extracts top songs by popularity from merged data.
"""

import os
import pandas as pd


def get_top_songs(merged_path, top_five_path, top_fifty_path):
    df = pd.read_csv(merged_path)

    top_five_songs = (
        df.groupby("Album")
        .apply(lambda x: x.nlargest(5, "Popularity"), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
        .sort_values(by="Popularity", ascending=False)
    )
    top_five_songs.to_csv(top_five_path, index=False)
    print(f"Top 5 per album: {len(top_five_songs)} tracks")

    top_fifty_songs = df.nlargest(50, "Popularity").reset_index(drop=True)
    top_fifty_songs.to_csv(top_fifty_path, index=False)
    print(f"Top 50 overall: {len(top_fifty_songs)} tracks")
