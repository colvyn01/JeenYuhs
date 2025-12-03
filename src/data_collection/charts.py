"""
Matches songs against Billboard Hot 100 chart data.
"""

import os
import pandas as pd


def normalize_title(title):
    return title.lower().strip().replace("'", "'")


def add_chart_positions(top_songs_path, billboard_path, output_path):
    top_songs = pd.read_csv(top_songs_path)
    billboard_chart = pd.read_csv(billboard_path)

    kanye_chart = billboard_chart[
        billboard_chart["performer"].str.contains("Kanye", case=False, na=False)
    ]

    kanye_peak_positions = kanye_chart.groupby("title").agg({
        "peak_pos": "min",
        "wks_on_chart": "max"
    }).reset_index()

    top_songs["normalized"] = top_songs["Track Name"].apply(normalize_title)
    kanye_peak_positions["normalized"] = kanye_peak_positions["title"].apply(normalize_title)

    merged = top_songs.merge(
        kanye_peak_positions[["normalized", "peak_pos", "wks_on_chart"]],
        on="normalized",
        how="left",
    )

    merged["Charted"] = merged["peak_pos"].notna()
    merged = merged.drop(columns=["normalized"])
    merged.to_csv(output_path, index=False)

    print(f"Billboard positions added: {merged['Charted'].sum()} charted songs")
