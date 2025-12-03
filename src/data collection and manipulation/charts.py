import os
import pandas as pd
import time
from datetime import datetime, timedelta


top_songs = pd.read_csv(
    os.path.join("..", "data", "processed", "kanye_top_fifty_songs.csv")
)

#src https://www.kaggle.com/datasets/elizabethearhart/billboard-hot-1001958-2024/data
billboard_chart = pd.read_csv(
    os.path.join("..", "data", "raw", "hot-100-current.csv")
)


kanye_chart = billboard_chart[billboard_chart["performer"].str.contains("Kanye", case=False, na=False)]


kanye_peak_positions = kanye_chart.groupby('title').agg({
    'peak_pos': 'min',
    'wks_on_chart': 'max'
}).reset_index()


def normalize_title(title):
    return title.lower().strip().replace("'","'")


top_songs['normalized'] = top_songs['Track Name'].apply(normalize_title)
kanye_peak_positions['normalized'] = kanye_peak_positions['title'].apply(normalize_title)


merged = top_songs.merge(
    kanye_peak_positions[['normalized', 'peak_pos', 'wks_on_chart']],
    on='normalized',
    how='left',
)


merged['Charted'] = merged['peak_pos'].notna()


merged = merged.drop(columns=['normalized'])


merged.to_csv(os.path.join("..", "data", "processed", "kanye_billboard_positions.csv"), index=False)


print("Successfully saved Billboard chart data!")
print(f"Total songs: {len(merged)}")
print(f"Songs that charted: {merged['Charted'].sum()}")
