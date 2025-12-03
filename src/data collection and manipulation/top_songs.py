import os
import pandas as pd


df = pd.read_csv(os.path.join("..", "data", "processed", "kanye_merged.csv"))

top_five_songs = (
    df.groupby("Album")
    .apply(lambda x: x.nlargest(5, "Popularity"), include_groups=False)
    .reset_index(level=0)
    .reset_index(drop=True)
    .sort_values(by="Popularity", ascending=False)
)

top_five_songs.to_csv(
    os.path.join("..", "data", "processed", "kanye_top_five_songs.csv"), index=False
)

print(f"Selected {len(top_five_songs)} tracks")
print(top_five_songs[["Album", "Track Name", "Popularity"]])


top_fifty_songs = df.nlargest(50, "Popularity").reset_index(drop=True)

top_fifty_songs.to_csv(
    os.path.join("..", "data", "processed", "kanye_top_fifty_songs.csv"), index=False
)

print(f"Selected {len(top_fifty_songs)} tracks")
print(top_fifty_songs[["Album", "Track Name", "Popularity"]])
