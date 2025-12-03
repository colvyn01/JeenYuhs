import os
import pandas as pd

df = pd.read_csv(os.path.join("..", "data", "processed", "kanye_billboard_positions.csv"))

charted_only = df[df['Charted'] == True]


charted_only = charted_only.sort_values('peak_pos')

charted_only.to_csv(os.path.join("..", "data", "processed", "kanye_charted_songs_only.csv"), index=False)

print(f"Saved {len(charted_only)} charted songs to: kanye_charted_songs_only.csv")
print(f"\nPreview:")
print(charted_only[['Track Name', 'Album', 'peak_pos', 'wks_on_chart']].head(10))
