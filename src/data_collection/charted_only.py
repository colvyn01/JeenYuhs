"""
Filters to only songs that charted on Billboard.
"""

import os
import pandas as pd


def filter_charted_songs(billboard_path, output_path):
    df = pd.read_csv(billboard_path)

    charted_only = df[df["Charted"] == True]
    charted_only = charted_only.sort_values("peak_pos")
    charted_only.to_csv(output_path, index=False)

    print(f"Charted songs only: {len(charted_only)} tracks")
