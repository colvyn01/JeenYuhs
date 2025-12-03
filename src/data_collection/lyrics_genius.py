"""
Collects song IDs and lyrics from Genius API.
"""

import os
import sys
import pandas as pd
import lyricsgenius

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GENIUS_ACCESS_TOKEN

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3)


def collect_lyrics(album_path, output_path):
    df = pd.read_csv(album_path)
    song_data = []
    
    for idx, row in df.iterrows():
        track = row["Track Name"]
        album = row["Album"]

        song = genius.search_song(track, "Kanye West")
        if song:
            genius_album = song._body.get("album", None)
            if genius_album and album.lower() == genius_album["name"].lower():
                song_data.append({
                    "Track Name": track,
                    "Genius Track ID": song._body["id"],
                    "Lyrics": song.lyrics,
                    "Album": album,
                })

    songId_df = pd.DataFrame(song_data)
    songId_df.to_csv(output_path, index=False)
    print("Saved lyrics data")
