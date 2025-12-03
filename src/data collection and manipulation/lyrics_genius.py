import os
import lyricsgenius
import pandas as pd


from config import GENIUS_ACCESS_TOKEN

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3)

csv_path = os.path.join("..", "data", "raw", "kanye_album_tracklist.csv")

songIds = []
lyrics = []


def collect_songid():
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        track = row["Track Name"]  # FIXED: Changed from 'track_name' to 'Track Name'
        album = row["Album"]

        song = genius.search_song(track, "Kanye West")
        if song:
            genius_album = song._body.get("album", None)
            if genius_album and album.lower() == genius_album["name"].lower():
                songIds.append(
                    {
                        "Track Name": track,
                        "Genius Track ID": song._body["id"],
                        "Lyrics": song.lyrics,
                        "Album": album,
                    }
                )
    songId_df = pd.DataFrame(songIds)
    songId_df.to_csv(os.path.join("..", "data", "raw", "kanye_lyrics.csv"), index=False)
    print(songId_df.head())
