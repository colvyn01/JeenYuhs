"""
(jeen-yuhs/src/album_collector.py)
@author Colvyn Harris Mathan Mohan
@version 11/20/2025

Collects tracklist data from Spotify's API and saves it to CSV.
"""

import os
import pandas as pd
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
    )
)

KANYE_ALBUMS = {
    "The College Dropout": "4Uv86qWpGTxf7fU7lG5X6F",
    "Late Registration": "5ll74bqtkcXlKE7wwkMq4g",
    "Graduation": "4SZko61aMnmgvNhfhgTuD3",
    "808s & Heartbreak": "3WFTGIO6E3Xh4paEOBY9OU",
    "My Beautiful Dark Twisted Fantasy": "20r762YmB5HeofjMCiPMLv",
    "Watch The Throne": "0OcMap99vLEeGkBCfCwRwS",
    "Yeezus": "7D2NdGvBHIavgLhmcwhluK",
    "The Life Of Pablo": "7gsWAHLeT0w7es6FofOXk1",
    "ye": "2Ek1q2haOnxVqhvVKqMvJe",
    "KIDS SEE GHOSTS": "6pwuKxMUkNg673KETsXPUV",
    "JESUS IS KING": "0FgZKfoU2Br5sHOfvZKTI9",
    "Donda": "5CnpZV3q5BcESefcB3WJmz",
}

album_tracklist = []


def collect_album_tracklist():
    for album_name, album_id in KANYE_ALBUMS.items():
        album_info = sp.album(album_id)
        release_date = album_info["release_date"]

        album_tracks = sp.album_tracks(album_id)["items"]

        for track in album_tracks:
            full_track_details = sp.track(track["id"])

            # Extract all artist names
            all_artists = [artist["name"] for artist in track["artists"]]
            primary_artist = all_artists[0] if all_artists else None
            featured_artists = all_artists[1:] if len(all_artists) > 1 else []

            track_data = {
                "Album": album_name,
                "Album ID": album_id,
                "Release Date": release_date,
                "Track Number": track["track_number"],
                "Disc Number": track["disc_number"],
                "Track Name": track["name"],
                "Track ID": track["id"],
                "Explicit": track["explicit"],
                "Duration": track["duration_ms"],
                "Primary Artist": primary_artist,
                "Featured Artists": ", ".join(
                    featured_artists
                ),  # Join as comma-separated string
                "Total Artists": len(all_artists),
                "Popularity": full_track_details["popularity"],
            }

            album_tracklist.append(track_data)

    df = pd.DataFrame(album_tracklist)
    df.to_csv(
        os.path.join("..", "data", "raw", "kanye_album_tracklist.csv"), index=False
    )
    print("Saved kanye_album_tracklist.csv")
