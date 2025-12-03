import os
import pandas as pd

from album_collector import collect_album_tracklist
from lyrics_genius import collect_songid

album_path = os.path.join("..", "data", "raw", "kanye_album_tracklist.csv")
lyric_path = os.path.join("..", "data", "raw", "kanye_lyrics.csv")


def main():
    if os.path.exists(album_path):
        print("already collected")
    else:
        try:
            print("Collecting Album Track Lists from Spotify")
            collect_album_tracklist()
        except Exception as e:
            print("Task Failed:", e)

    print("Successfully Collected Album Track Lists from Spotify")

    if os.path.exists(lyric_path):
        print("already collected")
    else:
        try:
            print("Collecting Song IDs for Genius")
            collect_songid()
        except Exception as e:
            print("Task Failed:", e)

    print("Successfully Collected Song IDs for Genius")

    lyrics = pd.read_csv(lyric_path)
    albums = pd.read_csv(album_path)

    print(lyrics.columns, "\n")
    print(albums.columns, "\n")
    print(albums.head(), "\n")
    print(lyrics.head(), "\n")


if __name__ == "__main__":
    main()
