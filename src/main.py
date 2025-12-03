"""
Main pipeline for Kanye West data collection and processing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_collection"))

from data_collection.album_collector import collect_album_tracklist
from data_collection.lyrics_genius import collect_lyrics
from data_collection.merge_clean import clean_and_merge
from data_collection.top_songs import get_top_songs
from data_collection.charts import add_chart_positions
from data_collection.charted_only import filter_charted_songs
from audio.audio_analysis import analyze_audio_tracks

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ALBUM_PATH = os.path.join(RAW_DIR, "kanye_album_tracklist.csv")
LYRICS_PATH = os.path.join(RAW_DIR, "kanye_lyrics.csv")
BILLBOARD_PATH = os.path.join(RAW_DIR, "hot-100-current.csv")
MERGED_PATH = os.path.join(PROCESSED_DIR, "merged.csv")
TOP_FIVE_PATH = os.path.join(PROCESSED_DIR, "top5_album.csv")
TOP_FIFTY_PATH = os.path.join(PROCESSED_DIR, "top50.csv")
BILLBOARD_POSITIONS_PATH = os.path.join(PROCESSED_DIR, "charts.csv")
CHARTED_ONLY_PATH = os.path.join(PROCESSED_DIR, "charted.csv")
AUDIO_ANALYSIS_PATH = os.path.join(OUTPUT_DIR, "kanye_track_analysis.csv")


def main():
    # Step 1: Collect album tracklist from Spotify
    if os.path.exists(ALBUM_PATH):
        print("Album tracklist already exists")
    else:
        print("Collecting album tracklist from Spotify...")
        collect_album_tracklist(ALBUM_PATH)

    # Step 2: Collect lyrics from Genius
    if os.path.exists(LYRICS_PATH):
        print("Lyrics data already exists")
    else:
        print("Collecting lyrics from Genius...")
        collect_lyrics(ALBUM_PATH, LYRICS_PATH)

    # Step 3: Merge and clean data
    if os.path.exists(MERGED_PATH):
        print("Merged data already exists")
    else:
        print("Merging and cleaning data...")
        clean_and_merge(ALBUM_PATH, LYRICS_PATH, MERGED_PATH)

    # Step 4: Get top songs
    if os.path.exists(TOP_FIFTY_PATH):
        print("Top songs data already exists")
    else:
        print("Extracting top songs...")
        get_top_songs(MERGED_PATH, TOP_FIVE_PATH, TOP_FIFTY_PATH)

    # Step 5: Add Billboard chart positions
    if os.path.exists(BILLBOARD_POSITIONS_PATH):
        print("Billboard positions already exists")
    else:
        print("Adding Billboard chart positions...")
        add_chart_positions(TOP_FIFTY_PATH, BILLBOARD_PATH, BILLBOARD_POSITIONS_PATH)

    # Step 6: Filter charted songs only
    if os.path.exists(CHARTED_ONLY_PATH):
        print("Charted songs data already exists")
    else:
        print("Filtering charted songs...")
        filter_charted_songs(BILLBOARD_POSITIONS_PATH, CHARTED_ONLY_PATH)

    # Step 7: Analyze audio tracks
    if os.path.exists(AUDIO_ANALYSIS_PATH):
        print("Audio analysis already exists")
    else:
        print("Analyzing audio tracks...")
        analyze_audio_tracks(AUDIO_DIR, AUDIO_ANALYSIS_PATH)

    print("Pipeline complete")


if __name__ == "__main__":
    main()
