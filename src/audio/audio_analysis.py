import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dir = os.path.dirname(__file__)


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

""" Scale degree definitions in semitones relative to the root note """
# Root (0), Major Second (2), Major Third (4), Perfect Fourth (5), Perfect Fifth (7), Major Sixth (9), and Major Seventh (11)
MAJOR_SCALE_DEGREES = np.array([0, 2, 4, 5, 7, 9, 11])
# Root (0), Major Second (2), Minor Third (3), Perfect Fourth (5), Perfect Fifth (7), Minor Sixth (8), and Minor Seventh (10)
NATURAL_MINOR_SCALE_DEGREES = np.array([0, 2, 3, 5, 7, 8, 10])
# Root (0), Major Second (2), Minor Third (3), Perfect Fourth (5), Perfect Fifth (7), Minor Sixth (8), and Major Seventh (11)
HARMONIC_MINOR_SCALE_DEGREES = np.array([0, 2, 3, 5, 7, 8, 11])
# Root (0), Major Second (2), Minor Third (3), Perfect Fourth (5), Perfect Fifth (7), Major Sixth (9), and Major Seventh (11)
MELODIC_MINOR_SCALE_DEGREES = np.array([0, 2, 3, 5, 7, 9, 11])
# Root (0), Major Second (2), Major Third (4), Perfect Fifth (7), and Major Sixth (9)
PENTATONIC_MAJOR_DEGREES = np.array([0, 2, 4, 7, 9])
# Root (0), Minor Third (3), Perfect Fourth (5), Perfect Fifth (7), and Minor Seventh (10)
PENTATONIC_MINOR_DEGREES = np.array([0, 3, 5, 7, 10])
# Root (0), Minor Third (3), Perfect Fourth (5), Diminished Fifth (6), Perfect Fifth (7), Minor Seventh (10)
BLUES_SCALE_DEGREES = np.array([0, 3, 5, 6, 7, 10])
# Root (0), Major Second (2), Major Third (4), Augmented Fourth (6), Augmented Fifth (8), Minor Seventh (10)
WHOLE_TONE_SCALE_DEGREES = np.array([0, 2, 4, 6, 8, 10])
# All semitone positions from 0 to 11
CHROMATIC_SCALE_DEGREES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


"""Weights for scale degrees based on their importance in establishing tonality"""
# Root (1st) is the most important
TONIC_WEIGHT = 1.0
# Dominant (5th) and Subdominant (4th) are important in establishing tonality
DOMINANT_WEIGHT = 0.9
# Subdominant (4th) is slightly less important than Dominant (5th)
SUBDOMINANT_WEIGHT = 0.7
# Mediant (3rd) and Submediant (6th) has moderate importance
NON_DIATONIC_WEIGHT = 0.3


LOAD_TRACK = {
    'Follow God': os.path.join(dir, "..", "..", "data", "audio", "Follow God.mp3"),
    'Stronger': os.path.join(dir, "..", "..", "data", "audio", "Stronger.mp3"),
}


track_details = {}


def detect_key_scale(samples, sample_rate):
    # trim silence from the beginning and end of the audio
    samples_trimmed, _ = librosa.effects.trim(samples)

    # remove percussive elements to focus on harmonic content
    filtered_samples, _ = librosa.effects.hpss(samples_trimmed)

    # chroma using constant-Q transform
    chroma_cq = librosa.feature.chroma_cqt(y=filtered_samples, sr=sample_rate, bins_per_octave=48)

    # compute the mean chroma vector for the overall pitch distribution
    chroma_mean = np.mean(chroma_cq, axis=1)

    # normalize the chroma vector values sum up to 1 to get a better representation of pitch class distribution
    chroma_mean /= np.sum(chroma_mean)

    # define scales to check against
    scales = {
        'Major': MAJOR_SCALE_DEGREES,
        'Natural Minor': NATURAL_MINOR_SCALE_DEGREES,
        'Harmonic Minor': HARMONIC_MINOR_SCALE_DEGREES,
        'Melodic Minor': MELODIC_MINOR_SCALE_DEGREES,
        'Pentatonic Major': PENTATONIC_MAJOR_DEGREES,
        'Pentatonic Minor': PENTATONIC_MINOR_DEGREES,
        'Blues': BLUES_SCALE_DEGREES,
        'Whole Tone': WHOLE_TONE_SCALE_DEGREES,
        # 'Chromatic' excluded as fallback only
    }

    best_key = None
    best_scale = None
    best_score = -np.inf  # negative infinity baseline

    # --- Evaluate every scale in every possible key (0-11) ---
    for scale_name, degrees in scales.items():
        for root in range(12):
            # Shift degrees so the root note becomes the tonic of the scale
            scale_positions = (degrees + root) % 12

            # Total chroma energy from all allowed scale tones
            match_energy = chroma_mean[scale_positions].sum()

            # Extra weights for harmonic anchors using declared constants
            tonic = chroma_mean[root]                           # Root
            dominant = chroma_mean[(root + 7) % 12]             # Perfect fifth
            subdominant = chroma_mean[(root + 5) % 12]          # Perfect fourth
            mediant = chroma_mean[(root + 4) % 12]              # Major/Minor third
            submediant = chroma_mean[(root + 9) % 12]           # Sixth degree

            # Weighted score: sum + tonic + dominant + subdominant + mediant + submediant
            score = (
                match_energy
                + TONIC_WEIGHT * tonic
                + DOMINANT_WEIGHT * dominant
                + SUBDOMINANT_WEIGHT * subdominant
                + NON_DIATONIC_WEIGHT * mediant
                + NON_DIATONIC_WEIGHT * submediant
            )

            # Store best result
            if score > best_score:
                best_score = score
                best_key = NOTE_NAMES[root]
                best_scale = scale_name

    return best_key, best_scale




def chart_finder(track_name):
    df = pd.read_csv(os.path.join(dir, "..", "..", "data", "processed", "charted.csv"))

    # Find the row for this track
    track_title = df[df["Track Name"] == track_name]

    # Pull artist/album from the actual columns in charted.csv
    track_artist = track_title["Primary Artist"].values[0]
    track_album = track_title["Album"].values[0]

    track_info = {
        "Artist": track_artist,
        "Album": track_album,
    }
    return track_info


def load_audio_track():
    # FIRST CREATE A LOOP TO LOAD THE TRACKS FROM THE LOAD_TRACK DICTIONARY
    for track_name, track_path in LOAD_TRACK.items():
        # Librosa resamples everything to 22050 Hz and also sets the audio as mono unless you tell it not to
        # Here sr=None preserves the native sampling rate while mono=True mixes down to mono
        samples, sample_rate = librosa.load(track_path, sr=None, mono=True)
        key, scale = detect_key_scale(samples, sample_rate)

        track_info = chart_finder(track_name)
        track_info['Key'] = key
        track_info['Scale'] = scale
        track_details[track_name] = track_info


if __name__ == "__main__":
    load_audio_track()
    print(track_details)
