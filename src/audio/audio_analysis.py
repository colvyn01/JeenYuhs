import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_DEGREES = [0, 2, 3, 5, 7, 8, 10]

TONIC_WEIGHT = 1.0
DOMINANT_WEIGHT = 0.9
SUBDOMINANT_WEIGHT = 0.7
NON_DIATONIC_WEIGHT = 0.3

LOAD_TRACK = {
    'Stronger': os.path.join(dir, "..", "..", "data", "audio", "Stronger.mp3"),
}

TRACK_DETAILS = {
    'Stronger': {
        'title': 'Stronger',
        'artist': 'Kanye West',
        'album': 'Graduation',
        'release_year': 2007,
        'key': '',
        'bpm': '',
        'time_signature': '',
    },
}

# Librosa resamples everything to 22050 Hz and also sets the audio as mono unless you tell it not to
samples, sample_rate = librosa.load(LOAD_TRACK['Stronger'], sr=None, mono=False)

# only trims the audio doesn't change sample rate or mono/stereo
samples_trimmed, trimmed_range = librosa.effects.trim(samples)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(samples_trimmed, sr=sample_rate)
plt.title("Stronger by Kanye West")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.show()

def DETECT_KEY(samples_trimmed, sample_rate):
    return 1

if __name__ == "__main__":
    print('')