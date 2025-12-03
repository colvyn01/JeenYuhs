"""
Audio analysis module for extracting musical features from tracks.

Chroma Position to Note Mapping:
Position 0  = C
Position 1  = C#/Db
Position 2  = D
Position 3  = D#/Eb
Position 4  = E
Position 5  = F
Position 6  = F#/Gb
Position 7  = G
Position 8  = G#/Ab
Position 9  = A
Position 10 = A#/Bb
Position 11 = B
"""

import os
import librosa
import pandas as pd
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

ENHARMONIC_PAIRS = {
    'C#': 'Db', 'Db': 'C#',
    'D#': 'Eb', 'Eb': 'D#',
    'F#': 'Gb', 'Gb': 'F#',
    'G#': 'Ab', 'Ab': 'G#',
    'A#': 'Bb', 'Bb': 'A#'
}


def load_audio(file_path):
    # Just loading the file here, getting the raw audio and sample rate
    audio_time_series, sample_rate = librosa.load(file_path)
    return audio_time_series, sample_rate


def extract_tempo(audio_time_series, sample_rate):
    # Finding where the beats hit (drums, vocals, etc...)
    onset_strength_envelope = librosa.onset.onset_strength(
        y=audio_time_series,
        sr=sample_rate
    )

    # Using those hits to calculate the BPM
    tempo_bpm, beat_frame_indices = librosa.beat.beat_track(
        onset_envelope=onset_strength_envelope,
        sr=sample_rate
    )

    return tempo_bpm.item(), beat_frame_indices, onset_strength_envelope


def estimate_time_signature(audio_time_series, sample_rate, beat_frame_indices, onset_env):
    """
    Trying to figure out the time signature. Using tempogram ratio and beat interval autocorrelation 
    because simple templates don't work well for hip-hop.
    """
    # First way: Tempogram ratio. Helps tell if it's duple or triple meter.
    # Calculating the tempogram here.
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=512,
        win_length=384  # ~8 seconds for stable meter estimation
    )

    # Looking for the main tempo patterns.
    # 4/4 usually shows up at 0.5, 1.0, 2.0.
    # 3/4 hits around 0.33, 0.67, 1.0, 1.33.
    # 6/8 is usually 0.67, 1.0, 1.33.
    # This gives us the ratios we need.
    try:
        ratio_features = librosa.feature.tempogram_ratio(
            tempogram=tempogram,
            sr=sample_rate,
            hop_length=512
        )
        # Averaging it out.
        ratio_mean = np.mean(ratio_features, axis=1)
        # Index 0 is half time (0.5), index 1 is 0.67, and so on.
        # Checking if it feels more like duple or triple.
        duple_strength = np.mean(ratio_mean[:2])  # ratios 0.5, 0.67
        triple_strength = np.mean(ratio_mean[2:4])  # ratios 1.0, 1.33
    except Exception:
        # If that fails, just default to 0.
        duple_strength = 0.0
        triple_strength = 0.0

    # Second way: Beat interval autocorrelation.
    # Turning beat frames into time intervals.
    beat_times = librosa.frames_to_time(beat_frame_indices, sr=sample_rate)
    if len(beat_times) < 4:
        return 4  # default

    # Getting the time between beats.
    intervals = np.diff(beat_times)
    if len(intervals) < 3:
        return 4

    # Checking for repeating patterns in the intervals.
    corr = np.correlate(intervals, intervals, mode='full')
    corr = corr[len(corr)//2:]  # keep only positive lags
    corr[:10] = 0  # zero out near-zero lag (trivial correlation)

    # Looking for peaks at lags 1, 2, and 3.
    peaks = []
    for lag in [1, 2, 3, 4]:
        if lag < len(corr):
            # Finding the highest points around this lag
            window = corr[max(0, lag-2):min(len(corr), lag+3)]
            peaks.append(np.max(window))
        else:
            peaks.append(0.0)

    # Normalizing the peaks
    peaks = np.array(peaks)
    if peaks.max() > 0:
        peaks /= peaks.max()

    # Combining both methods to make a decision.
    # Since it's hip-hop, it's probably 4/4, so I'm biasing towards that.
    confidence_4_4 = peaks[1] + duple_strength  # lag-2 periodicity + duple ratio
    confidence_3_4 = peaks[2] + triple_strength  # lag-3 periodicity + triple ratio

    # If 3/4 looks really strong, I'll go with that.
    if confidence_3_4 > confidence_4_4 * 1.2:
        return 3
    # Checking for 6/8 (strong at lag 2, weak at 1 and 3).
    elif peaks[2] > 0.6 and peaks[1] < 0.3 and peaks[3] < 0.3:
        return 6
    else:
        return 4


def identify_active_notes(chroma_frame, threshold=0.5):
    # If it's above the threshold, it's probably playing.
    # Everything else is likely just background noise or harmonics.
    active_notes = []
    for note_index in range(12):
        note_strength = chroma_frame[note_index]
        if note_strength > threshold:
            active_notes.append({
                'name': NOTE_NAMES[note_index],
                'strength': note_strength
            })

    return active_notes


def clean_audio_for_chroma(audio, sample_rate):
    """
    Cleaning up the audio for pitch analysis.
    Reducing rumble and separating harmonics to get rid of drums.
    """
    preemphasized = librosa.effects.preemphasis(audio, coef=0.97)
    harmonic, percussive = librosa.effects.hpss(preemphasized)
    return harmonic


def compute_enhanced_chroma(harmonic_audio, sample_rate, hop_length=512):
    """
    Making a smoother chroma for key detection.
    Using HPSS and median filtering like in the librosa examples.
    """
    # Getting the Constant-Q chroma from the harmonic part.
    chroma = librosa.feature.chroma_cqt(
        y=harmonic_audio,
        sr=sample_rate,
        hop_length=hop_length
    )

    # Smoothing it out with a nearest-neighbor filter.
    chroma_filtered = np.minimum(
        chroma,
        librosa.decompose.nn_filter(
            chroma,
            aggregate=np.median,
            metric='cosine'
        )
    )

    # Normalizing each frame.
    chroma_normalized = librosa.util.normalize(chroma_filtered, axis=0)

    return chroma_normalized


def get_relative_key(key_note, key_mode):
    """
    Figuring out the relative major or minor key.
    Returns tuple: (relative_note, relative_mode)
    """
    root_index = NOTE_NAMES.index(key_note)
    
    if key_mode == 'major':
        # Relative minor is 3 semitones down.
        relative_index = (root_index - 3) % 12
        relative_note = NOTE_NAMES[relative_index]
        relative_mode = 'minor'
    else:  # minor
        # Relative major is 3 semitones up.
        relative_index = (root_index + 3) % 12
        relative_note = NOTE_NAMES[relative_index]
        relative_mode = 'major'
    
    return relative_note, relative_mode


def estimate_key_with_candidates(chroma_matrix):
    """
    Using Krumhansl-Schmuckler to guess the key.
    Ignoring quiet parts and using cosine similarity.
    """
    # Getting rid of the quiet parts like fades or silence.
    frame_energy = chroma_matrix.sum(axis=0)
    mask = frame_energy > 0.1 * frame_energy.max()
    if np.any(mask):
        avg_chroma = np.mean(chroma_matrix[:, mask], axis=1)
    else:
        avg_chroma = np.mean(chroma_matrix, axis=1)

    # Normalizing the pitch vector.
    avg_chroma = (avg_chroma - np.mean(avg_chroma)) / (np.std(avg_chroma) + 1e-8)

    # Normalizing the profiles.
    major_norm = (MAJOR_PROFILE - MAJOR_PROFILE.mean()) / (MAJOR_PROFILE.std() + 1e-8)
    minor_norm = (MINOR_PROFILE - MINOR_PROFILE.mean()) / (MINOR_PROFILE.std() + 1e-8)

    candidates = []

    for root_index in range(12):
        rotated_major = np.roll(major_norm, root_index)
        rotated_minor = np.roll(minor_norm, root_index)

        maj_score = np.dot(avg_chroma, rotated_major) / (
            np.linalg.norm(avg_chroma) * np.linalg.norm(rotated_major) + 1e-8
        )
        min_score = np.dot(avg_chroma, rotated_minor) / (
            np.linalg.norm(avg_chroma) * np.linalg.norm(rotated_minor) + 1e-8
        )

        candidates.append({
            'key': NOTE_NAMES[root_index] + ' major',
            'score': maj_score
        })
        candidates.append({
            'key': NOTE_NAMES[root_index] + ' minor',
            'score': min_score
        })

    # Sorting them to find the best match.
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nKey detection - Top 5 candidates:")
    for i in range(5):
        print(f"  {i+1}. {candidates[i]['key']}: {candidates[i]['score']:.3f}")

    best = candidates[0]
    best_key, best_mode = best['key'].split()
    
    # Getting the relative key.
    relative_note, relative_mode = get_relative_key(best_key, best_mode)

    print(f"\nSelected: {best_key} {best_mode}")
    print(f"Relative key: {relative_note} {relative_mode}")
    
    # Checking for enharmonic equivalents.
    if best_key in ENHARMONIC_PAIRS:
        print(f"Enharmonic equivalent: {ENHARMONIC_PAIRS[best_key]} {best_mode}")

    return best_key, best_mode, avg_chroma


def get_diatonic_chords(key_note, key_mode):
    """Getting all the chords that fit in this key."""
    # Finding where the root note is.
    root_index = NOTE_NAMES.index(key_note)

    if key_mode == 'major':
        # Major key chords.
        chord_types = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
    else:  # minor
        # Minor key chords.
        chord_types = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']

    diatonic_chords = []
    for i, chord_type in enumerate(chord_types):
        chord_root_index = (root_index + i) % 12
        chord_root = NOTE_NAMES[chord_root_index]
        diatonic_chords.append(f"{chord_root}{chord_type}")

    return diatonic_chords


def detect_chord(chroma_frame, diatonic_chords):
    """
    Matching the chroma to the best chord in the key using cosine similarity.
    """
    chroma_vec = chroma_frame[:12].astype(float)
    chroma_norm = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-8)

    best_chord = "N"
    best_score = -1.0

    for chord_name in diatonic_chords:
        if chord_name.endswith('maj') or chord_name.endswith('min') or chord_name.endswith('dim'):
            root_note = chord_name[:-3]
            quality = chord_name[-3:]
        else:
            continue

        root_index = NOTE_NAMES.index(root_note)

        if quality == 'maj':
            chord_notes = [0, 4, 7]
        elif quality == 'min':
            chord_notes = [0, 3, 7]
        elif quality == 'dim':
            chord_notes = [0, 3, 6]
        else:
            continue

        template = np.zeros(12, dtype=float)
        for off in chord_notes:
            template[(root_index + off) % 12] = 1.0

        template_norm = template / (np.linalg.norm(template) + 1e-8)

        score = float(np.dot(chroma_norm, template_norm))

        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord, best_score


def explore_chroma(audio, sample_rate):
    # First, I'm cleaning the audio to get rid of drums and rumble.
    harmonic_audio = clean_audio_for_chroma(audio, sample_rate)

    # Getting the enhanced chroma over time.
    chroma_matrix = compute_enhanced_chroma(harmonic_audio, sample_rate)

    num_frames = chroma_matrix.shape[1]

    print(f"Chroma shape: {chroma_matrix.shape}")

    # Checking the first frame.
    first_frame = chroma_matrix[:, 0]
    print(f"\nRaw numbers from first frame (high-pass filtered):")
    print(first_frame)

    # Translating that to actual notes.
    print(f"\nActive notes (strength > 0.5):")
    active_notes = identify_active_notes(first_frame, threshold=0.5)
    for note in active_notes:
        print(f"  {note['name']}: {note['strength']:.3f}")

    # Checking a bit later in the song.
    frame_to_check = min(500, num_frames - 1)
    later_frame = chroma_matrix[:, frame_to_check]

    print(f"\nActive notes at frame {frame_to_check} (a few seconds into the song):")
    active_notes_later = identify_active_notes(later_frame, threshold=0.6)
    for note in active_notes_later:
        print(f"  {note['name']}: {note['strength']:.3f}")

    # Figuring out the overall key.
    key_center, key_mode, avg_chroma = estimate_key_with_candidates(chroma_matrix)

    # Getting the chords that fit.
    diatonic_chords = get_diatonic_chords(key_center, key_mode)
    print(f"\nDiatonic chords in {key_center} {key_mode}:")
    for i, chord in enumerate(diatonic_chords, 1):
        print(f"  {i}. {chord}")

    # Checking the chord progression for the first 20 frames.
    print(f"\nFirst 20 frames chord progression:")
    for frame in range(min(20, chroma_matrix.shape[1])):
        chord, score = detect_chord(chroma_matrix[:, frame], diatonic_chords)
        print(f"  Frame {frame:2d}: {chord} (conf: {score:.3f})")

    return key_center, key_mode, avg_chroma, diatonic_chords


def detect_key_for_export(chroma_matrix):
    """
    Key detection helper for CSV export.
    Returns key info and top 3 candidates.
    """
    frame_energy = chroma_matrix.sum(axis=0)
    mask = frame_energy > 0.1 * frame_energy.max()
    avg_chroma = np.mean(chroma_matrix[:, mask], axis=1) if np.any(mask) else np.mean(chroma_matrix, axis=1)
    avg_chroma = (avg_chroma - np.mean(avg_chroma)) / (np.std(avg_chroma) + 1e-8)

    major_norm = (MAJOR_PROFILE - MAJOR_PROFILE.mean()) / (MAJOR_PROFILE.std() + 1e-8)
    minor_norm = (MINOR_PROFILE - MINOR_PROFILE.mean()) / (MINOR_PROFILE.std() + 1e-8)

    candidates = []
    for root_index in range(12):
        rotated_major = np.roll(major_norm, root_index)
        rotated_minor = np.roll(minor_norm, root_index)
        maj_score = np.dot(avg_chroma, rotated_major) / (np.linalg.norm(avg_chroma) * np.linalg.norm(rotated_major) + 1e-8)
        min_score = np.dot(avg_chroma, rotated_minor) / (np.linalg.norm(avg_chroma) * np.linalg.norm(rotated_minor) + 1e-8)
        candidates.append((NOTE_NAMES[root_index] + ' major', maj_score))
        candidates.append((NOTE_NAMES[root_index] + ' minor', min_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_key_full, best_score = candidates[0]
    best_key, best_mode = best_key_full.split()
    relative_note, relative_mode = get_relative_key(best_key, best_mode)
    enharmonic = ENHARMONIC_PAIRS.get(best_key, None)

    return {
        'key': best_key,
        'mode': best_mode,
        'score': best_score,
        'relative_note': relative_note,
        'relative_mode': relative_mode,
        'enharmonic': enharmonic,
        'candidates': candidates[:3]
    }


def analyze_single_track(track_name, file_path, results_list):
    """Analyze one track and append results to list."""
    try:
        audio, sample_rate = load_audio(file_path)
        tempo, beat_frames, onset_env = extract_tempo(audio, sample_rate)
        time_sig = estimate_time_signature(audio, sample_rate, beat_frames, onset_env)

        harmonic = clean_audio_for_chroma(audio, sample_rate)
        chroma_matrix = compute_enhanced_chroma(harmonic, sample_rate)

        key_info = detect_key_for_export(chroma_matrix)

        results_list.append({
            'Track': track_name.replace('_', ' ').title(),
            'Tempo_BPM': round(tempo, 2),
            'Time_Signature': f'{time_sig}/4',
            'Detected_Key': f"{key_info['key']} {key_info['mode']}",
            'Key_Score': round(key_info['score'], 3),
            'Relative_Key': f"{key_info['relative_note']} {key_info['relative_mode']}",
            'Enharmonic': f"{key_info['enharmonic']} {key_info['mode']}" if key_info['enharmonic'] else '',
            'Top1_Key': f"{key_info['candidates'][0][0]} ({key_info['candidates'][0][1]:.3f})",
            'Top2_Key': f"{key_info['candidates'][1][0]} ({key_info['candidates'][1][1]:.3f})",
            'Top3_Key': f"{key_info['candidates'][2][0]} ({key_info['candidates'][2][1]:.3f})",
            'Chroma_Frames': chroma_matrix.shape[1]
        })

    except Exception as e:
        print(f"Error analyzing {track_name}: {e}")


def analyze_audio_tracks(audio_dir, output_path):
    """
    Main entry point for audio analysis.
    Analyzes all mp3 files in audio_dir and saves results to output_path.
    """
    print("Starting audio analysis...")

    # Find all mp3 files in the audio directory
    mp3_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    if not mp3_files:
        print(f"No mp3 files found in {audio_dir}")
        return

    results = []

    for filename in mp3_files:
        file_path = os.path.join(audio_dir, filename)
        track_name = os.path.splitext(filename)[0].lower().replace(' ', '_')

        print(f"\n{'-' * 70}")
        print(f"Processing {filename}:")
        print(f"{'-' * 70}")

        try:
            audio, sample_rate = load_audio(file_path)
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            continue

        # Print detailed analysis
        tempo, beat_frames, onset_env = extract_tempo(audio, sample_rate)
        print(f"Estimated tempo: {tempo:.2f} BPM")
        time_sig = estimate_time_signature(audio, sample_rate, beat_frames, onset_env)
        print(f"Estimated time signature: {time_sig}/4")
        explore_chroma(audio, sample_rate)

        # Store for CSV
        analyze_single_track(track_name, file_path, results)

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nResults exported to: {output_path}")


if __name__ == "__main__":
    # Standalone mode - use paths relative to this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    audio_dir = os.path.join(base_dir, 'data', 'audio')
    output_path = os.path.join(base_dir, 'outputs', 'track_analysis.csv')

    analyze_audio_tracks(audio_dir, output_path)