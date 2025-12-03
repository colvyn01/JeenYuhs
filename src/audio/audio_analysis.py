import os
import librosa
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join('..', '..', 'outputs')

"""
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
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    active_notes = []
    for note_index in range(12):
        note_strength = chroma_frame[note_index]
        if note_strength > threshold:
            active_notes.append({
                'name': note_names[note_index],
                'strength': note_strength
            })

    return active_notes


def clean_audio_for_chroma(audio_time_series, sample_rate):
    """
    Cleaning up the audio for pitch analysis.
    Reducing rumble and separating harmonics to get rid of drums.
    """
    y_pre = librosa.effects.preemphasis(audio_time_series, coef=0.97)
    y_harm, y_perc = librosa.effects.hpss(y_pre)
    return y_harm


def compute_enhanced_chroma(y_harm, sample_rate, hop_length=512):
    """
    Making a smoother chroma for key detection.
    Using HPSS and median filtering like in the librosa examples.
    """
    # Getting the Constant-Q chroma from the harmonic part.
    chroma = librosa.feature.chroma_cqt(
        y=y_harm,
        sr=sample_rate,
        hop_length=hop_length
    )

    # Smoothing it out with a nearest-neighbor filter.
    chroma_filt = np.minimum(
        chroma,
        librosa.decompose.nn_filter(
            chroma,
            aggregate=np.median,
            metric='cosine'
        )
    )

    # Normalizing each frame.
    chroma_norm = librosa.util.normalize(chroma_filt, axis=0)

    return chroma_norm


def get_relative_key(key_note, key_mode):
    """
    Figuring out the relative major or minor key.
    Returns tuple: (relative_note, relative_mode)
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    root_index = note_names.index(key_note)
    
    if key_mode == 'major':
        # Relative minor is 3 semitones down.
        relative_index = (root_index - 3) % 12
        relative_note = note_names[relative_index]
        relative_mode = 'minor'
    else:  # minor
        # Relative major is 3 semitones up.
        relative_index = (root_index + 3) % 12
        relative_note = note_names[relative_index]
        relative_mode = 'major'
    
    return relative_note, relative_mode


def estimate_key_with_candidates(chroma_pitch_matrix):
    """
    Using Krumhansl-Schmuckler to guess the key.
    Ignoring quiet parts and using cosine similarity.
    """
    # Getting rid of the quiet parts like fades or silence.
    frame_energy = chroma_pitch_matrix.sum(axis=0)
    mask = frame_energy > 0.1 * frame_energy.max()
    if np.any(mask):
        average_chroma = np.mean(chroma_pitch_matrix[:, mask], axis=1)
    else:
        average_chroma = np.mean(chroma_pitch_matrix, axis=1)

    # Normalizing the pitch vector.
    average_chroma = (average_chroma - np.mean(average_chroma)) / (
        np.std(average_chroma) + 1e-8
    )

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Standard profiles for major and minor keys.
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Normalizing the profiles.
    major_profile = (major_profile - major_profile.mean()) / (major_profile.std() + 1e-8)
    minor_profile = (minor_profile - minor_profile.mean()) / (minor_profile.std() + 1e-8)

    candidates = []

    for root_index in range(12):
        rotated_major = np.roll(major_profile, root_index)
        rotated_minor = np.roll(minor_profile, root_index)

        maj_score = np.dot(average_chroma, rotated_major) / (
            np.linalg.norm(average_chroma) * np.linalg.norm(rotated_major) + 1e-8
        )
        min_score = np.dot(average_chroma, rotated_minor) / (
            np.linalg.norm(average_chroma) * np.linalg.norm(rotated_minor) + 1e-8
        )

        candidates.append({
            'key': note_names[root_index] + ' major',
            'score': maj_score
        })
        candidates.append({
            'key': note_names[root_index] + ' minor',
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
    enharmonic_pairs = {
        'C#': 'Db', 'Db': 'C#',
        'D#': 'Eb', 'Eb': 'D#',
        'F#': 'Gb', 'Gb': 'F#',
        'G#': 'Ab', 'Ab': 'G#',
        'A#': 'Bb', 'Bb': 'A#'
    }
    if best_key in enharmonic_pairs:
        print(f"Enharmonic equivalent: {enharmonic_pairs[best_key]} {best_mode}")

    return best_key, best_mode, average_chroma


def get_diatonic_chords(key_note, key_mode):
    """Getting all the chords that fit in this key."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Finding where the root note is.
    root_index = note_names.index(key_note)

    if key_mode == 'major':
        # Major key chords.
        chord_types = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
    else:  # minor
        # Minor key chords.
        chord_types = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']

    diatonic_chords = []
    for i, chord_type in enumerate(chord_types):
        chord_root_index = (root_index + i) % 12
        chord_root = note_names[chord_root_index]
        diatonic_chords.append(f"{chord_root}{chord_type}")

    return diatonic_chords


def detect_chord(chroma_frame, diatonic_chords):
    """
    Matching the chroma to the best chord in the key using cosine similarity.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

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

        root_index = note_names.index(root_note)

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


def explore_chroma(audio_time_series, sample_rate):
    # First, I'm cleaning the audio to get rid of drums and rumble.
    clean_audio = clean_audio_for_chroma(audio_time_series, sample_rate)

    # Getting the enhanced chroma over time.
    chroma_pitch_matrix = compute_enhanced_chroma(clean_audio, sample_rate)

    num_pitch_classes = chroma_pitch_matrix.shape[0]
    num_time_frames = chroma_pitch_matrix.shape[1]

    print(f"Chroma shape: {chroma_pitch_matrix.shape}")

    # Checking the first frame.
    first_frame_pitches = chroma_pitch_matrix[:, 0]
    print(f"\nRaw numbers from first frame (high-pass filtered):")
    print(first_frame_pitches)

    # Translating that to actual notes.
    print(f"\nActive notes (strength > 0.5):")
    active_notes = identify_active_notes(first_frame_pitches, threshold=0.5)
    for note in active_notes:
        print(f"  {note['name']}: {note['strength']:.3f}")

    # Checking a bit later in the song.
    frame_to_check = min(500, num_time_frames - 1)
    later_frame_pitches = chroma_pitch_matrix[:, frame_to_check]

    print(f"\nActive notes at frame {frame_to_check} (a few seconds into the song):")
    active_notes_later = identify_active_notes(later_frame_pitches, threshold=0.6)
    for note in active_notes_later:
        print(f"  {note['name']}: {note['strength']:.3f}")

    # Figuring out the overall key.
    key_center, key_mode, avg_chroma = estimate_key_with_candidates(chroma_pitch_matrix)

    # Getting the chords that fit.
    diatonic_chords = get_diatonic_chords(key_center, key_mode)
    print(f"\nDiatonic chords in {key_center} {key_mode}:")
    for i, chord in enumerate(diatonic_chords, 1):
        print(f"  {i}. {chord}")

    # Checking the chord progression for the first 20 frames.
    print(f"\nFirst 20 frames chord progression:")
    for frame in range(min(20, chroma_pitch_matrix.shape[1])):
        chord, score = detect_chord(chroma_pitch_matrix[:, frame], diatonic_chords)
        print(f"  Frame {frame:2d}: {chord} (conf: {score:.3f})")

    return key_center, key_mode, avg_chroma, diatonic_chords


def analyze_and_store(track_name, file_path, results_list):
    """Extracting data for the CSV file."""
    try:
        audio_time_series, sample_rate = load_audio(file_path)
        tempo_bpm, beat_frame_indices, onset_env = extract_tempo(audio_time_series, sample_rate)
        time_signature = estimate_time_signature(audio_time_series, sample_rate, beat_frame_indices, onset_env)
        
        # Doing the chroma processing.
        clean_audio = clean_audio_for_chroma(audio_time_series, sample_rate)
        chroma_pitch_matrix = compute_enhanced_chroma(clean_audio, sample_rate)
        
        # Running the key detection logic.
        frame_energy = chroma_pitch_matrix.sum(axis=0)
        mask = frame_energy > 0.1 * frame_energy.max()
        average_chroma = np.mean(chroma_pitch_matrix[:, mask], axis=1) if np.any(mask) else np.mean(chroma_pitch_matrix, axis=1)
        average_chroma = (average_chroma - np.mean(average_chroma)) / (np.std(average_chroma) + 1e-8)

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile = (major_profile - major_profile.mean()) / (major_profile.std() + 1e-8)
        minor_profile = (minor_profile - minor_profile.mean()) / (minor_profile.std() + 1e-8)

        key_candidates = []
        for root_index in range(12):
            rotated_major = np.roll(major_profile, root_index)
            rotated_minor = np.roll(minor_profile, root_index)
            maj_score = np.dot(average_chroma, rotated_major) / (np.linalg.norm(average_chroma) * np.linalg.norm(rotated_major) + 1e-8)
            min_score = np.dot(average_chroma, rotated_minor) / (np.linalg.norm(average_chroma) * np.linalg.norm(rotated_minor) + 1e-8)
            key_candidates.append((note_names[root_index] + ' major', maj_score))
            key_candidates.append((note_names[root_index] + ' minor', min_score))
        
        key_candidates.sort(key=lambda x: x[1], reverse=True)
        best_key_full, best_score = key_candidates[0]
        best_key, best_mode = best_key_full.split()
        relative_note, relative_mode = get_relative_key(best_key, best_mode)
        
        enharmonic_pairs = {'C#': 'Db', 'Db': 'C#', 'D#': 'Eb', 'Eb': 'D#', 
                           'F#': 'Gb', 'Gb': 'F#', 'G#': 'Ab', 'Ab': 'G#', 
                           'A#': 'Bb', 'Bb': 'A#'}
        enharmonic_equiv = enharmonic_pairs.get(best_key, None)

        # Saving the data in a format that works for CSV.
        results_list.append({
            'Track': track_name.replace('_', ' ').title(),
            'Tempo_BPM': round(tempo_bpm, 2),
            'Time_Signature': f'{time_signature}/4',
            'Detected_Key': f'{best_key} {best_mode}',
            'Key_Score': round(best_score, 3),
            'Relative_Key': f'{relative_note} {relative_mode}',
            'Enharmonic': f'{enharmonic_equiv} {best_mode}' if enharmonic_equiv else '',
            'Top1_Key': f"{key_candidates[0][0]} ({key_candidates[0][1]:.3f})",
            'Top2_Key': f"{key_candidates[1][0]} ({key_candidates[1][1]:.3f})", 
            'Top3_Key': f"{key_candidates[2][0]} ({key_candidates[2][1]:.3f})",
            'Chroma_Frames': chroma_pitch_matrix.shape[1]
        })
        
    except Exception as e:
        print(f"Export error for {track_name}: {e}")

def analyze_track():
    print("Analyzing track...")
    
    tracks = {
        'all_falls_down': os.path.join('..', '..', 'data', 'audio', 'All Falls Down.mp3'),
        'follow_god': os.path.join('..', '..', 'data', 'audio', 'Follow God.mp3'),
        'gold_digger': os.path.join('..', '..', 'data', 'audio', 'Gold Digger.mp3'),
        'heartless': os.path.join('..', '..', 'data', 'audio', 'Heartless.mp3'),
        'love_lockdown': os.path.join('..', '..', 'data', 'audio', 'Love Lockdown.mp3'),
        'stronger': os.path.join('..', '..', 'data', 'audio', 'Stronger.mp3')
    }

    results = []
    
    for track_name, file_path in tracks.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {track_name}...")
        print(f"{'=' * 60}")

        try:
            audio_time_series, sample_rate = load_audio(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        # Running the analysis and printing the results.
        tempo_bpm, beat_frame_indices, onset_env = extract_tempo(audio_time_series, sample_rate)
        print(f"Estimated tempo: {tempo_bpm:.2f} BPM")
        time_signature = estimate_time_signature(audio_time_series, sample_rate, beat_frame_indices, onset_env)
        print(f"Estimated time signature: {time_signature}/4")
        explore_chroma(audio_time_series, sample_rate)
        
        # Exporting to CSV silently.
        analyze_and_store(track_name, file_path, results)

    # Saving the CSV file.
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, 'kanye_track_analysis.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to: {csv_path}")

if __name__ == "__main__":
    analyze_track()