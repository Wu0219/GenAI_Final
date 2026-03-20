import os
import numpy as np
import librosa
import soundfile as sf

# Configure paths
input_folder = "Symphony"
output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)

segment_duration = 10.0  # Set duration for each segment to 10 seconds
target_sr = 24000        # Target sample rate

print("Starting full audio processing and segmentation...")

# Get list of audio files
symphony_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.mp3', '.flac'))]

if not symphony_files:
    print("Alert: No audio files found in the 'Symphony' folder!")
else:
    for i, file in enumerate(symphony_files):
        file_path = os.path.join(input_folder, file)

        # Get total audio duration without loading the full file to save memory
        total_duration = librosa.get_duration(path=file_path)
        print(f"Processing: {file} | Total duration: {total_duration:.2f}s")

        # Calculate the number of complete 10s segments possible
        # Use np.ceil if you want to include the final segment even if it is shorter than 10s
        num_segments = int(total_duration // segment_duration)

        for seg_idx in range(num_segments):
            start_time = seg_idx * segment_duration

            # Load specific segment
            y, sr = librosa.load(
                file_path,
                sr=target_sr,
                offset=start_time,
                duration=segment_duration
            )

            # Generate output filename including original file index and segment index
            # Format: 7_ComplexMusic_fileIndex_segSegmentIndex.wav
            out_name = os.path.join(output_folder, f"7_ComplexMusic_{i + 1}_seg{seg_idx + 1}.wav")

            sf.write(out_name, y, sr)

        print(f"Completed! Extracted {num_segments} segments.")

print("--- All files processed ---")

