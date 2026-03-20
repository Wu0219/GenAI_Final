import os
import numpy as np
import soundfile as sf
from scipy import signal

# Define the output directory for generated audio files
output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

print("Starting generation of mathematical waveform control groups...")

# Core parameters
sr = 24000                  # Sample rate (Hz)
duration = 10.0             # Duration of each file in seconds
num_samples = int(sr * duration)
t = np.linspace(0, duration, num_samples, endpoint=False)
num_files_per_type = 15     # Number of files to generate for each waveform type

# Set a random seed to ensure reproducible frequency generation
np.random.seed(42)

# Define global amplitude to prevent clipping (0.5 represents 50% volume)
amplitude = 0.5

for i in range(num_files_per_type):
    # Generate a random fundamental frequency between 220Hz and 880Hz (spanning two octaves)
    f = np.random.uniform(220.0, 880.0)

    # ------------------------------------------------
    # 1. Pure Sine Wave - Contains only the fundamental frequency, no harmonics
    # ------------------------------------------------
    sine_wave = amplitude * np.sin(2 * np.pi * f * t)
    sf.write(os.path.join(output_folder, f"10_Sine_{i + 1:02d}.wav"), sine_wave, sr)

    # ------------------------------------------------
    # 2. Square Wave - Contains fundamental and odd harmonics (typical 8-bit game sound)
    # ------------------------------------------------
    square_wave = amplitude * signal.square(2 * np.pi * f * t)
    sf.write(os.path.join(output_folder, f"11_Square_{i + 1:02d}.wav"), square_wave, sr)

    # ------------------------------------------------
    # 3. Sawtooth Wave - Contains fundamental and all integer harmonics (bright and harsh)
    # ------------------------------------------------
    sawtooth_wave = amplitude * signal.sawtooth(2 * np.pi * f * t)
    sf.write(os.path.join(output_folder, f"12_Sawtooth_{i + 1:02d}.wav"), sawtooth_wave, sr)

    # ------------------------------------------------
    # 4. Half-wave Rectified Sine - Introduces nonlinear distortion
    # Formula: f(t) = max(0, sin(ωt))
    # ------------------------------------------------
    half_sine = amplitude * np.maximum(0, np.sin(2 * np.pi * f * t))
    sf.write(os.path.join(output_folder, f"13_HalfSine_{i + 1:02d}.wav"), half_sine, sr)

    # Print progress every 5 iterations
    if (i + 1) % 5 == 0:
        print(f"Generated {i + 1}/{num_files_per_type} sets of extreme waveforms...")

print("\nAll 60 extreme mathematical waveform samples have been successfully added to the dataset!")