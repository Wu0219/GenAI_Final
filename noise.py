import os
import numpy as np
import soundfile as sf

# Set the output directory for the dataset
output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

print("Starting batch generation of high-purity mathematical noise datasets...")

# Core parameters matching Encodec standards
sr = 24000          # Sample rate in Hz
duration = 10.0     # Duration of each audio file in seconds
num_samples = int(sr * duration)
num_files_per_type = 15

print(f"Configuration: Generating {num_files_per_type} samples per type, each {duration} seconds long at {sr}Hz.")

for i in range(num_files_per_type):
    # --- 1. White Gaussian Noise ---
    # Generate new random numbers from a normal distribution for each iteration
    white_noise = np.random.normal(0, 0.5, num_samples)
    name_white = os.path.join(output_folder, f"8_Noise_WhiteGaussian_{i + 1:02d}.wav")
    sf.write(name_white, white_noise, sr)

    # --- 2. Uniform White Noise ---
    # Generate random numbers from a uniform distribution
    uniform_noise = np.random.uniform(-0.5, 0.5, num_samples)
    name_uniform = os.path.join(output_folder, f"8_Noise_UniformWhite_{i + 1:02d}.wav")
    sf.write(name_uniform, uniform_noise, sr)

    # --- 3. Brownian Noise (Red Noise) ---
    # Generate by cumulatively summing Gaussian noise to simulate Brownian motion
    brown_noise = np.cumsum(np.random.normal(0, 0.05, num_samples))
    # Normalize strictly to prevent clipping (audio distortion)
    brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.5
    name_brown = os.path.join(output_folder, f"8_Noise_Brownian_{i + 1:02d}.wav")
    sf.write(name_brown, brown_noise, sr)

    # Print progress every 5 files or on the first iteration
    if (i + 1) % 5 == 0 or i == 0:
        print(f"  Generated {i + 1}/{num_files_per_type} sets...")

print("\nSuccessfully generated 45 high-quality pure noise samples!")