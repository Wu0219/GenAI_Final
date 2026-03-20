import os
import torch
import librosa
import numpy as np
from transformers import EncodecModel, AutoProcessor

# 1. Directory Configuration
input_folder = "dataset"
output_folder = "extracted_codes"
os.makedirs(output_folder, exist_ok=True)

print("Initializing Encodec model (facebook/encodec_24khz)...")
# Automatically download and load the pre-trained Encodec model and processor
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
model = EncodecModel.from_pretrained("facebook/encodec_24khz")

# Move the model to GPU if available for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device.upper()}")

# Retrieve all .ogg and .wav files from the dataset folder
audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.ogg'))]
audio_files.sort()  # Sort files to ensure consistent processing order

print(f"\nStarting processing of {len(audio_files)} audio files to extract discrete codes...")

for i, file_name in enumerate(audio_files):
    file_path = os.path.join(input_folder, file_name)

    # Core Detail 1: Format Normalization
    # librosa automatically handles format differences (ogg/wav) and resamples to 24000Hz mono.
    # This ensures consistent data distribution for the model input.
    audio_array, sr = librosa.load(file_path, sr=24000, mono=True)

    # Convert the audio array to the tensor format required by the model using the processor
    inputs = processor(raw_audio=audio_array, sampling_rate=sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    # Core Detail 2: Forward Inference to Extract Codebook
    with torch.no_grad():  # Disable gradient calculation for inference to save memory
        # Extract discrete codes (audio_codes)
        # Output shape is typically: [batch_size, num_quantizers, frames]
        encoder_outputs = model.encode(input_values, bandwidth=6.0)
        audio_codes = encoder_outputs.audio_codes[0]

    # Move tensor from GPU to CPU and convert to numpy array
    codes_np = audio_codes.cpu().numpy()

    # Core Detail 3: Persistent Storage
    # Remove the original extension (.ogg/.wav) and save as .npy
    base_name = os.path.splitext(file_name)[0]
    save_path = os.path.join(output_folder, f"{base_name}.npy")
    np.save(save_path, codes_np)

    # Print progress every 10 files or for the first file
    if (i + 1) % 10 == 0 or i == 0:
        print(f"Progress: [{i + 1}/{len(audio_files)}] Saved -> {base_name}.npy | Shape: {codes_np.shape}")

print("\nAudio feature extraction completed successfully! Feature matrices are saved in the 'extracted_codes' directory.")