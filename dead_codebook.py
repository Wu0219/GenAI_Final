import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Core Configuration and Filter Lists
# ==========================================
CODEBOOK_SIZE = 1024
INPUT_FOLDER = "extracted_codes"

# Exclude all synthetic and mathematical baselines; retain only real-world natural sounds
EXCLUDED_CATEGORIES = []

# Academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

print("Starting global codebook utilization scan for [Real Natural Audio]...")

# ==========================================
# 2. Global Data Accumulator
# ==========================================
# Initialize an array of length 1024 with zeros to accumulate frequencies
global_code_counts = np.zeros(CODEBOOK_SIZE, dtype=int)
processed_files = 0

# ==========================================
# 3. Iteration and Frequency Accumulation
# ==========================================
for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.endswith('.npy'):
        continue

    parts = file_name.split('_')
    if len(parts) >= 2:
        clean_name = parts[1]
    else:
        continue

    # Filter out non-real audio categories
    if clean_name in EXCLUDED_CATEGORIES:
        continue

    # Load extracted features
    codes_np = np.squeeze(np.load(os.path.join(INPUT_FOLDER, file_name)))
    layer1_flat = codes_np[0].flatten()

    # Count occurrences of each code (0-1023) in the current file and add to the global array
    counts = np.bincount(layer1_flat, minlength=CODEBOOK_SIZE)
    global_code_counts += counts
    processed_files += 1

print(f"Successfully scanned {processed_files} real audio files!")

# ==========================================
# 4. Detect "Dead Codes" and Calculate Utilization
# ==========================================
total_tokens_used = np.sum(global_code_counts)

# Find indices with zero frequency (codes that were never used)
dead_codes_indices = np.where(global_code_counts == 0)[0]
dead_codes_count = len(dead_codes_indices)
active_codes_count = CODEBOOK_SIZE - dead_codes_count
utilization_rate = (active_codes_count / CODEBOOK_SIZE) * 100

print("\n" + "="*50)
print("Codebook Collapse Report")
print("="*50)
print(f"Total Extracted Tokens: {total_tokens_used:,}")
print(f"Theoretical Dictionary Capacity: {CODEBOOK_SIZE}")
print(f"Active Tokens Count: {active_codes_count}")
print(f"Unused Dead Codes Count: {dead_codes_count}")
print(f"Global Dictionary Utilization Rate: {utilization_rate:.2f}%")

if dead_codes_count > 0:
    print(f"\nWarning! The following {dead_codes_count} tokens were never invoked:")
    # Format dead code list: print 10 per line to avoid excessive output
    formatted_dead_codes = [str(code).zfill(4) for code in dead_codes_indices]
    for i in range(0, len(formatted_dead_codes), 10):
        print("   " + ", ".join(formatted_dead_codes[i:i+10]))
else:
    print("\nPerfect! All 1024 dictionary tokens were invoked at least once; no absolute dead codes found!")
print("="*50 + "\n")


# ==========================================
# 5. Plot Panorama Frequency and Long-Tail Effect (Zipf's Law)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- Plot 1: Absolute distribution sorted by native index (0-1023) ---
# Use an area plot to visualize data density
ax1.fill_between(range(CODEBOOK_SIZE), global_code_counts, color='teal', alpha=0.7)
ax1.plot(range(CODEBOOK_SIZE), global_code_counts, color='darkcyan', linewidth=1)
ax1.set_title('Figure A: Global Codebook Frequency Distribution (Layer 1)', fontweight='bold', pad=10)
ax1.set_xlabel('Codebook Index (0 to 1023)')
ax1.set_ylabel('Total Frequency')
ax1.set_xlim(0, 1023)
# Mark the zero line to highlight dead code regions
ax1.axhline(0, color='red', linewidth=1.5, linestyle='--')

# --- Plot 2: Long-tail distribution sorted by frequency (Rank-Frequency Plot) ---
sorted_counts = np.sort(global_code_counts)[::-1]  # Sort in descending order
ax2.fill_between(range(CODEBOOK_SIZE), sorted_counts, color='coral', alpha=0.7)
ax2.plot(range(CODEBOOK_SIZE), sorted_counts, color='orangered', linewidth=1.5)
ax2.set_title("Figure B: Rank-Frequency Distribution", fontweight='bold', pad=10)
ax2.set_xlabel('Token Rank (Most Frequent to Least Frequent)')
ax2.set_ylabel('Total Frequency')
ax2.set_xlim(0, 1023)

# Dynamically annotate the starting position of dead codes on Plot 2
if dead_codes_count > 0:
    ax2.axvline(active_codes_count, color='black', linestyle='--', linewidth=2, label=f'Dead Codes Onset ({dead_codes_count} unused)')
    ax2.fill_betweenx([0, max(sorted_counts)], active_codes_count, CODEBOOK_SIZE, color='gray', alpha=0.3)
    ax2.legend(loc='upper right')

plt.tight_layout(pad=3.0)
fig.savefig("Global_Codebook_Utilization_and_Dead_Codes_ALL.png", dpi=300)
print("Panorama utilization and long-tail chart saved as: Global_Codebook_Utilization_and_Dead_Codes.png")