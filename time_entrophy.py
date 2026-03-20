import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ==========================================
# 1. Core Configuration and Filter List
# ==========================================
CODEBOOK_SIZE = 1024
INPUT_FOLDER = "extracted_codes"

# Core Operation: Define categories to exclude (Artificial/Mathematically synthesized waves)
EXCLUDED_CATEGORIES = []
# EXCLUDED_CATEGORIES = ['Sine', 'Square', 'Sawtooth', 'HalfSine']
# Plotting style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

print("Starting latent space dynamics analysis for pure natural audio...")

# ==========================================
# 2. Core Algorithms
# ==========================================
def calculate_entropy(codes_1d):
    """
    Calculates Shannon Entropy for a 1D array of codes.
    Measures the spatial breadth or diversity of the code usage.
    """
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    probs = counts / counts.sum()
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_tccr(codes_1d):
    """
    Calculates Temporal Code Change Rate (TCCR).
    Measures the frequency of code changes over time (temporal volatility).
    """
    if len(codes_1d) <= 1:
        return 0
    changes = np.sum(codes_1d[1:] != codes_1d[:-1])
    return changes / (len(codes_1d) - 1)

# ==========================================
# 3. Data Loading and Precise Filtering
# ==========================================
data_records = []

for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.endswith('.npy'):
        continue

    parts = file_name.split('_')
    if len(parts) >= 2:
        clean_name = parts[1]
    else:
        continue

    # Filter Logic: Skip if the audio belongs to artificial/mathematical wave categories
    if clean_name in EXCLUDED_CATEGORIES:
        continue

    # Load and process Layer 1 data
    codes_np = np.squeeze(np.load(os.path.join(INPUT_FOLDER, file_name)))
    layer1_flat = codes_np[0].flatten()

    ent = calculate_entropy(layer1_flat)
    tccr = calculate_tccr(layer1_flat)

    data_records.append({
        'Category': clean_name,
        'Shannon Entropy (bits)': ent,
        'Temporal Code Change Rate (TCCR)': tccr
    })

df = pd.DataFrame(data_records)
unique_cats = df['Category'].nunique()
print(f"Data filtering complete! Retained {unique_cats} real-world sound categories, totaling {len(df)} samples.")

# ==========================================
# 4. Statistical Testing (Core)
# ==========================================
r_stat, p_value = pearsonr(df['Shannon Entropy (bits)'], df['Temporal Code Change Rate (TCCR)'])

print("\n" + "="*40)
print("Statistical Correlation Test Results (Natural Audio Subset)")
print("="*40)
print(f"   - Pearson Correlation Coefficient (r): {r_stat:.4f}")
print(f"   - Confidence Level (P-value): {p_value:.2e}")
if p_value < 0.05:
    print("   Conclusion: Statistically significant (Reject Null Hypothesis)")
else:
    print("   Conclusion: Not statistically significant (Fail to Reject Null Hypothesis)")
print("="*40 + "\n")

# ==========================================
# 5. Plotting Regression for Pure Natural Audio
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7))

# Dynamically generate a color palette suitable for the number of retained categories
colors = sns.color_palette("tab10", unique_cats)

# Plot regression line with 95% confidence interval
sns.regplot(
    data=df,
    x='Shannon Entropy (bits)',
    y='Temporal Code Change Rate (TCCR)',
    scatter=False,
    color='gray',
    line_kws={"linestyle": "--", "alpha": 0.7, "linewidth": 2.5}
)

# Plot categorized scatter points
sns.scatterplot(
    data=df,
    x='Shannon Entropy (bits)',
    y='Temporal Code Change Rate (TCCR)',
    hue='Category',
    palette=colors,
    s=150,
    edgecolor='black',
    alpha=0.85,
    ax=ax
)

title_text = f'Real-World Audio Latent Dynamics\nPearson $r$ = {r_stat:.2f} (p = {p_value:.2e})'
ax.set_title(title_text, fontweight='bold', pad=15)
ax.set_xlabel('Spatial Breadth: Shannon Entropy (bits)', fontweight='bold')
ax.set_ylabel('Temporal Volatility: TCCR', fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Natural Audio Classes")
plt.tight_layout()
fig.savefig("Natural_Audio_TCCR_vs_Entropy.png", dpi=300)
print("Analysis chart saved as: Natural_Audio_TCCR_vs_Entropy.png")