import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ==========================================
# 1. Core Configuration
# ==========================================
CODEBOOK_SIZE = 1024
INPUT_FOLDER = "extracted_codes"

# Exclude artificially synthesized waveforms to focus on real-world natural audio analysis.
# Set this list to empty if you wish to include all categories.
EXCLUDED_CATEGORIES = ['Sine', 'Square', 'Sawtooth', 'HalfSine']

# Configure academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

print("Starting latent space dynamics analysis (Entropy vs. Robust Gini)...")


# ==========================================
# 2. Core Algorithms
# ==========================================
def calculate_entropy(codes_1d):
    """
    Calculate Shannon Entropy.
    Measures the breadth of information distribution.
    """
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    probs = counts / counts.sum()
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def calculate_tccr(codes_1d):
    """
    Calculate Temporal Code Change Rate (TCCR).
    Measures the volatility or rate of change in the code sequence over time.
    """
    if len(codes_1d) <= 1:
        return 0
    changes = np.sum(codes_1d[1:] != codes_1d[:-1])
    return changes / (len(codes_1d) - 1)


def calculate_gini_robust(codes_1d):
    """
    Calculate Robust Gini Coefficient.
    Measures inequality in resource (code) allocation.

    Note: Calculation is performed only on active codes (frequency > 0)
    to prevent statistical invalidity caused by extreme sparsity.
    """
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    active_counts = counts[counts > 0]

    if len(active_counts) <= 1:
        return 0

    active_counts = np.sort(active_counts)
    n = len(active_counts)
    index = np.arange(1, n + 1)
    # Standard Gini coefficient formula applied to sorted active counts
    return (np.sum((2 * index - n - 1) * active_counts)) / (n * np.sum(active_counts))


# ==========================================
# 3. Data Loading & Processing
# ==========================================
data_records = []

# Verify that the input folder exists
if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Folder '{INPUT_FOLDER}' not found. Please check your path.")
else:
    for file_name in os.listdir(INPUT_FOLDER):
        if not file_name.endswith('.npy'):
            continue

        # Parse category name from filename (assumes format like 'prefix_CategoryName_suffix.npy')
        parts = file_name.split('_')
        if len(parts) >= 2:
            clean_name = parts[1]
        else:
            continue

        # Filter out excluded artificial waveform categories
        if clean_name in EXCLUDED_CATEGORIES:
            continue

        # Load and flatten Layer 1 data
        codes_np = np.squeeze(np.load(os.path.join(INPUT_FOLDER, file_name)))
        layer1_flat = codes_np[0].flatten()

        # Calculate core metrics
        ent = calculate_entropy(layer1_flat)
        tccr = calculate_tccr(layer1_flat)
        gini = calculate_gini_robust(layer1_flat)

        data_records.append({
            'Category': clean_name,
            'Shannon Entropy (bits)': ent,
            'Temporal Code Change Rate (TCCR)': tccr,
            'Robust Gini (Inequality)': gini
        })

df = pd.DataFrame(data_records)
unique_cats = df['Category'].nunique()
print(f"Data processing complete! Retained {unique_cats} real-world sound categories, totaling {len(df)} samples.")

# ==========================================
# 4. Statistical Testing
# ==========================================
# Calculate Pearson correlation between Entropy and Gini Coefficient
r_stat_gini, p_value_gini = pearsonr(df['Shannon Entropy (bits)'], df['Robust Gini (Inequality)'])

print("\n" + "=" * 50)
print("Statistical Correlation: Shannon Entropy vs. Robust Gini (Sample Level)")
print("=" * 50)
print(f"   - Pearson Correlation Coefficient (r): {r_stat_gini:.4f}")
print(f"   - Confidence Level (P-value): {p_value_gini:.2e}")

if p_value_gini < 0.05:
    print("   Conclusion: Statistically significant (Reject Null Hypothesis)")
else:
    print("   Conclusion: Not statistically significant (Fail to Reject Null Hypothesis)")
print("=" * 50 + "\n")

# ==========================================
# 5. Plotting Regression Analysis
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7))

# Generate color palette dynamically based on number of unique categories
colors = sns.color_palette("tab10", unique_cats)

# 1. Plot linear regression fit line with 95% confidence interval
sns.regplot(
    data=df,
    x='Shannon Entropy (bits)',
    y='Robust Gini (Inequality)',
    scatter=False,
    color='gray',
    line_kws={"linestyle": "--", "alpha": 0.7, "linewidth": 2.5}
)

# 2. Plot scatter points colored by category
sns.scatterplot(
    data=df,
    x='Shannon Entropy (bits)',
    y='Robust Gini (Inequality)',
    hue='Category',
    palette=colors,
    s=150,
    edgecolor='black',
    alpha=0.85,
    ax=ax
)

# Set chart title and axis labels
title_text = f'Vocabulary Inequality vs. Information Breadth\nPearson $r$ = {r_stat_gini:.2f} (p = {p_value_gini:.2e})'
ax.set_title(title_text, fontweight='bold', pad=15)
ax.set_xlabel('Spatial Breadth: Shannon Entropy (bits)', fontweight='bold')
ax.set_ylabel('Resource Inequality: Robust Gini Coefficient', fontweight='bold')

# Adjust legend position to avoid obscuring data points
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Natural Audio Classes", frameon=True)

# Adjust layout and save figure
plt.tight_layout()
output_filename = "Natural_Audio_Gini_vs_Entropy.png"
fig.savefig(output_filename, dpi=300)
print(f"Analysis chart successfully saved as: {output_filename}")