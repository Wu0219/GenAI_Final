import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ==========================================
# 0. Configuration constants
# ==========================================
CLASS_NUM = 13
NUM_QUANTIZERS = 8
CODEBOOK_SIZE = 1024
INPUT_FOLDER = "extracted_codes"

# ==========================================
# 1. High-Contrast Academic Visualization Configuration
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = sns.color_palette("tab20", CLASS_NUM)

print("🚀 Starting to load panoramic feature matrices and perform multi-dimensional statistical analysis...")


# ==========================================
# 2. Core Algorithm Toolkit (All Corrected Metrics)
# ==========================================

def calculate_entropy(codes_1d):
    """Calculate Shannon Entropy of the code distribution."""
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def calculate_active_codes(codes_1d):
    """Count the number of unique codes used in the sequence."""
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    return np.sum(counts > 0)


def calculate_gini_robust(codes_1d):
    """
    Robust Gini Coefficient: Calculates allocation inequality ONLY for codes that have appeared.
    Filters out the massive number of unused codes to avoid sparsity-induced bias.
    """
    counts = np.bincount(codes_1d, minlength=CODEBOOK_SIZE)
    active_counts = counts[counts > 0]

    if len(active_counts) <= 1:
        return 0

    active_counts = np.sort(active_counts)
    n = len(active_counts)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * active_counts)) / (n * np.sum(active_counts))


def calculate_tccr(codes_1d):
    """Calculate Temporal Code Change Rate (TCCR)."""
    if len(codes_1d) <= 1:
        return 0
    changes = np.sum(codes_1d[1:] != codes_1d[:-1])
    return changes / (len(codes_1d) - 1)


# ==========================================
# 3. Data Storage Structures
# ==========================================
category_layer_entropies = defaultdict(lambda: [[] for _ in range(NUM_QUANTIZERS)])
category_raw_codes_layer1 = defaultdict(list)
category_active_codes = defaultdict(list)
sample_entropies_layer1 = defaultdict(list)
category_gini_layer1 = defaultdict(list)
category_tccr_layer1 = defaultdict(list)

# ==========================================
# 4. Data Processing Pipeline (Single Pass Efficiency)
# ==========================================
for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.endswith('.npy'):
        continue

    parts = file_name.split('_')
    if len(parts) >= 2:
        category = f"{parts[0]}_{parts[1]}"
    else:
        continue

    # Load and flatten the code matrix
    codes_np = np.squeeze(np.load(os.path.join(INPUT_FOLDER, file_name)))
    layer1_flat = codes_np[0].flatten()

    # Collect Layer 1 data seamlessly
    category_raw_codes_layer1[category].extend(layer1_flat.tolist())
    sample_entropies_layer1[category].append(calculate_entropy(layer1_flat))
    category_active_codes[category].append(calculate_active_codes(layer1_flat))
    category_gini_layer1[category].append(calculate_gini_robust(layer1_flat))
    category_tccr_layer1[category].append(calculate_tccr(layer1_flat))

    # Collect entropy data for all available layers
    for layer_idx in range(min(NUM_QUANTIZERS, codes_np.shape[0])):
        layer_codes = codes_np[layer_idx].flatten()
        category_layer_entropies[category][layer_idx].append(calculate_entropy(layer_codes))

# Prepare sorted categories and clean labels for plotting
sorted_categories = sorted(category_layer_entropies.keys())
clean_labels = [cat.split('_')[1] for cat in sorted_categories]

# Calculate averages for aggregation
avg_entropy = {cat: [np.mean(category_layer_entropies[cat][l]) for l in range(NUM_QUANTIZERS)] for cat in
               sorted_categories}
mean_ent = [np.mean(sample_entropies_layer1[cat]) for cat in sorted_categories]
robust_gini_means = [np.mean(category_gini_layer1[cat]) for cat in sorted_categories]
tccr_means = [np.mean(category_tccr_layer1[cat]) for cat in sorted_categories]

print("✅ Data aggregation complete! Starting to generate 7 high-order analysis charts...")

# ==========================================
# Figure 1: RVQ Information Entropy Trajectory
# ==========================================
fig1, ax1 = plt.subplots(figsize=(11, 7))
for i, cat in enumerate(sorted_categories):
    ax1.plot(range(1, NUM_QUANTIZERS + 1), avg_entropy[cat], marker='o', lw=3, label=clean_labels[i], color=colors[i])

ax1.set_title('Entropy Trajectory Across RVQ Layers', fontweight='bold', pad=15)
ax1.set_xlabel('Encodec RVQ Layer')
ax1.set_ylabel('Average Shannon Entropy (bits)')
ax1.set_ylim(0, 10.5)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
fig1.savefig("Figure_1_Entropy_Trajectory.png", dpi=300)

# ==========================================
# Figure 2: Global Acoustic Complexity Heatmap
# ==========================================
heatmap_data = np.array([avg_entropy[cat] for cat in sorted_categories])
fig2, ax2 = plt.subplots(figsize=(11, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="magma", xticklabels=range(1, NUM_QUANTIZERS + 1),
            yticklabels=clean_labels, ax=ax2)
ax2.set_title('Global Codebook Entropy Heatmap', fontweight='bold', pad=15)
ax2.set_xlabel('RVQ Layer Depth')
ax2.set_ylabel('Audio Category')
plt.tight_layout()
fig2.savefig("Figure_2_Entropy_Heatmap.png", dpi=300)

# ==========================================
# Figure 3: Layer 1 Codebook Usage Distribution
# ==========================================
fig3, axes = plt.subplots(5, 3, figsize=(18, 16), sharex=True)
fig3.suptitle('Codebook Usage Distribution (RVQ Layer 1)', fontsize=20, fontweight='bold', y=0.98)
axes = axes.flatten()

for i, cat in enumerate(sorted_categories):
    plot_data = category_raw_codes_layer1[cat][:30000]
    sns.histplot(plot_data, bins=256, ax=axes[i], color=colors[i], stat='density', element='step', alpha=0.6)
    axes[i].set_title(f'{clean_labels[i]}', fontsize=14, fontweight='bold')
    axes[i].set_xlim(0, CODEBOOK_SIZE)
    axes[i].set_ylabel('Density')

for j in range(len(sorted_categories), len(axes)):
    fig3.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig3.savefig("Figure_3_Full_Distributions.png", dpi=300)

# ==========================================
# Figure 4: Dictionary Sparsity (Bar Chart)
# ==========================================
mean_active = [np.mean(category_active_codes[cat]) for cat in sorted_categories]
std_active = [np.std(category_active_codes[cat]) for cat in sorted_categories]

fig4, ax4 = plt.subplots(figsize=(11, 6))
bars = ax4.bar(clean_labels, mean_active, yerr=std_active, capsize=5, color=colors, edgecolor='black', linewidth=1)
ax4.set_title('Active Codebook Size in Layer 1 (Out of 1024)', fontweight='bold', pad=15)
ax4.set_ylabel('Number of Unique Codes Used')
ax4.set_xticklabels(clean_labels, rotation=45, ha='right')
ax4.axhline(CODEBOOK_SIZE, color='red', linestyle='--', alpha=0.5, label='Max Capacity (1024)')
ax4.legend()
plt.tight_layout()
fig4.savefig("Figure_4_Active_Codes.png", dpi=300)

# ==========================================
# Figure 5: Intra-class Stability (Box Plot)
# ==========================================
box_data = [sample_entropies_layer1[cat] for cat in sorted_categories]
fig5, ax5 = plt.subplots(figsize=(11, 6))
sns.boxplot(data=box_data, palette=colors, ax=ax5)
ax5.set_title('Intra-class Variance of Entropy (Layer 1)', fontweight='bold', pad=15)
ax5.set_xticklabels(clean_labels, rotation=45, ha='right')
ax5.set_ylabel('Shannon Entropy Distribution (bits)')
plt.tight_layout()
fig5.savefig("Figure_5_Variance_Boxplot.png", dpi=300)

# ==========================================
# Figure 6: Vocabulary Inequality Analysis (Robust Gini)
# ==========================================
fig6, ax6 = plt.subplots(figsize=(11, 7))
sns.scatterplot(x=mean_ent, y=robust_gini_means, hue=clean_labels, palette=colors, s=350, edgecolor='black', alpha=0.9,
                ax=ax6)

ax6.set_title('Codebook Vocabulary Inequality Analysis', fontweight='bold', fontsize=16, pad=20)
ax6.set_xlabel('Shannon Entropy (Information Diversity)', fontweight='bold')
ax6.set_ylabel('Robust Gini Coefficient (Inequality)', fontweight='bold')

# Dynamically adjust the Y-axis range to clarify data distribution safely
y_min, y_max = min(robust_gini_means), max(robust_gini_means)
padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
ax6.set_ylim(max(0, y_min - padding), min(1.0, y_max + padding))

ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Audio Classes", frameon=True)
plt.tight_layout()
fig6.savefig("Figure_6_Robust_Gini_Analysis.png", dpi=300)

# ==========================================
# Figure 7: Temporal Volatility 2D Clustering (TCCR vs Entropy)
# ==========================================
fig7, ax7 = plt.subplots(figsize=(11, 7))
sns.scatterplot(x=mean_ent, y=tccr_means, hue=clean_labels, palette=colors, s=350, edgecolor='black', alpha=0.9, ax=ax7)

ax7.set_title('Temporal Latent Volatility (TCCR) vs. Spatial Complexity', fontweight='bold', pad=15)
ax7.set_xlabel('Shannon Entropy (Spatial Breadth)', fontweight='bold')
ax7.set_ylabel('Temporal Code Change Rate (TCCR)', fontweight='bold')

# Add quadrant reference lines based on global medians
ax7.axvline(np.median(mean_ent), color='gray', linestyle='--', alpha=0.5)
ax7.axhline(np.median(tccr_means), color='gray', linestyle='--', alpha=0.5)



ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Audio Classes")
plt.tight_layout()
fig7.savefig("Figure_7_Temporal_Volatility_2D.png", dpi=300)

print("\n🎉 Success! All 7 highly optimized academic charts have been generated.")