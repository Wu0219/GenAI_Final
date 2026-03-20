import os
import soundfile as sf
from collections import defaultdict

# Set the dataset path
DATASET_FOLDER = "dataset"


def get_dataset_stats_with_totals(folder):
    # Data structure: {category_name: [file_count, total_duration]}
    stats = defaultdict(lambda: [0, 0.0])

    print(f"Starting full data audit for: {folder} ...")

    # Filter for audio files with supported extensions
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.ogg', '.mp3'))]

    if not files:
        print("No audio files found.")
        return None

    for file_name in files:
        parts = file_name.split('_')
        # Extract category from filename (expects format like 'ID_Category_...')
        category = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "Unknown"
        file_path = os.path.join(folder, file_name)

        try:
            with sf.SoundFile(file_path) as f:
                # Calculate duration in seconds
                duration = len(f) / f.samplerate
                stats[category][0] += 1
                stats[category][1] += duration
        except Exception:
            # Skip files that cannot be read
            continue

    # --- Core Logic: Automatic Aggregation ---
    total_categories = len(stats)
    total_count = sum(item[0] for item in stats.values())
    total_duration = sum(item[1] for item in stats.values())
    overall_mean = total_duration / total_count if total_count > 0 else 0

    # Prepare formatted table header
    header = f"{'ID & Category':<22} | {'Count (N)':<10} | {'Total Dur (s)':<15} | {'Mean Dur (s)':<12}"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    # Sort categories: numeric ID first, then others
    sorted_keys = sorted(stats.keys(), key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999)

    # Iterate through sorted categories to print statistics
    for cat in sorted_keys:
        count, duration = stats[cat]
        mean_dur = duration / count if count > 0 else 0
        print(f"{cat:<22} | {count:<10} | {duration:<15.2f} | {mean_dur:<12.2f}")

    # Print the automatically generated summary row
    print(divider)
    print(
        f"{'TOTAL (' + str(total_categories) + ' Cats)':<22} | {total_count:<10} | {total_duration:<15.2f} | {overall_mean:<12.2f}")
    print(divider + "\n")

    return stats, (total_categories, total_count, total_duration, overall_mean)


# Execute the function and retrieve results
stats_data, summary = get_dataset_stats_with_totals(DATASET_FOLDER)