import requests
import os
import urllib.request

# 1. Insert your Freesound API Key here
API_KEY = "HIDDEN"

# 2. Define the sound categories to compare
# Format: "Category_Name": "Search Keyword"HIDDEN
CATEGORIES = {
    "1_Synthetic": "8-bit",  # Mathematical waveforms
    "2_Speech": "speech",  # Single human voice
    "3_Instrument": "piano",  # Melodic instruments
    "4_Percussion": "drum",  # Rhythm and transients
    "5_Crowd": "crowd talking",  # Complex human voices
    "6_Nature": "rain",  # Natural broadband sounds
    "7_ComplexMusic": "Symphony",  # Orchestral music
    "8_Noise": "",  # General noise (empty query may return broad results)
    "9_Impulsive": "Gun"  # Impulsive sounds like gunshots
}

# Create the directory to save the dataset if it doesn't exist
os.makedirs("dataset", exist_ok=True)

print("Starting dataset acquisition via Freesound APIv2...")

for category_name, query_text in CATEGORIES.items():
    print(f"\nSearching category: {category_name} (Keyword: '{query_text}')")

    # Build the request URL using the text search endpoint
    search_url = "https://freesound.org/apiv2/search/text/"

    # Configure core parameters
    params = {
        "query": query_text,
        "token": API_KEY,
        # Filter: Duration between 5.0 and 30.0 seconds, single event only
        "filter": "duration:[5.0 TO 30.0] single_event:true",
        # Optimization: Request preview links directly in the search results to avoid extra API calls
        "fields": "id,name,previews",
        "page_size": 30  # Limit results per category
    }

    response = requests.get(search_url, params=params)

    # Check if the request was successful
    if response.status_code != 200:
        error_detail = response.json().get('detail', 'Unknown error')
        print(f"Request failed: {response.status_code} - {error_detail}")
        continue

    results = response.json().get('results', [])

    # Iterate through the search results
    for i, sound in enumerate(results):
        sound_id = sound['id']

        # Get the high-quality OGG preview URL
        preview_url = sound['previews']['preview-hq-ogg']

        # Generate the filename
        file_name = f"dataset/{category_name}_{i + 1}_{sound_id}.ogg"

        print(f"  Downloading: {file_name}")

        # Download and save the audio file
        urllib.request.urlretrieve(preview_url, file_name)

print("\nDataset acquisition complete! All files are saved in the 'dataset/' directory.")