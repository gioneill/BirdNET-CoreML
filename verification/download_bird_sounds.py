#!/usr/bin/env python3
"""
download_bird_sounds_freesound.py

Fetch bird call recordings from the Freesound API.
Creates bird_sounds/<Species_Name>/ folders and downloads ~5 MP3 previews per species.
"""

import os
import time
import requests

# === CONFIGURATION ===
species_list = [
    "American Robin", "Northern Cardinal", "House Finch", "Song Sparrow",
    "Carolina Wren", "American Goldfinch", "Red-winged Blackbird",
    "Eastern Bluebird", "Common Grackle", "European Starling",
    "Great Horned Owl", "Barred Owl", "Barn Owl", "Eastern Screech-Owl",
    "Western Screech-Owl", "Downy Woodpecker", "Hairy Woodpecker",
    "Northern Flicker", "Mallard", "Red-tailed Hawk",
    "Mourning Dove", "Rock Pigeon", "Blue Jay", "Black-capped Chickadee",
    "Tufted Titmouse", "Canada Goose", "Killdeer", "Great Blue Heron",
    "American Coot", "Snowy Egret", "Osprey", "Bald Eagle",
    "Peregrine Falcon", "Common Loon", "Belted Kingfisher",
    "Cedar Waxwing", "Indigo Bunting", "Yellow Warbler",
    "White-throated Sparrow", "Northern Mockingbird"
]
recordings_per_species = 5
base_dir = "bird_sounds"
api_base = "https://freesound.org/apiv2"
token = os.getenv("FREESOUND_API_KEY")
if not token:
    raise RuntimeError("Please set your Freesound API key in the FREESOUND_API_KEY environment variable")

# === SCRIPT START ===
os.makedirs(base_dir, exist_ok=True)

for sp in species_list:
    safe_name = sp.replace(" ", "_").replace("-", "_")
    species_dir = os.path.join(base_dir, safe_name)
    os.makedirs(species_dir, exist_ok=True)

    params = {
        "query": sp,
        "filter": "duration:[1 TO 10]",      # only 1–10s clips
        "fields": "id,name,previews",        # get IDs, names, and preview URLs  [oai_citation:0‡freesound.org](https://freesound.org/docs/api/resources_apiv2.html?utm_source=chatgpt.com)
        "page_size": recordings_per_species,
        "token": token
    }
    search_url = f"{api_base}/search/text/"
    print(f"\n🔍 Searching Freesound for “{sp}” → {search_url}")
    resp = requests.get(search_url, params=params)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        print(f"  ⚠️  No Freesound results for {sp}.")
        continue

    for rec in results:
        rec_id = rec["id"]
        preview_url = rec["previews"]["preview-hq-mp3"]
        out_path = os.path.join(species_dir, f"{safe_name}_{rec_id}.mp3")

        if os.path.exists(out_path):
            print(f"  • Skipping existing {os.path.basename(out_path)}")
            continue

        print(f"  ↓ Downloading ID {rec_id}…", end="", flush=True)
        dl = requests.get(preview_url, stream=True)
        dl.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)
        print(" done.")
        time.sleep(1.0)  # respect Freesound’s rate limits

print("\n✅ All downloads complete. Check the 'bird_sounds/' directory.")
