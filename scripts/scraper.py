import requests
import os
import hashlib
import json

# Configuration
API_BASE_URL = "https://pogoapi.net/api/v1"
DATA_DIR = "data/raw"
HASH_FILE = "data/cache/api_hashes.json"
dataofinterest = [
    'pokemon_types.json',
    'pokemon_evolutions.json',
    'pokemon_powerup_requirements.json',
    'pokemon_candy_to_evolve.json',
    'pokemon_buddy_distances.json',
    'type_effectiveness.json',
    'pokemon_stats.json',
    'pokemon_rarity.json',
    'current_pokemon_moves.json',
    'pvp_fast_moves.json',
    'pvp_charged_moves.json',
    'pokemon_max_cp.json',
    'cp_multiplier.json'
    ] # the datasets to grab from API_BASE_URL

#TODO LATER:
#ADD OPTION TO ONLY UPDATE NECESSARY DATA EVERY SO OFTEN

def fetch_hashes():
    response = requests.get(f"{API_BASE_URL}/api_hashes.json")
    if response.status_code == 200: #success
        return response.json()
    else:
        print(f'ERROR UPDATING DATA FROM ONLINE REPOSITORY, ARE YOU SURE {API_BASE_URL} STILL EXISTS?')
    return None

def save_file(url, filename):
    #print(f"Attempting to download from: {url}")
    response = requests.get(url)
    #print(response)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

def check_and_update_data(hashes):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for api_filename, hash_data in hashes.items():
        #print(hash_data)
        #print(api_filename)
        if api_filename in dataofinterest:
            file_path = os.path.join(DATA_DIR, api_filename)
            if os.path.exists(file_path):
                # Compare hashes
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash == hash_data["hash_md5"]:
                    continue  # File is up-to-date
            
            # Download and save the updated file
            save_file(f"{API_BASE_URL}/{api_filename}", file_path)
            print(f"Updated {api_filename}")

def main():
    hashes = fetch_hashes()
    if hashes:
        check_and_update_data(hashes)
    else:
        print("ERROR, CHECK scripts/scraper.py")

if __name__ == "__main__":
    main()