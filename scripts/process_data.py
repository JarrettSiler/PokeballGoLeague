import json, os
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed/internal"
PROCESSING_DIR = "data/processing"
FULL_PROCESSING_DIR = Path(os.getcwd()) / 'data' / 'processing'
TOP_ML_POKEMON = "data/processed/external/MasterLeague_Threats.csv"

#DRASTICALLY CHANGE THE DATABASE SIZE *****************
IV_STEP = 15 #5  # Use every 5th IV
LEVEL_STEP = 25 #5  # Use every 5th level
#******************************************************

#To prepare data for dynamic merging we want to seperate static and dynamic stats
#static: pokemon properties, dynamic: stats that change due to a varying property, powering up or modification from player

#make important csv's and save as processed data
def prepare_static_dynamic_data():
    pd.read_csv(PROCESSING_DIR + '/pokemon_stats.csv', usecols=['pokemon_name','form','base_attack','base_defense','base_stamina']).to_csv(PROCESSED_DATA_DIR+'/base_stats.csv', index=False)
    pd.read_csv(PROCESSING_DIR + '/pokemon_types.csv', usecols=['pokemon_name','form','type']).to_csv(PROCESSED_DATA_DIR+'/pokemon_types.csv', index=False)
    pd.read_csv(PROCESSING_DIR + '/pokemon_evolutions.csv').to_csv(PROCESSED_DATA_DIR+'/pokemon_evolutions.csv', index=False)
    pd.read_csv(PROCESSING_DIR + '/pokemon_max_cp.csv', usecols=['pokemon_name','form','max_cp']).to_csv(PROCESSED_DATA_DIR+'/pokemon_max_cp.csv', index=False)
    te = pd.read_csv(PROCESSING_DIR + '/type_effectiveness.csv')
    te.insert(0, 'index', te.columns)
    te.to_csv(PROCESSED_DATA_DIR+'/type_effectiveness.csv', index=False)

    #pd.read_csv(PROCESSING_DIR + '/cp_multiplier.csv', usecols=['level','multiplier']).to_csv(PROCESSED_DATA_DIR+'/cp_multipliers.csv', index=False)
    pd.read_csv(PROCESSING_DIR + '/current_pokemon_moves.csv', usecols=['pokemon_name','form','fast_moves','elite_fast_moves','charged_moves','elite_charged_moves']).to_csv(PROCESSED_DATA_DIR+'/pokemon_moves.csv', index=False)
    fast_moves = pd.read_csv(PROCESSING_DIR + '/pvp_fast_moves.csv', usecols=['name','type','turn_duration','energy_delta','power'])
    charged_moves = pd.read_csv(PROCESSING_DIR + '/pvp_charged_moves.csv', usecols=['name','type','energy_delta','power'])
    fast_moves["move_type"], charged_moves["move_type"], charged_moves["turn_duration"] = "fast", "charged", ''
    charged_moves = charged_moves[['name', 'type', 'turn_duration', 'energy_delta', 'power', 'move_type']]
    all_moves = pd.concat([fast_moves, charged_moves], ignore_index=True)
    pokemon_moves = pd.read_csv(PROCESSED_DATA_DIR + '/pokemon_moves.csv')
    #differentiate between elite moves
    elite_fast_moves = pokemon_moves[['elite_fast_moves']].dropna().explode('elite_fast_moves').rename(columns={'elite_fast_moves': 'name'})
    elite_charged_moves = pokemon_moves[['elite_charged_moves']].dropna().explode('elite_charged_moves').rename(columns={'elite_charged_moves': 'name'})
    all_moves['is_elite'] = all_moves['name'].isin(elite_fast_moves['name']) | all_moves['name'].isin(elite_charged_moves['name'])
    all_moves.to_csv(PROCESSED_DATA_DIR + '/all_moves.csv', index=False)
    
    #master_league_threats = pd.read_csv(TOP_ML_POKEMON)
    #master_league_threats = master_league_threats.dropna(axis=1, how='all')
    #master_league_threats['is_shadow'] = master_league_threats['pokemon_name'].str.contains('Shadow', case=False, na=False)
    #master_league_threats['pokemon_name'] = master_league_threats['pokemon_name'].str.replace('Shadow ', '', case=False, regex=True)
    #master_league_threats.to_csv(TOP_ML_POKEMON, index=False)
#------------------------------------------------------------------

def extend_cp_multipliers(max_level=51):
    # Load existing data
    df = pd.read_csv(PROCESSING_DIR + '/cp_multiplier.csv')

    if df["level"].max() >= 51.0:
        print("CP multipliers already include levels up to 51")
        return
    
    # CP multipliers for levels 45.0 to 51.0, found online
    #Source: https://pogo.gamepress.gg/cp-multiplier
    new_multipliers = [
        (45.0, 0.81529999),
        (45.5, 0.81779999),
        (46.0, 0.82029999),
        (46.5, 0.82279999),
        (47.0, 0.82529999),
        (47.5, 0.82779999),
        (48.0, 0.83029999),
        (48.5, 0.83279999),
        (49.0, 0.83529999),
        (49.5, 0.83779999),
        (50.0, 0.84029999),
        (50.5, 0.84279999),
        (51.0, 0.84529999)
    ]
    
    # Create a DataFrame for the new multipliers
    new_data = pd.DataFrame(new_multipliers, columns=["level", "multiplier"])
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Remove duplicates (in case some levels already exist)
    df = df.drop_duplicates(subset=["level"], keep="last")
    df = df.sort_values(by="level").reset_index(drop=True)
    
    # Save the updated data
    df.to_csv(PROCESSED_DATA_DIR + '/cp_multipliers.csv', index=False)
    print("CP multipliers extended to level 51 (51 is best buddy boost)")


#In JSON files there appears to be lists of dictionaries and 
#dictionaries of lists of dictionaries in the obtained data

def json_to_csv(json_file, csv_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    # Handle dictionaries of lists of dictionaries and dicts of dicts
    if isinstance(data, dict):
        # Flatten the data into a list of dictionaries
        # all first level keys are already in the dictionaries
        flattened_data = []
        for key, entry in data.items():
            if isinstance(entry, list):
                for dicty in entry:
                    flattened_data.append(dicty)
            elif isinstance(entry, dict):
                # Handle cases where the value is not a list
                flattened_data.append(entry)
        df = pd.DataFrame(flattened_data)
    
    # Handle lists of dictionaries
    elif isinstance(data, list):
        df = pd.DataFrame(data)

    # Handle unsupported structures
    else:
        raise ValueError(f"Unsupported JSON structure in {json_file}")

    df.to_csv(csv_file, index=False)

def preprocess_data():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(PROCESSING_DIR):
        os.makedirs(PROCESSING_DIR)

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".json"):
            json_file = os.path.join(RAW_DATA_DIR, filename)
            csv_file = os.path.join(PROCESSING_DIR, filename.replace(".json", ".csv"))
            json_to_csv(json_file, csv_file)
            #print(f"Rewrote {csv_file}")

#---------------------------------------------------------------------------------------

def main():
    print("Updating Database...")
    preprocess_data()
    print("Consolidating Data...")
    extend_cp_multipliers()
    prepare_static_dynamic_data()
    print("data prepared")
    for file in FULL_PROCESSING_DIR.iterdir():
        if file.is_file():
            file.unlink()
    print("removed unused files")