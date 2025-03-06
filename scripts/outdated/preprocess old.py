import json, os, datetime, ast, itertools
import pandas as pd
import numpy as np

#THIS FILE IS OBSOLETE, IT WAS USED TO MAKE A DATABASE VIA PER-MERGING
#BUT THE PROJECT WAS SWITCHED TO DYNAMIC MERGING TO PREVENT BLOATING

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed/internal"
battlebuilds_sheet_dir = 'data/processed/app_calcs/battlebuilds.csv'
move_types = ['fast_moves', 'elite_fast_moves', 'charged_moves', 'elite_charged_moves']
battlebuilds_column_order = [
    'pokemon_name', 'form', 'type', 'base_attack', 'base_defense', 'base_stamina', 'max_cp',
    'fast_moves', 'type_fast', 'duration', 'energy_delta', 'power',
    'charged_moves', 'type_charged', 'duration_charged', 'energy_delta_charged', 'power_charged',
    'name_effect', 'effect_chance', 'target', 'stat_changed', 'amount'
]
#use data pulled from api
stats = pd.read_csv('data/processed/internal/pokemon_stats.csv', usecols=['pokemon_name','form','base_attack','base_defense','base_stamina'])
types = pd.read_csv('data/processed/internal/pokemon_types.csv', usecols=['pokemon_name','form','type'])
max_cp = pd.read_csv('data/processed/internal/pokemon_max_cp.csv', usecols=['pokemon_name','form','max_cp'])
pokemon_moves = pd.read_csv('data/processed/internal/current_pokemon_moves.csv', usecols=['pokemon_name','form','fast_moves','elite_fast_moves','charged_moves','elite_charged_moves'])
fast_moves = pd.read_csv('data/processed/internal/fast_moves.csv', usecols=['name','type','duration','energy_delta','power'])
charged_moves = pd.read_csv('data/processed/internal/charged_moves.csv', usecols=['name','type','duration','energy_delta','power']) #duration does not matter in charged moves for pvp?
move_effects = pd.read_csv('data/processed/external/PVP_Charged_Move_Effects.csv', usecols=['stat_changed','effect_chance','target','amount','name'])

#There appears to be lists of dictionaries and 
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
    
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".json"):
            json_file = os.path.join(RAW_DATA_DIR, filename)
            csv_file = os.path.join(PROCESSED_DATA_DIR, filename.replace(".json", ".csv"))
            json_to_csv(json_file, csv_file)
            #print(f"Rewrote {csv_file}")

def generate_move_combinations(row):
    fast_moves = row['fast_moves'] + row['elite_fast_moves']
    charged_moves = row['charged_moves'] + row['elite_charged_moves']
    return list(itertools.product(fast_moves, charged_moves))

def check_battlebuildscsv(sheet_dir):
    if not os.path.exists(sheet_dir):
        print(f"Creating {sheet_dir}")
        merge1 = pd.merge(stats,types,on=['pokemon_name','form'],how='inner')
        merge2 = pd.merge(merge1,pokemon_moves,on=['pokemon_name','form'],how='inner')
        merge3 = pd.merge(merge2,max_cp,on=['pokemon_name','form'],how='inner')
        #calculates the maximum length of move lists across all move columns
        for type in move_types:
            #convert the move columns from string lists to lists
            merge3[type] = merge3[type].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
        merge3['move_combinations'] = merge3.apply(generate_move_combinations, axis=1)
        merge3 = merge3.explode('move_combinations') #make battlebuildcsv have all combos of moves
        # Split the move combinations into separate columns
        merge3[['fast_moves', 'charged_moves']] = pd.DataFrame(merge3['move_combinations'].tolist(), index=merge3.index)
        # Drop the temporary 'move_combinations' column
        merge3 = merge3.drop(columns=['move_combinations'])
        # Explode the move columns to create one row per move
        merge4 = pd.merge(merge3, fast_moves, left_on='fast_moves', right_on='name', how='left', suffixes=('', '_fast'))
        merge5 = pd.merge(merge4, charged_moves, left_on='charged_moves', right_on='name', how='left', suffixes=('', '_charged'))
        merge6 = pd.merge(merge5, move_effects, left_on='charged_moves', right_on='name', how='left', suffixes=('', '_effect'))
        #cols = merge5.columns
        #print(cols)
        #merge5 = merge5.drop(merge5.columns[-5], axis=1) #we dont need this
        #rearrange for readability
        merge6 = merge6[battlebuilds_column_order]
        #fill in all NaN values with ''
        for col in merge6.columns:
            merge6[col] = merge6[col].fillna('')
        merge6.to_csv(battlebuilds_sheet_dir,index=False)
    else:
        create_time = os.path.getctime(sheet_dir)
        mod_time = os.path.getmtime(sheet_dir)
        print(f"{sheet_dir} created: {datetime.datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Last modified: {datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    print("Updating Database...")
    preprocess_data()
    print("Consolidating Data...")
    check_battlebuildscsv(battlebuilds_sheet_dir)

if __name__ == "__main__":
    main()