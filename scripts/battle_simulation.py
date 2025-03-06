import numpy as np
import os, ast
import pandas as pd
from pathlib import Path
from scripts.analyze_pokemon import main as get_pokemon_data

#UPDATE TO HANDLE POKEMON FORM FROM ML LIST
TOP_ML_POKEMON = "data/processed/external/MasterLeague_Threats.csv"
MASTER_LG_DIR = Path(os.getcwd()) / 'data' / 'processed' / 'battle_leagues' / 'ML_META'
PROCESSED_DATA_DIR = "data/processed/internal"
CALC_DIR = "data/processed/app_calcs"

def update_ML_sheets():
    #create all ref files for meta pokemon if not already made
    master_league_threats = pd.read_csv(TOP_ML_POKEMON)
    for _, row in master_league_threats.iterrows():
        if row['is_shadow']:
            if row['form'] != 'Normal': file_name = f"shadow_{row['form']}_{row['pokemon_name']}.csv"
            else: file_name = f"shadow_{row['pokemon_name']}.csv"
        else:
            if row['form'] != 'Normal': file_name = f"{row['form']}_{row['pokemon_name']}.csv"
            else: file_name = f"{row['pokemon_name']}.csv"
        if not (MASTER_LG_DIR / file_name).exists():
            get_pokemon_data(row['pokemon_name'], row['form'], row['is_shadow'], print_out=True, print_dir=MASTER_LG_DIR, moveset='best') #print means save csv

def calc_battle_score(battle_dict, type_effectiveness, attacker_shields=2, defender_shields=2):

    #calculate the battle score considering shields for both the attacker and defender and type effectiveness
    #shields reduce the effectiveness of charged moves for the analysis, later this could be updated
    
    # Get attacker and defender types, shortend the variables as I am tied of typing
    at1, at2 = battle_dict['attacker_type_1'], battle_dict['attacker_type_2'] if battle_dict['attacker_type_2'] != '' else False
    dt1, dt2 = battle_dict['defender_type_1'], battle_dict['defender_type_2'] if battle_dict['defender_type_2'] != '' else False

    # Lookup attacker move types (fast, first_charged, second_charged)
    afm_type, acm1_type, acm2_type = battle_dict['attacker_fast_move_type'], battle_dict['attacker_first_charged_move_type'], battle_dict['attacker_second_charged_move_type']
    dfm_type, dcm1_type, dcm2_type = battle_dict['defender_fast_move_type'], battle_dict['defender_first_charged_move_type'], battle_dict['defender_second_charged_move_type']
    # Lookup defender's types in the type effectiveness table
    deff_fast = type_effectiveness.loc[dfm_type, at1] #effectiveness on type 1
    if at2: deff_fast = (deff_fast * type_effectiveness.loc[dfm_type, at2]) #effectiveness on type 2
    deff_charged1 = type_effectiveness.loc[dcm1_type, at1]
    if at2: deff_charged1 = (deff_charged1 * type_effectiveness.loc[dcm1_type, at2])
    deff_charged2 = type_effectiveness.loc[dcm2_type, at1]
    if at2: deff_charged2 = (deff_charged2 * type_effectiveness.loc[dcm2_type, at2])

    # Lookup defender's types in the type effectiveness table
    aeff_fast = type_effectiveness.loc[afm_type, dt1] #effectiveness on type 1
    if dt2: aeff_fast = (aeff_fast * type_effectiveness.loc[afm_type, dt2]) #effectiveness on type 2
    aeff_charged1 = type_effectiveness.loc[acm1_type, dt1]
    if dt2: aeff_charged1 = (aeff_charged1 * type_effectiveness.loc[acm1_type, dt2])
    aeff_charged2 = type_effectiveness.loc[acm2_type, dt1]
    if dt2: aeff_charged2 = (aeff_charged2 * type_effectiveness.loc[acm2_type, dt2])

    # Calculate the attacker's total damage output, incorporating effectiveness
    weights = [1, 1, 1]
    attacker_batstat = np.array([battle_dict['attacker_TDO']*weights[0], battle_dict['attacker_DPT']*weights[1], battle_dict['attacker_DPE']*weights[2]])
    defender_batstat = np.array([battle_dict['defender_TDO']*weights[0], battle_dict['defender_DPT']*weights[1], battle_dict['defender_DPE']*weights[2]])
    var = [1,2,4]
    win_counter, loss_counter = 0,0
    #If 2 shields are given each then this loop setup is inefficient, but possibly in the future
    #a better battle simulator will be made which will account for shield variations in a battle
    for ats in range(attacker_shields,-1,-1):
        for dfs in range(defender_shields,-1,-1):
            asa = (attacker_batstat*aeff_fast)+((attacker_batstat*aeff_charged1)/var[dfs])+((attacker_batstat*aeff_charged2)/var[dfs])
            dsa = (defender_batstat*deff_fast)+((defender_batstat*deff_charged1)/var[ats])+((defender_batstat*deff_charged2)/var[ats])
            abe, dbe = (asa[1]/asa[2])*asa[0], (dsa[1]/dsa[2])*dsa[0] #idea is DPT/DPE = E/T, (E/T)*TDO is battle efficiency
            if abe > dbe: loss_counter += 1
            else: win_counter += 1
    if win_counter > loss_counter: return 1
    else: return 0

def make_battlesheets(pokemon_name, form_pokemon, is_shadow, IVs, lvl, moves, print_out): #moveset none is the optimal
    defending_pokemon = get_pokemon_data(pokemon_name, form_pokemon, is_shadow, IVs, LEVELS=lvl, print_out=print_out, moveset=moves).pkm_data #add moveset later
    move_types = pd.read_csv(PROCESSED_DATA_DIR+'/all_moves.csv',usecols=['name','type'])
    pokemon_types = pd.read_csv(PROCESSED_DATA_DIR+'/pokemon_types.csv')
    pokemon_types['type'] = pokemon_types['type'].apply(ast.literal_eval) #convert string lists to lists
    type_effectiveness = pd.read_csv(PROCESSED_DATA_DIR+'/type_effectiveness.csv', index_col=0)
    battle_data = []

    final_columns = ['defender_pokemon_name',
                'defender_level','defender_CP','defender_HP',
                'defender_IV_atk','defender_IV_def','defender_IV_sta',
                'defender_type_1','defender_type_2','defender_fast_move_type',
                'defender_first_charged_move_type','defender_second_charged_move_type',
                'defender_TDO','defender_DPT','defender_DPE',
                'battle_score']
    final_frame= pd.DataFrame(columns=final_columns)

    data_points = defending_pokemon.shape[0] #gets number of variations of the pokemon data in defending_pokemon
    for defending_pokemon_index in range(0,data_points):
        for attacking_pokemon_file in MASTER_LG_DIR.iterdir():
            pokemon_used = defending_pokemon.iloc[[defending_pokemon_index]]
            attacker, defender = pd.read_csv(attacking_pokemon_file), pokemon_used
            attacker_types = pokemon_types.loc[(pokemon_types['form'] == attacker['form'].values[0]) &
                    (pokemon_types['pokemon_name'] == attacker['pokemon_name'].values[0]), 'type'].values[0]
            defender_types = pokemon_types.loc[(pokemon_types['form'] == defender['form'].values[0]) &
                    (pokemon_types['pokemon_name'] == defender['pokemon_name'].values[0]), 'type'].values[0]
            attacker_type_1, attacker_type_2 = attacker_types[0],attacker_types[1] if len(attacker_types) > 1 else ''
            defender_type_1, defender_type_2 = defender_types[0],defender_types[1] if len(defender_types) > 1 else ''
            
            # Lookup move types for each move (fast_move, first_charged_move, second_charged_move)
            attacker_fast_move_type = move_types.loc[(move_types['name'] == attacker['fast_move'].values[0]), 'type'].values[0]
            attacker_first_charged_move_type = move_types.loc[(move_types['name'] == attacker['first_charged_move'].values[0]), 'type'].values[0]
            attacker_second_charged_move_type = move_types.loc[(move_types['name'] == attacker['second_charged_move'].values[0]), 'type'].values[0]

            defender_fast_move_type = move_types.loc[(move_types['name'] == defender['fast_move'].values[0]), 'type'].values[0]
            defender_first_charged_move_type = move_types.loc[(move_types['name'] == defender['first_charged_move'].values[0]), 'type'].values[0]
            defender_second_charged_move_type = move_types.loc[(move_types['name'] == defender['second_charged_move'].values[0]), 'type'].values[0]

            battle_row = {
                # Attacker features
                'attacker_pokemon_name': attacker['pokemon_name'].values[0],
                'attacker_pokemon_form': attacker['form'].values[0],
                'attacker_level': attacker['level'].values[0],
                'attacker_IV_atk': attacker['IV_atk'].values[0],
                'attacker_IV_def': attacker['IV_def'].values[0],
                'attacker_IV_sta': attacker['IV_sta'].values[0],
                'attacker_CP': attacker['CP'].values[0],
                'attacker_HP': attacker['HP'].values[0],
                'attacker_TDO': round(attacker['TDO'].values[0],2),
                'attacker_DPT': round(attacker['DPT'].values[0],2),
                'attacker_DPE': round(attacker['DPE'].values[0],2),
                'attacker_fast_move': attacker['fast_move'].values[0],
                'attacker_first_charged_move': attacker['first_charged_move'].values[0],
                'attacker_second_charged_move': attacker['second_charged_move'].values[0],
                
                # Attacker types
                'attacker_type_1': attacker_type_1,
                'attacker_type_2': attacker_type_2,
                
                # Defender features
                'defender_pokemon_name': defender['pokemon_name'].values[0],
                'defender_level': defender['level'].values[0],
                'defender_IV_atk': defender['IV_atk'].values[0],
                'defender_IV_def': defender['IV_def'].values[0],
                'defender_IV_sta': defender['IV_sta'].values[0],
                'defender_CP': defender['CP'].values[0],
                'defender_HP': defender['HP'].values[0],
                'defender_TDO': round(defender['TDO'].values[0],2),
                'defender_DPT': round(defender['DPT'].values[0],2),
                'defender_DPE': round(defender['DPE'].values[0],2),
                'defender_fast_move': defender['fast_move'].values[0],
                'defender_first_charged_move': defender['first_charged_move'].values[0],
                'defender_second_charged_move': defender['second_charged_move'].values[0],
                
                # Defender types
                'defender_type_1': defender_type_1,
                'defender_type_2': defender_type_2,
                
                # Add move types for the attacker and defender
                'attacker_fast_move_type': attacker_fast_move_type,
                'attacker_first_charged_move_type': attacker_first_charged_move_type,
                'attacker_second_charged_move_type': attacker_second_charged_move_type,
                'defender_fast_move_type': defender_fast_move_type,
                'defender_first_charged_move_type': defender_first_charged_move_type,
                'defender_second_charged_move_type': defender_second_charged_move_type,

                # Shadow status (attacker in this case, but could use defender if needed)
                'is_shadow_attacker': attacker['is_shadow'].values[0],
                'is_shadow_defender': defender['is_shadow'].values[0]
            }

            #determine this battle a win or loss
            battle_row['win'] = calc_battle_score(battle_row, type_effectiveness)
            battle_data.append(battle_row)
    
        battle_dataset = pd.DataFrame(battle_data) #dataset for one variation of defender against all attackers
        battle_data = [] #reset for next iteration
        total_wins, num_battles = battle_dataset['win'].sum(), battle_dataset['win'].count()
        viability = round(total_wins/num_battles,3)
        battle_dataset['battle_score'] = viability
        common_columns = battle_dataset.columns.intersection(final_frame.columns)
        battles_row_data_frame = battle_dataset.loc[0,common_columns].to_frame().T
        if final_frame is None or final_frame.empty: final_frame = battles_row_data_frame.copy()
        else: final_frame = pd.concat([final_frame, battles_row_data_frame], ignore_index=True)
        
    #final_frame.to_csv(CALC_DIR+'/battle.csv',index=False)
    return final_frame

#model will have two jobs:
#1. will the pokemon specified at its IVs be capable of beating a pokemon on the master league list (classification)
#2. At what level does this pokemon begin to win? (Regression)
#features of self: level, IV_atk, IV_def, IV_sta, CP, HP, TDO, DPT, DPE
#features of opponent: level_opp, IV_atk_opp, IV_def_opp, IV_sta_opp, CP_opp, HP_opp, TDO_opp, DPT_opp, DPE_opp
#move types: fast_move_type, charged_move_1_type, charged_move_2_type
#fast_move_opp_type, charged_move_1_opp_type, charged_move_2_opp_type
#attack_effectiveness, defense_effectiveness
#shields_used_self, shields_used_opp

#train the model on the battle dataset
def main(pokemon_name, form_pokemon, is_shadow, IVs, lvl, moves='best', print_out=False):
    sheet = make_battlesheets(pokemon_name, form_pokemon, is_shadow, IVs, lvl, moves, print_out)
    return sheet
    #print(viability)