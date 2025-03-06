import os, ast, joblib, sys, math
import pandas as pd
from pathlib import Path
from scripts.scraper import main as fetch_data
from scripts.process_data import main as process_data
from scripts.analyze_pokemon import main as get_pokemon_data
from scripts.exploratory import main as run_EDA
from scripts.battle_simulation import main as battle
from scripts.battle_simulation import update_ML_sheets as update_Master_LG
from scripts.train_model import main as train_models
#from scripts.train_model import main as train_model
#from scripts.predict import main as predict

#---------------------------------------------------------------------------------
fetch_new_data = False #retrieves new data from the pokemon go database api
recalculate_pokemon_data_files = False #will remove datafiles from the masterleague dataset
rewrite_internal_data = False #rewrites all csv files in /internal from scraped data
EDA = False #runs the EDA on pokemon specified and stored in /data/cache

GEN_BATTLE_DB = False #only do this when the meta changes (for each new MasterLeague_Threats.csv) - updates database for model training
TRAIN_MODELS = False # make new classification and regression models- only run if a model does not exist or parameters are changed
model_dir = 'data/models/'
threshold = 0.25 #has a threshold*100% chance of winning in the current ML meta (keep below 0.35)

#------------------------
PREDICT_USING_MODEL = True
pokemon_name = 'Melmetal'
form_pokemon = 'Normal'
moveset = 'best' #or specific moveset ['Psystrike','Shadow Claw','Ominous Wind'] (not optimized)
is_shadow = False
IVs = [14,14,14]
LEVEL = 40
#------------------------

BATTLE_TEST = False #for testing the battle_simulation.py script (DEBUGGING)
data_dirs = [Path(os.getcwd()) / 'data' / 'processed' / 'raw', 
             Path(os.getcwd()) / 'data' / 'processed' / 'internal',
             Path(os.getcwd()) / 'data' / 'processing'
             ]
poke_dir = Path(os.getcwd()) / 'data' / 'processed' / 'battle_leagues' / 'ML_META'
calc_dir = Path(os.getcwd()) / 'data' / 'processed' / 'app_calcs'
cache_dir = Path(os.getcwd()) / 'data' / 'cache'

#1,0,0,0,45,24,0.31,1.04,0.2
#1,15,15,15,51,26,0.37,1.11,0.21
if fetch_new_data:
    fetch_data()
else:
    data_dirs = data_dirs[1:(len(data_dirs)+1)]

if rewrite_internal_data: #reloads all fetchable data and dependant calcs
    for dir in data_dirs:
        if dir.exists():
            for file in dir.iterdir():
                if file.is_file():
                    file.unlink()
    print("Removed all internal data and ready to re-fetch")

    print("Updating data...")
    process_data()
    print(".. Data up to date")

if recalculate_pokemon_data_files:
    for file in list(poke_dir.iterdir()):
        if file.is_file():
            file.unlink()

if EDA:

    pokemon_info = ["Melmetal","Normal",False] #for testing purposes (bool is shadow)
    CLEAR = False #If you clear it will take a long time to generate the pokemon file again
    if CLEAR:
        for file in cache_dir.iterdir():
            if file.is_file():
                file.unlink()
    IVS = [range(0, 16), range(0, 16), range(0, 16)]
    LEVELS = range(1, 51)
    if CLEAR:
        get_pokemon_data(pokemon_info[0],pokemon_info[1],pokemon_info[2],
                        IVS=IVS,LEVELS=LEVELS,print_dir=cache_dir,print_out=True)
    #TIME ESTIMATES: unknown
    run_EDA(pokemon_info[1],pokemon_info[0])

if BATTLE_TEST:
    for file in list(calc_dir.iterdir()):
        if file.is_file():
            file.unlink()
    update_Master_LG()
    battle(pokemon_name, form_pokemon, is_shadow, IVs, LEVEL, moveset, print_out=True)

if GEN_BATTLE_DB:

    final_columns = ['defender_pokemon_name',
        'defender_level','defender_CP','defender_HP',
        'defender_IV_atk','defender_IV_def','defender_IV_sta',
        'defender_type_1','defender_type_2','defender_fast_move_type',
        'defender_first_charged_move_type','defender_second_charged_move_type',
        'battle_score','viable']
    main_frame= pd.DataFrame(columns=final_columns)

    pokemon_evolutions = pd.read_csv('data/processed/internal/pokemon_evolutions.csv')
    all_pokemon = pd.read_csv('data/processed/internal/pokemon_types.csv', usecols=['pokemon_name','form'])
    pokemon_moves = pd.read_csv('data/processed/internal/pokemon_moves.csv', usecols=['pokemon_name','form','charged_moves'])
    pokemon_max_cp = pd.read_csv('data/processed/internal/pokemon_max_cp.csv')
    list_of_pokemon = []
    ivs_to_include = [range(10,16,5),range(10,16,5),range(10,16,5)] #create IVS that are a variation
    LEVELS = range(30,51,5)

    for i in range(0,all_pokemon['pokemon_name'].count()):
        #only add final evolutions to the pokemon list
        pokemon_name, pokemon_form = all_pokemon['pokemon_name'].values[i], all_pokemon['form'].values[i]
        if pokemon_max_cp[(pokemon_max_cp['pokemon_name'] == all_pokemon['pokemon_name'].values[i])&
                            (pokemon_max_cp['form'] == all_pokemon['form'].values[i])]['max_cp'].values[0]>3200:
            #if not ((pokemon_evolutions['pokemon_name'] == all_pokemon['pokemon_name'].values[i]).any()):#if pokemon is a final evolution
            pokemon_moves_row = pokemon_moves[(pokemon_moves['pokemon_name'] == pokemon_name)&(pokemon_moves['form'] == pokemon_form)]
            pokemon_charged_moves = pokemon_moves_row['charged_moves'].apply(ast.literal_eval).values[0]
            #print(pokemon_charged_moves)
            if len(pokemon_charged_moves) > 1:
                list_of_pokemon.append((all_pokemon['form'].values[i],all_pokemon['pokemon_name'].values[i]))
            #else: print(pokemon_name) #pokemon removed for not having more than one available charged move
    #remove copies
    list_of_pokemon = [
        pkm for pkm in list_of_pokemon 
        if pkm[0] not in ['S', 'Copy_2019'] and pkm[1] not in ['Unown','Spindra', 'Florges', 'Silvally', 'Vivillon', 'Arceus','Sawsbuck']
    ]
    #reduce the number of samples
    #list_of_pokemon = list_of_pokemon[::2]
    dataset_size = len(list_of_pokemon)
    #print(list_of_pokemon) #DEBUGGING

    update_Master_LG()
    for pokemon in list_of_pokemon:
        place = list_of_pokemon.index(pokemon)
        time_estimate_minutes = round((((dataset_size-place)*30)/60),1)
        print(f"Adding pokemon {place}/{dataset_size}, approx {time_estimate_minutes} minutes remaining to completion of database")
        #non_shadows
        battle_data = battle(pokemon[1], pokemon[0], False, ivs_to_include, LEVELS)
        if main_frame is None or main_frame.empty: main_frame = battle_data.copy()
        else: main_frame = pd.concat([main_frame,battle_data], ignore_index=True)
        
        #save progress
        if (place in range(0,dataset_size,5)) or (place == dataset_size):
            print("UPDATING DATABASE FILE battle.csv")
            #update column names for accessibility
            main_frame_renamed = main_frame.rename(columns={'defender_pokemon_name': 'name','defender_IV_atk': 'IV_atk', 'defender_IV_def': 'IV_def',
                   'defender_IV_sta': 'IV_sta', 'defender_CP': 'CP', 'defender_HP': 'HP', #'defender_level': 'level',
                   'defender_TDO': 'TDO', 'defender_DPT': 'DPT', 'defender_DPE': 'DPE',
                   'defender_type_1': 'type_1', 'defender_type_2': 'type_2', 'defender_fast_move_type': 'fm_type',
                   'defender_first_charged_move_type': 'cm1_type',
                   'defender_second_charged_move_type': 'cm2_type',
                   })
            main_frame_renamed.to_csv("data/processed/app_calcs/battle.csv",index=False) #save intermitently
        

if TRAIN_MODELS:
    #add column for viability
    main_frame = pd.read_csv("data/processed/app_calcs/battle.csv")
    main_frame['viable'] = (main_frame['battle_score'] > threshold).astype(int)
    main_frame.to_csv("data/processed/app_calcs/battle.csv",index=False)
    train_models(threshold)

if PREDICT_USING_MODEL:
    viability_model = joblib.load(model_dir+'viability_model.pkl')
    minlvl_regressor_model = joblib.load(model_dir+'minlvl_regressor_model.pkl')
    
    if is_shadow: print(f"Asking the machine if a {IVs} Shadow {pokemon_name} at lvl {LEVEL} is viable in Master League...")
    else: print(f"Asking the machine if a {IVs} {pokemon_name} at lvl {LEVEL} is viable in Master League...")
    print("")

    try: pokemon_data_for_min_lvl = get_pokemon_data(pokemon_name,form_pokemon,is_shadow,IVS=IVs,LEVELS=[30,50],print_out=False)
    except:
        print("Pokemon Name or Type does not exist... Exiting")
        sys.exit()
    #make a database row that matches the regression model's stats delta format 
    dfml = pokemon_data_for_min_lvl.pkm_data[['IV_atk','IV_def','IV_sta','CP','HP','TDO','DPT','DPE']].copy() #simplify name
    result = {
        "IV_atk": dfml.iloc[0]["IV_atk"],
        "IV_def": dfml.iloc[0]["IV_def"],
        "IV_sta": dfml.iloc[0]["IV_sta"],
        "CP": dfml.iloc[1]["CP"] - dfml.iloc[0]["CP"],
        "HP": dfml.iloc[1]["HP"] - dfml.iloc[0]["HP"],
        "TDO": dfml.iloc[1]["TDO"] - dfml.iloc[0]["TDO"],
        "DPT": dfml.iloc[1]["DPT"] - dfml.iloc[0]["DPT"],
        "DPE": dfml.iloc[1]["DPE"] - dfml.iloc[0]["DPE"],
    }
    pokemon_data_df2 = pd.DataFrame([result])
    
    try: pokemon_data = get_pokemon_data(pokemon_name,form_pokemon,is_shadow,IVS=IVs,LEVELS=[LEVEL],print_out=False)
    except:
        print("Pokemon Name or Type does not exist... Exiting")
        sys.exit()
    pokemon_data_df1 = pokemon_data.pkm_data[['IV_atk','IV_def','IV_sta','CP','HP','TDO','DPT','DPE']].copy()
    
    #update pokemon type and move types (df2 is updated at encoder step)
    try: type_2 = pokemon_data.pokemon_typing[1]
    except IndexError: type_2 = pokemon_data.pokemon_typing[0]
    pokemon_data_df1.loc[:, 'type_1'],pokemon_data_df1.loc[:, 'type_2'] = pokemon_data.pokemon_typing[0],type_2
    pokemon_data_df1.loc[:, 'fm_type'],pokemon_data_df1.loc[:, 'cm1_type'],pokemon_data_df1.loc[:, 'cm2_type'] = pokemon_data.fm_type,pokemon_data.cm1_type,pokemon_data.cm2_type

    #print(pokemon_data_df.loc['type_1'].values[0])
    #print(pokemon_data_df.loc['type_2'].values[0])
    
    #encode necessary info
    encoder_type_1 = joblib.load(model_dir+'encoder_type_1.pkl')
    encoder_type_2 = joblib.load(model_dir+'encoder_type_2.pkl')
    encoder_fm = joblib.load(model_dir+'encoder_fm.pkl')
    encoder_cm1 = joblib.load(model_dir+'encoder_cm1.pkl')
    encoder_cm2 = joblib.load(model_dir+'encoder_cm2.pkl')

    pokemon_data_df1['type_1'] = pokemon_data_df2['type_1'] = encoder_type_1.transform(pokemon_data_df1['type_1'])
    pokemon_data_df1['type_2'] = pokemon_data_df2['type_2'] = encoder_type_2.transform(pokemon_data_df1['type_2'])
    pokemon_data_df1['fm_type'] = pokemon_data_df2['fm_type'] = encoder_fm.transform(pokemon_data_df1['fm_type'])
    pokemon_data_df1['cm1_type'] = pokemon_data_df2['cm1_type'] = encoder_cm1.transform(pokemon_data_df1['cm1_type'])
    pokemon_data_df1['cm2_type'] = pokemon_data_df2['cm2_type'] = encoder_cm2.transform(pokemon_data_df1['cm2_type'])

    #if 
    viable = viability_model.predict(pokemon_data_df1)
    viable_lvl = math.ceil(minlvl_regressor_model.predict(pokemon_data_df2)[0]*2)/2
    if LEVEL >= viable_lvl: viable = 1 #bridge the gap between the two models
    if (LEVEL < viable_lvl) and viable == 0: viable = 2
    evolved_pokemon = pokemon_data.pokemon_name
    if is_shadow:
        evolved_pokemon = "Shadow " + evolved_pokemon
        pokemon_name = "Shadow " + pokemon_name

    
    print('--------------------------------------RESULTS--------------------------------------')
    if viable_lvl > 49:
        print(f"      A {pokemon_name} if fully evolved and powered up with these stats will most")
        print(f"  likely fail to hold its weight in Master League battles... Maybe your {pokemon_name}")
        print(f"          could be used to take on Rocket Grunts if trained well...")
    else:
        if pokemon_name != evolved_pokemon:
            print(f"           A {pokemon_name} if evolved into a {evolved_pokemon} at lvl {LEVEL}")
            print(f"            and IVs of {IVs} should have a CP of {pokemon_data.pkm_data['CP'].values[0]}")
            if viable == 1:
                print(f"   ... you could use this {evolved_pokemon} in Master League at these values if evolved!")
                print("")
                print(f"        Acording to my calculations a {evolved_pokemon} at or above level {viable_lvl}")
                print( "             with these IVs should stand a chance in the Master League")
            if viable == 0:
                print("")
                print(f"  ... it might be best not to use a {pokemon_name} at this level ")
                print(f"    in the Master League until it is powered up to level {viable_lvl}!")
                print(f"If you don't have the resources for that you could continue the hunt for a higher IV!")
        else:
            print(f"     A {pokemon_name} at lvl {LEVEL} and IVs of {IVs} should have a CP of {pokemon_data.pkm_data['CP'].values[0]}")
            if viable == 1:
                print(f"  ... you could use this {pokemon_name} in Master League at these values!")
                print("")
                print(f"       Acording to my calculations a {pokemon_name} at or above level {viable_lvl}")
                print("                 should stand a chance in the Master League")
            if viable == 0:
                print(f"     ... it might be best not to use this {pokemon_name} in the Master League until it")
                print(f"  is powered up around level {viable_lvl}! A higher IV pokemon may be viable, however...")
    if viable == 2: 
        print(f" ...this {pokemon_name} could hold potential at it's current power if it is fully evolved")
        print(f"          but it will do better as it gets closer to {viable_lvl}")
    print('-----------------------------------------------------------------------------------')

