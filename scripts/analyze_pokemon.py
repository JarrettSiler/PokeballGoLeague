import ast, itertools
from itertools import product
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = "data/processed/internal"
MOVE_EFFECTS = "data/processed/external/PVP_Charged_Move_Effects.csv"
SCALING_FACTOR = 1.07 #used for adjusting the impact of attack defence modifiers play on battle stats

#Script flow
#Step 1: Load Pokémon data (base stats, moves, etc.).
#Step 2: Find the optimal moveset (fast_move and charged_move/s) accounting for STAB and move effects.
#Step 3: Generate IV and level combinations, calculating CP and HP.
#Step 4: Apply the optimal moveset to all IV-level combinations and compute TDO, DPT, and DPE.

class PokemonAnalyzer:
    def __init__(self, pokemon_name, pokemon_form, is_shadow, IVS, LEVELS,
                    importance_variables = [0.4, 0.3, 0.3], print_out=False, print_dir="data/processed/app_calcs", moveset='best'): #assume retrieve best stats if not specified
        self.print_to_csv = print_out
        self.output_dir = print_dir
        self.pokemon_name = pokemon_name
        self.form = pokemon_form

        #get the last evolution
        self.pokemon_evolutions = pd.read_csv(PROCESSED_DATA_DIR + '/pokemon_evolutions.csv')
        while ((self.pokemon_evolutions['pokemon_name'] == self.pokemon_name) & (self.pokemon_evolutions['form'] == self.form)).any():
            row = self.pokemon_evolutions[(self.pokemon_evolutions['pokemon_name'] == self.pokemon_name) & (self.pokemon_evolutions['form'] == self.form)]['evolutions'].apply(ast.literal_eval).iloc[0][0]
            self.pokemon_name, self.form =  row['pokemon_name'], row['form']

        self.is_shadow = is_shadow
        if is_shadow:
            self.shadow_atk_def_mult = [1.2,0.833]
            #print(f"generating data for Shadow {self.form} {self.pokemon_name}")
            self.pkm_data_dir = f'{print_dir}/Shadow_{self.form}_{self.pokemon_name}.csv'
            if self.form == 'Normal':
                self.pkm_data_dir = f'{print_dir}/Shadow_{self.pokemon_name}.csv'
        else:
            self.shadow_atk_def_mult = [1,1]
            #print(f"generating data for {self.form} {self.pokemon_name}")
            self.pkm_data_dir = f'{print_dir}/{self.form}_{self.pokemon_name}.csv'
            if self.form == 'Normal':
                self.pkm_data_dir = f'{print_dir}/{self.pokemon_name}.csv'
        self.importance_variables = importance_variables
        self.LEVELS = LEVELS
        self.IVS = IVS
        self.load_data()
        self.optimal_moveset = self.find_optimal_moveset() if moveset == 'best' else moveset
        self.calculate_ivs_cp_hp()
        self.generate_pokemon_specific_data()

    def load_data(self):
        # Load data for the specified Pokémon and form
        self.base_stats = pd.read_csv(PROCESSED_DATA_DIR + '/base_stats.csv')
        self.base_stats = self.base_stats[(self.base_stats['pokemon_name'] == self.pokemon_name)&(self.base_stats['form'] == self.form)]
        
        self.base_atk, self.base_def, self.base_sta = self.base_stats['base_attack'].values[0], self.base_stats['base_defense'].values[0], self.base_stats['base_stamina'].values[0]
        self.pokemon_moves = pd.read_csv(PROCESSED_DATA_DIR + '/pokemon_moves.csv')
        self.pokemon_moves = self.pokemon_moves[(self.pokemon_moves['pokemon_name'] == self.pokemon_name)&(self.pokemon_moves['form'] == self.form)]
        self.pokemon_moves['fast_moves'] = self.pokemon_moves['fast_moves'].apply(ast.literal_eval)
        self.pokemon_moves['charged_moves'] = self.pokemon_moves['charged_moves'].apply(ast.literal_eval)
        self.pokemon_moves['elite_fast_moves'] = self.pokemon_moves['elite_fast_moves'].apply(ast.literal_eval)
        self.pokemon_moves['elite_charged_moves'] = self.pokemon_moves['elite_charged_moves'].apply(ast.literal_eval)
        # Merge Elite Moves into Regular Moves
        
        self.pokemon_moves['fast_moves'] = self.pokemon_moves.apply(lambda row: [move for move in (row['fast_moves'] + row['elite_fast_moves']) if move not in [None, "", np.nan]],axis=1)
        self.pokemon_moves['charged_moves'] = self.pokemon_moves.apply(lambda row: [move for move in (row['charged_moves'] + row['elite_charged_moves']) if move not in [None, "", np.nan]],axis=1)
        # Create Elite Indicator Columns
        self.pokemon_moves['is_elite_fast'] = self.pokemon_moves['fast_moves'].apply(lambda moves: [move in self.pokemon_moves['elite_fast_moves'].values[0] for move in moves])
        self.pokemon_moves['is_elite_charged'] = self.pokemon_moves['charged_moves'].apply(lambda moves: [move in self.pokemon_moves['elite_charged_moves'].values[0] for move in moves])
        self.move_effects = pd.read_csv(MOVE_EFFECTS)
        self.all_moves = pd.read_csv(PROCESSED_DATA_DIR + '/all_moves.csv')

        self.pokemon_types = pd.read_csv(PROCESSED_DATA_DIR + '/pokemon_types.csv')
        self.cp_multipliers = pd.read_csv(PROCESSED_DATA_DIR + '/cp_multipliers.csv')

    def calculate_ivs_cp_hp(self):
        # Generate all combinations of IVs, levels, and CP
        if isinstance(self.IVS[0], int):  ivs = [tuple(self.IVS)]
        else:  # If IVS contains ranges like [[0:15], [0:15], [0:15]]
            ivs = []
            # Generate all combinations of IVs within the given ranges
            for iv_range in self.IVS:
                ivs.append(list(iv_range))
            ivs = list(itertools.product(*ivs))

        levels = self.LEVELS
        self.ivs_cp_levels = []
        for _, row in self.base_stats.iterrows():
            base_atk, base_def, base_sta = row['base_attack'], row['base_defense'], row['base_stamina']
            
            for iv_atk, iv_def, iv_sta in ivs:
                for level in levels:
                    
                    cpm = self.cp_multipliers.loc[self.cp_multipliers['level'] == level, 'multiplier'].values[0]
                    cp = ((base_atk + iv_atk) * ((base_def + iv_def) ** 0.5) * ((base_sta + iv_sta) ** 0.5) * (cpm ** 2)) / 10 #the CP formula
                    cp = max(10, round(cp)) #min CP is 10
                    hp = np.floor((base_sta + iv_sta) * cpm)
                    
                    self.ivs_cp_levels.append([self.pokemon_name, self.form, level, iv_atk, iv_def, iv_sta, cp, round(cpm,4), int(hp)])

        self.ivs_cp_levels = pd.DataFrame(self.ivs_cp_levels, columns=['pokemon_name', 'form', 'level', 'IV_atk', 'IV_def', 'IV_sta', 'CP', 'cpm', 'HP'])
        self.ivs_cp_levels.drop(['pokemon_name', 'form'], axis=1, inplace=True)
        #self.ivs_cp_levels.to_csv(f"{PROCESSED_DATA_DIR}/ivs_cp_levels.csv", index=False) #debugging

    def find_optimal_moveset(self): #Will be used later
        #THE OPTIMAL MOVESET IS DETEMINED BY WHICH HAS THE BEST COMBINED WEIGHTED TDO, DPT, and DPE ******

        fixed_level = 40  # Use level 40 as a reference for finding our moveset
        cpm = self.cp_multipliers.loc[self.cp_multipliers['level'] == fixed_level, 'multiplier'].values[0]

        # Get all available fast and charged moves
        fast_moves = self.pokemon_moves['fast_moves'].values[0]
        charged_moves = self.pokemon_moves['charged_moves'].values[0]
        all_movesets = []

        # Generate movesets with one charged move
        '''for fast_move, charged_move in product(fast_moves, charged_moves):
            all_movesets.append((fast_move, charged_move))
        '''

        # Generate movesets with two charged moves
        for fast_move, (charged_move1, charged_move2) in product(fast_moves, itertools.combinations(charged_moves, 2)):
            all_movesets.append((fast_move, charged_move1, charged_move2))

        #remove movesets with charged moves that use two of the same typing
        for moveset in all_movesets:
            type1 = self.all_moves.loc[self.all_moves['name'] == moveset[1], 'type'].values[0]
            type2 = self.all_moves.loc[self.all_moves['name'] == moveset[2], 'type'].values[0]
            if type1 == type2:
                all_movesets.remove(moveset)
        #print("Generated movesets:", all_movesets)
        # Find the best moveset based on TDO,DPE,DPT we'll use base stats at first (IVs = 0) and a fixed level/CP (cmp at lvl 40)
        # later we will find the TDO, DPE, DPT for the optimal moveset at all CP and IV combinations
        return max(
                    all_movesets,
                    key=lambda moveset: self.calculate_weighted_sum(moveset, cpm)
                )
        
    def calculate_weighted_sum(self, moveset, cpm):
        # Helper function to calculate the weighted sum for a moveset for finding best moveset
        tdo, dpt, dpe = self.calculate_stats_for_moveset(moveset, 0, 0, 0, cpm)
        # Ensure tdo, dpt, and dpe are scalars
        if isinstance(tdo, pd.Series):
            tdo = tdo.item()  # or tdo.values[0]
        if isinstance(dpt, pd.Series):
            dpt = dpt.item()  # or dpt.values[0]
        if isinstance(dpe, pd.Series):
            dpe = dpe.item()  # or dpe.values[0]
        # Calculate the weighted sum
        weighted_sum = (
            tdo * self.importance_variables[0] +
            dpt * self.importance_variables[1] +
            dpe * self.importance_variables[2]
        )
        #print(f"tdo is {tdo}")
        #print(f"weighted sum is {weighted_sum}")
        return weighted_sum

    def get_moveset_data(self, move_name):
        move_data = self.all_moves[self.all_moves['name'] == move_name].iloc[0]
        return move_data
    
    def calculate_stats_for_moveset(self, moveset, iv_atk, iv_def, iv_sta, cpm):
        fast_move = moveset[0]
        fast_move = self.get_moveset_data(fast_move) #get the data for that move
        pokemon_types = self.pokemon_types[(self.pokemon_types['pokemon_name'] == self.pokemon_name) & (self.pokemon_types['form'] == self.form)]['type'].apply(ast.literal_eval).values[0]
        self.pokemon_typing = pokemon_types
        stamina = (self.base_sta + iv_sta) * cpm

        if len(moveset) == 2:
            # One charged move
            charged_move = moveset[1]
            charged_move = self.get_moveset_data(charged_move) #get the data for that move
            dpt_multiplier, hp_multiplier = self.get_move_effect_multiplier(charged_move)
            dpt = self.calculate_dpt(fast_move, charged_move, pokemon_types, dpt_multiplier, iv_atk, cpm)
            hp = stamina * hp_multiplier
            tdo = self.calculate_tdo(dpt, hp, iv_def, cpm)
            dpe = self.calculate_dpe(charged_move, pokemon_types, iv_atk, cpm)
            
            # Ensure tdo, dpt, and dpe are scalars
            if isinstance(tdo, pd.Series):
                tdo = tdo.item()  # or tdo.values[0]
            if isinstance(dpt, pd.Series):
                dpt = dpt.item()  # or dpt.values[0]
            if isinstance(dpe, pd.Series):
                dpe = dpe.item()  # or dpe.values[0]

            self.fm_type, self.cm1_type= fast_move['type'], charged_move['type']
            return tdo, dpt, dpe
        
        else:
            # Two charged moves (average their TDO and DPT)
            charged_move1, charged_move2 = moveset[1], moveset[2]
            charged_move1 = self.get_moveset_data(charged_move1) #get the data for that move
            charged_move2 = self.get_moveset_data(charged_move2) #get the data for that move
            self.fm_type, self.cm1_type, self.cm2_type = fast_move['type'], charged_move1['type'], charged_move2['type']
            return self.stats_for_two_charged_moves(fast_move, charged_move1, charged_move2, pokemon_types, stamina, iv_atk, iv_def, cpm)

    def stats_for_two_charged_moves(self, fast_move, charged_move1, charged_move2, pokemon_types, stamina, iv_atk, iv_def, cpm):
        # Compute TDO,DPT, and DPE separately for each charged move
        dpt_multiplier1, hp_multiplier = self.get_move_effect_multiplier(charged_move1)
        dpt1 = self.calculate_dpt(fast_move, charged_move1, pokemon_types, dpt_multiplier1, iv_atk, cpm)
        hp = stamina * cpm * hp_multiplier
        tdo1 = self.calculate_tdo(dpt1, hp, iv_def, cpm)
        dpe1 = self.calculate_dpe(charged_move1, pokemon_types, iv_atk, cpm)

        dpt_multiplier2, hp_multiplier = self.get_move_effect_multiplier(charged_move2)
        dpt2 = self.calculate_dpt(fast_move, charged_move2, pokemon_types, dpt_multiplier2, iv_atk, cpm)
        hp = stamina * cpm * hp_multiplier
        tdo2 = self.calculate_tdo(dpt2, hp, iv_def, cpm)
        dpe2 = self.calculate_dpe(charged_move2, pokemon_types, iv_atk, cpm)

        # Return the averaged values
        return round((tdo1+tdo2)/2,2) , round((dpt1+dpt2)/2,2), round((dpe1+dpe2)/2,2)

    def get_move_effect_multiplier(self, charged_move):
        
        # Get the effect of the charged move
        move_effect = self.move_effects[self.move_effects['name'] == charged_move['name']]
        if not move_effect.empty and 'stat_changed' in move_effect.columns:
            effect_chance = 0
            if pd.notna(move_effect['effect_chance'].values[0]): #handle NaN error
                effect_chance = int(float(move_effect['effect_chance'].values[0]))
            
            stat_changed = move_effect['stat_changed'].values[0]
            
            amount = 0
            if pd.notna(move_effect['amount'].values[0]): #handle NaN error
                try: amount = int(float(move_effect['amount'].values[0]))
                except: effect_chance = 0 #data contained error
            
            who = move_effect['target'].values[0]

            delt_self_defense, delt_enemy_defense, delt_self_attack, delt_enemy_attack = 0,0,0,0
            #If the move changes someones defense
            if stat_changed == 'def' and amount != 0:
                if who == 'self': delt_self_defense += effect_chance*amount  
                if who == 'enemy': delt_enemy_defense += effect_chance*amount 
            #If the move lowers the users defense or attack, decrease TDO
            if stat_changed == 'atk' and amount != 0:
                if who == 'self': delt_self_attack += effect_chance*amount  
                if who == 'enemy': delt_enemy_attack += effect_chance*amount 
        
            # Improved multipliers using exponential scaling
            DPT_mult = (SCALING_FACTOR ** delt_self_attack) * (SCALING_FACTOR ** -delt_enemy_defense)
            effective_hp_mult = (SCALING_FACTOR ** delt_self_defense) * (SCALING_FACTOR ** -delt_enemy_attack)
            DPT_mult = max(0.8, min(1.2, DPT_mult))  # Buffs/nerfs stay between 80% and 120%
            effective_hp_mult = max(0.8, min(1.2, effective_hp_mult)) # Buffs/nerfs stay between 80% and 120%

            return DPT_mult, effective_hp_mult
        
        return 1,1 #no effects from moves

    def calculate_stab(self, move_type, pokemon_types):
        # Calculate STAB (Same Type Attack Bonus)
        return 1.2 if move_type in pokemon_types else 1

    def calculate_dpe(self, charged_move, pokemon_types, iv_atk, cpm):
        # Calculate DPE (Damage Per Energy)
        STAB = self.calculate_stab(charged_move['type'], pokemon_types)
    
        # Adjust the attack power based on IVs and level
        attack = (self.base_atk + iv_atk) * cpm  # Adjust attack by IVs and STAB
        damage = charged_move['power'] * (attack/self.base_atk) * STAB * self.shadow_atk_def_mult[0]
        energy_cost = abs(charged_move['energy_delta'])
        return round((damage/energy_cost), 2)

    def calculate_dpt(self, fast_move, charged_move, pokemon_types, effects_multiplier, iv_atk, cpm):
        
        # Calculate DPT (Damage Per Turn)
        fast_stab = self.calculate_stab(fast_move['type'], pokemon_types)
        charged_stab = self.calculate_stab(charged_move['type'], pokemon_types)
        attack = (self.base_atk+iv_atk)*cpm
        fast_damage = fast_stab*(attack/self.base_atk)*fast_move['power']*self.shadow_atk_def_mult[0]
        charged_damage = charged_stab*(attack/self.base_atk)*charged_move['power']*self.shadow_atk_def_mult[0]

        # Number of fast attacks to charge the charged attack
        fatk_per_catk = np.ceil(abs(charged_move['energy_delta']) / fast_move['energy_delta'])
        # Total energy generated and unused energy
        en_gen = fatk_per_catk * fast_move['energy_delta']
        unused_en = en_gen - abs(charged_move['energy_delta'])
        # Time used for the attack sequence
        turns_used = fatk_per_catk * fast_move['turn_duration']
        # Damage from unused energy
        dmg_unused_en = (unused_en / abs(charged_move['energy_delta'])) * charged_damage
        # Total damage
        tot_dmg = (charged_damage + (fast_damage * fatk_per_catk) + dmg_unused_en)
        #print(f'fast per charged: {fatk_per_catk}, unused en {unused_en}, time used {turns_used}, total damage {tot_dmg}')
        # Effective DPT
        #print(effects_multiplier)
        return round((tot_dmg/turns_used)*effects_multiplier, 2)

    def calculate_tdo(self, dpt, hp, iv_def, cpm, opponent_dpt=15): #10 will be a default value until we simulate a battle
        # Calculate TDO (Total Damage Output)
        defense = (self.base_def+iv_def)*(cpm**2)*self.shadow_atk_def_mult[1]
        damage_taken_per_turn = opponent_dpt * (100 / defense)
        ttf = hp / damage_taken_per_turn  # Turns to faint (placeholder for incoming damage)
        return round(dpt * ttf, 2)

    def generate_pokemon_specific_data(self):
        # Calculate TDO, DPT, and DPE for all IV and level combinations using the optimal moveset
        self.pkm_data = self.ivs_cp_levels.copy()
        
        #update all stats based off of CP and IVs
        pkm_stats = self.pkm_data.apply(lambda row: self.calculate_stats_for_moveset(self.optimal_moveset, row['IV_atk'], row['IV_def'], row['IV_sta'], row['cpm']), axis=1)
        self.pkm_data['TDO'], self.pkm_data['DPT'], self.pkm_data['DPE'] = zip(*pkm_stats)
        #give important data
        '''if len(self.optimal_moveset) == 3:
            print(f"best PVP moveset for {self.pokemon_name} ({self.form}) is {self.optimal_moveset[0]} with {self.optimal_moveset[1]} and {self.optimal_moveset[2]}")
        else:
            print(f"best PVP moveset for {self.pokemon_name} ({self.form}) is {self.optimal_moveset[0]} with {self.optimal_moveset[1]}")
        '''
        #removed data that is no longer relevant
        self.pkm_data.drop('cpm', axis=1, inplace=True)
        #add data for ref later
        self.pkm_data['fast_move'], self.pkm_data['first_charged_move'], self.pkm_data['second_charged_move'] = self.optimal_moveset[0], self.optimal_moveset[1], self.optimal_moveset[2]
        self.pkm_data.insert(0, 'is_shadow', self.is_shadow)
        self.pkm_data.insert(1, 'form', self.form)
        self.pkm_data.insert(2, 'pokemon_name', self.pokemon_name)
        # Save data to a file for EDA analysis
        if self.print_to_csv:
            self.pkm_data.to_csv(self.pkm_data_dir, index=False)


def main(pokemon_name, pokemon_form, is_shadow, IVS=[15,15,15], LEVELS=[50], print_out=False, print_dir="data/processed/app_calcs", moveset='best'):
    #these variables are the weighted importance of TDO, DPT, and DPE for
    #finding the optimal moveset in PVP (example: [0.3, 0.3, 0.4] means DPE matters most)

    importance_variables = [0.5, 0.3, 0.2] 
    return PokemonAnalyzer(pokemon_name, pokemon_form, is_shadow, IVS, LEVELS, importance_variables,print_out,print_dir,moveset=moveset)
