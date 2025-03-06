import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
PROCESSED_DATA_DIR = "data/processed/internal"

#pokemon viability in master league is dependant on the battle stats (TDO, DPT, DPE)
#but also on its typing (ie 'grass'), we will not consider type in this EDA, and instead 
#run an EDA to see how IVs (atk,def,sta) and level(mostly CP) impact these battle parameters
# #This will give us a good starting point on what to prioritize when finding a viable pokemon for PVP 

def correlation_matrix(pkm):

    correlation_matrix = pkm.corr()
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Pokémon Stats")
    plt.show()

def feature_importance(pkm):

    y = ["DPT","DPE","TDO"]
    for entry in y:
        #we see in the correlation matrix that level is extremely important,
        #lets analyze the other variables
        X = pkm[["IV_atk", "IV_def", "IV_sta", "IV_tot"]] 
        y = pkm[entry]
        # Train a Random Forest model
        model = RandomForestRegressor()
        model.fit(X, y)

        # Plot feature importance
        importances = model.feature_importances_
        features = X.columns
        plt.barh(features, importances)
        plt.title(f"Importance for Predicting {entry}")
        plt.show()

def plot_pair_plot(pkm, specified_levels):
    x_vars = ['level', 'IV_atk', 'IV_def', 'IV_sta', 'IV_tot']
    y_vars = ['DPT', 'DPE', 'TDO']

    # Filter data for specified levels
    filtered_data = pkm[pkm['level'].isin(specified_levels)]

    # Define filtering conditions
    atk_condition = (filtered_data["IV_def"] == 0) & (filtered_data["IV_sta"] == 0)
    def_condition = (filtered_data["IV_atk"] == 15) & (filtered_data["IV_sta"] == 0)
    sta_condition = (filtered_data["IV_atk"] == 15) & (filtered_data["IV_def"] == 15)

    # Create filtered datasets for each IV type
    atk_filtered = filtered_data[atk_condition & (filtered_data["IV_atk"].between(0, 15))]
    def_filtered = filtered_data[def_condition & (filtered_data["IV_def"].between(0, 15))]
    sta_filtered = filtered_data[sta_condition & (filtered_data["IV_sta"].between(0, 15))]

    # Merge all relevant filtered datasets for general use
    tot_filtered = pd.concat([atk_filtered, def_filtered, sta_filtered])

    # Create a grid of subplots
    fig, axes = plt.subplots(len(y_vars), len(x_vars), figsize=(15, 10))
    fig.suptitle("Pair Plot of IVs and Level vs. Battle Stats", y=1.02)

    # Define colors for different levels
    level_colors = sns.color_palette("husl", len(specified_levels))

    # Plot each combination of x_vars and y_vars
    for i, y_var in enumerate(y_vars):
        for j, x_var in enumerate(x_vars):
            ax = axes[i, j]

            # Choose dataset based on x_var
            if x_var == 'level':
                sns.lineplot(data=pkm, x=x_var, y=y_var, ax=ax, color='blue', marker='o')
            else:
                # Determine the relevant dataset for the IV variable
                if x_var == 'IV_atk':
                    data = atk_filtered
                    color = 'red'
                elif x_var == 'IV_def':
                    data = def_filtered
                    color = 'green'
                elif x_var == 'IV_sta':
                    data = sta_filtered
                    color = 'purple'
                else:
                    data = tot_filtered
                    color = 'black'

                # Scatter plot of data points
                sns.scatterplot(data=data, x=x_var, y=y_var, ax=ax, alpha=0.6, color=color)

                # Connect points **level-wise** using a line
                for idx, level in enumerate(specified_levels):
                    level_data = data[data['level'] == level]
                    if not level_data.empty:
                        sns.lineplot(
                            data=level_data, x=x_var, y=y_var, ax=ax, 
                            color=level_colors[idx], marker='o', linewidth=1.5
                        )
                        # Label the last data point of each level’s line
                        last_row = level_data.iloc[-1]
                        ax.text(last_row[x_var], last_row[y_var], f"L{level}",
                                color=level_colors[idx], fontsize=9, weight='bold')

            # Set labels
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)

    # Remove legend box (no labels for the levels in the legend)
    ax.legend([], [], frameon=False)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def linear_regression_analysis_of_IVs(df, target):

    # Define features (IVs only of interest)
    X = df[['IV_atk', 'IV_def', 'IV_sta', 'IV_tot']] 
    X = sm.add_constant(X)  # Add constant (intercept)
    y = df[target] # Define dependent variable (target)

    model = sm.OLS(y, X).fit()
    print(f"Linear Regression results for {target}:")
    print(model.summary())

    # F-statistic and p-value
    f_stat = model.fvalue
    p_value = model.f_pvalue
    print(f"F-statistic: {f_stat}, p-value: {p_value}")

def main(pokemon_form,pokemon_name):

    pkm_data_dir = f'data/processed/app_calcs/{pokemon_form}_{pokemon_name}.csv'
    pkm_data_dir2 = f'data/processed/app_calcs/{pokemon_name}.csv'

    try:
        df = pd.read_csv(pkm_data_dir)
    except:
        df = pd.read_csv(pkm_data_dir2)
    
    #Features: Level, IVs of attack, defense, stamina, and total IV factor
    #target variables: CP, HP, TDO, DPT, DPE
    #EDA is performed to correlate 

    #add analysis of combined IVs
    df['IV_tot'] = df['IV_atk'] + df['IV_def'] + df['IV_sta']
    specified_levels = [10,20,30,40,50]
    
    correlation_matrix(df)
    plot_pair_plot(df, specified_levels)
    feature_importance(df)
    for battle_stat in ['TDO', 'DPT', 'DPE']:
        linear_regression_analysis_of_IVs(df, battle_stat)
