import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, auc
from scipy.interpolate import interp1d
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

CALC_DIR = "data/processed/app_calcs"
MODEL_DIR = "data/models/"
df = pd.read_csv(CALC_DIR + '/battle.csv')

def train_minimum_level(df,threshold): #regression
    #reframe database to be the difference in stats between lvl 30 and 50
    results = []
    grouped = df.groupby(["name", "IV_atk", "IV_def", "IV_sta"])
    for group_key, group in grouped:
        group = group.sort_values(by="defender_level")
        if group["battle_score"].max() < threshold: min_viable_level = 51 #this pokemon will not be viable regardless of level
        else:
            above_threshold, below_threshold = group[group["battle_score"] >= threshold], group[group["battle_score"] < threshold]
            if len(above_threshold) == 0 or len(below_threshold) == 0: min_viable_level = -1 # interpolation will not work for this pokemon
            else:
                just_below, just_above = below_threshold.iloc[-1], above_threshold.iloc[0]
                x = [just_below["defender_level"], just_above["defender_level"]]
                y = [just_below["battle_score"], just_above["battle_score"]]
                interp_func = interp1d(y, x, fill_value="extrapolate")
                min_viable_level = interp_func(threshold)
                cp_diff = group.iloc[-1]["CP"] - group.iloc[0]["CP"]
                hp_diff = group.iloc[-1]["HP"] - group.iloc[0]["HP"]
                tdo_diff = group.iloc[-1]["TDO"] - group.iloc[0]["TDO"]
                dpt_diff = group.iloc[-1]["DPT"] - group.iloc[0]["DPT"]
                dpe_diff = group.iloc[-1]["DPE"] - group.iloc[0]["DPE"]

                result = {
                    "name": group.iloc[0]["name"],
                    "IV_atk": group.iloc[0]["IV_atk"],
                    "IV_def": group.iloc[0]["IV_def"],
                    "IV_sta": group.iloc[0]["IV_sta"],
                    "CP": cp_diff,
                    "HP": hp_diff,
                    "TDO": tdo_diff,
                    "DPT": dpt_diff,
                    "DPE": dpe_diff,
                    "type_1": group.iloc[0]["type_1"],
                    "type_2": group.iloc[0]["type_2"],
                    "fm_type": group.iloc[0]["fm_type"],
                    "cm1_type": group.iloc[0]["cm1_type"],
                    "cm2_type": group.iloc[0]["cm2_type"],
                    "min_viable_level": min_viable_level
                }

            results.append(result)

    df_minlvl = pd.DataFrame(results)
    df_minlvl.to_csv("data/processed/app_calcs/battle_stats_delta.csv", index=False)
    df_minlvl.drop(columns=['name'],inplace=True)

    X = df_minlvl.drop(columns=["min_viable_level"])
    y = df_minlvl["min_viable_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    
    #------------------------------------------------------------------------
    # Hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 1000, 5000],
        "max_depth": [None, 3, 6, 9],
        "learning_rate": [0.01, 0.25, 0.5],
        "min_child_weight": [2, 4, 8],
        "gamma": [0.1, 0.2, 0.4]
    }

    random_search = RandomizedSearchCV(
            XGBRegressor(random_state=38),
            param_distributions=param_grid,
            n_iter=50, 
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=39, n_jobs=-1  # Parallelize
        )
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error of XGBRegressor: {mse}")
    print(f"r^2 Score: {r2}")
    print("Saving XGBRegressor model as data/models/minlvl_regressor_model.pkl...")
    print('')

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='red')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='green')  # Diagonal line
    plt.xlabel('Actual Minimum Viable Levels')
    plt.ylabel('Predicted Minimum Viable Level')
    plt.title('Actual vs Predicted Level Scatter Plot')
    plt.show()

    joblib.dump(model, MODEL_DIR+'minlvl_regressor_model.pkl')
    
def train_viability(threshold):
    #----------------------------Prepare dataframe for model
    
    #df.drop(columns=['fm_type','cm1_type','cm2_type'],inplace=True)
    #df.drop(columns=['type_1','type_2'],inplace=True)
    
    #update empty entries, make the second type the first type if only one type
    df['type_2'] = df['type_2'].fillna(df['type_1']) 

    #encode labels
    encoder_type_1 = LabelEncoder()
    encoder_type_2 = LabelEncoder()
    encoder_fm = LabelEncoder()
    encoder_cm1 = LabelEncoder()
    encoder_cm2 = LabelEncoder()
    df['type_1'] = encoder_type_1.fit_transform(df['type_1'])
    df['type_2'] = encoder_type_2.fit_transform(df['type_2'])
    df['fm_type'] = encoder_fm.fit_transform(df['fm_type'])
    df['cm1_type'] = encoder_cm1.fit_transform(df['cm1_type'])
    df['cm2_type'] = encoder_cm2.fit_transform(df['cm2_type'])

    train_minimum_level(df,threshold)
    

    #drop unused data
    df.drop(columns=['name','defender_level','battle_score'],inplace=True)

    #preparing the features and the target columns
    X = df.drop(columns=["viable"])  # Features
    y = df["viable"]  # Target variable
    #split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    #first model- Random Forest Classifier
    modelRF = RandomForestClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, n_estimators=1000, random_state=42, class_weight="balanced")
    modelRF.fit(X_train, y_train)
    y_pred = modelRF.predict(X_test)
    RF_acc = accuracy_score(y_test, y_pred)
    RF_precision = precision_score(y_test, y_pred, average='binary')
    RF_recall = recall_score(y_test, y_pred, average='binary')
    print("Accuracy of Random Forest Classifier:", RF_acc)
    print("Precision of Random Forest Classifier:", RF_precision)
    print("Recall of Random Forest Classifier:", RF_recall)
    print("")
    #print(classification_report(y_test, y_pred))

    #second model- Logistic Regression
    modelLR = LogisticRegression(max_iter=5000) #increase iterations because this is a complex model
    modelLR.fit(X_train, y_train)
    y_pred2 = modelLR.predict(X_test)
    LR_acc = accuracy_score(y_test, y_pred2)
    LR_precision = precision_score(y_test, y_pred2, average='binary')
    LR_recall = recall_score(y_test, y_pred2, average='binary')
    print("Accuracy of Logistic Regression:", LR_acc)
    print("Precision of Logistic Regression:", LR_precision)
    print("Recall of Logistic Regression:", LR_recall)
    print("")
    #print(classification_report(y_test, y_pred))

    if LR_acc > RF_acc:
        print("Logistic Regression had highest accuracy, saving as data/models/viability_model.pkl...")
        best_model = modelLR
        y_pred = y_pred2
    else:
        print("Random Forest Classifier had highest accuracy, saving as data/models/viability_model.pkl...")
        best_model = modelRF

    joblib.dump(best_model, MODEL_DIR+'viability_model.pkl')
    joblib.dump(encoder_type_1, MODEL_DIR+'encoder_type_1.pkl')
    joblib.dump(encoder_type_2, MODEL_DIR+'encoder_type_2.pkl')
    joblib.dump(encoder_fm, MODEL_DIR+'encoder_fm.pkl')
    joblib.dump(encoder_cm1, MODEL_DIR+'encoder_cm1.pkl')
    joblib.dump(encoder_cm2, MODEL_DIR+'encoder_cm2.pkl')
    return best_model

def main(threshold):
    train_viability(threshold)