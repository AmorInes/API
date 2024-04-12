import numpy as np
import pandas as pd 
import json
from flask import Flask, request, jsonify
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import xgboost
import lightgbm 
import catboost 
import warnings
warnings.filterwarnings("ignore", category=Warning)

NB_PRIX = 5

class XGBRegressorInt(xgboost.XGBRegressor):
        def predict(self, data):
            _y = super().predict(data)
            return np.asarray(_y, dtype=np.intc)

    



def XgBoostRegressor(X_train, y_train):
    #XGBOOSTRegressor hyperparameters :
    xgb = XGBRegressorInt()
    param_grid = { 
                'objective':['reg:squarederror'],
                'learning_rate' : [0.03,0.05,0.07],
                'reg_lambda': [1],
                'n_estimators' : [1000],
                'max_depth':[3,5,7],
                'min_child_weight':[6,7,8,9],
                'gamma' : [0],
                'colsample_bytree':[0.8],
                #'nthread' : [4],
                'eval_metric' : ['mae'], 
    }


    # finding the best estimator :
    tscv = TimeSeriesSplit(n_splits=5)
    grid_xgb = GridSearchCV(xgb, param_grid, n_jobs=-1, cv=tscv)
    grid_xgb.fit(X_train, y_train)
    model_xgb =  grid_xgb.best_estimator_


    # fitting the model : 
    model_xgb.fit(X_train, y_train)

    return model_xgb


class LightBMRegressorInt(lightgbm.LGBMRegressor):
        def predict(self, data):
            _y = super().predict(data)
            return np.asarray(_y, dtype=np.intc)

def LightBMRegressor(X_train, y_train):
    #LightGBM hyperparameter 
    lgb = LightBMRegressorInt()

    param_grid = {
                'num_leaves': [8,16,32,64],  
                'reg_lambda': [1], 
                'min_child_samples' : [4,6,8],
                'learning_rate': [0.03, 0.05, 0.07], #Comme pour xgboost c'est l'apport d'information après chaque arbres
                'subsample'    : [0.6],
                'max_depth': [3,5,7], #Profondeur maximal des arbres 
                'n_estimators': [1000],
                'force_row_wise': [True]  # Activer l'option force_row_wise pour la perf mémoire
                
        
        }

    #start = time()

    tscv = TimeSeriesSplit(n_splits=5)
    grid_lgb = GridSearchCV(lgb, param_grid,  n_jobs=-1, cv=tscv)
    
    grid_lgb.fit(X_train, y_train)
    
    # fitting the model : 
    model_lgb =  grid_lgb.best_estimator_
    model_lgb.fit(X_train, y_train)

    return model_lgb





def ModelPrediction(model, X_test): 
    # make prediction :
    y_pred= model.predict(X_test)

    y_pred [y_pred < 0] = 0

    return y_pred

def get_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0

def get_feature_importance(model, final_df, target, exogenous, importance_type='gain'):
    """
    Get feature importance of an XGBoost or LightGBM model and calculate elasticity.
    """
    if isinstance(model, XGBRegressorInt):  # Check if the model is XGBoost
        # Get feature importance from XGBoost model
        importance = model.get_booster().get_score(importance_type=importance_type)
        #print(f"L'image a été téléchargée avec succès : {model}")
    elif isinstance(model, LightBMRegressorInt):  # Check if the model is LightGBM
        # Get feature importance from LightGBM model
        importance_score = model.feature_importances_
        feature_names = model.booster_.feature_name()
        importance = {feature_names[i]: importance_score[i] for i in range(len(feature_names))}
        #print(f"L'image a été téléchargée avec succès : {model}")
    else:
        raise ValueError("Model type not supported. Please provide an XGBoost or LightGBM model.")

    # Normalize feature importance
    total_importance = sum(importance.values()) if isinstance(importance, dict) else sum(importance)
    normalized_importance = {k: v / total_importance for k, v in importance.items()} if isinstance(importance, dict) else {exogenous[i]: importance[i] / total_importance for i in range(len(exogenous))}

    # Add a constant to the model (the intercept)
    X = sm.add_constant(final_df[exogenous])

    # Fit the regression model
    regression_model = sm.OLS(final_df[target], X).fit()

    # Calculate elasticity for each feature
    elasticities = {}
    for feature in exogenous:
        if feature in regression_model.params:
            elasticity = (regression_model.params[feature] * final_df[feature].mean()) / final_df[target].mean()
            elasticities[feature] = elasticity

    # Combine feature importance with elasticity
    combined_importance = {feature: round(normalized_importance.get(feature, 0) * get_sign(elasticities.get(feature, 0)) * 100, 2)
                           for feature in exogenous}

    return combined_importance



# def XgBoostget_feature_importance(model, final_df, target, exogenous, importance_type='gain'):
#     """
#     Get feature importance of an XGBoost model and calculate elasticity.
#     """
#     # Get feature importance from XGBoost model
#     importance = model.get_booster().get_score(importance_type=importance_type)

#     # Normalize feature importance
#     total_importance = sum(importance.values())
#     normalized_importance = {k: v / total_importance for k, v in importance.items()}

#     # Add a constant to the model (the intercept)
#     X = sm.add_constant(final_df[exogenous])

#     # Fit the regression model
#     regression_model = sm.OLS(final_df[target], X).fit()

#     # Calculate elasticity for each feature
#     elasticities = {}
#     for feature in exogenous:
#         if feature in regression_model.params:
#             elasticity = (regression_model.params[feature] * final_df[feature].mean()) / final_df[target].mean()
#             elasticities[feature] = elasticity

#     # Combine feature importance with elasticity
#     combined_importance = {feature: round(normalized_importance.get(feature, 0) * get_sign(elasticities.get(feature, 0))  * 100,2)
#                            for feature in exogenous}

#     return combined_importance


def XgBoost_elasticities(X_test, exogenous, model):
    pct_change = 0.1
 
    elasticities = {"variables":[], "elasticity" :[]}
 
    for variable_name in exogenous:
        x1 = X_test[exogenous]
        x2 = X_test[exogenous]

        if "JS_" in variable_name or "VACANCES" in variable_name:
            x1.loc[:, variable_name] = 0
            x2.loc[:, variable_name] = 1
        else:
            x1.loc[:, variable_name] = x1[variable_name] * (1 + pct_change)
            x2.loc[:, variable_name] = x2[variable_name] * (1 - pct_change)
 
        y1 = model.predict(x1)
        y2 = model.predict(x2)
 
        elasticity = np.mean((y1 - y2) / (x1[variable_name] - x2[variable_name]))
 
        elasticities['elasticity'] +=  [elasticity]
        elasticities['variables'] += [variable_name]
 
 
    return elasticities



def split_df(x_final) : 
    nb_jour = min(90,int(len(x_final)*0.3))
    x_train = x_final[:-nb_jour]
    x_test = x_final[-nb_jour:]
    return x_train, x_test


def process_data(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []


    # Loop through both lists simultaneously
    if Product_features_json and Product_quantity_json and len(Product_features_json) == len(Product_quantity_json):
        for features, quantity in zip(Product_features_json, Product_quantity_json):
            try:
                df = pd.DataFrame([features])
                df['QUANTITE'] = quantity
                dfs.append(df)
            except Exception as e:
                print(f"Error processing features: {features}, quantity: {quantity}. Error: {e}")

        if not dfs:
            print("No dataframes were created. Check the input data.")
    else:
        print(f"Input data lists are empty or of different lengths.{len(Product_features_json)}{len(Product_quantity_json)}")

    # Concatenate all DataFrames together

    final_df = pd.concat(dfs, ignore_index=True)

    final_df = final_df.set_index('DATE_IMPORT')
    

    # Convert the received JSON data to a DataFrame : 
    target = 'QUANTITE'
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]

    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    
    final_df[target] = final_df[target].apply(lambda x: 0 if x < 0 else x)


    #Handle NaNs: 
    #fill them with a specific value, like zero: 
    final_df.fillna(0, inplace=True)
    
    
    x_future = pd.DataFrame(Product_future_features_json) 
 
    # The choice of prediction set 
    nb_jours = len(x_future)
    target = 'QUANTITE'


    x_future =  x_future.set_index('DATE_TMP')
    
        # Convert object columns to numeric
    for col in exogenous:
        if x_future[col].dtype == 'object':
            try:
                x_future[col] = pd.to_numeric(x_future[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")

    # Drop any columns that still have object dtype
    final_df = final_df.select_dtypes(exclude=['object'])

    return x_future, final_df, target, nb_jours, exogenous



def process_product(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    X_train, X_test = split_df(final_df)

    # Train and test XGBoost model
    model_xgb = XgBoostRegressor(X_train[exogenous], X_train[target])
    #print(f"best model : {model_xgb}")

    y_pred_xgb = ModelPrediction(model_xgb, X_test[exogenous])

    # mae_xgb = mean_absolute_error(y_pred_xgb, X_test[target])
    # rmse_xgb = np.sqrt(mean_squared_error(y_pred_xgb, X_test[target]))
    error_xgb = float(abs(sum(y_pred_xgb) - sum(X_test[target])))

    # Train and test CatBoost model
    model_Light = LightBMRegressor(X_train[exogenous], X_train[target])
    y_pred_Light = ModelPrediction(model_Light,X_test[exogenous])

    # mae_catboost = mean_absolute_error(y_pred_Light, X_test[target])
    # rmse_catboost = np.sqrt(mean_squared_error(y_pred_Light, X_test[target]))
    error_Light = float(abs(sum(y_pred_Light) - sum(X_test[target])))


    # Choose the best model based on error metric
    errors = {
        'xgboost': error_xgb,
        'lightgbm': error_Light,
    }

    best_model = min(errors, key=errors.get)
    best_error = errors[best_model]



    # print("Error:", errors)
    # print("Best model:", best_model)
    # print("Error:", best_error)


    
    #Retraing the best model on all data 
    if best_model == 'xgboost':
        model = XgBoostRegressor(final_df[exogenous], final_df[target])
    elif best_model == 'lightgbm':
        model = LightBMRegressor(final_df[exogenous], final_df[target])
  
    #print(type(model))

    # #Retraing the best model on all data 
    # model = XgBoostRegressor(final_df[exogenous], final_df[target])


    # Forecasting with the last price
    preds['QUANTITE_AJUSTE'] = ModelPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = ModelPrediction(model, x_future[exogenous])
    
    
    # Some variation test
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'].iloc[-1]

    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # Change prediction values with the price
    cpt = 0
    for prix in vec_prix_test: 
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = ModelPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = ModelPrediction(model, x_future[exogenous])
        
    
    
    # Convert predictions to the desired format
    preds_converted = {}
    for key, values in preds.items():
        if key != 'QUANTITE_AJUSTE' : 
            # Check if 'values' is a list or numpy array
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(x_future.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values

        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = get_feature_importance(model,final_df, target, exogenous)



    # x_train, x_test = split_df(final_df)
    # model_elasticity = XgBoostRegressor(X_train[exogenous], X_train[target])
    # coefficients = XgBoost_elasticities(X_test, exogenous, model_elasticity)


    # preds_converted['MAE']=mae
    # preds_converted['RMSE']=rmse
    preds_converted['ERRORS']=errors
    preds_converted['ERROR']=best_error
    preds_converted['MODEL']=best_model

    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)



    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)



    return json_output







