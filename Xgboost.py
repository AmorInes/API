import numpy as np
import pandas as pd 
import json
from flask import Flask, request, jsonify
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
import xgboost

NB_PRIX = 5



def XgBoostRegressor(X_train, y_train):
    #XGBOOSTRegressor hyperparameters :
    xgb = xgboost.XGBRegressor()
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

def XgBoostPrediction(model_xgb, X_test): 
    # make prediction :
    y_pred_xgboost = model_xgb.predict(X_test)
    
    # print(y_pred_xgboost)
    return y_pred_xgboost


def XgBoostget_feature_importance(model, importance_type='gain'):
    """
    get feature importance of an XGBoost model.
    
    :param model: The trained XGBoost model.
    :param importance_type: Type of importance measure. One of 'weight', 'gain', or 'cover'.
    """
    # Get feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Create a dataframe for visualization    
    # print(importance)
            
    
    return importance



def process_data_XgBoost(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []

    # print(f'len of the sales quantity {len(Product_quantity_json)}')
    # print(f'len of the feature quantity {len(Product_features_json)}')


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
    # print(f'Original dataset columns {final_df.columns}')


    # final_df['DATE_IMPORT'] = pd.to_datetime(final_df['DATE_IMPORT'], format='%d/%m/%Y')
    final_df = final_df.set_index('DATE_IMPORT')
    
    # We then have to see if our date are continuous : 
    # print(f'index of the dataframe : {final_df.index}')
    
    # Convert the received JSON data to a DataFrame : 
    target = 'QUANTITE'
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]
    # print(f'Used Features : {exogenous}')
    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    
    #Handle NaNs: 
    # final_df.dropna(inplace=True)
    #fill them with a specific value, like zero: 
    final_df.fillna(0, inplace=True)
    
    
    x_future = pd.DataFrame(Product_future_features_json) 
    # print(x_future.columns)
    # The choice of prediction set 
    nb_jours = len(x_future)
    target = 'QUANTITE'

    # print(f'features contained in {x_future.columns}')
    # x_future['DATE_TMP'] = pd.to_datetime(x_future['DATE_TMP'], format='%d/%m/%Y')
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
    
    # print(len(x_future.columns))

    # print(json_output)
    return x_future, final_df, target, nb_jours, exogenous



def process_product_Xgboost(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = XgBoostRegressor(final_df[exogenous], final_df[target])
    print(f"len final_df {len(final_df[exogenous])}")
    print(f"len x_future {len(x_future[exogenous])}")
    
    preds['QUANTITE_AJUSTE'] = XgBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = XgBoostPrediction(model, x_future[exogenous])

    print(f"len preds[QUANTITE_AJUSTE] {len(preds['QUANTITE_AJUSTE'])}")
    print(f"len preds[QUANTITE_0] {len(preds['QUANTITE_0'])}")
    
    
    # Some variation test
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]

    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # Change prediction values with the price
    cpt = 0
    for prix in vec_prix_test: 
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = XgBoostPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = XgBoostPrediction(model, x_future[exogenous])
        
    
    
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
                # print(key)
        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = XgBoostget_feature_importance(model)
    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)

    print(f"La quantite ajuste est de len {len(preds_converted['QUANTITE_AJUSTE'])}")
    

    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)



    return json_output







