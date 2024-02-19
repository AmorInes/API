import numpy as np
import pandas as pd 
import json
from flask import Flask, request, jsonify
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, exc
import pymysql

import statsmodels.api as sm
import xgboost

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from darts import TimeSeries
from darts.metrics import mape, mase, rmse, mae, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.dataprocessing.transformers import Scaler
from darts.models import XGBModel
from darts.utils.likelihood_models import GaussianLikelihood,PoissonLikelihood, QuantileRegression
from darts.explainability.shap_explainer import ShapExplainer

import shap

import pickle

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import(
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)

from pytorch_lightning.callbacks import EarlyStopping
import mysql.connector
from mysql.connector import Error


NB_PRIX = 5
MAX_N_EPOCHS = 50
NB_JOUR_PRED = -60 






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


    # fitting the model : (I have to watch it again)
    model_xgb.fit(X_train, y_train)

    return model_xgb

def XgBoostPrediction(model_xgb, X_test): 
    # make prediction :
    y_pred_xgboost = model_xgb.predict(X_test)

    y_pred_xgboost [y_pred_xgboost < 0] = 0

    return y_pred_xgboost


def get_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


def XgBoostget_feature_importance(model, final_df, target, exogenous, importance_type='gain'):
    """
    Get feature importance of an XGBoost model and calculate elasticity.
    """
    # Get feature importance from XGBoost model
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Normalize feature importance
    total_importance = sum(importance.values())
    normalized_importance = {k: v / total_importance for k, v in importance.items()}

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
    combined_importance = {feature: round(normalized_importance.get(feature, 0) * get_sign(elasticities.get(feature, 0))  * 100,2)
                           for feature in exogenous}

    return combined_importance


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






def process_data_XgBoost(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
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



def process_product_Xgboost(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    X_train, X_test = split_df(final_df)

    model_test = XgBoostRegressor(X_train[exogenous], X_train[target])

    y_pred_test = XgBoostPrediction(model_test, X_test[exogenous])

    mae = mean_absolute_error(y_pred_test, X_test[target])
    error = float(abs(sum(y_pred_test) - sum(X_test[target])))

    # Forecasting with the last price
    model = XgBoostRegressor(final_df[exogenous], final_df[target])

    
    preds['QUANTITE_AJUSTE'] = XgBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = XgBoostPrediction(model, x_future[exogenous])
    
    
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

        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = XgBoostget_feature_importance(model,final_df, target, exogenous)



    # x_train, x_test = split_df(final_df)
    # model_elasticity = XgBoostRegressor(x_train[exogenous], x_train[target])
    # coefficients = XgBoost_elasticities(x_test, exogenous, model_elasticity)


    preds_converted['MAE']=mae
    preds_converted['ERROR']=error

    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)



    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)



    return json_output


### Dans cette version on ajoute un test pour vérifier si l'id du produit est déjà dans  la base de donné on ajoute le modèle dans le cas contraire : ###
####### Cette version n'utilise pas les trials comme on pourra le faire en utilisant Optuna. #######

def process_product_XgboostII(x_future, final_df, target, nb_jours, exogenous,id_produit):
    # Create a dictionary which contains all information needed 
    preds = {}

    table_name = 'example'
    id_column_name = 'product_id'


    db_config = {
        'user': 'root',
        'password': 'Booper2014!',
        'host': 'localhost',
        'database': table_name 
    }




    if not check_if_id_exists(id_produit, table_name, id_column_name, db_config) :
        # Forecasting with the last price
        model = XgBoostRegressor(final_df[exogenous], final_df[target])
        save_the_model(model,id_produit,db_config)

    else : 
         
        y_pred,model = model_load(x_future, exogenous, id_produit, db_config)
        print(y_pred)


    
    preds['QUANTITE_AJUSTE'] = XgBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = XgBoostPrediction(model, x_future[exogenous])
    
    
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

        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = XgBoostget_feature_importance(model,final_df, target, exogenous)



    # x_train, x_test = split_df(final_df)
    # model_elasticity = XgBoostRegressor(x_train[exogenous], x_train[target])
    # coefficients = XgBoost_elasticities(x_test, exogenous, model_elasticity)

    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)



    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)



    return json_output



def save_the_model(model_xgb,Product_Id_produit_json,db_config): 
    # Use the id of the product : 
    print(Product_Id_produit_json)
    product_id = Product_Id_produit_json


    serialized_model = pickle.dumps(model_xgb)
    # Database connection parameters


    # Establish connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # SQL query to insert the model with both product_id and model_data
    query = "INSERT INTO models (product_id, model_data) VALUES (%s, %s)"

    # Execute the query
    cursor.execute(query, (product_id, serialized_model))


    # Commit the transaction
    conn.commit()

    cursor.close()
    conn.close()

def model_load(data, exogenous, model_id,db_config) : 
    """
    Load the model if the id exist : 

    :param model_id: Is the product_id if we are in a unique magasin else we have 
    """

    # Establish connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    # SQL query to fetch the model
     # Replace with the actual ID of your model
    query = "SELECT model_data, product_id FROM models WHERE product_id = %s"

    # Execute the query
    cursor.execute(query, (model_id,))


    # Fetch the result
    result = cursor.fetchone()

    # Consume any remaining results to avoid the 'Unread result found' error
    cursor.fetchall()

    serialized_model = result[0] if result else None
    print(type(serialized_model))

    if serialized_model:
        model = pickle.loads(serialized_model)
        print("Model loaded successfully.")
        # Example: Making a prediction (assuming the model is a regressor and X_test is available)
        y_pred = model.predict(data[exogenous])
    else:
        print("Model not found.")


    cursor.close()
    conn.close()


    return y_pred,model 

def check_if_id_exists(id_to_check, table_name, id_column_name, db_config):
    """
    Check if an ID exists in a MySQL database.

    :param id_to_check: The ID to check in the database.
    :param table_name: Name of the table in the database.
    :param id_column_name: Name of the column that contains the ID.
    :param db_config: A dictionary containing database connection parameters.
    :return: True if the ID exists, False otherwise.
    """
    try:
        # Establish connection to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Prepare the query
        query = f"SELECT EXISTS(SELECT 1 FROM models WHERE {id_column_name}  = %s)"
        
        # Execute the query
        cursor.execute(query, (id_to_check,))
        result = cursor.fetchone()
        print(result)

        # Close the connection
        cursor.close()
        conn.close()

        bool_test = result[0] == 1
        print(bool_test)

        # Return True if the ID exists, False otherwise
        return bool_test

    except Error as e:
        print(f"Error: {e}")
        return False



#####################################################################################################################
                    #### Utilisations et ajout d'Optuna dans l'utilisations d'XGBoost ####
                            #### Optimisation avec l'influence des shaps local ####
#####################################################################################################################
    
"""In this part of our work I will try various process of XAI. The main goal in this process is to see if I can select a model because of 
the good explanation espacially because there existe a some problem for compuiting these indicators.  
"""
    
# def get_shap_Model(model_xgb, train_df_smooth_1_histo) : 
#     explainer = shap.Explainer(model_xgb)
#     explainerII = shap.TreeExplainer(model_xgb).shap_interaction_values(train_df_smooth_1_histo[[x for x in train_df_smooth_1_histo.columns if x != 'QUANTITE']]._get_numeric_data())
#     shap_values = explainer(train_df_smooth_1_histo[[x for x in train_df_smooth_1_histo.columns if x != 'QUANTITE']]._get_numeric_data())
#     print(shap_values)



"""To use this function we need a dataFrame with the date as index :"""




# That's the function to creat usable dataset for darts : 
# The orinale version of this come from utilsTCN
def transforme_data(data,name_date): 

    
    print("DataFrame columns before TimeSeries.from_dataframe:", data.columns)
    print("Is 'DATE_IMPORT' in DataFrame columns?", name_date in data.columns)
    
    # Ensure the index is in the expected format
    # if not isinstance(data.index, pd.DatetimeIndex):
    #     # Convert the index to a datetime index if necessary
    #     data.index = pd.to_datetime(data.index)

    # # Group by date (assuming the index is a datetime)
    # grouped_data = data.groupby(data.index).mean()
    
    if name_date == "DATE_IMPORT" : 
        
        
        df_test_produit_day_avg = (
            data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
        )
        

        series_en = fill_missing_values(
            TimeSeries.from_dataframe(
                df_test_produit_day_avg , fill_missing_dates=True, freq="D", time_col= name_date
            ),
            "auto",
        )


        # list(df_smooth_1_histo.index)[nb_train_smooth_1_histo]
        
        print(list(data.index)[int(len(data)*0.75)])

        # scale
        scaler_en = Scaler()
        series_en_transformed = scaler_en.fit_transform(series_en)
        train_en_transformed, val_en_transformed = series_en_transformed.split_after(
            list(data.index)[int(len(data)*0.75)]
        )
        return train_en_transformed, val_en_transformed
    
    if name_date == 'DATE_TMP': 
        df_test_produit_day_avg = (
            data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
        )
    
        series_en = fill_missing_values(
            TimeSeries.from_dataframe(
                df_test_produit_day_avg , fill_missing_dates=True, freq="D", time_col= name_date
            ),
            "auto",
        )
        scaler_en = Scaler()
        series_en_transformed = scaler_en.fit_transform(series_en)
        return series_en_transformed


def rescale_data(predic_non_scaled, data_corrected_test_eval) : 
    
    scaler_skQuantite = MaxAbsScaler()
    scaled_series = scaler_skQuantite.fit_transform(np.array(data_corrected_test_eval['QUANTITE']).reshape(-1,1))

    list_prev = predic_non_scaled['QUANTITE'].values()

    prediction_scaled = [x[0] for x in list_prev]
    prediction_original = scaler_skQuantite.inverse_transform(np.array(prediction_scaled).reshape(-1,1))
    data_for_metrics = np.array(prediction_original).reshape(1,-1)
    
    return data_for_metrics

def rescale_data_total(predic_non_scaled, data_corrected_test_eval) : 
    scaler_skQuantite = MaxAbsScaler()
    scaled_series = scaler_skQuantite.fit_transform(np.array(data_corrected_test_eval).reshape(-1,1))

    list_prev = predic_non_scaled.values()

    prediction_scaled = [x[0] for x in list_prev]
    prediction_original = scaler_skQuantite.inverse_transform(np.array(prediction_scaled).reshape(-1,1))
    data_for_metrics = np.array(prediction_original).reshape(1,-1)
    
    return data_for_metrics


# Metric for the objective function in optuna :  
def compute_rmse(list1, list2):
    # Convert lists to numpy arrays for efficient computation
    

    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Ensure the arrays are of the same length
    if arr1.shape != arr2.shape:
        raise ValueError("Input lists must have the same length!")

    # Compute squared differences
    squared_diffs = (arr1 - arr2) ** 2

    # Mask where either array has a NaN
    valid_values_mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    # Calculate mean of valid squared differences
    mean_squared_diff = np.mean(squared_diffs[valid_values_mask])

    # Compute RMSE
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def process_data_Darts(request, Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []
    target = 'QUANTITE'
   

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
        
        
    final_df = pd.concat(dfs, ignore_index=True)
    target = 'QUANTITE'
    
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]
    # print(f'Used Features : {exogenous}')
    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    

    # Concatenate all DataFrames together
    name_date = 'DATE_IMPORT'

    # print(f'Original dataset columns {final_df.columns}')
    # Example: Checking the DataFrame columns
    print(final_df.columns)  # Check the actual column names
    
    final_df[name_date] = pd.to_datetime(final_df[name_date], dayfirst=True, format='%d/%m/%Y')
    # Assuming x_future is your existing DataFrame
    # Step 2: Set 'DATE_TMP' as the index
    final_df = final_df.set_index(name_date, drop = False)

    

    #Handle NaNs: 
    final_df.dropna(inplace=True)
    #fill them with a specific value, like zero: 
    final_df.asfreq('D', fill_value=0) 
    
    print(final_df.columns)
    
    if name_date not in final_df.columns:
        final_df[name_date] = final_df.index
    
    #Inspect Problematic Columns: 
    # The choice of prediction set     
    train_en_transformed, val_en_transformed = transforme_data(final_df, name_date)
    
    
    
    
    name_date = 'DATE_TMP'
    x_future = pd.DataFrame(Product_future_features_json) 
    print(x_future.columns)
    
    # Convert 'DATE_TMP' to datetime and set as index
    x_future['DATE_TMP'] = pd.to_datetime(x_future['DATE_TMP'], dayfirst=True, format='%d/%m/%Y')
    x_future = x_future.set_index('DATE_TMP', drop = False)

    
       
    for col in exogenous:
        if x_future[col].dtype == 'object':
            try:
                x_future[col] = pd.to_numeric(x_future[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                

    x_future_scaled = transforme_data(x_future, name_date)
    # print(json_output)
    return x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous



def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    

# This function is helping us to compiut the new hyperparameter :
def objective(trial,train_en_transformed,val_en_transformed):
    
    early_stopping_rounds = 10
    
    # Define the hyperparameter search space
    
    param = {
        "objective": trial.suggest_categorical("regression", ["reg:tweedie","reg:absoluteerror","reg:squarederror"]),
        "eval_metric": trial.suggest_categorical("eval_metric",["rmse", "mae"]),
        "booster": trial.suggest_categorical("booster", ["gbtree","dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1e-5, log = True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e-5, log = True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 3, 7)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1e-5, log = True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1e-5, log = True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1e-5, log = True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1e-5, log = True)

    evals_result = {}  # To store evaluation results
    model = XGBModel(
        #lags = 12,
        output_chunk_length = 30,
        lags_past_covariates = 15,
        lags = 15,
        early_stopping_rounds=early_stopping_rounds,
        evals_result = evals_result,
        verbosity = 0,
        **param
    )
    
    # Check for constant validation scores
    last_n = 5  # for example
    if evals_result.get('valid_0') and 'rmse' in evals_result['valid_0'] and len(evals_result['valid_0']['rmse']) >= last_n:
        last_scores = evals_result['valid_0']['rmse'][-last_n:]
        if abs(last_scores[0] - last_scores[last_n])/last_scores[0] < 0.01 :  # all values are the same
            print('on est ici')
            return float('inf')  # return a large value to indicate this configuration is not desirable
    
    # Train the model
    model.fit(series = train_en_transformed['QUANTITE'], past_covariates = train_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']], val_series = val_en_transformed['QUANTITE'][:-90], val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:-90], verbose = False)


    # Validate the model
    forecast = model.predict(n=60, series = val_en_transformed['QUANTITE'][:-60], past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']])  #, series= target[-150:-60], past_covariates = future_covariates[-180:-30], future_covariates = future_covariates[-150:-23])

    error = rmse(val_en_transformed['QUANTITE'][-60:], forecast)
    error_val = np.mean(error)

    return error_val

def create_database(db_name):
    
    try:
        # Connect to the MySQL server
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='JCricklin2023'
        )
        
        # Create a cursor object using the cursor() method
        cursor = connection.cursor()
        
        # Drop database if it already exists and then create it
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database `{db_name}` created successfully.")
        
    except Error as e:
        print(f"Error: '{e}' occurred while creating database.")
    
    finally:
        # Close the cursor and connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


# Make a future forcast :
def model_XGB_prev(model_XgbOptimRMSE, train_en_transformed, val_en_transformed, exogenous, feature_forecast, nb_in, nombrePred):
    
    combined_series = val_en_transformed[exogenous].append(feature_forecast[exogenous])

    # Make predictions
    predictions = model_XgbOptimRMSE.predict(n= nombrePred, series = val_en_transformed['QUANTITE'][nb_in:], past_covariates = combined_series)  #, series= target[-150:-60], past_covariates = future_covariates[-180:-30], future_covariates = future_covariates[-150:-23])
    
    # I may rescal the forecast 
    return predictions


#Il faut que fasse attention à toutes ces parties pour finir mon module Darts : 
def historical_forecast_XGboost(nb_in, nb_out, model_XGboost, final_df, val_en_transformed, train_en_transformed) : 
    # Glue up the training and validation set :
    combined_series = train_en_transformed.append(val_en_transformed)  

    # Make historical forecast :
    histo_reg_XGB  = model_XGboost.historical_forecasts(
    series = combined_series['QUANTITE'],
    past_covariates = combined_series[[x for x in combined_series.columns if x != 'QUANTITE']],
    forecast_horizon = nb_out,  # forecast horizon
    stride = 15,  # generate a forecast at every time step
    retrain = False,  # whether to retrain the model at every step
    verbose = False
    )
    
    rescale_data(histo_reg_XGB, final_df)
    
    return histo_reg_XGB

def create_database_if_not_exists(database_name):
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='JCricklin2023'  # Replace with your MySQL root password
        )
        cursor = connection.cursor()
        # Check if the database already exists
        cursor.execute(f"SHOW DATABASES LIKE '{database_name}'")
        if cursor.fetchone():
            print(f"Database {database_name} already exists.")
        else:
            # Create a new database
            cursor.execute(f"CREATE DATABASE {database_name}")
            print(f"Database {database_name} created successfully.")
    except Error as e:
        print(f"Error: '{e}'")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# That's the function where we are training the model using if it's possible the older studies :
# Il faut que j'ajoute à mon study_name l'id_produit, l'id_so, ainsi que eventuellement la date !!!!!!!!!!!!!!!!!!!!!!!!!  
def creat_and_use_optunaStudies(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous) :
    # Replace 'NouvelleBase' with your desired database name
    create_database_if_not_exists('BaseTestSave')               
    storage_url = "mysql://root:JCricklin2023@localhost/BaseTestSave"
    study_name = f"Optuna_Study_Example_XGBoost"

    try:
        study_XGBoost = optuna.study.load_study(study_name=study_name, storage=storage_url)
        print("Study loaded successfully.")
    except KeyError:
        study_XGBoost  = optuna.create_study(
            storage=storage_url,
            direction="minimize",
            study_name=study_name
        )


    if len(study_XGBoost.trials) < 100 : 
        study_XGBoost.optimize(lambda trial: objective(trial, train_en_transformed, train_en_transformed ), n_trials=100, callbacks=[print_callback])
            # Kill connection after optimization
            # To cancel database connection problems
            # kill_sleeping_processes('example', 'root', 'NewPassword')

    # # Close connection
    # connection.close()
    best_params =  study_XGBoost.best_params
    
    print(f'les meillieurs parametres sont {best_params}')

    # Training a new model with best hyperparameters
    model_XgbOptimRMSE = XGBModel(
        output_chunk_length = 10,
        lags_past_covariates = 15,
        lags=15,
        verbosity = 0,
        **best_params
    )
    
    model_XgbOptimRMSE.fit( series = train_en_transformed['QUANTITE'],
        past_covariates = train_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
        val_series = val_en_transformed['QUANTITE'],
        val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
    verbose=False)
    
    return model_XgbOptimRMSE


def GetFeaturesInterpretationDarts(model_XgbOptimRMSE,final_df, train_en_transformed, val_en_transformed, exogenous, feature_forecast, nb_in, nombrePred): 
    combined_series = val_en_transformed[exogenous].append(feature_forecast[exogenous])
    parameters = model_XgbOptimRMSE.kwargs
    print(parameters)
    if parameters['booster'] == "gdtree" : 
        shap_explainer = ShapExplainer(model_XgbOptimRMSE)
        results = shap_explainer.explain()
        shap_explainer.summary_plot
        
    else :
        model = 
        results = XgBoostget_feature_importance(model,final_df, target, exogenous)
    
    return(results)
    

# Il faut que je change tout les -60 qui on peut lieu d'être pour les hypers paramêtres mais ça reste à vérifier : 
def process_product_Darts_XGBoost(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous) :
    
    print(len(x_future_scaled))
    nb_pred = len(x_future)
    nb_in = 15
    nb_out = 10

    #Creat a dictionary which contains all information needed 
    preds = {}

    # forecasting with the last price :
    model_XGboost = creat_and_use_optunaStudies(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous)
    
    
    preds['QUANTITE_AJUSTE'] = historical_forecast_XGboost(nb_in, nb_out, model_XGboost, final_df, val_en_transformed, train_en_transformed)
    preds['QUANTITE_0'] = model_XGB_prev(model_XGboost, train_en_transformed, val_en_transformed,  exogenous, x_future_scaled, nb_in, nb_pred)
    
    # We are testing the prevision for five differents prices : 
    # Remark in the Deep learning cases we have to scale the feature for each prices. 
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]
    vec_prix_test = [prix_min + i*(prix_max - prix_min) / (NB_PRIX - 1 ) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    
    
    # Change prediction values with the price :
    cpt=0
    for prix in vec_prix_test : 
        cpt+=1
        x_future['PARAM_PRIX'] = [prix] * nb_pred
        #Scaling : 
        x_future_scaled = transforme_data( x_future, 'DATE_TMP')
        preds[f'PRIX_{cpt}'] = model_XGB_prev(model_XGboost, train_en_transformed, val_en_transformed, exogenous, x_future_scaled, nb_in, nb_pred)
        # x_future_scaled = rescale_data_TCN(preds, x_future)
 
 
    # Change prediction values with the promotion value :  
       
    x_future['PARAM_PRIX'] = [last_price] * nb_pred 
   
    for promo in vec_promo_test : 

        x_future['PARAM_PROMO'] = [promo] * nb_pred
        #Scaling : 
        x_future_scaled = transforme_data(x_future, 'DATE_TMP')
        preds[f'PROMO_{int(promo*100)}'] = model_XGB_prev(model_XGboost, train_en_transformed, val_en_transformed, exogenous, x_future_scaled, nb_in, nb_pred)
    
    coefficients = GetFeaturesInterpretationDarts(model_XGboost,final_df, train_en_transformed, val_en_transformed, exogenous, x_future_scaled, nb_in, nb_pred)
    preds['ELASTICITE'] = coefficients

    
    # Convertir chaque série pandas en liste, en incluant les dates
    preds_converted = {
        key: [{"date": str(date), "value": value} for date, value in zip(value.index, value)] 
        if hasattr(value, 'index') else value 
        for key, value in preds.items()
    }

    # Traiter spécifiquement l'élasticité si c'est un dictionnaire ou une série pandas
    if isinstance(preds['ELASTICITE'], (dict, pd.Series)):
        preds_converted['ELASTICITE'] = preds['ELASTICITE'].to_dict()  if isinstance(preds['ELASTICITE'], pd.Series)  else preds['ELASTICITE']

    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # Convertir le dictionnaire en JSON
    json_output = json.dumps(preds_converted, indent=4)
    return json_output