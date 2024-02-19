from flask import Flask, request, jsonify
import logging
# test afin  de pouvoir répliquer arima
import utilsTCN
from AutoArima import TrainAutoArima, PredictAutoArima, GetFeaturesInterpretation
import AutoArima
import Xgboost
import LightGbm
import CatBoost
import RForest
from waitress import serve
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import data_prep
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
NB_PRIX = 5


#On a desoin de plusieurs version de prepare data



def process_product_Xgboost(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = Xgboost.XgBoostRegressor(final_df[exogenous], final_df[target])
    preds['QUANTITE_AJUSTE'] = Xgboost.XgBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = Xgboost.XgBoostPrediction(model, x_future[exogenous])
    
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
        preds[f'PRIX_{cpt}'] = Xgboost.XgBoostPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = Xgboost.XgBoostPrediction(model, x_future[exogenous])
        
    coefficients = Xgboost.get_feature_importance(model)
    preds['ELASTICITE'] = coefficients
    
    
    # Convert predictions and other data to a format suitable for JSON serialization
    preds_converted = {}
    for key, value in preds.items():
        if isinstance(value, pd.DataFrame):
            # Convert DataFrame to a list of dictionaries (one for each row)
            preds_converted[key] = value.to_dict(orient='records')
        elif isinstance(value, (np.ndarray, pd.Series)):
            # Convert NumPy arrays and pandas Series to lists
            preds_converted[key] = value.tolist()
        else:
            preds_converted[key] = value

    # Convert vec_prix_test to a list if it's a NumPy array
    if isinstance(vec_prix_test, np.ndarray):
        vec_prix_test = vec_prix_test.tolist()

    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted, indent=4)
    
    print(json_output)





    # # Convert predictions to a format suitable for JSON serialization
    # preds_converted = {
    #     key: value.tolist() if isinstance(value, (np.ndarray, pd.Series)) else value
    #     for key, value in preds.items()
    # }

    # # Convert vec_prix_test to a list if it's a NumPy array
    # if isinstance(vec_prix_test, np.ndarray):
    #     vec_prix_test = vec_prix_test.tolist()

    # preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # # Convert the dictionary to JSON
    # json_output = json.dumps(preds_converted, indent=4)
    
    # print(json_output)
    return json_output



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
    final_df.dropna(inplace=True)
    #fill them with a specific value, like zero: 
    final_df.fillna(0, inplace=True)
    
    
    x_future = pd.DataFrame(Product_future_features_json) 
    print(x_future.columns)
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
    
    print(len(x_future.columns))

    # print(json_output)
    return x_future, final_df, target, nb_jours, exogenous




def process_data_TCN(request, Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []
    target = 'QUANTITE'

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

    # # Ensure the index is a DatetimeIndex for resampling
    # if not isinstance(x_future.index, pd.DatetimeIndex):
    #     x_future.index = pd.to_datetime(x_future.index)

    # # Resample the data
    # try:
    #     x_future = x_future.resample('D').first()
    # except Exception as e:
    #     print(f"Error during resampling: {e}")
    
    
    # # Assuming x_future is your existing DataFrame
    # # Step 1: Create a copy of 'DATE_TMP' column
    # x_future[name_date] = pd.to_datetime(x_future[name_date], dayfirst=True, format='%d/%m/%Y')
    # x_future['DATE_TMP_copy'] = x_future[name_date]
    # # Step 2: Set 'DATE_TMP' as the index
    # # Optionally, you can rename the copied column back to 'DATE_TMP'
    # x_future.rename(columns={'DATE_TMP_copy': name_date}, inplace=True)
    
    # x_future = x_future.set_index(name_date, drop = False)
    # x_future = x_future.resample('D').first()
    x_future_scaled = transforme_data(x_future, name_date)
    # print(json_output)
    return x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous
# x_future scaled features version 
# train_en_transformed, val_en_transformed scaled trained and validation version 



@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    

    
    # print(Product_features_json)
    # print(Product_Id_produit_json)
    # print(Product_future_features_json)

    print(f'features {Product_future_features_json}')


    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 
        
        # data processing of parametric processis : 
        # x_future, final_df, target, nb_jours, exogenous = AutoArima.process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
        # # Result of parametric processis : 
        # results = AutoArima.process_product_ARIMA(x_future, pd.DataFrame(Product_features_json), final_df, target, nb_jours, exogenous)
        
        
        # data processing of tree based model : 
        # x_future, final_df, target, nb_jours, exogenous = process_data_XgBoost(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
        # # Result of tree based forcasting :
        # # results = LightGbm.process_product_LightGBM(x_future,final_df,target,nb_jours,exogenous) 
        # results = Xgboost.process_product_Xgboost(x_future,final_df,target,nb_jours,exogenous) 
        # results = RForest.process_product_RandomForest(x_future,final_df,target,nb_jours,exogenous) 
        # results = CatBoost.process_product_CATBoost(x_future, final_df, target, nb_jours, exogenous)
        #print(results['QUANTITE_0'])

        # Create an exemple with Darts Models :
        #x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous = utilsTCN.process_data_TCN(request, Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
        #results = utilsTCN.process_product_TCN(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous)
        x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous = Xgboost.process_data_Darts(request, Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
        print(f'target {target}')
        results =  Xgboost.process_product_Darts_XGBoost(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous)
        print(f'result of our first test {results}')
        elasticite = GetFeaturesInterpretationDarts(nb_pred, nb_in, model_XGboost, train_en_transformed, val_en_transformed, x_future_scaled, x_future, final_df,  target, exogenous)
        print(f'elasticite {elasticite}')
        
        # Creat a 
        # Create a Pool of procedure : (for compiutation)
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     print(f'Received len {len(Product_Id_produit_json)}')
        #     futures = [executor.submit(process_product_ARIMA, x_future, pd.DataFrame(Product_features_json), final_df, target, nb_jours, exogenous) for _ in range(len(Product_Id_produit_json))]
        #     results = [future.result() for future in futures]
        
        # results = list(executor.map(lambda x: process_product_ARIMA(x_future, pd.DataFrame(Product_features_json), final_df, target, nb_jours, exogenous), [Product_Id_produit_json]))
        # print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
        return results, 200
    # else : 


if __name__ == '__main__':
    # In the app section directly
    with ThreadPoolExecutor(max_workers=50) as executor:
        app.run(debug=True ,host='0.0.0.0', port=5000)


    # Might be better for multi processing use : 
    # with ProcessPoolExecutor(max_workers=50) as executor:
    #     executor.map(app.run(debug=True ,host='0.0.0.0', port=5000))
    # # serve(app, host='0.0.0.0', port=5000)
