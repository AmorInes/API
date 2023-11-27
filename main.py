from flask import Flask, request, jsonify
import logging

from datetime import datetime
# test afin  de pouvoir répliquer arima
from AutoArima import TrainAutoArima, PredictAutoArima, GetFeaturesInterpretation
# import utilsTCN
from waitress import serve
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import data_prep
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
NB_PRIX = 5


#On a desoin de plusieurs version de prepare data

def process_product_ARIMA(x_future,final_df,target,nb_jours,exogenous, feature_quantite_final, product_id) :

    start_time = datetime.now()
    print(f"Start processing product {product_id} at {start_time}")

    #Creat a dictionary which contains all information need
    preds = {}

    # forecasting with the last price :
    model = TrainAutoArima(final_df, exogenous, target)
    preds['QUANTITE_AJUSTE'] = PredictAutoArima(model, feature_quantite_final, exogenous)
    preds['QUANTITE_0'] = PredictAutoArima(model, x_future, exogenous)
    
    # print(preds['QUANTITE_0'])
    # some variation test :
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]

    vec_prix_test = [prix_min + i*(prix_max - prix_min) / (NB_PRIX - 1 ) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    
    # Change prediction values with the price :
    cpt=0
    for prix in vec_prix_test : 
        cpt+=1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = PredictAutoArima(model, x_future, exogenous)
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 
    # Change prediction values with the promotion value :  
    
    for promo in vec_promo_test : 

        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo*100)}'] = PredictAutoArima(model, x_future, exogenous)
        
    coefficients = GetFeaturesInterpretation(model)
    # if coefficients is not None:
    #     # print("Model Coefficients:", coefficients)
    preds['ELASTICITE'] = coefficients
            
    # df_prediction = pd.DataFrame(preds)
    # print(df_prediction)
    # Check if the series is not empty (it change)

    # json_string = df_prediction.to_json()
    # #print(json_string)  # This will print the JSON representation of your series
    # preds_converted = {key: value.dt.strftime('%Y-%m-%d').tolist() if hasattr(value, 'dt') else value.tolist() for key, value in preds.items()}
    # # preds['ELASTICITE_NAME']
    # print(preds_converted['ELASTICITE'])
    
    # test_json =  json.dumps(preds_converted)
    # print(test_json)
    # return test_json ,200
    
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

    preds_converted['OK'] = 1
    preds_converted['IDProduit'] = product_id

    end_time = datetime.now()
    print(f"Finished processing product {product_id} at {end_time}")


    # Convertir le dictionnaire en JSON
    json_output = json.dumps(preds_converted, indent=4)
    return json_output

def process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
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
    #Inspect Problematic Columns:       
    jour_max_connu = int(len(final_df)*0.70)
    
    
    X_train = final_df[:jour_max_connu]
    X_test = final_df[jour_max_connu:]
    x_future = pd.DataFrame(Product_future_features_json) 
    
    # The choice of prediction set 
    nb_jours = len(x_future)
    target = 'QUANTITE'


    # print(f'features contained in {x_future.columns}')
    # x_future['DATE_TMP'] = pd.to_datetime(x_future['DATE_TMP'], format='%d/%m/%Y')
    x_future =  x_future.set_index('DATE_TMP')
    # print(json_output)
    return x_future, final_df, target, nb_jours, exogenous

@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    

    
    # print(Product_features_json)
    # print(Product_Id_produit_json)
    # print(Product_future_features_json)

    print(f'features len {len(Product_features_json)}')
    print(f'quantity len {len(Product_quantity_json)}')

    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 
        # Create a list to store DataFrames : 


        # x_future, final_df, target, nb_jours, exogenous = process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
        # results = process_product_ARIMA(x_future, final_df, target, nb_jours, exogenous , pd.DataFrame(Product_features_json), Product_Id_produit_json)


        with ThreadPoolExecutor(max_workers=10) as executor:
            # futures = []

            x_future, final_df, target, nb_jours, exogenous = process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
            results = process_product_ARIMA(x_future, final_df, target, nb_jours, exogenous , pd.DataFrame(Product_features_json), Product_Id_produit_json)
            # future = executor.submit(process_product_ARIMA, x_future, final_df, target, nb_jours, exogenous, pd.DataFrame(Product_features_json), Product_Id_produit_json)
            # futures.append(future)
            # results = [future.result() for future in futures]

            print(jsonify(results))
            print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
            return results, 200


        # Create a Pool of procedure : (for compiutation)
        # Parallel processing
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = [executor.submit(process_product_ARIMA, process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json), pd.DataFrame(Product_features_json), Product_Id_produit_json)]
        #     results = [future.result() for future in futures]
        
        # results = list(executor.map(lambda x: process_product_ARIMA(x_future, pd.DataFrame(Product_features_json), final_df, target, nb_jours, exogenous), [Product_Id_produit_json]))

    else : 
        results = {}
        results['OK'] = 0
        json_string = json.dumps(results)
        return json_string , 200




if __name__ == '__main__':
    # In the app section directly
    # with ThreadPoolExecutor(max_workers=50) as executor:
    #     executor.map(app.run(debug=True ,host='0.0.0.0', port=5000))


    # Might be better for multi processing use : 
    # with ProcessPoolExecutor(max_workers=50) as executor:
    #     executor.map(app.run(debug=True ,host='0.0.0.0', port=5000))
    serve(app, host='0.0.0.0', port=5000)
