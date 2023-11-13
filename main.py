from flask import Flask, request, jsonify
# test afin  de pouvoir répliquer arima
from AutoArima import TrainAutoArima, PredictAutoArima, GetFeaturesInterpretation
# import utilsTCN
import data_prep
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
NB_PRIX = 5


#On a desoin de plusieurs version de prepare data





@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():
    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    
    # print(Product_features_json)
    # print(Product_quantity_json)
    # print(Product_future_features_json)
    # Create a list to store DataFrames
    
    #Creat a dictionary which contains all information need
    preds = {}
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    
    dfs = []

    # Loop through both lists simultaneously
    for features, quantity in zip(Product_features_json, Product_quantity_json):
        # Convert the features dictionary to DataFrame
        df = pd.DataFrame([features])
        # Add the quantity to the DataFrame
        df['QUANTITE'] = quantity
        dfs.append(df)


    # Concatenate all DataFrames together
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Final_df['DATE_IMPORT'] = pd.to_datetime(final_df['DATE_IMPORT'])
    final_df = final_df.set_index('DATE_IMPORT')
    
    # We then have to see if our date are continuous : 
    
    
    # Convert the received JSON data to a DataFrame
    target = 'QUANTITE'
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]
    
    
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
    
    # x_future['DATE_IMPORT'] = pd.to_datetime(x_future['DATE_IMPORT'])
    x_future =  x_future.set_index('DATE_TMP')

    # forecasting with the last price :
    model = TrainAutoArima(final_df, exogenous, target)
    preds['Prix_0'] = PredictAutoArima(model, x_future, exogenous)
    
    print(preds['Prix_0'])
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
    if coefficients is not None:
        print("Model Coefficients:", coefficients)
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
        preds_converted['ELASTICITE'] = preds['ELASTICITE'].to_dict() if isinstance(preds['ELASTICITE'], pd.Series) else preds['ELASTICITE']

    # Convertir le dictionnaire en JSON
    json_output = json.dumps(preds_converted, indent=4)
    print(json_output)


    print("je suis là !")
    return json_output ,200


if __name__ == '__main__':
    app.run(debug=True ,host='0.0.0.0', port=5000)