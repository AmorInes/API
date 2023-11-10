from flask import Flask, request, jsonify
# test afin  de pouvoir répliquer arima
from AutoArima import TrainAutoArima, PredictAutoArima
# import utilsTCN
import data_prep
import pandas as pd
import numpy as np

app = Flask(__name__)
NB_PRIX = 5


#On a desoin de plusieurs version de prepare data


# @app.route('/api/modelbooper',  methods=['POST'])


# @app.route('/api/modelbooper/prixpermanent', methods=['POST'])
# def api_endpoint():
#     #data = request.json  # Récupère le JSON envoyé
#     #if not data:
#     #    return jsonify({"message": "Pas de données JSON fournies"}), 400
    
#     # Traite les données reçues (ici, on les double simplement)
#     #new_data = { key: value * 2 for key, value in data.items() }
#     new_data = { "key": 2, "value" : 1 ,"user":'JC'}
    
#     # random
    
#     # Renvoie le nouveau JSON
#     return jsonify(new_data), 200


# def train_Arima(nb_jours, y_train, y_test, exogenous, target): 
    
#     model_input = 'arima_auto'

#     if model_input == 'arima_auto':
#         model = pm.auto_arima(y_train['QUANTITE'],seasonal=False,m=12,stepwise=True,trace=True,start_p=0,start_q=0,start_P=0,start_Q=0,max_p=2,max_q=2,maxiter=50000,with_intercept=True,trend='ct')
#     elif model_input == 'arima':
#         model = pm.arima.ARIMA(order=(1,0,1),seasonal=False,m=12,stepwise=True,trace=True,maxiter=6000,with_intercept=True,trend='ct')

#     # Train on x_train, y_train
#     model.fit(y_train[target],X=y_train[exogenous])

#     # Predict on x_test
#     predsautoARIMA = model.predict(n_periods= len(y_test), X=y_test[exogenous])
    
#     return predsautoARIMA 


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

    final_df = final_df.set_index('DATE_IMPORT')
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

    
    # forecasting with the last price :
    model = TrainAutoArima(final_df, exogenous, target)
    preds['last_price'] = PredictAutoArima(model, x_future, exogenous, target)
    
    print(preds['last_price'])
    # some variation test :
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    vec_prix_test = [prix_min + i*(prix_max - prix_min) / (NB_PRIX - 1 ) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # We look if we should set the date as index : 
    x_future = x_future.set_index('DATE_TMP')
    print(x_future) 
    
    # See the price values : 
    name_Quantitefuture = [column for column in x_future.columns if 'PARAM_PRIX' in column]
    print(name_Quantitefuture)
    
    # Change prediction value with the price :
    for prix in vec_prix_test : 
        x_future['PARAM_PRIX']= [prix] * nb_jours
        preds[f'prix_{prix}'] = PredictAutoArima(model, x_future, exogenous, target)
    
    df_prediction = pd.DataFrame(preds)
    
    # Check if the series is not empty (it change)
 
    json_string = df_prediction.to_json()
    print(json_string)  # This will print the JSON representation of your series

    
    # if preds:
    #     json_string = json.dumps(preds.tolist())
    #     print(json_string)  # This will print the JSON representation of your list
    # else:
    #     print("The list is empty!")
        
    # print(f"Data predicted : {preds}")
    # print(request.json['LIST_PARAMETRE'])
    print("je suis là !")
    return json_string ,200


if __name__ == '__main__':
    app.run(debug=True ,host='0.0.0.0', port=5000)