from flask import Flask, request, jsonify
import logging
# test afin  de pouvoir répliquer arima
import AllModels
from waitress import serve
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
import pandas as pd
import numpy as np
import json





app = Flask(__name__)




NB_PRIX = 5

def Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json, So_Id_json):
    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
    results = AllModels.process_product(x_future,final_df,target,nb_jours,exogenous) 
    # mae = json.loads(results).get('MAE')
    errors = json.loads(results).get('ERRORS')
    error = json.loads(results).get('ERROR')
    model = json.loads(results).get('MODEL')
    print(f"Produit {Product_Id_produit_json} dans le magasin {So_Id_json} -- Errors = {errors} -- BestModel = {model} -- BestError = {error} ")
    return results


@app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
def receive_data2():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    So_Id_json = request.json['ID_SO']


    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 
            
        return  Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json, So_Id_json) 

    else : 
        results = {}
        results['OK'] = 0
        json_string = json.dumps(results)
        return json_string , 200
    


@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    So_Id_json = request.json['ID_SO']


    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 
        # Create a list to store DataFrames : 
        result = Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json, So_Id_json) 
        # json_string = json.dumps(results)
        return result,200


    else : 
        results = {}
        results['OK'] = str(0)
        json_string = json.dumps(results)
        return json_string 
 
    
    
    
if __name__ == '__main__':
    # In the app section directly
    app.debug = True
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(app.run(host='0.0.0.0', port=80))
