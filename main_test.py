from flask import Flask, request, jsonify
import logging
# test afin  de pouvoir répliquer arima
import Xgboost
from waitress import serve
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
import data_prep
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
NB_PRIX = 5

def XGBoost_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json):
    x_future, final_df, target, nb_jours, exogenous = Xgboost.process_data_XgBoost(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
    results = Xgboost.process_product_Xgboost(x_future,final_df,target,nb_jours,exogenous) 
    print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
    return results

@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    


    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 
        # Create a list to store DataFrames : 
        result = XGBoost_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) 
        # json_string = json.dumps(results)
        return result,200


    else : 
        results = {}
        results['OK'] = str(0)
        json_string = json.dumps(results)
        return json_string 
 


    
@app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
def receive_data2():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    
    print(f'len list histo {len(Product_features_json)}')
    print(f'len list quantite { len(Product_quantity_json)}')
    
    print('User party')

    if len(Product_features_json) != 0 and len(Product_quantity_json) != 0 : 

        
  

        # with ThreadPoolExecutor(max_workers=1) as executor:
            
        #     futures = executor.submit(XGBoost_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json))
        #     # result = future[0].result() # Wait for the function to complete and get the result
        #     results = [future.result() for future in futures]
        #     print(results)

        # with ThreadPoolExecutor(max_workers=1) as executor:   
        #     print(Product_Id_produit_json)
        #     futures = [executor.submit(XGBoost_direct,
        #                                Product_features_json,
        #                                Product_quantity_json, 
        #                                Product_future_features_json,  
        #                                Product_Id_produit_json)]
        #     wait(futures, return_when=ALL_COMPLETED) 
        #     results = [future.result() for future in futures]
            
        return  XGBoost_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) 

            


    else : 
        results = {}
        results['OK'] = 0
        json_string = json.dumps(results)
        return json_string , 200
    
    
    
    
if __name__ == '__main__':
    # In the app section directly
    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(app.run(debug=True ,host='0.0.0.0', port=80))
