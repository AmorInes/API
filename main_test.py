from flask import Flask, request, jsonify
# test afin  de pouvoir répliquer arima
import AllModels
from waitress import serve
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
import pandas as pd
import numpy as np
import json
import torch
import multiprocessing




app = Flask(__name__)




NB_PRIX = 6

def Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json):
    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
    results = AllModels.process_product(x_future,final_df,target,nb_jours,exogenous) 
    # mae = json.loads(results).get('MAE')
    errors = json.loads(results).get('ERRORS')
    error = json.loads(results).get('ERROR')
    # com_eroor = json.loads(results).get('COM_ERROR')

    model = json.loads(results).get('MODEL')
    print(f"Produit {Product_Id_produit_json} -- Errors = {errors} -- BestModel = {model} -- BestError = {error}")
    return results

def version_SET(Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json) : 

    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
    # print(f"L{Product_Id_produit_json}")
    results = AllModels.SET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, Product_Id_produit_json, ID_SO_request)
    print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
    return results

def version_GET(Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json, Product_ID_TARIF, features_model, parm_model, date_import, model_name) : 

    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json)
    #print(Product_Id_produit_json)
    results = AllModels.GET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, Product_Id_produit_json, ID_SO_request, date_import, model_name, features_model, parm_model)
    # print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
    return results


# @app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
# def receive_data2():

#     Product_features_json = request.json['LIST_HISTO']
#     Product_quantity_json = request.json['LIST_QUANTITE']
#     Product_future_features_json = request.json['LIST_FUTURE']
#     Product_Id_produit_json = request.json['ID_PRODUIT']
#     # So_Id_json = request.json['ID_SO']
#     # Tarif_Id_json = request.json['ID_TARIF']



#     if len(Product_features_json) >6 and len(Product_quantity_json) >6 : 
            
#         return  Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) 

#     else : 
#         results = {}
#         results['OK'] = 0
#         json_string = json.dumps(results)
#         return json_string , 200
    
@app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
def receive_data2():

    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    So_Id_json = request.json.get('ID_SO', 0)
    date_import = request.json['DATE_IMPORT']
    Product_ID_TARIF = 0
    features_model = request.json["FEATURES_MODEL"]
    parm_model = request.json["PARAM_MODEL"]
    date_import = request.json["DATE_IMPORT"]
    model_name = request.json["MODEL_NAME"]

    print(len(Product_features_json))

    if len(Product_features_json) > 0 and len(Product_quantity_json) > 0 : 
        # user = request.json['user']
        # password = request.json['password']
        # host = request.json['host']
        # port = request.json['port']
        # service_name = request.json['service_name']

        if features_model != "-1" and parm_model != "-1" and date_import != "-1 " and model_name != "-1": 
            print(f"Je charge un ancien modèle pour le produit {Product_Id_produit_json}")
            result =  version_GET(Product_features_json, Product_quantity_json, Product_future_features_json, So_Id_json, Product_Id_produit_json, Product_ID_TARIF, features_model, parm_model, date_import, model_name)
            return result,200
        else :  
            #params = oracledb.ConnectParams(user="SYSTEM_U", password="SYSTEM_U1",host="51.68.86.221", port= 1521, service_name="ONE")
            # with oracledb.connect( params = params) as connection:   
            result =  version_SET(Product_features_json, Product_quantity_json, Product_future_features_json, So_Id_json, Product_Id_produit_json)
            # result =  XGBoost_optuna_version(Product_features_json, Product_quantity_json, Product_future_features_json, So_Id_json, Product_Id_produit_json)
            return result,200
    
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
    ID_SO_request = request.json.get('ID_SO', 0)
    print(f"J'ai le produit {Product_Id_produit_json}")
    # date_import = request.json['DATE_IMPORT']

    if len(Product_features_json) > 0 and len(Product_quantity_json) > 0 : 
        # user = request.json['user']
        # password = request.json['password']
        # host = request.json['host']
        # port = request.json['port']
        # service_name = request.json['service_name']
        # params = oracledb.ConnectParams(user = user, password = password, host = host, port = port, service_name= service_name)
        # dsn = cx_Oracle.makedsn("51.68.86.221", 1521, service_name="ONE")

        # with oracledb.connect( params = params) as connection:

       
    
        result = version_SET(Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json)

        return result,200


    else : 
        results = {}
        results['OK'] = str(0)
        json_string = json.dumps(results)
        return json_string, 200 
    
    

# @app.route('/api/modelbooper/prixpermanent', methods=['POST'])
# def receive_data():

#     Product_features_json = request.json['LIST_HISTO']
#     Product_quantity_json = request.json['LIST_QUANTITE']
#     Product_future_features_json = request.json['LIST_FUTURE']
#     Product_Id_produit_json = request.json['ID_PRODUIT']
#     # So_Id_json = request.json['ID_SO']


#     if len(Product_features_json) > 6 and len(Product_quantity_json) > 6 : 
#         # Create a list to store DataFrames : 

#         result = Model_direct(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) 

#         # json_string = json.dumps(results)
#         return result,200


#     else : 
#         results = {}
#         results['OK'] = str(0)
#         json_string = json.dumps(results)
#         return json_string 
 

    
if __name__ == '__main__':
    # In the app section directly


    # app.debug = False
    #app.run(host='0.0.0.0', port=5000)

    # with ThreadPoolExecutor(max_workers=15) as executor:
    #     executor.map(app.run(host='0.0.0.0', port=5000))

    #extractor = parallelTestModule.ParallelExtractor()
    #extractor.runInParallel()
    multiprocessing.freeze_support()

    #Il sera peut être bon de modifier de nouveaux
    #Ca ne marche pas


    app.run(host='0.0.0.0', port=5000)

    

    #with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    #    futures = [executor.submit(app.run(host='0.0.0.0', port=5000))]
    #    str_lst=[]
    #    for future in concurrent.futures.as_completed(futures):
    #        out_text = ""
            

#     with ThreadPoolExecutor(max_workers=50) as executor:
#         executor.map(app.run(host='0.0.0.0', port=80))
