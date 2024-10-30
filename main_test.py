from flask import Flask, request, jsonify
# test afin  de pouvoir répliquer arima
import AllModels
from waitress import serve
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
import pandas as pd
import numpy as np
import json







app = Flask(__name__)




NB_PRIX = 6





def version_SET(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json) : 
    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json)
    results = AllModels.SET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, Product_Id_produit_json, ID_SO_request)
    print(f"je suis là ! Avec le produit {Product_Id_produit_json}")
    return results

def version_GET(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json, Product_ID_TARIF, features_model, parm_model, date_import, model_name) : 

    x_future, final_df, target, nb_jours, exogenous = AllModels.process_data(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json)
    results = AllModels.GET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, Product_Id_produit_json, ID_SO_request, date_import, model_name, features_model, parm_model)
    return results



    
@app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
def receive_data2():
    Product_parametre_json = request.json['LIST_PARAMETRE']
    Product_features_json = request.json['LIST_HISTO']
    Product_quantity_json = request.json['LIST_QUANTITE']
    Product_future_features_json = request.json['LIST_FUTURE']
    Product_Id_produit_json = request.json['ID_PRODUIT']
    So_Id_json = request.json.get('ID_SO', 0)
    Product_ID_TARIF = 0
    features_model = request.json["FEATURES_MODEL"]
    parm_model = request.json["PARAM_MODEL"]
    date_import = request.json["DATE_IMPORT"]
    model_name = request.json["MODEL_NAME"]




    if len(Product_features_json) > 0 and len(Product_quantity_json) > 0 : 
        

        if features_model != "-1" and parm_model != "-1" and date_import != "-1 " and model_name != "-1": 
            print(f"Je charge un ancien modèle pour le produit {Product_Id_produit_json}")
            result =  version_GET(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json, So_Id_json, Product_Id_produit_json, Product_ID_TARIF, features_model, parm_model, date_import, model_name)
            return result,200
        else :  
               
            result =  version_SET(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json, So_Id_json, Product_Id_produit_json)
            return result,200
    
    else : 
        results = {}
        results['OK'] = 0
        json_string = json.dumps(results)
        return json_string , 200
    


@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():
    try : 
        Product_parametre_json = request.json['LIST_PARAMETRE']
        Product_features_json = request.json['LIST_HISTO']
        Product_quantity_json = request.json['LIST_QUANTITE']
        Product_future_features_json = request.json['LIST_FUTURE']
        Product_Id_produit_json = request.json['ID_PRODUIT']
        ID_SO_request = request.json.get('ID_SO', 0)

  

        if not Product_features_json or not Product_quantity_json:
            return jsonify({'OK': 0, 'message': 'Données manquantes'}), 400

        if len(Product_features_json) > 0 and len(Product_quantity_json) > 0:
            result = version_SET(Product_parametre_json, Product_features_json, Product_quantity_json, Product_future_features_json, ID_SO_request, Product_Id_produit_json)
            return result

        else:
            return jsonify({'OK': 0, 'message': 'Les données sont vides après vérification'}), 400

    except IndexError as e:
        return jsonify({'OK': 0, 'message': 'Erreur d\'index : vérifiez les données entrantes'}), 400
    except Exception as e:
        return jsonify({'OK': 0, 'message': str(e)}), 500
    
    


    


 

    
if __name__ == '__main__':
    print("Hello! API BOOPER!")
    app.debug = False
    serve(app, host='0.0.0.0', port=8080, threads=10)
    #with ThreadPoolExecutor(max_workers=10) as executor:
    #    executor.map(app.run(host='0.0.0.0', port=8080))