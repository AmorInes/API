from flask import Flask, request, jsonify
import AllModelsGPU
from waitress import serve
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.core')
warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch

import os




app = Flask(__name__)

NB_PRIX = 6

#logging.basicConfig(level=logging.INFO)

# Nombre de GPU disponibles
NUM_GPU = torch.cuda.device_count()

USE_GPU = torch.cuda.is_available()


# Détecter le nombre de threads selon CPU/GPU
if USE_GPU:
    MAX_THREADS = NUM_GPU * 2
else:
    MAX_THREADS = min(5, os.cpu_count())  # par défaut 4 ou moins si CPU faible



def process_single_product(product_input, product_idx=0):

     # Assigner le device
    if USE_GPU and product_idx >= 0:
        torch.cuda.set_device(product_idx)
    # sinon CPU, rien à faire pour torch

    #print(USE_GPU)

    Product_parametre_json = product_input['LIST_PARAMETRE']
    Product_features_json = product_input['LIST_HISTO']
    Product_quantity_json = product_input['LIST_QUANTITE']
    Product_future_features_json = product_input.get('LIST_FUTURE', [])
    Product_Id_produit_json = product_input['ID_PRODUIT']
    So_Id_json = product_input.get('ID_SO', 0)
    features_model = product_input.get("FEATURES_MODEL", "-1")
    parm_model = product_input.get("PARAM_MODEL", "-1")
    date_import = product_input.get("DATE_IMPORT", "-1")
    model_name = product_input.get("MODEL_NAME", "-1")

    # Vérifie si ancien modèle
    if features_model != "-1" and parm_model != "-1" and date_import != "-1" and model_name != "-1":
        result = AllModelsGPU.GET_process_product_Version(
            *AllModelsGPU.process_data(Product_parametre_json, Product_features_json, Product_quantity_json, Product_future_features_json),
            Product_Id_produit_json, So_Id_json, date_import, model_name, features_model, parm_model
        )
    else:
        result = AllModelsGPU.SET_process_product_Version(
            *AllModelsGPU.process_data(Product_parametre_json, Product_features_json, Product_quantity_json, Product_future_features_json),
            Product_Id_produit_json, So_Id_json, gpu_id=product_idx if USE_GPU else 0  # utiliser CPU comme GPU 0 fictif
        )

    return {Product_Id_produit_json: json.loads(result)}




# Compteur global pour distribuer les produits sur les GPU
product_counter = 0


@app.route('/api/modelbooper/prixpermanent', methods=['POST'])
def receive_data():
    global product_counter
    try:
        product_input = request.json

        # Choisir le GPU seulement si USE_GPU = True
        gpu_id = product_counter % NUM_GPU if USE_GPU else -1
        product_counter += 1

        # process_single_product doit respecter USE_GPU
        result = process_single_product(product_input, product_idx=gpu_id)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'OK': 0, 'message': str(e)}), 500



# Route pour un produit (user)
@app.route('/api/modelbooper/prixpermanent/user', methods=['POST'])
def receive_data2():
    try:
        product_input = request.json
        # Assigner le GPU 0 par défaut pour un seul produit
        result = process_single_product(product_input, product_idx=0)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'OK': 0, 'message': str(e)}), 500



if __name__ == '__main__':
    #print(MAX_THREADS)
    if USE_GPU:
        print("Hello! API BOOPER GPU !")
    else:
        print("Hello! API BOOPER CPU !")
    
    app.debug = False
    serve(app, host='0.0.0.0', port=8080, threads=MAX_THREADS)
