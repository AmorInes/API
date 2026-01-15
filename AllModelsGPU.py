import numpy as np
import pandas as pd 
import json
from flask import Flask, request, jsonify
from joblib import Parallel, parallel_backend
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import statsmodels.api as sm
import warnings
import ast
from datetime import datetime, timedelta
import time
import optuna
import xgboost as xgb
import lightgbm as lgb
#import cupy  # pour s'assurer que LightGBM/XGBoost utilisent GPU
from functools import partial
import torch  # optionnel, juste pour vérifier GPU
from joblib import Parallel, delayed


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")
# Ignorer les avertissements spécifiques de XGBoost
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.core')
warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
optuna.logging.set_verbosity(optuna.logging.WARNING)





NB_PRIX = 6

#print("CUDA available:", torch.cuda.is_available(), "Num GPUs:", torch.cuda.device_count())

USE_GPU = torch.cuda.is_available()

#XGBOOST
class XGBRegressorInt(xgb.XGBRegressor):
    def predict(self, data):
        return np.asarray(super().predict(data), dtype=np.intc)



def XgBoostRegressor(X_train, y_train, n_trials=20, gpu_id=0):
    """Optimisation Optuna XGBoost sur CPU ou GPU"""

    def objective(trial, gpu_idx):
        param = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if USE_GPU else 'hist',
            'gpu_id': gpu_idx if USE_GPU else -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
            'n_estimators': 1000,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        y_pred = model.predict(X_train)
        return mean_squared_error(y_train, y_pred)

    num_gpu = torch.cuda.device_count() if USE_GPU else 1
    trials_per_gpu = max(1, n_trials // num_gpu)

    def optimize_gpu(gpu_idx):
        study = optuna.create_study(direction='minimize')
        study.optimize(partial(objective, gpu_idx=gpu_idx), n_trials=trials_per_gpu, show_progress_bar=False)
        return study.best_params

    # Lancer Optuna en parallèle sur tous les GPU ou CPU
    best_params_list = Parallel(n_jobs=num_gpu)(delayed(optimize_gpu)(gpu) for gpu in range(num_gpu))

    best_params = best_params_list[0]  # simple pour garder compatible

    # Mettre les bons paramètres GPU/CPU pour l'entraînement final
    if USE_GPU:
        best_params.update({'tree_method': 'gpu_hist', 'gpu_id': gpu_id})
    else:
        best_params.update({'tree_method': 'hist', 'gpu_id': -1})

    model_xgb = XGBRegressorInt(**best_params)
    model_xgb.fit(X_train, y_train)

    return model_xgb, best_params




#LIGHTGBM

class LightBMRegressorInt(lgb.LGBMRegressor):
    def predict(self, data):
        return np.asarray(super().predict(data), dtype=np.intc)

def LightBMRegressor(X_train, y_train, n_trials=20, gpu_id=0):
    """Optimisation Optuna LightGBM sur CPU ou GPU"""

    def objective(trial, gpu_idx):
        param = {
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'n_estimators': 1000,
            'objective': 'regression',
            'device': 'gpu' if USE_GPU else 'cpu',
            'gpu_device_id': gpu_idx if USE_GPU else -1,
            'verbosity': -1
        }
        model = LightBMRegressorInt(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_absolute_error(y_train, preds)

    num_gpu = torch.cuda.device_count() if USE_GPU else 1
    trials_per_gpu = max(1, n_trials // num_gpu)

    def optimize_gpu(gpu_idx):
        study = optuna.create_study(direction='minimize')
        study.optimize(partial(objective, gpu_idx=gpu_idx), n_trials=trials_per_gpu, show_progress_bar=False)
        return study.best_params

    best_params_list = Parallel(n_jobs=num_gpu)(delayed(optimize_gpu)(gpu) for gpu in range(num_gpu))
    best_params = best_params_list[0]

    # Mettre à jour les params finaux selon CPU ou GPU
    if USE_GPU:
        best_params.update({'device': 'gpu', 'gpu_device_id': gpu_id, 'objective': 'regression'})
    else:
        best_params.update({'device': 'cpu', 'objective': 'regression'})

    model_lgb = LightBMRegressorInt(**best_params)
    model_lgb.fit(X_train, y_train)
    
    return model_lgb, best_params






def GETLightBMRegressor(X_train, y_train, best_hyperparam):
    default_params = {
        'random_state': None,
        'verbose': -1
        # Ajoutez d'autres paramètres par défaut si nécessaire
    }
    # Mettre à jour les paramètres par défaut avec ceux fournis
    params = {**default_params, **best_hyperparam}

    #LightGBM hyperparameter 
    lgb = LightBMRegressorInt(**params)
    model_lgb = lgb.fit(X_train, y_train)

    return model_lgb



def GETXgboostRegressor(X_train, y_train, best_hyperparam):
    #LightGBM hyperparameter 
    xgb = XGBRegressorInt(**best_hyperparam)
    model_xgb = xgb.fit(X_train, y_train)

    return model_xgb


def GET_Date_Product(id_produit, id_so, connection):

    with connection.cursor() as cursor:
        query = f"""SELECT MAX(Date_Entrainement) AS Derniere_Date
                        FROM TRAIN_HISTORIC
                        WHERE id_produit = {id_produit} AND id_so = {id_so}"""

        cursor.execute(query)
        last_calc_date = cursor.fetchone()
        if not last_calc_date:
            print("Produit non trouvé.")
            return None

        return last_calc_date[0]
    




def ModelPrediction(model, X_test):
    """
    Prédit les valeurs pour X_test et s'assure que toutes les valeurs négatives sont à 0.
    Compatible XGBoost et LightGBM.
    """
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0
    return y_pred


def get_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


def get_feature_importance(model, final_df, target, exogenous, importance_type='gain'):
    """
    Récupère l'importance des features et calcule les élasticités pour XGBoost ou LightGBM.
    Compatible avec multi-modèles pour ModelChoice.
    """
    # Récupération importance selon type de modèle
    if isinstance(model, XGBRegressorInt):  # XGBoost
        importance = model.get_booster().get_score(importance_type=importance_type)
    elif isinstance(model, LightBMRegressorInt):  # LightGBM
        importance_score = model.feature_importances_
        feature_names = model.booster_.feature_name()
        importance = {feature_names[i]: importance_score[i] for i in range(len(feature_names))}
    else:
        raise ValueError("Model type not supported. Please provide an XGBoost or LightGBM model.")

    # Normalisation importance
    total_importance = sum(importance.values()) if isinstance(importance, dict) else sum(importance)
    normalized_importance = {k: v / total_importance for k, v in importance.items()} \
        if isinstance(importance, dict) else {exogenous[i]: importance[i] / total_importance for i in range(len(exogenous))}

    # Ajout constante pour régression
    X = sm.add_constant(final_df[exogenous])
    regression_model = sm.OLS(final_df[target], X).fit()

    # Calcul des élasticités
    elasticities = {}
    for feature in exogenous:
        if feature in regression_model.params:
            elasticity = (regression_model.params[feature] * final_df[feature].mean()) / final_df[target].mean()
            # règles spécifiques
            if feature == 'PARAM_PRIX' and elasticity > 0:
                elasticity = 0
            if 'PARAM_PROMO' in feature and elasticity < 0:
                elasticity = 0
            if 'PARAM_CONC' in feature and elasticity < 0:
                elasticity = 0
            elasticities[feature] = elasticity

    # Combine importance normalisée et signe de l'élasticité
    combined_importance = {feature: round(normalized_importance.get(feature, 0) * get_sign(elasticities.get(feature, 0)) * 100, 2)
                           for feature in exogenous}

    return combined_importance


def split_df(x_final):
    """
    Split train/test selon les 30% derniers jours ou max 365 jours.
    Compatible avec ModelChoice.
    """
    nb_jour = min(365, int(len(x_final) * 0.3))
    x_train = x_final[:-nb_jour]
    x_test = x_final[-nb_jour:]
    return x_train, x_test





def ModelChoice(final_df, exogenous, target, gpu_id=0):
    import time
    start_time = time.time()

    X_train, X_test = split_df(final_df)
    sum_target = sum(X_test[target])

    print(f"XGB train ")
    model_xgb, best_params_xgb  = XgBoostRegressor(X_train[exogenous], X_train[target], gpu_id=gpu_id)
    y_pred_xgb = ModelPrediction(model_xgb, X_test[exogenous])
    error_xgb = float(abs(sum(y_pred_xgb) - sum(X_test[target])))

    print(f"LGBM train ")
    model_Light, best_params_Light  = LightBMRegressor(X_train[exogenous], X_train[target], gpu_id=gpu_id)
    y_pred_Light = ModelPrediction(model_Light, X_test[exogenous])
    error_Light = float(abs(sum(y_pred_Light) - sum(X_test[target])))

    # Choose the best model based on error metric
    errors = {
        'xgboost': error_xgb,
        'lightgbm': error_Light,
    }

    best_model = min(errors, key=errors.get)
    best_error = errors[best_model]

    # Refit the best model on all data using the best params
    if best_model == 'xgboost':
        print("XGB final")
        if USE_GPU:
            best_params_xgb['tree_method'] = 'gpu_hist'
            best_params_xgb['gpu_id'] = gpu_id
        else:
            best_params_xgb['tree_method'] = 'hist'
            best_params_xgb.pop('gpu_id', None) 
    
        model = GETXgboostRegressor(final_df[exogenous], final_df[target], best_params_xgb)
        best_params = best_params_xgb
        mae = mean_absolute_error(y_pred_xgb, X_test[target])
        rmse = np.sqrt(mean_squared_error(y_pred_xgb, X_test[target]))

        if error_xgb == 0:
            percentage_accuracy_global = 100
        elif sum_target == 0:
            percentage_accuracy_global = None
        else:
            percentage_accuracy_global = max(0, 100 - (error_xgb / sum_target) * 100)
        percentage_accuracy = None

    else:  # lightgbm
        print("LGBM final")
        model = GETLightBMRegressor(final_df[exogenous], final_df[target], best_params_Light)
        best_params = best_params_Light
        mae = mean_absolute_error(y_pred_Light, X_test[target])
        rmse = np.sqrt(mean_squared_error(y_pred_Light, X_test[target]))

        if error_Light == 0:
            percentage_accuracy_global = 100
        elif sum_target == 0:
            percentage_accuracy_global = None
        else:
            percentage_accuracy_global = max(0, 100 - (error_Light / sum_target) * 100)
        percentage_accuracy = None

    print(f"Le temps d'execution est {(time.time()-start_time):.2f} seconds.")

    return best_model, model, best_error, percentage_accuracy, percentage_accuracy_global, best_params, mae, rmse

def useHyperParamII(features_model, param_model):
    features_Model_str = features_model
    Param_Model_str = param_model

    actual_features_list = ast.literal_eval(features_Model_str)
    actual_Param_Model = ast.literal_eval(Param_Model_str)
   
    return actual_features_list, actual_Param_Model


def process_data(Product_parametre_json, Product_features_json, Product_quantity_json, Product_future_features_json):
    dfs = []

    if Product_features_json and Product_quantity_json and len(Product_features_json) == len(Product_quantity_json):
        for features, quantity in zip(Product_features_json, Product_quantity_json):
            try:
                df = pd.DataFrame([features])
                df['QUANTITE'] = quantity
                dfs.append(df)
            except Exception as e:
                print("Error processing features")
        if not dfs:
            print("No dataframes created. Check input data.")
    else:
        print(f"Input data lists are empty or of different lengths: {len(Product_features_json)} {len(Product_quantity_json)}")

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.set_index('DATE_IMPORT')

    target = 'QUANTITE'
    exogenous = [x for x in Product_parametre_json if x != target]

    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    final_df[target] = final_df[target].apply(lambda x: 0 if x < 0 else x)
    final_df.fillna(0, inplace=True)

    x_future = pd.DataFrame(Product_future_features_json)
    nb_jours = len(x_future)
    if nb_jours > 0:
        if 'DATE_TMP' in x_future.columns:
            x_future = x_future.set_index('DATE_TMP')
        for col in exogenous:
            if x_future[col].dtype == 'object':
                x_future[col] = pd.to_numeric(x_future[col], errors='coerce')

    final_df = final_df.select_dtypes(exclude=['object'])

    return x_future, final_df, target, nb_jours, exogenous




def GET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, id_produit, id_so, date_import, model_name, feature_Model, parm_model):
    preds = {}
    list_featureModel, dic_parm_model = useHyperParamII(feature_Model, parm_model)
    final_features = [i for i in list_featureModel if i != 'QUANTITE']

    # TRAIN / RETRAIN MODEL avec best hyperparams trouvés par Optuna
    if model_name == 'xgboost':
        model = GETXgboostRegressor(final_df[final_features], final_df[target], dic_parm_model)
    elif model_name == 'lightgbm':
        model = GETLightBMRegressor(final_df[final_features], final_df[target], dic_parm_model)

    # Forecasting
    preds['QUANTITE_AJUSTE'] = ModelPrediction(model, final_df[final_features])
    preds['QUANTITE_0'] = ModelPrediction(model, x_future[final_features])

    # Variation prix
    last_price = final_df['PARAM_PRIX'].iloc[-1]
    prix_min = last_price * 0.9
    prix_max = last_price * 1.1
    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(1, NB_PRIX)]
    cpt = 0
    for prix in vec_prix_test:
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = ModelPrediction(model, x_future[final_features])
    x_future['PARAM_PRIX'] = [last_price] * nb_jours

    # Variation promo
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    cols_type_promo = [col for col in final_features if col.startswith("PARAM_PROMO_")]
    if cols_type_promo:
        for col in cols_type_promo:
            for promo in vec_promo_test:
                x_future_copy = x_future.copy()
                x_future_copy[col] = [promo] * nb_jours
                preds[f'{col}_POURCENT_{int(promo*100)}'] = ModelPrediction(model, x_future_copy[final_features])
    else:
        for promo in vec_promo_test:
            x_future_copy = x_future.copy()
            x_future_copy['PARAM_PROMO'] = [promo] * nb_jours
            preds[f'PROMO_{int(promo*100)}'] = ModelPrediction(model, x_future_copy[final_features])

    # Convert predictions
    preds_converted = {}
    for key, values in preds.items():
        if isinstance(values, (list, np.ndarray)):
            df_source = x_future if key != 'QUANTITE_AJUSTE' else final_df
            list_dic_values = [{"date": str(date), "value": str(value)} for date, value in zip(df_source.index, values)]
            preds_converted[key] = list_dic_values

    # Feature importance / élasticités
    coefficients = get_feature_importance(model, final_df, target, final_features)
    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)

    return json.dumps(preds_converted)






def SET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, id_produit, id_so, gpu_id):
    preds = {}

    best_model, model, best_error, percentage_accuracy, percentage_accuracy_global, best_param, mae, rmse = ModelChoice(final_df, exogenous, target, gpu_id=gpu_id )
    bestFeatures = exogenous

    # Forecasting
    preds['QUANTITE_AJUSTE'] = ModelPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = ModelPrediction(model, x_future[exogenous])

    last_price = final_df['PARAM_PRIX'].iloc[-1]
    prix_min = last_price * 0.9
    prix_max = last_price * 1.1
    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(1, NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()

    # Variation prix
    cpt = 0
    for prix in vec_prix_test:
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = ModelPrediction(model, x_future[exogenous])
    x_future['PARAM_PRIX'] = [last_price] * nb_jours

    # Variation promo
    for promo in vec_promo_test:
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo*100)}'] = ModelPrediction(model, x_future[exogenous])

    # Convert predictions
    preds_converted = {}
    for key, values in preds.items():
        if isinstance(values, (list, np.ndarray)):
            df_source = x_future if key != 'QUANTITE_AJUSTE' else final_df
            list_dic_values = [{"date": str(date), "value": str(value)} for date, value in zip(df_source.index, values)]
            preds_converted[key] = list_dic_values

    # Feature importance
    coefficients = get_feature_importance(model, final_df, target, exogenous)

    bestHyperParams_str = str(best_param)
    bestFeatures_str = str(list(bestFeatures))

    preds_converted.update({
        "ACCURACY_GLOBAL": percentage_accuracy_global,
        "ACCURACY": percentage_accuracy,
        "MAE_LAST_MONTH": mae,
        "RMSE_TRAIN": rmse,
        "ERROR_GLOBAL": best_error,
        "FEATURES_MODEL": bestFeatures_str,
        "PARAM_MODEL": bestHyperParams_str,
        "MODEL_NAME": best_model,
        "MODEL": best_model,
        "ELASTICITE": coefficients,
        "PRIX_INTERVAL": vec_prix_test,
        "OK": str(1)
    })

    return json.dumps(preds_converted)



def format_date_in_predictions(preds_converted):
    """Helper function to format dates in predictions"""
    for key, values in preds_converted.items():
        if isinstance(values, list) and values and isinstance(values[0], dict) and 'date' in values[0]:
            for pred in values:
                # Convert the date string to datetime then back to desired format
                date_obj = datetime.strptime(pred['date'], '%Y-%m-%d %H:%M:%S')
                pred['date'] = date_obj.strftime('%d/%m/%Y')
    return preds_converted


def GET_process_product_Version_chain(x_future, final_df, target, nb_jours, exogenous, id_produit, id_so, model_list):
    final_df.index = pd.to_datetime(final_df.index,format='%d/%m/%Y')
    
  # Extract lists from model_list DataFrame
    feature_Model_list = model_list['FEATURES_MODEL'].tolist()
    model_name_list =  model_list['MODEL_NAME'].tolist()
    parm_model_list = model_list['PARAM_MODEL'].tolist()
    date_import_list = model_list['DATE_IMPORT'].tolist()
    
    #print(date_import_list[0])
    #print(type(date_import_list[0]))
    # Sort the indices based on dates
    sorted_indices = sorted(range(len(date_import_list)), 
                          key=lambda i: datetime.strptime(date_import_list[i], '%d/%m/%Y'))
    
    # Sort all lists using the sorted indices
    feature_Model_list = [feature_Model_list[i] for i in sorted_indices]
    parm_model_list = [parm_model_list[i] for i in sorted_indices]
    date_import_list = [date_import_list[i] for i in sorted_indices]
    
    # Initialize combined predictions dictionary
    combined_preds = {}
    combined_preds['QUANTITE_AJUSTE'] = []
    combined_preds['QUANTITE_0'] = []
    
    # Initialize price variation predictions
    last_price = final_df['PARAM_PRIX'].iloc[-1]
    prix_min = last_price * 0.9
    prix_max = last_price * 1.1
    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(1, NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    for prix_idx in range(len(vec_prix_test)):
        combined_preds[f'PRIX_{prix_idx + 1}'] = []
    
    for promo in vec_promo_test:
        combined_preds[f'PROMO_{int(promo * 100)}'] = []
    
    # Process each model
    for i in range(len(date_import_list)):
        # Get the date range for this model
        # print(f"Processing Model {i + 1}/{len(date_import_list)}")
        # print(date_import_list[i])


        # print(f"\nOriginal date string: {date_import_list[i]}")
        start_date = datetime.strptime(date_import_list[i], '%d/%m/%Y')
        #print(f"Parsed start_date: {start_date}")


        if i < len(date_import_list) - 1:
            end_date = datetime.strptime(date_import_list[i + 1], '%d/%m/%Y')
        else:
            # For the last model, use the last date in x_future
            end_date = final_df.index[-1]
        #print(f"Parsed end_date: {end_date}")
        # print(f"\nBefore filtering:")
        # print(f"final_df date range: {final_df.index.min()} to {final_df.index.max()}")
        # print(f"final_df shape: {final_df.shape}")

        # Filter data for this date range
        mask = (final_df.index <= start_date)
        period_df = final_df[mask].copy()
        # print(f"\nAfter historical filtering (period_df):")
        # print(f"Date range: {period_df.index.min()} to {period_df.index.max()}")
        # print(f"Shape: {period_df.shape}")
        
        future_mask = (final_df.index >= start_date) & (final_df.index < end_date)
        period_future = final_df[future_mask].copy()
        #print(len(period_future))
        # print(f"\nAfter future filtering (period_future):")
        # print(f"Date range: {period_future.index.min()} to {period_future.index.max()}")
        # print(f"Shape: {period_future.shape}")
        

        # Get predictions for this period
        if len(period_df) > 0 or len(period_future) > 0:
            # print("\nCalling GET_process_product_Version with:")
            # print(f"period_future shape: {period_future.shape}")
            # print(f"period_df shape: {period_df.shape}")
            # print(f"nb_jours: {len(period_future)}")
            # print(f"model_name: {model_name_list[i]}")
            period_preds = GET_process_product_Version(
                period_future, 
                period_df,
                target,
                len(period_future),
                exogenous,
                id_produit,
                id_so,
                start_date,
                model_name_list[i],
                feature_Model_list[i],
                parm_model_list[i]
            )
            
            # Parse predictions and append to combined results
            period_preds = json.loads(period_preds)
            # print("\nPrediction Results:")
            # for key in period_preds:
            #     if isinstance(period_preds[key], list):
            #         print(f"{key}: {len(period_preds[key])} predictions")
            #     else:
            #         print(f"{key}: {type(period_preds[key])}")
            
            # Extend combined predictions
            for key in combined_preds.keys():
                if key in period_preds:
                    original_len = len(combined_preds[key])
                    combined_preds[key].extend(period_preds[key])
                    # print(f"\nExtended {key}: {original_len} -> {len(combined_preds[key])}")


    # Convert combined predictions to the desired format
    preds_converted = {}
    for key, values in combined_preds.items():
        preds_converted[key] = values
    
    # print(preds_converted['QUANTITE_0'])
    # Add additional information
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)

    preds_converted = format_date_in_predictions(preds_converted)

    # Final results summary with new format
    # print("\n" + "="*50)
    # print("Final Results Summary (with formatted dates):")
    # print("="*50)
    # for key in preds_converted:
    #     if isinstance(preds_converted[key], list):
    #         print(f"{key}: {len(preds_converted[key])} total predictions")
    #         if len(preds_converted[key]) > 0:
    #             print(f"First prediction: {preds_converted[key][0]}")
    #             print(f"Last prediction: {preds_converted[key][-1]}")

    # Convert to JSON
    return json.dumps(preds_converted)


