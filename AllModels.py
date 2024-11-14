import numpy as np
import pandas as pd 
import json
from flask import Flask, request, jsonify
from joblib import Parallel, parallel_backend
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import statsmodels.api as sm
import xgboost
import lightgbm 
import warnings
import ast
from datetime import datetime, timedelta
import time
from joblib import parallel_backend
import joblib 
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV





warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")
# Ignorer les avertissements spécifiques de XGBoost
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.core')


NB_PRIX = 6





# def XgBoostRegressor(X_train, y_train):

#     xgb = XGBRegressorInt()
#     param_dist = { 
#         'objective': ['reg:squarederror'],
#         'learning_rate': [0.03],
#         'reg_lambda': [1],
#         'n_estimators': [1000],
#         'max_depth': [3, 5, 7],
#         'min_child_weight': [6, 7, 8, 9],
#         'gamma': [0],
#         'colsample_bytree': [0.8],
#     }

#     n_folds = min(3, X_train.shape[0])
#     tscv = TimeSeriesSplit(n_splits=n_folds)

#     random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=tscv, n_jobs=2)
    
#     max_time = 60  # Augmenté à 60 secondes
#     start_time = time.time()

#     try:
#         with parallel_backend('loky', n_jobs=2, inner_max_num_threads=1):
#             with joblib.parallel.parallel_memory_limit(max_memory=1024 * 1024 * 1024):  # 1GB
#                 random_search.fit(X_train, y_train)
#     except Exception as e:
#         print(f"An error occurred during fitting: {str(e)}")
#         return None, None

#     if (time.time() - start_time) >= max_time:
#         print(f"RandomizedSearchCV XGB fitting timed out after {max_time} seconds.")
#         return None, None

#     model_xgb = random_search.best_estimator_
#     model_xgb.fit(X_train, y_train)

#     print(f'Ceci est mon modèle xgboost {model_xgb}')

#     return model_xgb, random_search.best_params_
n_jobs  = -1
class XGBRegressorInt(xgboost.XGBRegressor):
        def predict(self, data):
            _y = super().predict(data)
            return np.asarray(_y, dtype=np.intc)


def XgBoostRegressor(X_train, y_train):
   
    #XGBOOSTRegressor hyperparameters :
    xgb = XGBRegressorInt()
    param_grid = { 
                'objective':['reg:squarederror'],
                # 'learning_rate' : [0.03,0.05,0.07],
                'learning_rate' : [0.03],
                'reg_lambda': [1],
                'n_estimators' : [1000],
                'max_depth':[3,5,7],
                'min_child_weight':[6,7,8,9],
                'gamma' : [0],
                'colsample_bytree':[0.8],
                #'nthread' : [4],
                # 'eval_metric' : ['mae'], 
    }


    # finding the best estimator :
    # start_time = datetime.now()

    n_folds = 3
   

    # Assurer que le nombre de plis n'est pas supérieur au nombre d'échantillons
    if n_folds > X_train.shape[0]:
        raise ValueError(f"Le nombre de plis ({n_folds}) est supérieur au nombre d'échantillons ({X_train.shape[0]})")
    
    tscv = TimeSeriesSplit(n_splits=n_folds)

    grid_xgb = DaskGridSearchCV(xgb, param_grid, n_jobs=n_jobs, cv=tscv) 
    
    # grid_xgb.fit(X_train, y_train)
    # Set the maximum execution time in seconds
    max_time = 20

    start_time = time.time()

    while (time.time() - start_time) < max_time:
        try:
            
            grid_xgb.fit(X_train, y_train)
            break  # Exit the loop if fit completes within the time limit
        except TimeoutError:
           pass

    # Check if fitting completed within the time limit
    if (time.time() - start_time) >= max_time:
        print("GridSearchCV XGB fitting timed out after", max_time, "seconds.")



    model_xgb =  grid_xgb.best_estimator_
    # fitting the model : 
    model_xgb.fit(X_train, y_train)


    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print("Temps écoulé XBG:", elapsed_time)

    return model_xgb, grid_xgb.best_params_


class LightBMRegressorInt(lightgbm.LGBMRegressor):

    def predict(self, data):
        _y = super().predict(data)
        return np.asarray(_y, dtype=np.intc)

def LightBMRegressor(X_train, y_train):
    
    #LightGBM hyperparameter 
    lgb = LightBMRegressorInt()

    param_grid = {
                'num_leaves': [8,16,32,64],  
                'reg_lambda': [1], 
                'min_child_samples' : [4,6,8],
                # 'learning_rate' : [0.03,0.05,0.07],
                'learning_rate': [0.03], #Comme pour xgboost c'est l'apport d'information après chaque arbres
                'subsample'    : [0.6],
                'max_depth': [3,5,7], #Profondeur maximal des arbres 
                'n_estimators': [1000],
                'force_row_wise': [True],  # Activer l'option force_row_wise pour la perf mémoire
                'verbose' : [-1]
        
        }

    # start_time = datetime.now()

    n_folds = 3


    # Assurer que le nombre de plis n'est pas supérieur au nombre d'échantillons
    if n_folds > X_train.shape[0]:
        raise ValueError(f"Le nombre de plis ({n_folds}) est supérieur au nombre d'échantillons ({X_train.shape[0]})")
    
    
    tscv = TimeSeriesSplit(n_splits=3)
    grid_lgb = DaskGridSearchCV(lgb, param_grid,  n_jobs=n_jobs, cv=tscv)
    
    # grid_lgb.fit(X_train, y_train)
    # Set the maximum execution time in seconds
    max_time = 20

    start_time = time.time()
    while (time.time() - start_time) < max_time:
        try:
            
            grid_lgb.fit(X_train, y_train)
            break  # Exit the loop if fit completes within the time limit
        except TimeoutError:
            pass  # Handle timeout gracefully (optional)
    # Check if fitting completed within the time limit
    if (time.time() - start_time) >= max_time:
        print("GridSearchCV LGBM fitting timed out after", max_time, "seconds.")

    # fitting the model : 
    
    model_lgb =  grid_lgb.best_estimator_
    
    model_lgb.fit(X_train, y_train)
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print("Temps écoulé LBM:", elapsed_time)

    return model_lgb, grid_lgb.best_params_



# def LightBMRegressor(X_train, y_train):
#     #LightGBM hyperparameter 
#     lgb = LightBMRegressorInt()

#     param_grid = {
#                 'num_leaves': [8,16,32,64],  
#                 'reg_lambda': [1], 
#                 'min_child_samples' : [4,6,8],
#                 # 'learning_rate' : [0.03,0.05,0.07],
#                 'learning_rate': [0.03], #Comme pour xgboost c'est l'apport d'information après chaque arbres
#                 'subsample'    : [0.6],
#                 'max_depth': [3,5,7], #Profondeur maximal des arbres 
#                 'n_estimators': [1000],
#                 'force_row_wise': [True],  # Activer l'option force_row_wise pour la perf mémoire
#                 'verbose' : [-1]
        
#         }

#     # start_time = datetime.now()

#     n_folds = 3


#     # Assurer que le nombre de plis n'est pas supérieur au nombre d'échantillons
#     if n_folds > X_train.shape[0]:
#         raise ValueError(f"Le nombre de plis ({n_folds}) est supérieur au nombre d'échantillons ({X_train.shape[0]})")
    
    
#     tscv = TimeSeriesSplit(n_splits=3)
#     grid_lgb = GridSearchCV(lgb, param_grid,  n_jobs=-1, cv=tscv)
    
#     # grid_lgb.fit(X_train, y_train)
#     # Set the maximum execution time in seconds
#     random_search = RandomizedSearchCV(lgb, param_distributions=param_grid, n_iter=10, cv=tscv, n_jobs=2)
    
#     max_time = 60  # Augmenté à 60 secondes
#     start_time = time.time()

#     try:
#         with parallel_backend('loky', n_jobs=2, inner_max_num_threads=1):
#             with joblib.parallel.parallel_memory_limit(max_memory=1024 * 1024 * 1024):  # 1GB
#                 random_search.fit(X_train, y_train)
#     except Exception as e:
#         print(f"An error occurred during fitting: {str(e)}")
#         return None, None

#     if (time.time() - start_time) >= max_time:
#         print(f"RandomizedSearchCV XGB fitting timed out after {max_time} seconds.")
#         return None, None
#     # fitting the model : 
    
#     model_lgb =  grid_lgb.best_estimator_
    
#     model_lgb.fit(X_train, y_train)
#     print(f'Ceci est mon modèle lightgbm {model_lgb}')
#     # end_time = datetime.now()
#     # elapsed_time = end_time - start_time
#     # print("Temps écoulé LBM:", elapsed_time)

#     return model_lgb, grid_lgb.best_params_


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
    
def useHyperParamII(features_model, param_model):

    features_Model_str = features_model
    Param_Model_str = param_model
    # print(f"dictionaire hyper {Param_Model_str}")
    # print(f"listfeatrue type {type(features_Model_str)} dictionaire hyper {type(Param_Model_str)}")

    actual_features_list = ast.literal_eval(features_Model_str)
    actual_Param_Model = ast.literal_eval(Param_Model_str)
    # params_json = json.dumps(Param_Model_str, defaul=str)

    # print(params_json)

    # d = json.loads(Param_Model_str)
    # print(d)


    # print(f"listfeatrue type {type(actual_features_list)} dictionaire hyper {type(actual_Param_Model)}")
   
    return actual_features_list, actual_Param_Model


# def useHyperParam(param_model):

#     Param_Model_str = param_model
#     print(f" dictionaire hyper {Param_Model_str}")

#     # actual_features_list = ast.literal_eval(features_Model_str)
#     actual_Param_Model = ast.literal_eval(Param_Model_str)

#     print(f"dictionaire hyper {type(actual_Param_Model)}")
#     return actual_Param_Model

# def useHyperParamIII(features_model, param_model):

#     features_Model_str = features_model
#     Param_Model_str = param_model
#     print(f"dictionaire hyper {Param_Model_str}")
#     # print(f"listfeatrue type {type(features_Model_str)} dictionaire hyper {type(Param_Model_str)}")

#     actual_features_list = ast.literal_eval(features_Model_str)
#     # actual_Param_Model = ast.literal_eval(Param_Model_str)

#     params_json = json.dumps(Param_Model_str, default=str) 
#     d = json.loads(params_json)
#     # actual_Param_Model = {k: (float('nan') if v == '__NaN__' else v) for k, v in d.items()}
#     filtered_params = {k: v for k, v in Param_Model_str.items() if v is not None}

#     print(f"listfeatrue type {type(actual_features_list)} dictionaire hyper {type(filtered_params)}")
   
#     return actual_features_list, filtered_params


# def check_table_exists(connection, table_name):
#     with connection.cursor() as cursor:
#         query = "SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = :table_name"
#         cursor.execute(query, table_name = table_name.upper())

#         (count,) = cursor.fetchone()
#         return count > 0



def ModelPrediction(model, X_test): 
    # make prediction :
    y_pred= model.predict(X_test)

    y_pred [y_pred < 0] = 0

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
    Get feature importance of an XGBoost or LightGBM model and calculate elasticity.
    """
    if isinstance(model, XGBRegressorInt):  # Check if the model is XGBoost
        # Get feature importance from XGBoost model
        importance = model.get_booster().get_score(importance_type=importance_type)
        #print(f"L'image a été téléchargée avec succès : {model}")
    elif isinstance(model, LightBMRegressorInt):  # Check if the model is LightGBM
        # Get feature importance from LightGBM model
        importance_score = model.feature_importances_
        feature_names = model.booster_.feature_name()
        importance = {feature_names[i]: importance_score[i] for i in range(len(feature_names))}
        #print(f"L'image a été téléchargée avec succès : {model}")
    else:
        raise ValueError("Model type not supported. Please provide an XGBoost or LightGBM model.")

    # Normalize feature importance
    total_importance = sum(importance.values()) if isinstance(importance, dict) else sum(importance)
    normalized_importance = {k: v / total_importance for k, v in importance.items()} if isinstance(importance, dict) else {exogenous[i]: importance[i] / total_importance for i in range(len(exogenous))}

    # Add a constant to the model (the intercept)
    X = sm.add_constant(final_df[exogenous])

    # Fit the regression model
    regression_model = sm.OLS(final_df[target], X).fit()

    # Calculate elasticity for each feature
    elasticities = {}
    for feature in exogenous:
        if feature in regression_model.params:
            elasticity = (regression_model.params[feature] * final_df[feature].mean()) / final_df[target].mean()
            #mettre electicité prix >0 nulle 
            if feature == 'PARAM_PRIX' and elasticity > 0:
                elasticity = 0
            if 'PARAM_PROMO' in feature and elasticity < 0: 
                elasticity = 0
            if 'PARAM_CONC' in feature and elasticity < 0: 
                elasticity = 0
            elasticities[feature] = elasticity

    # Combine feature importance with elasticity
    combined_importance = {feature: round(normalized_importance.get(feature, 0) * get_sign(elasticities.get(feature, 0)) * 100, 2)
                           for feature in exogenous}

    return combined_importance



# def save_intheOracleBase(model_name, id_produit, id_so, current_date, Date_Import, bestHyperParams, bestFeatures, Rmse_model, connection) : 

#     # Constructing column names and placeholders for the SQL query
#     bestHyperParams_str = str(bestHyperParams)
#     bestFeatures_str = str(list(bestFeatures))
#     nb_caractHyperParams = len(bestHyperParams_str)
#     nb_caractFeature = len(bestFeatures_str) 
#     save_quality = 1  

#     try:
#         with connection.cursor() as cursor:
#             # SQL query to insert into the dynamically named table
#             # query = f"INSERT INTO TRAIN_HISTORIC  ( id_produit, ID_SO, DATE_IMPORT, Date_Entrainement, Model_Name, PARAM_MODEL, features_Model, RMSE_TRAIN, Product_ID_TARIF) VALUES (:IDProduit, :IDSo, :DATE_IMPORT, :DATE_UPDATE_request, :NomModel, :bestHyperParams, :bestFeatures, :RMSE_TRAIN, :Product_ID_TARIF)"
#             query = f"INSERT INTO TRAIN_HISTORIC  (id_produit, ID_SO, DATE_IMPORT, Date_Entrainement, Model_Name, PARAM_MODEL, features_Model, RMSE_TRAIN) VALUES (:IDProduit, :IDSo, :DATE_IMPORT, :DATE_UPDATE_request, :NomModel,:bestHyperParams, :bestFeatures, :RMSE_TRAIN)"

#             # Prepare values for the dynamic part of the query
#                             # Prepare values for the dynamic part of the query
#             values = {
#                 'IDProduit': int(id_produit),
#                 'IDSo': int(id_so),
#                 'DATE_IMPORT': Date_Import,
#                 'DATE_UPDATE_request': current_date,
#                 'NomModel': model_name,
#                 'bestHyperParams': bestHyperParams_str,
#                 'bestFeatures': bestFeatures_str,
#                 'RMSE_TRAIN': float(Rmse_model)
#                 # 'Product_ID_TARIF': int(Product_ID_TARIF)
#             }
            
#             # Execute the query
#             cursor.execute(query, values)

#             # Commit the transaction
#             connection.commit()
#     except oracledb.DatabaseError as e:
#             print(f"Failed to save model data due to a database error: {e}")
#             connection.rollback()
#             raise



# def load_from_OracleBase(id_produit, id_so, connection):
#     try:
#         with connection.cursor() as cursor:
#             query = """
#                 SELECT *
#                 FROM TRAIN_HISTORIC
#                 WHERE DATE_ENTRAINEMENT = (
#                     SELECT MAX(DATE_ENTRAINEMENT)
#                     FROM TRAIN_HISTORIC
#                     WHERE ID_PRODUIT = :IDProduit AND ID_SO = :IDSo
#                 )
#                 AND RMSE_TRAIN = (
#                     SELECT MIN(RMSE_TRAIN)
#                     FROM TRAIN_HISTORIC
#                     WHERE DATE_ENTRAINEMENT = (
#                         SELECT MAX(DATE_ENTRAINEMENT)
#                         FROM TRAIN_HISTORIC
#                         WHERE ID_PRODUIT = :IDProduit AND ID_SO = :IDSo
#                     )
#                 )
#                 AND ID_PRODUIT = :IDProduit AND ID_SO = :IDSo
#             """
#             values = {
#                 'IDProduit': id_produit,
#                 'IDSo': id_so,
#             }
#             cursor.execute(query, values)
#             rows = cursor.fetchall()  # Fetch all matching rows

#             if rows:
#                 headers = [i[0] for i in cursor.description]
#                 df_temp = pd.DataFrame(rows, columns=headers)
#                 print("Most recent model loaded successfully.")
#                 return df_temp
#             else:
#                 print("No data found for the specified criteria.")
#                 return pd.DataFrame()  # Return an empty DataFrame

#     except oracledb.DatabaseError as e:
#         print(f"Database error occurred: {e}")
#         return None


def XgBoost_elasticities(X_test, exogenous, model):
    pct_change = 0.1
 
    elasticities = {"variables":[], "elasticity" :[]}
 
    for variable_name in exogenous:
        x1 = X_test[exogenous]
        x2 = X_test[exogenous]

        if "JS_" in variable_name or "VACANCES" in variable_name:
            x1.loc[:, variable_name] = 0
            x2.loc[:, variable_name] = 1
        else:
            x1.loc[:, variable_name] = x1[variable_name] * (1 + pct_change)
            x2.loc[:, variable_name] = x2[variable_name] * (1 - pct_change)
 
        y1 = model.predict(x1)
        y2 = model.predict(x2)
 
        elasticity = np.mean((y1 - y2) / (x1[variable_name] - x2[variable_name]))
 
        elasticities['elasticity'] +=  [elasticity]
        elasticities['variables'] += [variable_name]
 
 
    return elasticities



def split_df(x_final) : 
    nb_jour = min(365,int(len(x_final)*0.3)) 
    x_train = x_final[:-nb_jour]
    x_test = x_final[-nb_jour:]
    return x_train, x_test




def ModelChoice(final_df, exogenous, target) : 
    start_time = time.time()

    X_train, X_test = split_df(final_df)

    sum_target = sum(X_test[target])

    print("XGB train")
    model_xgb, best_params_xgb  = XgBoostRegressor(X_train[exogenous], X_train[target])
    y_pred_xgb = ModelPrediction(model_xgb, X_test[exogenous])
    error_xgb = float(abs(sum(y_pred_xgb) - sum(X_test[target])))
    # # Calcul de MAE
    # mae = mean_absolute_error(y_pred_xgb, X_test[target])
    # print("MAE:", mae)

    # # Calcul de MSE
    # mse = mean_squared_error(y_pred_xgb, X_test[target])
    # print("MSE:", mse)

    # # Calcul de RMSE
    # rmse = np.sqrt(mse)
    # print("RMSE:", rmse)

    # # Calcul de MAPE
    # mape = np.mean(np.abs((X_test[target] - y_pred_xgb) / X_test[target])) * 100
    # print("MAPE:", mape)
    print("LGBM train")
    model_Light, best_params_Light  = LightBMRegressor(X_train[exogenous], X_train[target])
    y_pred_Light = ModelPrediction(model_Light,X_test[exogenous])
    error_Light = float(abs(sum(y_pred_Light) - sum(X_test[target])))


    # Choose the best model based on error metric
    errors = {
        'xgboost': error_xgb,
        'lightgbm': error_Light,
    }

    best_model = min(errors, key=errors.get)
    best_error = errors[best_model]

    
    
    #Retraing the best model on all data 
    if best_model == 'xgboost':
        print("XGB final")
        #model, best_params = XgBoostRegressor(final_df[exogenous], final_df[target]) # pas besoin de relancer le gridsearch --> on refit uniquement avec les best-param
        model = GETXgboostRegressor(final_df[exogenous], final_df[target],best_params_xgb)
        best_params = best_params_xgb
        #precision = accuracy_score(X_test[target], y_pred_xgb)
        #percentage_accuracy = precision * 100
        percentage_accuracy = None

        ##Precision selon erreur Booper
        if error_xgb == 0: 
            percentage_error_global = float(0)  
            percentage_accuracy_global = 100 - percentage_error_global
        else : 
            # if sum_target == 0 :   
            #     # Gérer le cas où la somme est 0
            #     percentage_error_global = float(100)  
            #     percentage_accuracy_global = 100 - percentage_error_global
            if sum_target == 0 :   
                # Gérer le cas où le produit est très peu vendu 
                percentage_accuracy_global = None
            else :
                # Calcul du pourcentage d'erreur
                percentage_error_global = (error_xgb / sum_target) * 100
                percentage_accuracy_global = 100 - percentage_error_global


        # ##recision selon erreur Booper
        # if sum_target <= 10:
        #     # Gérer le cas où la somme est 0
        #     percentage_error_global = float(0)  
        #     percentage_accuracy_global = 100 - percentage_error_global
        # else:
        #     # Calcul du pourcentage d'erreur
        #     percentage_error_global = (error_xgb / sum_target) * 100
        #     percentage_accuracy_global = 100 - percentage_error_global

    elif best_model == 'lightgbm':
        print("LGBM final")
        #model,best_params = LightBMRegressor(final_df[exogenous], final_df[target]) # pas besoin de relancer le gridsearch --> on refit uniquement avec les best-param
        model = GETLightBMRegressor(final_df[exogenous], final_df[target], best_params_Light)
        best_params = best_params_Light
        #precision = accuracy_score(X_test[target], y_pred_Light)
        #percentage_accuracy = precision * 100
        percentage_accuracy = None



        ##Precision selon erreur Booper
        if error_Light == 0: 
            percentage_error_global = float(0)  
            percentage_accuracy_global = 100 - percentage_error_global
        else : 
            # if sum_target == 0 :   
            #     # Gérer le cas où la somme est 0
            #     percentage_error_global = float(100)  
            #     percentage_accuracy_global = 100 - percentage_error_global
            if sum_target == 0 :   
                # Gérer le cas où le la somme des quantité reelle dans test est 0 (division impossible)
                percentage_accuracy_global = None
            else :
                # Calcul du pourcentage d'erreur
                percentage_error_global = (error_Light / sum_target) * 100
                percentage_accuracy_global = 100 - percentage_error_global

        # ##Precision selon erreur Booper
        # if sum_target <= 10:
        #     # Gérer le cas où la somme est 0
        #     percentage_error_global = float(0)  
        #     percentage_accuracy_global = 100 - percentage_error_global
        # else:
        #     # Calcul du pourcentage d'erreur
        #     percentage_error_global = (error_Light / sum_target) * 100
        #     percentage_accuracy_global = 100 - percentage_error_global

    if percentage_accuracy_global < 0 :
         percentage_accuracy_global = 0
        
    #print(f"Percentage Precicion : {percentage_accuracy}")
        

    print(f"Le temps d'execution est {time.time()-start_time} seconds.")

    return best_model, model, best_error, percentage_accuracy, percentage_accuracy_global, best_params






def process_data(Product_parametre_json,Product_features_json, Product_quantity_json, Product_future_features_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []


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

    final_df = final_df.set_index('DATE_IMPORT')
    

    # Convert the received JSON data to a DataFrame : 
    target = 'QUANTITE'
    exogenous = [x for x in Product_parametre_json if x != target]
    #print(exogenous)

    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    
    final_df[target] = final_df[target].apply(lambda x: 0 if x < 0 else x)


    #Handle NaNs: 
    #fill them with a specific value, like zero: 
    final_df.fillna(0, inplace=True)

    x_future = pd.DataFrame(Product_future_features_json) 
    nb_jours = len(x_future)
    if nb_jours >0 : 
    
        # The choice of prediction set 
        
        target = 'QUANTITE'

        if 'DATE_TMP' in x_future.columns :
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

    return x_future, final_df, target, nb_jours, exogenous 




def GET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, id_produit, id_so, date_import, model_name, feature_Model, parm_model):

    # Create a dictionary which contains all information needed 
    preds = {}


    # list_featureModel = feature_Model
    list_featureModel, dic_parm_model = useHyperParamII(feature_Model,parm_model)

    final_features = [i for i in list_featureModel if i != 'QUANTITE']     
    #Retraing the best model on all data 
    if model_name == 'xgboost':
        model = GETXgboostRegressor(final_df[final_features], final_df[target], dic_parm_model)
    elif model_name  == 'lightgbm':
        model = GETLightBMRegressor(final_df[final_features], final_df[target], dic_parm_model)


    # Forecasting with the last price
    preds['QUANTITE_AJUSTE'] = ModelPrediction(model, final_df[final_features])
    preds['QUANTITE_0'] = ModelPrediction(model, x_future[final_features])
    
    
    # Some variation test
    #prix_min = min(final_df['PARAM_PRIX'])
    # prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'].iloc[-1]
    
    prix_min = last_price * 0.9
    prix_max = last_price * 1.1

    # print("prix_min", prix_min  )
    # print("Last Price", last_price )

    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(1,NB_PRIX)]
    #print(vec_prix_test)
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # Change prediction values with the price
    cpt = 0
    for prix in vec_prix_test: 
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = ModelPrediction(model, x_future[final_features])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = ModelPrediction(model, x_future[final_features])
        
    
    
    # Convert predictions to the desired format
    preds_converted = {}
    for key, values in preds.items():
        if key != 'QUANTITE_AJUSTE' : 
            # Check if 'values' is a list or numpy array
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(x_future.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values

        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = get_feature_importance(model,final_df, target, final_features)



    # X_train, X_test = split_df(final_df)
    # model_elasticity = XgBoostRegressor(X_train[exogenous], X_train[target])
    # coefficients = XgBoost_elasticities(X_test, exogenous, model_elasticity)


    # # preds_converted['MAE']=mae
    # # preds_converted['RMSE']=rmse
    # preds_converted['ERRORS']=errors
    # preds_converted['ERROR']=best_error

    # # preds_converted['COM_ERROR']= com_error
    # preds_converted['MODEL']=best_model

    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test
    preds_converted['OK'] = str(1)



    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)

    return json_output






def SET_process_product_Version(x_future, final_df, target, nb_jours, exogenous, id_produit, id_so):
    # Create a dictionary which contains all information needed 
    preds = {}
    # current_date = datetime.now()
    #print(final_df['PARAM_PRIX'])

    # if GET_Date_Product(id_produit, id_so, connection) is not None :
    #     Date_Import = GET_Date_Product(id_produit, id_so, connection)
    # else : 
    #     Date_Import = current_date


    best_model, model, best_error, percentage_accuracy, percentage_accuracy_global, best_param = ModelChoice(final_df, exogenous, target)
    bestFeatures = exogenous
    # bestHyperParams = model.get_params()
    # print(f"best type {bestHyperParams}")
    # print(f"best param grid {best_param}") 
    # model.set_params(bestHyperParams)
    
    # save_intheOracleBase(best_model, id_produit, id_so, current_date, date_import, bestHyperParams, bestFeatures, best_error)



    # Forecasting with the last price
    preds['QUANTITE_AJUSTE'] = ModelPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = ModelPrediction(model, x_future[exogenous])
    
    
    # Some variation test
    #prix_min = min(final_df['PARAM_PRIX'])
    # prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'].iloc[-1]
    
    prix_min = last_price * 0.9
    prix_max = last_price * 1.1

    # print("prix_min", prix_min  )
    # print("Last Price", last_price )

    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(1,NB_PRIX)]
    # print(vec_prix_test)
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # Change prediction values with the price
    cpt = 0
    for prix in vec_prix_test: 
        # print(prix)
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = ModelPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = ModelPrediction(model, x_future[exogenous])
        
    
    
    # Convert predictions to the desired format
    preds_converted = {}
    for key, values in preds.items():
        if key != 'QUANTITE_AJUSTE' : 
            # Check if 'values' is a list or numpy array
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(x_future.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values

        else : 
            if isinstance(values, (list, np.ndarray)):
                list_dic_values = []
                for date, value in zip(final_df.index, values):
                    date_value = {"date": str(date), "value": str(value)}
                    list_dic_values.append(date_value)
                preds_converted[key] = list_dic_values


    coefficients = get_feature_importance(model,final_df, target, exogenous)

    bestHyperParams_str = str(best_param)
    bestFeatures_str = str(list(bestFeatures))

    preds_converted["ACCURACY_GLOBAL"] = percentage_accuracy_global
    preds_converted["ACCURACY"] = percentage_accuracy 
    preds_converted["MAE_LAST_MONTH"] = None 
    preds_converted["ERROR_GLOBAL"] = best_error
    preds_converted["FEATURES_MODEL"] = bestFeatures_str
    preds_converted["PARAM_MODEL"] = bestHyperParams_str
    preds_converted["MODEL_NAME"] = best_model
    # preds_converted["DATE_IMPORT"] = date_import

    #preds_converted['ERROR']=best_error

    # preds_converted['COM_ERROR']= com_error
    preds_converted['MODEL']=best_model
    preds_converted['ELASTICITE'] = coefficients
    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    preds_converted['OK'] = str(1)



    # # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted)


    return json_output


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
        # print(f"Parsed start_date: {start_date}")


        if i < len(date_import_list) - 1:
            end_date = datetime.strptime(date_import_list[i + 1], '%d/%m/%Y')
        else:
            # For the last model, use the last date in x_future
            end_date = final_df.index[-1]

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


