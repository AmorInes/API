import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit 
from catboost import CatBoostRegressor
import json

NB_PRIX = 5


# Assuming LightBMRegressor and LightBMPrediction are defined as per your previous message

def process_product_CATBoost(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = CATBoostRegressor(final_df[exogenous], final_df[target])
    preds['QUANTITE_AJUSTE'] =  CATBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] =  CATBoostPrediction(model, x_future[exogenous])
    
    # Some variation test
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]

    vec_prix_test = [prix_min + i * (prix_max - prix_min) / (NB_PRIX - 1) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    # Change prediction values with the price
    cpt = 0
    for prix in vec_prix_test: 
        cpt += 1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] =  CATBoostPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] =  CATBoostPrediction(model, x_future[exogenous])
        
    # Get feature importance
    coefficients = CatBoostGetFeatureImportance(model)
    preds['ELASTICITE'] = coefficients.to_dict()  # Convert DataFrame to dict for JSON serialization
    
    # Convert predictions and other data to a format suitable for JSON serialization
    preds_converted = {}
    for key, value in preds.items():
        if isinstance(value, pd.DataFrame):
            preds_converted[key] = value.to_dict(orient='records')
        elif isinstance(value, (np.ndarray, pd.Series)):
            preds_converted[key] = value.tolist()
        else:
            preds_converted[key] = value

    # Convert vec_prix_test to a list if it's a NumPy array
    if isinstance(vec_prix_test, np.ndarray):
        vec_prix_test = vec_prix_test.tolist()

    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted, indent=4)
    
    return json_output

def CATBoostRegressor(X_train, y_train):
    # CatBoost hyperparameter : 

    cat = CatBoostRegressor()

    param_grid = {
        'loss_function' : ['RMSE', 'MAE'],
        'depth' : [3,5,7], 
        'learning_rate' : [0.03, 0.05, 0.15, 0.3,],
        'iterations' : [1000], 
        'random_seed' : [20,30], 
        'od_type' : ['Iter'],
        'od_wait' : [20]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_cat = GridSearchCV(cat, param_grid, n_jobs=-1, cv=tscv, verbose=2)
    grid_cat.fit(X_train, y_train)
    model_cat =  grid_cat.best_estimator_


    # fitting the model : 
    model_cat.fit(X_train, y_train)
    return model_cat


def CATBoostPrediction(model_cat,X_train):
    # make prediction :
    y_pred_cat = model_cat.predict(X_train)
    return y_pred_cat

def CatBoostGetFeatureImportance(model):
    """
    Get feature importance of a CatBoost model.

    :param model: The trained CatBoost model.
    """
    # Get feature importance
    importance = model.get_feature_importance()

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': model.feature_names_,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df