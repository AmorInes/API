import numpy as np
import pandas as pd 
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

NB_PRIX = 5

def process_product_RandomForest(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = RFregressor(final_df[exogenous], final_df[target])
    preds['QUANTITE_AJUSTE'] =  RandomForestPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] =  RandomForestPrediction(model, x_future[exogenous])
    
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
        preds[f'PRIX_{cpt}'] =  RandomForestPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] =  RandomForestPrediction(model, x_future[exogenous])
        
    # Get feature importance
    coefficients = RandomForestGetFeatureImportance(model)
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

def RFregressor(X_train, y_train):
    # Random Forest hyperparameters
    rf = RandomForestRegressor()

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_rf = GridSearchCV(rf, param_grid, n_jobs=-1, cv=tscv, verbose=2)
    grid_rf.fit(X_train, y_train)
    
    # Fitting the model
    model_rf = grid_rf.best_estimator_
    model_rf.fit(X_train, y_train)
    return model_rf


def RandomForestPrediction(model_rf, X_test): 
    # Make prediction
    y_pred_rf = model_rf.predict(X_test)
    return y_pred_rf


def RandomForestGetFeatureImportance(model):
    # Get feature importance
    importance = model.feature_importances_

    # Assuming feature names are provided
    feature_names = [f'feature_{i}' for i in range(len(importance))]

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df
