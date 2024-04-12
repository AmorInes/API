import pandas as pd 
import numpy as np
import json
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from lightgbm import LGBMRegressor
from darts import TimeSeries

NB_PRIX = 5




def process_product_LightGBM(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = LightBMRegressor(final_df[exogenous], final_df[target])
    preds['QUANTITE_AJUSTE'] = LightBMPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = LightBMPrediction(model, x_future[exogenous])
    
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
        preds[f'PRIX_{cpt}'] = LightBMPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = LightBMPrediction(model, x_future[exogenous])
        
    # Get feature importance
    coefficients = LightBMGet_feature_importance(model)
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


def LightBMRegressor(X_train, y_train):
    #LightGBM hyperparameter 
    lgb = LGBMRegressor()

    param_grid = {
                'num_leaves': [8,16,32,64],  
                'reg_lambda': [1], 
                'min_child_samples' : [4,6,8],
                'learning_rate': [0.03, 0.05, 0.07], #Comme pour xgboost c'est l'apport d'information après chaque arbres
                'subsample'    : [0.6],
                'max_depth': [3,5,7], #Profondeur maximal des arbres 
                'n_estimators': [1000]
        
        }

    #start = time()

    tscv = TimeSeriesSplit(n_splits=5)
    grid_lgb = GridSearchCV(lgb, param_grid,  n_jobs=-1, cv=tscv, verbose=2)
    
    grid_lgb.fit(X_train, y_train)
    
    # fitting the model : 
    model_lgb =  grid_lgb.best_estimator_
    model_lgb.fit(X_train, y_train)
    return model_lgb



def LightBMPrediction(model_lgb, X_test): 
    # make prediction :
    y_pred_lgb = model_lgb.predict(X_test)
    # print(y_pred_xgboost)
    return y_pred_lgb


def LightBMGet_feature_importance(model):
    # Get feature importance
    importance = model.feature_importances_

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': model.feature_name_,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df



# # we can see the error : 
# mse = mean_squared_error(y_test_df_smooth_1_histo, y_pred_lgb)
# rmse = np.sqrt(mean_squared_error(y_test_df_smooth_1_histo, y_pred_lgb))
# mae = mean_absolute_error(y_test_df_smooth_1_histo, y_pred_lgb)



# print("\tMean Squared error (MSE): %0.2f " % mse) 
# print("\tMean Squared error (RMSE): %0.2f " % rmse) 
# print("\tMean absolute error (MAE) : %0.2f " % mae)