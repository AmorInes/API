import numpy as np
import pandas as pd 
import json
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from xgboost.sklearn import XGBRegressor

NB_PRIX = 5



def XgBoostRegressor(X_train, y_train):
    #XGBOOSTRegressor hyperparameters :
    xgb = XGBRegressor()
    param_grid = { 
                'objective':['reg:squarederror'],
                'learning_rate' : [0.03,0.05,0.07],
                'reg_lambda': [1],
                'n_estimators' : [1000],
                'max_depth':[3,5,7],
                'min_child_weight':[6,7,8,9],
                'gamma' : [0],
                'colsample_bytree':[0.8],
                #'nthread' : [4],
                'eval_metric' : ['mae'], 
    }


    # finding the best estimator :
    tscv = TimeSeriesSplit(n_splits=5)
    grid_xgb = GridSearchCV(xgb, param_grid, n_jobs=-1, cv=tscv, verbose=2 )
    grid_xgb.fit(X_train, y_train)
    model_xgb =  grid_xgb.best_estimator_


    # fitting the model : 
    model_xgb.fit(X_train, y_train)

    return model_xgb

def XgBoostPrediction(model_xgb, X_test): 
    # make prediction :
    y_pred_xgboost = model_xgb.predict(X_test)
    # print(y_pred_xgboost)
    return y_pred_xgboost


def XgBoostget_feature_importance(model, importance_type='gain'):
    """
    get feature importance of an XGBoost model.
    
    :param model: The trained XGBoost model.
    :param importance_type: Type of importance measure. One of 'weight', 'gain', or 'cover'.
    """
    # Get feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': importance.keys(),
        'Importance': importance.values()
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df



def process_product_Xgboost(x_future, final_df, target, nb_jours, exogenous):
    # Create a dictionary which contains all information needed 
    preds = {}

    # Forecasting with the last price
    model = XgBoostRegressor(final_df[exogenous], final_df[target])
    preds['QUANTITE_AJUSTE'] = XgBoostPrediction(model, final_df[exogenous])
    preds['QUANTITE_0'] = XgBoostPrediction(model, x_future[exogenous])
    
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
        preds[f'PRIX_{cpt}'] = XgBoostPrediction(model, x_future[exogenous])
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 

    # Change prediction values with the promotion value
    for promo in vec_promo_test: 
        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo * 100)}'] = XgBoostPrediction(model, x_future[exogenous])
        
    coefficients = XgBoostget_feature_importance(model)
    preds['ELASTICITE'] = coefficients
    
    
    # Convert predictions and other data to a format suitable for JSON serialization
    preds_converted = {}
    for key, value in preds.items():
        if isinstance(value, pd.DataFrame):
            # Convert DataFrame to a list of dictionaries (one for each row)
            preds_converted[key] = value.to_dict(orient='records')
        elif isinstance(value, (np.ndarray, pd.Series)):
            # Convert NumPy arrays and pandas Series to lists
            preds_converted[key] = value.tolist()
        else:
            preds_converted[key] = value

    # Convert vec_prix_test to a list if it's a NumPy array
    if isinstance(vec_prix_test, np.ndarray):
        vec_prix_test = vec_prix_test.tolist()

    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # Convert the dictionary to JSON
    json_output = json.dumps(preds_converted, indent=4)
    
    print(json_output)


    return json_output






# # we can see the error : 
# mse = mean_squared_error(y_test_df_smooth_1_histo, y_pred_xgboost)
# rmse = np.sqrt(mean_squared_error(y_test_df_smooth_1_histo, y_pred_xgboost))
# mae = mean_absolute_error(y_test_df_smooth_1_histo, y_pred_xgboost)



# print("\tMean Squared error (MSE): %0.2f " % mse) 
# print("\tMean Squared error (RMSE): %0.2f " % rmse) 
# print("\tMean absolute error (MAE) : %0.2f " % mae)
