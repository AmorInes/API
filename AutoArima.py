import pmdarima as pm
import pandas as pd


def TrainAutoArima(X_historique, exogenous, target): 
    model_input = 'arima_auto'

    if model_input == 'arima_auto':
        model = pm.auto_arima(X_historique[target], X = X_historique[exogenous] ,seasonal=True,m=12,stepwise=True,trace=True,start_p=0,start_q=0,start_P=0,start_Q=0,max_p=1,max_q=1,maxiter=50000,with_intercept=True,trend='ct')
    elif model_input == 'arima':
        model = pm.arima.ARIMA(order=(1,0,1),seasonal=False,m=12,stepwise=True,trace=True,maxiter=6000,with_intercept=True,trend='ct')
    # Train on x_train, y_train
    model.fit(X_historique[target], X = X_historique[exogenous])
    
    return model  
    



def PredictAutoArima(model, x_future,exogenous): 

    # Predict on x_test
    predsautoARIMA = model.predict(n_periods= len(x_future), X=x_future[exogenous])
    print(list(predsautoARIMA))
    predsautoARIMA = pd.Series(list(predsautoARIMA), index=x_future.index)
    
    return predsautoARIMA 


def GetFeaturesInterpretation(model):
    """
    Get the parameters (coefficients) of a fitted ARIMAX model.

    Parameters:
    model (pmdarima ARIMA): A fitted ARIMAX model from the pmdarima library.

    Returns:
    pandas.Series: A series of model coefficients.
    """
    try:
        # Retrieve and return the model's coefficients directly as a property
        return model.arima_res_.params
    except AttributeError as e:
        # Handle the case where the model might not be fitted or is not a valid ARIMAX model
        print(f"Error in retrieving model parameters: {e}")
        return None