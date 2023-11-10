import pmdarima as pm


def TrainAutoArima(X_historique, exogenous, target): 
    model_input = 'arima_auto'

    if model_input == 'arima_auto':
        model = pm.auto_arima(X_historique[target],seasonal=True,m=12,stepwise=True,trace=True,start_p=0,start_q=0,start_P=0,start_Q=0,max_p=4,max_q=4,maxiter=50000,with_intercept=True,trend='ct')
    elif model_input == 'arima':
        model = pm.arima.ARIMA(order=(1,0,1),seasonal=False,m=12,stepwise=True,trace=True,maxiter=6000,with_intercept=True,trend='ct')
    # Train on x_train, y_train
    model.fit(X_historique[target], X = X_historique[exogenous])
    
    return model  
    



def PredictAutoArima(model, x_future, exogenous, target): 

    # Predict on x_test
    predsautoARIMA = model.predict(n_periods= len(x_future), X=x_future[exogenous])
    
    return predsautoARIMA 