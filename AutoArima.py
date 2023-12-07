import pmdarima as pm
import pandas as pd


def process_product_ARIMA(x_future, feature_quantite_final ,final_df,target,nb_jours,exogenous) :

    #Creat a dictionary which contains all information needed 
    preds = {}

    # forecasting with the last price :
    model = TrainAutoArima(final_df, exogenous, target)
    preds['QUANTITE_AJUSTE'] = PredictAutoArima(model, feature_quantite_final, exogenous)
    preds['QUANTITE_0'] = PredictAutoArima(model, x_future, exogenous)
    
    # print(preds['QUANTITE_0'])
    # some variation test :
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]

    vec_prix_test = [prix_min + i*(prix_max - prix_min) / (NB_PRIX - 1 ) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    
    # Change prediction values with the price :
    cpt=0
    for prix in vec_prix_test : 
        cpt+=1
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}'] = PredictAutoArima(model, x_future, exogenous)
    
    x_future['PARAM_PRIX'] = [last_price] * nb_jours 
    # Change prediction values with the promotion value :  
    
    for promo in vec_promo_test : 

        x_future['PARAM_PROMO'] = [promo] * nb_jours
        preds[f'PROMO_{int(promo*100)}'] = PredictAutoArima(model, x_future, exogenous)
        
    coefficients = GetFeaturesInterpretation(model)
    # if coefficients is not None:
    #     # print("Model Coefficients:", coefficients)
    preds['ELASTICITE'] = coefficients
            
    # df_prediction = pd.DataFrame(preds)
    # print(df_prediction)
    # Check if the series is not empty (it change)

    # json_string = df_prediction.to_json()
    # #print(json_string)  # This will print the JSON representation of your series
    # preds_converted = {key: value.dt.strftime('%Y-%m-%d').tolist() if hasattr(value, 'dt') else value.tolist() for key, value in preds.items()}
    # # preds['ELASTICITE_NAME']
    # print(preds_converted['ELASTICITE'])
    
    # test_json =  json.dumps(preds_converted)
    # print(test_json)
    # return test_json ,200
    
    # Convertir chaque série pandas en liste, en incluant les dates
    preds_converted = {
        key: [{"date": str(date), "value": value} for date, value in zip(value.index, value)] 
        if hasattr(value, 'index') else value 
        for key, value in preds.items()
    }

    # Traiter spécifiquement l'élasticité si c'est un dictionnaire ou une série pandas
    if isinstance(preds['ELASTICITE'], (dict, pd.Series)):
        preds_converted['ELASTICITE'] = preds['ELASTICITE'].to_dict()  if isinstance(preds['ELASTICITE'], pd.Series)  else preds['ELASTICITE']

    preds_converted['PRIX_INTERVAL'] = vec_prix_test

    # Convertir le dictionnaire en JSON
    json_output = json.dumps(preds_converted, indent=4)
    return json_output

def process_data_ARIMA(Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []

    # print(f'len of the sales quantity {len(Product_quantity_json)}')
    # print(f'len of the feature quantity {len(Product_features_json)}')
   

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
    # print(f'Original dataset columns {final_df.columns}')


    # final_df['DATE_IMPORT'] = pd.to_datetime(final_df['DATE_IMPORT'], format='%d/%m/%Y')
    final_df = final_df.set_index('DATE_IMPORT')
    
    # We then have to see if our date are continuous : 
    # print(f'index of the dataframe : {final_df.index}')
    
    # Convert the received JSON data to a DataFrame : 
    target = 'QUANTITE'
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]
    # print(f'Used Features : {exogenous}')
    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    
    #Handle NaNs: 
    final_df.dropna(inplace=True)
    #fill them with a specific value, like zero: 
    final_df.fillna(0, inplace=True)
    
    
    x_future = pd.DataFrame(Product_future_features_json) 
    print(x_future.columns)
    # The choice of prediction set 
    nb_jours = len(x_future)
    target = 'QUANTITE'

    # print(f'features contained in {x_future.columns}')
    # x_future['DATE_TMP'] = pd.to_datetime(x_future['DATE_TMP'], format='%d/%m/%Y')
    x_future =  x_future.set_index('DATE_TMP')
    
    # print(json_output)
    return x_future, final_df, target, nb_jours, exogenous




def TrainAutoArima(X_historique, exogenous, target): 
    model_input = 'arima_auto'

    if model_input == 'arima_auto':
        #Using AutoArima  
        model = pm.auto_arima(X_historique[target], X = X_historique[exogenous] ,seasonal=False,m=0,stepwise=True,trace=True,start_p=0,start_q=0,start_P=0,start_Q=0,max_p=1,max_q=1,maxiter=50000,with_intercept=True,trend='ct')
    elif model_input == 'arima':
        # Initialize the ARIMA model
        model = pm.arima.ARIMA(X_historique[target], order=(1, 0, 1), maxiter=6000, with_intercept=True, trend='ct')# Train on x_train, y_train

    model.fit(X_historique[target], X = X_historique[exogenous])
    
    return model  
    



def PredictAutoArima(model, x_future,exogenous): 

    # Predict on x_test
    predsautoARIMA = model.predict(n_periods= len(x_future), X=x_future[exogenous])
    # print(list(predsautoARIMA))
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