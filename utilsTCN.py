import pandas as pd
import matplotlib.pyplot as plt
import darts
import numpy as np 
import oracledb
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import statsmodels.api as sm
import scipy.stats as st

from datetime import datetime
import matplotlib.dates as mdates

# Some test to understand how features are interpreted by models

from captum.attr import (
    KernelShap,
    GradientShap,
    IntegratedGradients,
    FeaturePermutation,
    FeatureAblation,
    Occlusion
)

import shap
import lime


#Some statistical test :
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping

import shutil
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, accuracy_score,  r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# technics for deep learning  
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

# test afin  de pouvoir répliquer arima
import pmdarima as pm

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel, TFTModel, CatBoostModel, XGBModel, LightGBMModel, KalmanForecaster, TCNModel, NBEATSModel, TransformerModel
from darts.utils.likelihood_models import GaussianLikelihood, PoissonLikelihood, QuantileRegression
from darts.metrics import mape, mase, rmse, mae, smape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.explainability.shap_explainer import ShapExplainer
from darts.models import RegressionModel
import scipy.stats as stats
import json

NB_PRIX = 5
BATCH_SIZE = 32
MAX_N_EPOCHS = 50
NR_EPOCHS_VAL_PERIOD = 1
MAX_SAMPLES_PER_TS = 100
NB_JOUR_PRED = -60 

def transforme_data(data,name_date): 
    # #On veut pouvoir ressevoir une matrice en json et lui donné la bonne forme pour que l'on puisse utiliser Darts :
    # df_smooth_1_histo_day_avg = (
    #     data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    # )
    
    print("DataFrame columns before TimeSeries.from_dataframe:", data.columns)
    print("Is 'DATE_IMPORT' in DataFrame columns?", name_date in data.columns)
    
    # Ensure the index is in the expected format
    # if not isinstance(data.index, pd.DatetimeIndex):
    #     # Convert the index to a datetime index if necessary
    #     data.index = pd.to_datetime(data.index)

    # # Group by date (assuming the index is a datetime)
    # grouped_data = data.groupby(data.index).mean()
    
    if name_date == "DATE_IMPORT" : 
        
        
        df_test_produit_day_avg = (
            data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
        )
        

        series_en = fill_missing_values(
            TimeSeries.from_dataframe(
                df_test_produit_day_avg , fill_missing_dates=True, freq="D", time_col= name_date
            ),
            "auto",
        )


        # list(df_smooth_1_histo.index)[nb_train_smooth_1_histo]
        
        print(list(data.index)[int(len(data)*0.75)])

        # scale
        scaler_en = Scaler()
        series_en_transformed = scaler_en.fit_transform(series_en)
        train_en_transformed, val_en_transformed = series_en_transformed.split_after(
            list(data.index)[int(len(data)*0.75)]
        )
        return train_en_transformed, val_en_transformed
    
    if name_date == 'DATE_TMP': 
        df_test_produit_day_avg = (
            data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
        )
    
        series_en = fill_missing_values(
            TimeSeries.from_dataframe(
                df_test_produit_day_avg , fill_missing_dates=True, freq="D", time_col= name_date
            ),
            "auto",
        )
        scaler_en = Scaler()
        series_en_transformed = scaler_en.fit_transform(series_en)
        return series_en_transformed
        
        






def process_data_TCN(request, Product_features_json, Product_quantity_json, Product_future_features_json, Product_Id_produit_json) :
    ##### 
    #Il faut que face une fonction pour retraiter les données : 
    #####
    dfs = []
    target = 'QUANTITE'

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
        
        
    final_df = pd.concat(dfs, ignore_index=True)
    target = 'QUANTITE'
    
    exogenous = [x for x in request.json['LIST_PARAMETRE'] if x != target]
    # print(f'Used Features : {exogenous}')
    
    #Convert to Numeric: 
    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target] = pd.to_numeric(final_df[target], errors='coerce')
    

    # Concatenate all DataFrames together
    name_date = 'DATE_IMPORT'

    # print(f'Original dataset columns {final_df.columns}')
    # Example: Checking the DataFrame columns
    print(final_df.columns)  # Check the actual column names
    
    final_df[name_date] = pd.to_datetime(final_df[name_date], dayfirst=True, format='%d/%m/%Y')
    # Assuming x_future is your existing DataFrame
    # Step 2: Set 'DATE_TMP' as the index
    final_df = final_df.set_index(name_date, drop = False)

    

    #Handle NaNs: 
    final_df.dropna(inplace=True)
    #fill them with a specific value, like zero: 
    final_df.asfreq('D', fill_value=0) 
    
    print(final_df.columns)
    
    if name_date not in final_df.columns:
        final_df[name_date] = final_df.index
    
    #Inspect Problematic Columns: 
    # The choice of prediction set     
    train_en_transformed, val_en_transformed = transforme_data(final_df, name_date)
    
    
    
    
    name_date = 'DATE_TMP'
    x_future = pd.DataFrame(Product_future_features_json) 
    print(x_future.columns)
    
    # Convert 'DATE_TMP' to datetime and set as index
    x_future['DATE_TMP'] = pd.to_datetime(x_future['DATE_TMP'], dayfirst=True, format='%d/%m/%Y')
    x_future = x_future.set_index('DATE_TMP', drop = False)

    # # Ensure the index is a DatetimeIndex for resampling
    # if not isinstance(x_future.index, pd.DatetimeIndex):
    #     x_future.index = pd.to_datetime(x_future.index)

    # # Resample the data
    # try:
    #     x_future = x_future.resample('D').first()
    # except Exception as e:
    #     print(f"Error during resampling: {e}")
    
       
    for col in exogenous:
        if x_future[col].dtype == 'object':
            try:
                x_future[col] = pd.to_numeric(x_future[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                
    # # Assuming x_future is your existing DataFrame
    # # Step 1: Create a copy of 'DATE_TMP' column
    # x_future[name_date] = pd.to_datetime(x_future[name_date], dayfirst=True, format='%d/%m/%Y')
    # x_future['DATE_TMP_copy'] = x_future[name_date]
    # # Step 2: Set 'DATE_TMP' as the index
    # # Optionally, you can rename the copied column back to 'DATE_TMP'
    # x_future.rename(columns={'DATE_TMP_copy': name_date}, inplace=True)
    
    # x_future = x_future.set_index(name_date, drop = False)
    # x_future = x_future.resample('D').first()
    x_future_scaled = transforme_data(x_future, name_date)
    # print(json_output)
    return x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous
# x_future scaled features version 
# train_en_transformed, val_en_transformed scaled trained and validation version 


def rescale_data_TCN(predic_non_scaled, data_corrected_test_eval) : 
    
    scaler_skQuantite = MaxAbsScaler()
    scaled_series = scaler_skQuantite.fit_transform(np.array(data_corrected_test_eval['QUANTITE']).reshape(-1,1))

    list_prev = predic_non_scaled['QUANTITE'].values()

    prediction_scaled = [x[0] for x in list_prev]
    prediction_original = scaler_skQuantite.inverse_transform(np.array(prediction_scaled).reshape(-1,1))
    data_for_metrics = np.array(prediction_original).reshape(1,-1)
    
    return data_for_metrics

def rescale_data_TCN_total(predic_non_scaled, data_corrected_test_eval) : 
    scaler_skQuantite = MaxAbsScaler()
    scaled_series = scaler_skQuantite.fit_transform(np.array(data_corrected_test_eval).reshape(-1,1))

    list_prev = predic_non_scaled.values()

    prediction_scaled = [x[0] for x in list_prev]
    prediction_original = scaler_skQuantite.inverse_transform(np.array(prediction_scaled).reshape(-1,1))
    data_for_metrics = np.array(prediction_original).reshape(1,-1)
    
    return data_for_metrics
    
def forecast_TCN(nb_pred, nb_in, model_TCN, val_en_transformed, feature_forecast, Original_Dataset, target, exogenous) :
    

        
    combined_series = val_en_transformed[exogenous].append(feature_forecast[exogenous])
    
    
    
    prev_test_TCN_unScale = model_TCN.predict(
        n = nb_pred,
        series = val_en_transformed['QUANTITE'][nb_in:], 
        past_covariates = combined_series, 
        num_samples = 100, 
        verbose=True)
    
    prev_test_TCN_Scale = rescale_data_TCN(prev_test_TCN_unScale, Original_Dataset)
    
    return prev_test_TCN_Scale



def historical_forecast_TCN(nb_in, nb_out, model_TCN, final_df, val_en_transformed, train_en_transformed) : 
    # Glue up the training and validation set :
    combined_series = train_en_transformed.append(val_en_transformed)  
    print(combined_series)
    
    # Make historical forecast :
    histo_reg_TCN  = model_TCN.historical_forecasts(
    series = combined_series['QUANTITE'],
    past_covariates = combined_series[[x for x in combined_series.columns if x != 'QUANTITE']],
    forecast_horizon = nb_out,  # forecast horizon
    stride = 15,  # generate a forecast at every time step
    retrain = False,  # whether to retrain the model at every step
    verbose = False
    )
    
    rescale_data_TCN(histo_reg_TCN, final_df)
    
    return histo_reg_TCN 



        

def train_TCN(train_en_transformed, val_en_transformed, nb_in, nb_out) :
    
    model_TCN = TCNModel(
    input_chunk_length= nb_in,
    output_chunk_length= nb_out,
    n_epochs=50,
    dropout=0.1,
    dilation_base=2,
    weight_norm=True,
    kernel_size= 7,
    num_filters= len(train_en_transformed.columns),
    nr_epochs_val_period=1,
    random_state=0,
    )
    
    model_TCN.fit( series = train_en_transformed['QUANTITE'],
    past_covariates = train_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
    val_series = val_en_transformed['QUANTITE'],
    val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
    verbose=False)
    
    # prev_test_TCN = model_TCN.predict(
    # n = - nb_pred,
    # series =  val_en_transformed['QUANTITE'][:nb_pred], 
    # past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:-nb_out], num_samples = 100, verbose=True)
    return  model_TCN



class MyTCNWrapper(RegressionModel):
    def __init__(self, model_TCN):
        super().__init__()
        self.model_TCN = model_TCN

    def fit(self, *args, **kwargs):
        return self.model_TCN.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model



# # fearture explanation 
# def GetFeaturesInterpretation_TCN(nb_pred, nb_in, model_TCN, train_en_transformed, val_en_transformed, feature_forecast, x_future, final_df, target, exogenous) : 
#     # Create a wrapper function for the model's predict method
#     # Initialize a SHAP explainer (Deep or Gradient)
#     # Note: You need to adapt this part based on your model and data specifics
#     shap_explain = ShapExplainer(model = model_TCN)
#     results = shap_explain.explain(feature_forecast)
#     shap_explain.summary_plot()
    
#     return results

#########
# This version of the feature interpretation use captum feature interpretation technics :
# the features are assumed independant : 

def GetFeaturesInterpretation_TCN(nb_pred, nb_in, model_TCN, train_en_transformed, val_en_transformed, feature_forecast, x_future, final_df, target, exogenous):
    
    # Convert the TimeSeries object to a Pandas DataFrame for SHAP
    future_df = feature_forecast.pd_dataframe() 
    # Define a wrapper function for your model's forecast function
    def model_wrapper(x):

            
        # Convert x (which will be a NumPy array) back to a TimeSeries object
        x_df = pd.DataFrame(x, columns= x_future.columns)
        x_timeseries = transforme_data(x_df, 'DATE_TMP')
        
        # Vérifier les indices de temps
        last_index_val = val_en_transformed[exogenous].time_index[-1]
        first_index_feature = x_timeseries.time_index[0]

        # S'assurer que les séries temporelles sont contiguës
        if last_index_val + pd.Timedelta(1, unit='D') != first_index_feature:
            # Call the forecast function
            end_index = feature_forecast[exogenous].time_index.get_loc(first_index_feature)
            sliced_ts = feature_forecast[exogenous][:end_index]
            sliced_ts = sliced_ts[~sliced_ts.index.duplicated(keep='first')]
            # Reset the index before appending
            sliced_ts = sliced_ts.reset_index(drop=True)

            x_timeseries = sliced_ts.append(x_timeseries[exogenous])
       
            forecast = forecast_TCN(len(x), nb_in, model_TCN,  val_en_transformed, x_timeseries, final_df, target, exogenous)
            
        else : 
            forecast = forecast_TCN(len(x), nb_in, model_TCN,  val_en_transformed, x_timeseries, final_df, target, exogenous)
        # Convert the forecast TimeSeries back to a NumPy array for SHAP
        return forecast

    future_value_forShapTrain =  x_future.values 
    # Create the SHAP explainer using the wrapper function and the DataFrame
    explainer = shap.KernelExplainer(model_wrapper,  future_value_forShapTrain[:100])
    # print(explainer)
    # Ca me sempble un peu compliqué mais : 
    # On va utiliser la prévision
    # Compute SHAP values for the instances you want to explain
    shap_values = explainer.shap_values(future_value_forShapTrain[:100])
    
    print(shap_values)

    return shap_values









# #It might be near try to use shap on a darts TCN 

def GetFeaturesInterpretation_TCN(nb_pred, nb_in, model_TCN, train_en_transformed, val_en_transformed, feature_forecast, x_future, final_df, target, exogenous):
    
    # Convert the TimeSeries object to a Pandas DataFrame for SHAP
    future_df = feature_forecast.pd_dataframe() 
    # Define a wrapper function for your model's forecast function
    def model_wrapper(x):

            
        # Convert x (which will be a NumPy array) back to a TimeSeries object
        x_df = pd.DataFrame(x, columns= x_future.columns)
        x_timeseries = transforme_data(x_df, 'DATE_TMP')
        
        # Vérifier les indices de temps
        last_index_val = val_en_transformed[exogenous].time_index[-1]
        first_index_feature = x_timeseries.time_index[0]

        # S'assurer que les séries temporelles sont contiguës
        if last_index_val + pd.Timedelta(1, unit='D') != first_index_feature:
            # Call the forecast function
            end_index = feature_forecast[exogenous].time_index.get_loc(first_index_feature)
            sliced_ts = feature_forecast[exogenous][:end_index]
            sliced_ts = sliced_ts[~sliced_ts.index.duplicated(keep='first')]
            # Reset the index before appending
            sliced_ts = sliced_ts.reset_index(drop=True)

            x_timeseries = sliced_ts.append(x_timeseries[exogenous])
       
            forecast = forecast_TCN(len(x), nb_in, model_TCN,  val_en_transformed, x_timeseries, final_df, target, exogenous)
            
        else : 
            forecast = forecast_TCN(len(x), nb_in, model_TCN,  val_en_transformed, x_timeseries, final_df, target, exogenous)
        # Convert the forecast TimeSeries back to a NumPy array for SHAP
        return forecast

    future_value_forShapTrain =  x_future.values 
    # Create the SHAP explainer using the wrapper function and the DataFrame
    explainer = shap.KernelExplainer(model_wrapper,  future_value_forShapTrain[:100])
    # print(explainer)
    # Ca me sempble un peu compliqué mais : 
    # On va utiliser la prévision
    # Compute SHAP values for the instances you want to explain
    shap_values = explainer.shap_values(future_value_forShapTrain[:100])
    
    print(shap_values)

    return shap_values


# def GetFeaturesInterpretation_TCN(nb_in, model_TCN, val_en_transformed, x_future, final_df, target, exogenous):
#     # Convert the TimeSeries object to a Pandas DataFrame for SHAP
#     col_names = x_future.columns
#     nb_pred = len(x_future)

#     # Define a wrapper function for your model's forecast function
#     def model_wrapper(x): 
        
#         # Convert x (NumPy array) to DataFrame
#         x_df = pd.DataFrame(x, columns=col_names)
#         print(f'The number of features is {len(col_names)}')

#         # Call the forecast function and get forecast
#         forecast = forecast_TCN(nb_pred, nb_in, model_TCN, val_en_transformed, x_df , final_df, target, exogenous)
#         # forecast =  historical_forecast_TCN(nb_out, nb_in, model_TCN, final_df, val_en_transformed, train_en_transformed)
#         # Return forecast values as NumPy array
        
#         return forecast

#     # Convert final_df to NumPy array for SHAP explainer
#     future_array = x_future.values
    
#     # Create the SHAP explainer using the wrapper function and the DataFrame
#     explainer = shap.KernelExplainer(model_wrapper, future_array[:-120])
#     print(f'explainer is ok !')

#     # Compute SHAP values for the instances you want to explain
#     shap_values = explainer.shap_values(future_array[-119:])
#     print(f'it the shap value obtained {shap_values}')
#     return shap_values
    

        

    


# Il faut que je change tout les -60 qui on peut lieu d'être pour les hypers paramêtres mais ça reste à vérifier : 
def process_product_TCN(x_future_scaled, x_future, train_en_transformed, val_en_transformed, final_df, target, exogenous) :
    
    nb_pred = len(x_future)
    nb_in = 15
    nb_out = 10

    #Creat a dictionary which contains all information needed 
    preds = {}

    # forecasting with the last price :
    model = train_TCN(train_en_transformed, val_en_transformed, nb_in, nb_out)
    
    
    preds['QUANTITE_AJUSTE'] = historical_forecast_TCN(nb_in, nb_out, model, final_df, val_en_transformed, train_en_transformed)
    preds['QUANTITE_0'] = forecast_TCN(nb_pred, nb_in, model, val_en_transformed, x_future_scaled, final_df, target, exogenous)
    
    # We are testing the prevision for five differents prices : 
    # Remark in the Deep learning cases we have to scale the feature for each prices. 
    prix_min = min(final_df['PARAM_PRIX'])
    prix_max = max(final_df['PARAM_PRIX'])
    last_price = final_df['PARAM_PRIX'][-1]
    vec_prix_test = [prix_min + i*(prix_max - prix_min) / (NB_PRIX - 1 ) for i in range(NB_PRIX)]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()
    
    
    
    # Change prediction values with the price :
    cpt=0
    for prix in vec_prix_test : 
        cpt+=1
        x_future['PARAM_PRIX'] = [prix] * nb_pred
        #Scaling : 
        x_future_scaled = transforme_data( x_future, 'DATE_TMP')
        preds[f'PRIX_{cpt}'] = forecast_TCN(nb_pred, nb_in, model, val_en_transformed, x_future_scaled, final_df, target, exogenous)
        # x_future_scaled = rescale_data_TCN(preds, x_future)
 
 
    # Change prediction values with the promotion value :  
       
    x_future['PARAM_PRIX'] = [last_price] * nb_pred 
   
    for promo in vec_promo_test : 

        x_future['PARAM_PROMO'] = [promo] * nb_pred
        #Scaling : 
        x_future_scaled = transforme_data(x_future, 'DATE_TMP')
        preds[f'PROMO_{int(promo*100)}'] = forecast_TCN(nb_pred, nb_in, model, val_en_transformed, x_future_scaled, final_df, target, exogenous)
    
    coefficients = GetFeaturesInterpretation_TCN(nb_pred, nb_in, model, train_en_transformed, val_en_transformed, x_future_scaled, x_future, final_df,  target, exogenous)
    preds['ELASTICITE'] = coefficients

    
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



#df_smooth_1_histo_day_avg
def Use_Create_Study(objective_tcn) : 
    
    early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
    callback = [early_stopper]
    
    study_TCN = optuna.create_study(direction="minimize")
    study_TCN.optimize(objective_tcn, n_trials=100, callbacks=[print_callback])
    
    

# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

def model_tcn_prev(study, train_transformed, val_transformed) :
    early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [early_stopper]
    
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0
    
    pl_trainer_kwargs = {
        "accelerator": "auto",
        "callbacks": callbacks,
    }

    model_tcn = TCNModel(
        input_chunk_length = study.best_params['in_len'],
        output_chunk_length = study.best_params['out_len'],
        n_epochs = 50,
        dropout = study.best_params['dropout'],
        dilation_base = study.best_params['dilation_base'],
        weight_norm = study.best_params['weight_norm'],
        kernel_size = study.best_params['kernel_size'],
        num_filters = study.best_params['num_filters'],
        optimizer_kwargs = {"lr": study.best_params['lr']},
        pl_trainer_kwargs = pl_trainer_kwargs,
        nr_epochs_val_period = 1,
        random_state =0,
    )

    model_tcn.fit(series = train_transformed['QUANTITE'],
        past_covariates =  train_transformed[[x for x in train_transformed.columns if x != 'QUANTITE']],
        val_series = val_transformed['QUANTITE'],
        val_past_covariates = val_transformed[[x for x in train_transformed.columns if x != 'QUANTITE']],
        verbose = False,
        num_loader_workers = num_workers
    )

    preds = model_tcn.predict(n= -NB_JOUR_PRED, series = val_transformed['QUANTITE'][:NB_JOUR_PRED], past_covariates = val_transformed[[x for x in val_transformed.columns if x != 'QUANTITE']] )
    print(preds)
    return preds




# Creat an hyperparameter study 
def creat_study_Darts(train_en_transformed,val_en_transformed) : 
    id_so = 805
    NB_JOUR_PRED = -120 
    nb_in = 30
    nb_out = 15
    cpt = 0 
    dic_result_pred = {}
    # The docment is to keep track of the experiment : 
    file_path = r'C:\Users\jcric\testApi.txt' 
    # the 'a' mode the program will add to the end of the txt without erasing any thing : 
    with open(file_path , 'a') as f :
        
        def objective_TCN(trial):
            # select input and output chunk lengths
            in_len = trial.suggest_int("in_len", 12, 36)
            out_len = trial.suggest_int("out_len", 1, in_len-1)

            # Other hyperparameters
            kernel_size = trial.suggest_int("kernel_size", 2, 5)
            num_filters = trial.suggest_int("num_filters", 1, 5)
            weight_norm = trial.suggest_categorical("weight_norm", [False, True])
            dilation_base = trial.suggest_int("dilation_base", 2, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
            include_year = trial.suggest_categorical("year", [False, True])

            # throughout training we'll monitor the validation loss for both pruning and early stopping
            pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=5, verbose=True)
            callbacks = [pruner, early_stopper]

            # detect if a GPU is available
            if torch.cuda.is_available():
                num_workers = 4
            else:
                num_workers = 0

            pl_trainer_kwargs = {
                "accelerator": "auto",
                "callbacks": callbacks,
            }

            # optionally also add the (scaled) year value as a past covariate
            if include_year:
                encoders = {"datetime_attribute": {"past": ["year"]},
                            "transformer": Scaler()}
            else:
                encoders = None

            # reproducibility
            torch.manual_seed(42)
            


            # build the TCN model
            model_tcn = TCNModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                batch_size=32,
                n_epochs=100,
                nr_epochs_val_period=1,
                kernel_size=kernel_size,
                num_filters=num_filters,
                weight_norm=weight_norm,
                dilation_base=dilation_base,
                dropout=dropout,
                optimizer_kwargs={"lr": lr},
                add_encoders=encoders,
                likelihood=GaussianLikelihood(),
                pl_trainer_kwargs=pl_trainer_kwargs,
                model_name="tcn_model",
                force_reset=True,
                save_checkpoints=True,
            )


            # when validating during training, we can use a slightly longer validation

            model_tcn.fit(series = train_en_transformed['QUANTITE'],
                past_covariates =  train_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
                val_series = val_en_transformed['QUANTITE'][:NB_JOUR_PRED ] ,
                val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:NB_JOUR_PRED],
                verbose = False,
                num_loader_workers = num_workers
            )
            
            #### Question 1: Is the error computed on the original scale data, or should the optimization be done on the scaled data?

            preds = model_tcn.predict(n= -NB_JOUR_PRED, series= val_en_transformed['QUANTITE'][:NB_JOUR_PRED], past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']])
            rmse_pred  = rmse(val_en_transformed['QUANTITE'][NB_JOUR_PRED:], preds, n_jobs=-1, verbose=True)
            rmse_pred_val = np.mean(rmse_pred)

            return rmse_pred_val if rmse_pred_val != np.nan else float("inf")
        
        
        
                
        storage_url = "mysql://root:Booper2014%40@localhost/example"
        study_name = f"distributed-exampleTCN{id_Produit}"

        try:
            study_TCN = optuna.study.load_study(study_name=study_name, storage=storage_url)
            print("Study loaded successfully.")
        except KeyError:
            study_TCN = optuna.create_study(
                storage=storage_url,
                direction="minimize",
                study_name=study_name
            )
        
        if len(study_TCN.trials) < 50 : 
            study_TCN.optimize(objective_TCN, n_trials=30, callbacks=[print_callback])
            # Kill the connection after optimization
            connection = pymysql.connect(host='localhost',
                                        user='root',
                                        password='Booper2014@',
                                        db='example')
            
            try:
                with connection.cursor() as cursor:
                    # Get connection ID
                    try:
                        cursor.execute("SELECT CONNECTION_ID();")
                        connection_id = cursor.fetchone()[0]
                        print(f"The process we want to kill: {connection_id }")
                    
                        # Kill connection
                        cursor.execute(f"KILL {connection_id};")
                        print(f"KILL {connection_id};")
                        
                    except exc.OperationalError as e:
                        if e.orig.args[0] == 1317:  # Check if error code is 'Query execution was interrupted'
                            print("Query was interrupted!")
                            # Handle the error as necessary, e.g., retry the operation, log the error, etc.
                        else:
                            print(f"OperationalError occurred: {e.orig.args[1]}")
                        
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
            finally:
                connection.close()


        
        prev_optim = model_tcn_prev(study_TCN, train_en_transformed, val_en_transformed)
        preds_affichage = list(df_test_produit['QUANTITE'][:-60]) + list(pred_WellScaled(prev_optim, df_test_produit)[0])
        
        # print(preds_affichage[-60:])
        panda_dataframe_produit = pd.DataFrame({'DATE_IMPORT': df_test_produit.index, 'QUANTITE': preds_affichage})
        panda_dataframe_produit.set_index('DATE_IMPORT', inplace=True)
        dic_result_pred[id_Produit]['pred_optim'] = list(pred_WellScaled(prev_optim, df_test_produit)[0])
        vecteur_comparaison = [list(ts[0][0][0].values()[0])[0] for ts in list(val_en_transformed['QUANTITE'][NB_JOUR_PRED:])]
        dic_result_pred[id_Produit]['rmse'] = compute_rmse(vecteur_comparaison, pred_WellScaled(prev_optim, df_test_produit)[0])
        predicted_values = dic_result_pred[id_Produit]['pred_optim']
        rmse_pred = dic_result_pred[id_Produit]['rmse']
        
        
        f.write(f'Id : {id_Produit}' + '\n')
        f.write(f'predicted : {predicted_values}' + '\n')
        f.write(f'rmse : {rmse_pred}' + '\n')
        print(f'On a fini les opérations avec le produit dont lID est {id_Produit} ')