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
from darts.utils.likelihood_models import GaussianLikelihood,PoissonLikelihood, QuantileRegression
from darts.metrics import mape, mase, rmse, mae, smape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values

import scipy.stats as stats



DataVal = []
DataTrain = []
BATCH_SIZE = 32
MAX_N_EPOCHS = 50
NR_EPOCHS_VAL_PERIOD = 1
MAX_SAMPLES_PER_TS = 100
NB_JOUR_PRED = -60 



quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]


# Lisez le fichier JSON dans un DataFrame
data_setCompited = pd.read_json("data.json")



def transforme_data(data,nb_train): 
    #On veut pouvoir ressevoir une matrice en json et lui donné la bonne forme pour que l'on puisse utiliser Darts :
    df_smooth_1_histo_day_avg = (
        data.groupby(data.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    )

    #df_smooth_1_histo_day_avg

    series_en = fill_missing_values(
        TimeSeries.from_dataframe(
            data , fill_missing_dates=True, freq="D", time_col="DATE_IMPORT"
        ),
        "auto",
    )


    # list(df_smooth_1_histo.index)[nb_train_smooth_1_histo]

    # scale
    scaler_en = Scaler()
    series_en_transformed = scaler_en.fit_transform(series_en)
    train_en_transformed, val_en_transformed = series_en_transformed.split_after(
        list(data.index)[nb_train]
    )
    
    return train_en_transformed, val_en_transformed

#On continue les testes avec optuna pour voir si l'on arrive à faire mieux : 

nb_train = int(len(data_setCompited)*0.7)
train_en_transformed, val_en_transformed = transforme_data(data_setCompited, nb_train)


# define objective function
def objective_tft(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len-1)

    series_transformed = DataTrain['QUANTITE']
    future_covariates = DataTrain[[x for x in DataTrain.columns if x != 'QUANTITE']]
    
    
    #num_static_components = count_statics(train_en_transformed['QUANTITE']),
    add_encoders =  trial.suggest_categorical("add_encoders", [False, True])
    
    
    # kernel_size = trial.suggest_int("kernel_size", 2, 5)
    # num_filters = trial.suggest_int("num_filters", 1, 5)
    # weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    # dilation_base = trial.suggest_int("dilation_base", 2, 4)
    
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_year = trial.suggest_categorical("year", [False, True])
    lstm_layers = trial.suggest_int("lstm_layers",1,4)
    hidden_size = trial.suggest_int("hidden_size",32,128)
    num_attention_heads = trial.suggest_int("num_attention_heads",4,12)
    

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="train_loss")
    early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
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
    # if include_year:
    #     encoders = {"datetime_attribute": {"past": ["year"]},
    #                 "transformer": Scaler()}
    # else:
    #     encoders = None

    # reproducibility
    torch.manual_seed(42)
    


    # build the TCN model
    model_tft = TFTModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=100,
        # nr_epochs_val_period=1,
        hidden_size = hidden_size,
        lstm_layers = lstm_layers,
        num_attention_heads = num_attention_heads,
        dropout=dropout,
        likelihood= QuantileRegression(
            quantiles=quantiles
        ), 
        pl_trainer_kwargs=pl_trainer_kwargs,
        optimizer_kwargs={"lr": 1e-3},
        model_name="tft_model",
        force_reset=True,
        save_checkpoints=True,
    )


    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    #model_val_set = scaler.transform(series[-(VAL_LEN + in_len) :])
    
    # train the model
    model_tft.fit(series = series_transformed ,
        future_covariates =  future_covariates,
        val_series = val_en_transformed['QUANTITE'][:-60] ,
        val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:-60],
        verbose = True,
        num_loader_workers = num_workers
    )

    # reload best model over course of training
    # model = TFTModel.load_from_checkpoint("tft_model")

    # pred_series = model_TFTModel.predict(n= - nb_joursPred, series = val_en_transformed['QUANTITE'][:nb_joursPred], future_covariates =  val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']])
    # Evaluate how good it is on the validation set, using sMAPE
    
    
    preds = model_tft.predict(n= -NB_JOUR_PRED, series=DataVal['QUANTITE'][:NB_JOUR_PRED], past_covariates = DataVal[[x for x in train_en_transformed.columns if x != 'QUANTITE']] , future_covariates = DataVal[[x for x in train_en_transformed.columns if x != 'QUANTITE']] )
    smapes = smape(DataVal['QUANTITE'][NB_JOUR_PRED:], preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")


# df_smooth_0_histo_day = df_smooth_0_histo.resample('D').first() # remarque on ne peut utiliser cette ligne qu'avec le temps placé en index
dates_du_produit = list(set(df_smooth_0_histo['DATE_IMPORT']))

df_smooth_1_histo_day_avg = (
    df_smooth_1_histo.groupby(df_smooth_1_histo.index.astype(str).str.split(" ").str[0]).mean().reset_index()
)

#df_smooth_1_histo_day_avg

# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":
    study_TFT = optuna.create_study(
        direction="minimize")
    study_TFT.optimize(objective_tft, n_trials=100, callbacks=[print_callback])

def model_tft_prev(study) :
    early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    pl_trainer_kwargs = {
        "accelerator": "auto",
        "callbacks": callbacks,
    }

    
    model_tft = TFTModel(
            input_chunk_length = study_TFT.best_params['in_len'],
            output_chunk_length= study_TFT.best_params['out_len'],
            batch_size=32,
            n_epochs=100,
            # nr_epochs_val_period=1,
            hidden_size = study_TFT.best_params['hidden_size'],
            lstm_layers = study_TFT.best_params['lstm_layers'],
            num_attention_heads = study_TFT.best_params['num_attention_heads'],
            dropout = study_TFT.best_params['dropout'],
            add_encoders = study_TFT.best_params['add_encoders'],
            likelihood= QuantileRegression(
                quantiles=quantiles
            ), 
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs={'lr': study_TFT.best_params['lr']},
    )

    model_tft.fit(series = series_transformed ,
        future_covariates =  future_covariates,
        past_covariates= future_covariates,
        verbose = True,
        num_loader_workers = num_workers
    )

    preds = model_tft.predict(n= -NB_JOUR_PRED, series = DataVal['QUANTITE'][:NB_JOUR_PRED], past_covariates = DataVal[[x for x in DataTrain.columns if x != 'QUANTITE']] , future_covariates = DataVal[[x for x in DataTrain.columns if x != 'QUANTITE']] )

    return preds 

