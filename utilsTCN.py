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




BATCH_SIZE = 32
MAX_N_EPOCHS = 50
NR_EPOCHS_VAL_PERIOD = 1
MAX_SAMPLES_PER_TS = 100
NB_JOUR_PRED = -60 


def train_TCN(train_en_transformed, val_en_transformed, nb_pred, nb_in, nb_out) :
    
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
    val_series = val_en_transformed['QUANTITE'][:nb_pred],
    val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:nb_pred],
    verbose=False)
    
    prev_test_TCN = model_TCN.predict(
    n = - nb_pred,
    series =  val_en_transformed['QUANTITE'][:nb_pred], 
    past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:-nb_out], num_samples = 100, verbose=True)
    
    
    return  prev_test_TCN


# Lisez le fichier JSON dans un DataFrame
BOOPERdata_set = pd.read_json("data.json")


NB_JOUR_PRED = -60

def objective_TCN(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len-1)
    
    
    #num_static_components = count_statics(train_en_transformed['QUANTITE']),
    encoders =  trial.suggest_categorical("add_encoders", [False, True])
    
    
    # kernel_size = trial.suggest_int("kernel_size", 2, 5)
    # num_filters = trial.suggest_int("num_filters", 1, 5)
    # weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    # dilation_base = trial.suggest_int("dilation_base", 2, 4)

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
    

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="train_loss")
    early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=10, verbose=True)
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
        add_encoders= encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )


    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    #model_val_set = scaler.transform(series[-(VAL_LEN + in_len) :])
    
    model.fit(series = train_en_transformed['QUANTITE'],
        past_covariates = train_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']],
        val_series = val_en_transformed['QUANTITE'][:-60] ,
        val_past_covariates = val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']][:-60],
        # max_samples_per_ts = MAX_SAMPLES_PER_TS,
        verbose = False,
        num_loader_workers = num_workers,
    )

    # reload best model over course of training
    # model = TFTModel.load_from_checkpoint("tft_model")

    # pred_series = model_TFTModel.predict(n= - nb_joursPred, series = val_en_transformed['QUANTITE'][:nb_joursPred], future_covariates =  val_en_transformed[[x for x in train_en_transformed.columns if x != 'QUANTITE']])
    # Evaluate how good it is on the validation set, using sMAPE
    
    #### Question 1: Is the error computed on the original scale data, or should the optimization be done on the scaled data?
    
    preds = model_tcn.predict(n= -NB_JOUR_PRED, series= val_transformed['QUANTITE'][:NB_JOUR_PRED], future_covariates = val_transformed[[x for x in train_transformed.columns if x != 'QUANTITE']])
    rmse_pred = rmse(val_transformed['QUANTITE'][NB_JOUR_PRED:], preds, n_jobs=-1, verbose=True)
    rmse_pred_val = np.mean(rmse_pred)

    return rmse_pred_val if rmse_pred_val != np.nan else float("inf")


#df_smooth_1_histo_day_avg

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
        val_series = val_transformed['QUANTITE'][:-60] ,
        val_past_covariates = val_transformed[[x for x in train_transformed.columns if x != 'QUANTITE']][:-60],
        verbose = False,
        num_loader_workers = num_workers
    )

    preds = model_tcn.predict(n= -NB_JOUR_PRED, series = val_transformed['QUANTITE'][:NB_JOUR_PRED], past_covariates = val_transformed[[x for x in val_transformed.columns if x != 'QUANTITE']] )
    return preds