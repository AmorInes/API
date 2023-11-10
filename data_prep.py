### In this spreadsheet, we perform data processing.

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np 
import oracledb
from datetime import datetime, timedelta


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, accuracy_score,  r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.dataprocessing.transformers import Scaler
# from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel, TFTModel, CatBoostModel, XGBModel, LightGBMModel, KalmanForecaster, TCNModel, NBEATSModel, TransformerModel
from darts.utils.likelihood_models import GaussianLikelihood,PoissonLikelihood, QuantileRegression
from darts.metrics import mape, mase, rmse, mae, smape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values




PARAMS = oracledb.ConnectParams(host="141.94.64.164", port= 1521, service_name="PRICEONE")

def add_time_series_features(df, target_col):
    """
    Adds time series features to the DataFrame.

    Parameters:
    - df: pandas DataFrame with a DateTimeIndex.
    - target_col: the name of the column containing the time series data.

    Returns:
    - df: pandas DataFrame with new features.
    """

    # Ensure the DataFrame is sorted by the index
    df = df.sort_index()

    # Daily features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear

    # Monthly features
    df['month'] = df.index.month
    df['month_start'] = df.index.is_month_start.astype(int)
    df['month_end'] = df.index.is_month_end.astype(int)

    # Annual features
    df['year'] = df.index.year

    # Rolling window features
    window_sizes = [7, 30, 90]  # Example window sizes: 1 week, 1 month, 3 months
    for window in window_sizes:
        df[f'rolling_mean_{window}d'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_var_{window}d'] = df[target_col].rolling(window=window).var()

    # Expanding window features
    df['expanding_mean'] = df[target_col].expanding().mean()
    df['expanding_var'] = df[target_col].expanding().var()

    # Drop any rows with NaN values that were created by rolling functions
    df = df.dropna()

    return df



#def test_multi_price() : 




def add_next_day(dates):
    new_dates = []
    for i in range(len(dates)):
        if not dates[i]:  # If empty
            previous_date = datetime.strptime(new_dates[-1], "%Y-%m-%d")  # Assuming date format is "YYYY-MM-DD"
            next_date = previous_date + timedelta(days=1)
            new_dates.append(next_date.strftime("%Y-%m-%d"))
        else:
            new_dates.append(dates[i])
    return new_dates



#Function to try without any call to the application in order to make ours tests : 

def connection_data_frame(id_produit, id_so) :
    header_list = []
    raw_data = []
    
    #On construit les headers
    #with oracledb.connect(user="LAPEYRE", password="LAPEYRE", params=params) as connection :
    with oracledb.connect(user="MEGA_MARKET", password="MEGA_MARKET", params = PARAMS) as connection : 
        with connection.cursor() as cursor :
            command = cursor.execute("select COLUMN_NAME from ALL_TAB_COLUMNS where TABLE_NAME=\'DONNEE_MODEL\'")
            row = command.fetchone() 
            header_list.append(row[0])
            while row :  
                row = cursor.fetchone()
                if row :
                    header_list.append(row[0])
                    
        #on construit une dataframe afin de stocker les données d'oracle
        with connection.cursor() as cursor:
            cursor.execute("select * from DONNEE_MODEL where ID_PRODUIT= %s AND ID_SO =%s order by DATE_IMPORT"%(id_produit, id_so))
            #cursor.description
            row = cursor.fetchone() 
            ligne = list(row)
            raw_data.append(ligne)
            while row :  
                row = cursor.fetchone()
                if row :
                    ligne = list(row)
                    raw_data.append(ligne)
    data_frame_produit_test = pd.DataFrame(raw_data, columns = header_list)
    return data_frame_produit_test



### (for later :) To got better performances we can use some pooling library to avoid high-demand situation :

def fast_import_fonction(id_produit, id_so): 
    ligne_magasin = []
    raw_data_magasin =[]
     
    commande_sql = "SELECT id_produit, id_so, date_import, SUM(quantite) AS quantite FROM histo_produit_releve_sop WHERE id_so = 805 AND id_produit = %s GROUP BY id_produit , id_so, date_import ORDER BY id_produit, id_so, date_import"%id_produit
    #print(commande_sql) 
    header_list = ['ID_PRODUIT', 'ID_SO', 'DATE_IMPORT', 'QUANTITE']
    list_col_NonGrata = ['ID_PRODUIT','ID_SO','PREV','DATE_IMPORT']

    with oracledb.connect(user="MEGA_MARKET", password="MEGA_MARKET", params = PARAMS) as connection:
        with connection.cursor() as cursor : 
            cursor.execute(commande_sql)
            row = cursor.fetchone()
            ligne_magasin = list(row)
            raw_data_magasin.append(ligne_magasin)
            while row :
                row = cursor.fetchone()
                if row : 
                    ligne_magasin = list(row)
                    raw_data_magasin.append(ligne_magasin)
    #on a alors notre data_frame : 
    df_temp_produit_quantite = pd.DataFrame(raw_data_magasin, columns = header_list)
    
    
    df_temp_produit = connection_data_frame(id_produit,  id_so)
    # print(max(df_temp_produit['DATE_IMPORT']))
    df_temp_produit = df_temp_produit.drop_duplicates(subset = ['DATE_IMPORT'])
    df_temp_produit = df_temp_produit.set_index('DATE_IMPORT', drop = False)
    df_temp_produit = df_temp_produit.resample('D').first()
    df_temp_produit = df_temp_produit[df_temp_produit['PREV'] == 0][:-4]
    
    for col in df_temp_produit.columns : 
        if df_temp_produit[col].isna().all() == False and col != "DATE_IMPORT" :
            df_temp_produit[col] = df_temp_produit[col].fillna((df_temp_produit[col].ffill() + df_temp_produit[col].bfill()) / 2).fillna(method='ffill').fillna(method='bfill')
        if col == "DATE_IMPORT" : 
            add_next_day(df_temp_produit[col])
            
    
    
    for sale_days in df_temp_produit['DATE_IMPORT']:
        # Get the corresponding quantity from df_temp_produit_quantite
        quantite_value = df_temp_produit_quantite.loc[df_temp_produit_quantite['DATE_IMPORT'] == sale_days, 'QUANTITE']
        
        if not quantite_value.empty:
            # Update the 'QUANTITE' column in df_temp_produit
            df_temp_produit.loc[df_temp_produit['DATE_IMPORT'] == sale_days, 'QUANTITE'] = quantite_value.values[0]
       
    for col in df_temp_produit[[x for x in df_temp_produit.columns if x not in list_col_NonGrata]].columns :
        nan_count = np.sum(pd.isna(df_temp_produit[col]))
        if (nan_count > len(df_temp_produit) * 0.5) : 
            list_col_NonGrata += [col]
        
        list_sans_na = df_temp_produit[col].fillna(method='ffill').fillna(method='bfill')
        if (np.var(list_sans_na) == 0) : 
            df_temp_produit = df_temp_produit.drop([col],axis = 1)  
            list_col_NonGrata += [col]  

    # rows_with_nan = df_temp_produit[df_temp_produit.isnull().any(axis=1)] #Pour tester 
    
    #Il faut que l'on prenne le min des dates
    
    return df_temp_produit[[x for x in df_temp_produit.columns if x not in list_col_NonGrata]]




#Transform for Darts model forecasting 

def prepareDataForDarts(df_test_produit) :
    #On commence par compléter les dates manquantes de notre base de donnée : 


    df_test_produit_day_avg = (
        df_test_produit.groupby(df_test_produit.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    )

    #df_smooth_1_histo_day_avg

    series_en = fill_missing_values(
        TimeSeries.from_dataframe(
            df_test_produit_day_avg , fill_missing_dates=True, freq="D", time_col="DATE_IMPORT"
        ),
        "auto",
    )


    # list(df_smooth_1_histo.index)[nb_train_smooth_1_histo]

    # scale
    scaler_en = Scaler()
    series_en_transformed = scaler_en.fit_transform(series_en)
    train_en_transformed, val_en_transformed = series_en_transformed.split_after(
        list(df_test_produit.index)[min(int(len(df_test_produit)*0.75),len(df_test_produit)-90)]
    )
    
    return train_en_transformed, val_en_transformed 


def pred_WellScaled(predic_non_scaled, data_corrected_test_eval):
    scaler_skQuantite = MaxAbsScaler()
    scaled_series = scaler_skQuantite.fit_transform(np.array(data_corrected_test_eval['QUANTITE']).reshape(-1,1))

    list_prev = predic_non_scaled['QUANTITE'].values()

    prediction_scaled = [x[0] for x in list_prev]
    prediction_original = scaler_skQuantite.inverse_transform(np.array(prediction_scaled).reshape(-1,1))
    data_for_metrics = np.array(prediction_original).reshape(1,-1)
    
    return data_for_metrics


# Helps to classify  times series helpfull at list of my test : 

def average_demand_interval(demand_series):
    """
    Compute the Average Demand Interval (ADI) from a given demand series.
    
    Parameters:
    - demand_series (list): A list of demand values over time.
    
    Returns:
    - float: The computed ADI value.
    """
    
    # Count non-zero demands
    non_zero_demands = sum(1 for demand in demand_series if demand > 0)
    
    # If there are no non-zero demands, return a large number to indicate infinity (or you can return None)
    if non_zero_demands == 0:
        return float('inf')
    
    # Compute the ADI
    adi = len(demand_series) / non_zero_demands
    
    return adi


def type_Produit(Liste_Quantite_vente) :
    CV_product = np.square(np.var(Liste_Quantite_vente)/ np.mean(Liste_Quantite_vente)) 
    AID_product = average_demand_interval(Liste_Quantite_vente)
    print(AID_product)
    print(CV_product)
    print(f"The Average Demand Interval (ADI) is: {AID_product}")
    if (CV_product  < 0.49) & (AID_product  > 1.32) > 0 :
        if (CV_product  < 0.49) & (AID_product  > 1.32) :
            return 'Intermittent'    
        if (CV_product  < 0.49) & (AID_product  < 1.32) :
            return 'Smooth'
        if  (CV_product  > 0.49) & (AID_product  > 1.32) :
            return 'Lumpy'
        if  (CV_product  > 0.49) & (AID_product  < 1.32) :
            return 'Erratic'
        


# Helps to compute relevant metrics; these methods are created to handle NaN values:

def compute_rmse(list1, list2):
    # Convert lists to numpy arrays for efficient computation
    

    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Ensure the arrays are of the same length
    if arr1.shape != arr2.shape:
        raise ValueError("Input lists must have the same length!")

    # Compute squared differences
    squared_diffs = (arr1 - arr2) ** 2

    # Mask where either array has a NaN
    valid_values_mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    # Calculate mean of valid squared differences
    mean_squared_diff = np.mean(squared_diffs[valid_values_mask])

    # Compute RMSE
    rmse = np.sqrt(mean_squared_diff)
    return rmse