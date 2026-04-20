import numpy as np
import pandas as pd
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import xgboost
import lightgbm
import warnings
import ast
from datetime import datetime
import time
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.core')
warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")

NB_PRIX = 5
n_jobs = -1
VEC_PRIX_FACTORS = [0.80, 0.90, 1.00, 1.10, 1.20]

# Seuils segmentation 
ADI_SEUIL = 1.32
CV2_SEUIL = 0.49


# CLASSES MODÈLES
class XGBRegressorInt(xgboost.XGBRegressor):
    #Prédit en float32 — les slow movers (0.3 u/j) ne tombent pas à 0
    def predict(self, data):
        _y = super().predict(data)
        return np.maximum(np.asarray(_y, dtype=np.float32), 0.0)


class LightBMRegressorInt(lightgbm.LGBMRegressor):
    #Prédit en float32 — les slow movers (0.3 u/j) ne tombent pas à 0
    def predict(self, data):
        _y = super().predict(data)
        return np.maximum(np.asarray(_y, dtype=np.float32), 0.0)




# SEGMENTATION ADI / CV2
def _segment_produit(y):
 
    ventes_positives = y[y > 0]
    nb_periodes      = len(y)
    nb_ventes        = len(ventes_positives)

    if nb_ventes == 0:
        return 'lumpy', 999.0, 999.0

    adi = nb_periodes / nb_ventes
    cv2 = (ventes_positives.std() / max(ventes_positives.mean(), 1e-6)) ** 2

    if adi <= ADI_SEUIL and cv2 <= CV2_SEUIL:
        segment = 'fast_regulier'
    elif adi <= ADI_SEUIL and cv2 > CV2_SEUIL:
        segment = 'fast_irregulier'
    elif adi > ADI_SEUIL and cv2 <= CV2_SEUIL:
        segment = 'slow'
    else:
        segment = 'lumpy'

    return segment, round(adi, 3), round(cv2, 3)


# UTILITAIRES MÉTRIQUES
def _wmape(y_pred, y_true): 
    #WMAPE en % — robuste fast ET slow
    #Si total réel < 1 (quasi nul) → fallback MAE normalisé sur 1 unité pour éviter division par zéro, mais toujours en %.
    
    denom = max(float(np.sum(y_true)), 1.0)
    return float(np.sum(np.abs(y_pred - y_true)) / denom) * 100


def _to_list_dict(values, index):
    return [{"date": str(d), "value": str(v)} for d, v in zip(index, values)]


def get_sign(number):
    if number > 0: return  1
    if number < 0: return -1
    return 0


def get_profil_produit(final_df, target):
    
    y = final_df[target]
    segment, adi, cv2 = _segment_produit(y)

    labels = {
        'fast_regulier':   'Fast Régulier',
        'fast_irregulier': 'Fast Irrégulier',
        'slow':            'Slow Moving',
        'lumpy':           'Lumpy',
    }

    return {
        'segment':    segment,
        'libelle':    labels.get(segment, segment),
        'adi':        round(adi, 3),
        'cv2':        round(cv2, 3),
    }


# EARLY STOPPING 
def _get_early_stopping_val(X_train, y_train, val_days=30):
  
    n = len(X_train)
    n_val = min(val_days, max(int(n * 0.10), 10))
    return (X_train.iloc[:-n_val], X_train.iloc[-n_val:],
            y_train.iloc[:-n_val], y_train.iloc[-n_val:])



# HYPERPARAMÉTRAGE XGBoost
def XgBoostRegressor(X_train, y_train, patience=3, tol=1e-4):
    param_grid = {
        'objective':        ['reg:squarederror'],
        'learning_rate':    [0.03],
        'reg_lambda':       [1],
        'n_estimators':     [500],            
        'max_depth':        [3, 5, 7],
        'min_child_weight': [6, 7, 8, 9],
        'gamma':            [0],
        'colsample_bytree': [0.8],
    }

    n_folds = 3
    if n_folds > X_train.shape[0]:
        raise ValueError(f"Plis ({n_folds}) > échantillons ({X_train.shape[0]})")

    tscv     = TimeSeriesSplit(n_splits=n_folds)
    grid_xgb = DaskGridSearchCV(XGBRegressorInt(random_state=42),
                                param_grid, n_jobs=n_jobs, cv=tscv)

    best_score = -np.inf; no_improvement = 0
    start_time = time.time(); max_time = 20

    try:
        grid_xgb.fit(X_train, y_train)
        results = pd.DataFrame(grid_xgb.cv_results_)
        for _, row in results.iterrows():
            score = row['mean_test_score']
            if score > best_score + tol:
                best_score = score; no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= patience or (time.time() - start_time) > max_time:
                print("XGB Early stopping GridSearch activé")
                break
        best_params = results.loc[results['mean_test_score'].idxmax(), "params"]
    except Exception as e:
        print(f"Erreur GridSearchCV XGB : {e}")
        best_params = {
            'objective': 'reg:squarederror', 'learning_rate': 0.03,
            'reg_lambda': 1, 'n_estimators': 1000, 'max_depth': 3,
            'min_child_weight': 6, 'gamma': 0, 'colsample_bytree': 0.8,
        }

    # Réentraînement sur 100% des données 
    params    = {**best_params, 'random_state': 42, 'n_estimators': 1000}
    model_xgb = XGBRegressorInt(**params)
    model_xgb.fit(X_train, y_train)

    return model_xgb, {k: v for k, v in best_params.items()}


# HYPERPARAMÉTRAGE LightGBM
def LightBMRegressor(X_train, y_train, patience=3, tol=1e-4):
    param_grid = {
        'num_leaves':        [8, 16, 32, 64],
        'max_depth':         [3, 5, 7],
        'reg_lambda':        [1],
        'min_child_samples': [4, 6, 8],
        'learning_rate':     [0.03],
        'subsample':         [0.6],
        'subsample_freq':    [1],          
        'colsample_bytree':  [0.8],
        'n_estimators':      [500],        
        'force_row_wise':    [True],
        'verbose':           [-1],
    }

    n_folds = 3
    if n_folds > X_train.shape[0]:
        raise ValueError(f"Plis ({n_folds}) > échantillons ({X_train.shape[0]})")

    tscv     = TimeSeriesSplit(n_splits=n_folds)
    grid_lgb = DaskGridSearchCV(LightBMRegressorInt(random_state=42),
                                param_grid, n_jobs=n_jobs, cv=tscv)

    best_score = -np.inf; no_improvement = 0
    start_time = time.time(); max_time = 20

    try:
        grid_lgb.fit(X_train, y_train)
        results = pd.DataFrame(grid_lgb.cv_results_)
        for _, row in results.iterrows():
            score = row['mean_test_score']
            if score > best_score + tol:
                best_score = score; no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= patience or (time.time() - start_time) > max_time:
                print("LGBM Early stopping GridSearch activé")
                break
        best_params = results.loc[results['mean_test_score'].idxmax(), "params"]
    except Exception as e:
        print(f"Erreur GridSearchCV LGBM : {e}")
        best_params = {
            'num_leaves': 16, 'max_depth': 5, 'reg_lambda': 1,
            'min_child_samples': 4, 'learning_rate': 0.03,
            'subsample': 0.6, 'subsample_freq': 1, 'colsample_bytree': 0.8,
            'n_estimators': 1000, 'force_row_wise': True, 'verbose': -1,
        }

    # Réentraînement sur 100% des données 
    params    = {**best_params, 'random_state': 42, 'verbose': -1, 'n_estimators': 1000}
    model_lgb = LightBMRegressorInt(**params)
    model_lgb.fit(X_train, y_train)

    return model_lgb, {k: v for k, v in best_params.items()}


# GET : réentraîne sur 100% des données avec les hyperparamètres stockés en base
def GETXgboostRegressor(X_train, y_train, best_hyperparam):
    params = {**best_hyperparam, 'random_state': 42}
    model  = XGBRegressorInt(**params)
    model.fit(X_train, y_train)
    return model


def GETLightBMRegressor(X_train, y_train, best_hyperparam):
    params = {**best_hyperparam, 'random_state': 42, 'verbose': -1}
    model  = LightBMRegressorInt(**params)
    model.fit(X_train, y_train)
    return model





# SPLIT & SÉLECTION MODÈLE
def split_df(x_final):
    nb_jour = min(365, int(len(x_final) * 0.3))
    return x_final[:-nb_jour], x_final[-nb_jour:]


def ModelChoice(final_df, exogenous, target):
    start_time = time.time()
    y = final_df[target]

    
    segment, adi, cv2 = _segment_produit(y)
    print(f"  Segment : {segment.upper()} (ADI={adi}, CV2={cv2})")

    
    X_train, X_test = split_df(final_df)
    sum_target      = float(X_test[target].sum())

    print("XGB train")
    model_xgb, best_params_xgb = XgBoostRegressor(X_train[exogenous], X_train[target])
    y_pred_xgb = ModelPrediction(model_xgb, X_test[exogenous])
    error_xgb  = _wmape(y_pred_xgb, X_test[target].values)

    print("LGBM train")
    model_Light, best_params_Light = LightBMRegressor(X_train[exogenous], X_train[target])
    y_pred_Light = ModelPrediction(model_Light, X_test[exogenous])
    error_Light  = _wmape(y_pred_Light, X_test[target].values)

    errors     = {'xgboost': error_xgb, 'lightgbm': error_Light}
    best_model = min(errors, key=errors.get)
    best_error = errors[best_model]

    if best_model == 'xgboost':
        print("XGB final")
        model       = GETXgboostRegressor(final_df[exogenous], final_df[target], best_params_xgb)
        best_params = best_params_xgb
        y_pred_best = y_pred_xgb
    else:
        print("LGBM final")
        model       = GETLightBMRegressor(final_df[exogenous], final_df[target], best_params_Light)
        best_params = best_params_Light
        y_pred_best = y_pred_Light

    mae  = float(mean_absolute_error(y_pred_best, X_test[target]))
    rmse = float(np.sqrt(mean_squared_error(y_pred_best, X_test[target])))

    
    error_booper   = float(abs(np.sum(y_pred_best) - sum_target))
    pct_err_booper = (error_booper / max(sum_target, 1.0)) * 100
    accuracy_global = max(100 - pct_err_booper, 0)

   
    accuracy_wmape = max(100 - best_error, 0)

    print(f"  Temps : {(time.time() - start_time):.1f}s | "
          f"Modèle : {best_model} | WMAPE={best_error:.1f}% | Booper={error_booper:.1f}")

    return (best_model, model, best_error, accuracy_wmape,
            accuracy_global, best_params, mae, rmse, error_booper)



# ÉLASTICITÉS PAR PERTURBATION : e = (ΔQ/Q) / (ΔX/X) 
def compute_elasticites(final_df, target, exogenous, model):

    SKIP     = {'PARAM_ANNEE', 'PARAM_JOUR'}
    BINAIRES = {'PARAM_VACANCE', 'PARAM_STOCK'}
    ORDINAUX = {'PARAM_MOIS', 'PARAM_JOUR_SEMAIN'}

    df     = final_df.copy()
    Q_base = model.predict(df[exogenous])
    Q_sum  = max(float(Q_base.sum()), 1.0)
    result = {}

    for feat in exogenous:

        if feat in SKIP:
            continue
        if feat not in df.columns or df[feat].std() < 1e-6:
            result[feat] = 0.0
            continue

        # BINAIRES 0/1 
        if feat in BINAIRES or feat.startswith('PARAM_JS_'):
            mask_0 = df[feat] == 0
            if mask_0.sum() < 10:
                result[feat] = 0.0
                continue
            df_0       = df[mask_0].copy()
            df_1       = df_0.copy()
            df_1[feat] = 1
            Q_0 = model.predict(df_0[exogenous]).sum()
            Q_1 = model.predict(df_1[exogenous]).sum()
            e   = (Q_1 - Q_0) / max(Q_0, 1.0)

        # ORDINAUX 
        elif feat in ORDINAUX:
            vals    = sorted(df[feat].unique())
            impacts = [float(model.predict(df.assign(**{feat: v})[exogenous]).sum())
                       for v in vals]
            q_med  = np.median(impacts)
            q_high = np.percentile(impacts, 75)
            e      = (q_high - q_med) / max(q_med, 1.0)

        # PRIX : multi-chocs [-1%, -5%, -10%] 
        elif feat == 'PARAM_PRIX':
            shocks        = [-0.01, -0.05, -0.10]
            elasticities  = []
            for shock in shocks:
                df_p       = df.copy()
                df_p[feat] = df_p[feat] * (1 + shock)
                Q_p        = model.predict(df_p[exogenous]).sum()
                dQ         = (Q_p - Q_sum) / Q_sum
                elasticities.append(dQ / shock)
            e = min(float(np.median(elasticities)), 0.0)

        # PROMO continue [0,1] : profondeur remise 
        # PARAM_PROMO = 0.20 → remise de 20%
        # Interprétation : e=1.5 → remise 20% → (1.5 × 20%) +30% ventes 
        elif 'PARAM_PROMO' in feat:
            mask_promo = df[feat] > 0
            mask_sans  = df[feat] == 0
            nb_promo   = int(mask_promo.sum())
            nb_sans    = int(mask_sans.sum())

            if nb_promo < 5:
                result[feat] = 0.0
                continue

            q_avec_jour = float(final_df[mask_promo][target].mean())
            q_sans_jour = float(final_df[mask_sans][target].mean()) if nb_sans > 0 else 0.0

            # Cas prioritaire : produit vendu UNIQUEMENT en promo
            if q_sans_jour < 0.01 and q_avec_jour > 0.0:
                result[feat] = 1.0
                continue

            # Garde empirique 15% : pas d'effet promo actionnable
            ratio_emp = abs(q_avec_jour - q_sans_jour) / max(q_sans_jour, 0.01)
            if ratio_emp < 0.15:
                result[feat] = 0.0
                continue

            # Référence : tout le df sans promo
            df_ref = df.copy()
            df_ref[feat] = 0.0
            Q_ref = float(model.predict(df_ref[exogenous]).sum())

            # Multi-niveaux de profondeur → médiane des élasticités par niveau
            promo_levels = [0.10, 0.20, 0.30, 0.40, 0.50]
            elasticities = []
            for level in promo_levels:
                df_p       = df.copy()
                df_p[feat] = level
                Q_p        = float(model.predict(df_p[exogenous]).sum())
                dQ         = (Q_p - Q_ref) / max(Q_ref, 1.0)
                elasticities.append(dQ / level)   

            e = max(float(np.median(elasticities)), 0.0)

            # Garde modèle 5% : filtre le bruit
            if e < 0.05:
                result[feat] = 0.0
                continue

        # CONCURRENT : cross-élasticité prix
        elif 'PARAM_CONC' in feat:
            feat_mean = float(df[feat].mean())
            if abs(feat_mean) < 1e-6:
                result[feat] = 0.0
                continue
            shocks       = [0.01, 0.05, 0.10]
            elasticities = []
            for shock in shocks:
                df_p       = df.copy()
                df_p[feat] = df_p[feat] * (1 + shock)
                Q_p        = model.predict(df_p[exogenous]).sum()
                dQ         = (Q_p - Q_sum) / Q_sum
                elasticities.append(dQ / shock)
            e = max(float(np.median(elasticities)), 0.0)

        # CONTINUS : multi-chocs [+1%, +5%] 
        else:
            feat_mean = float(df[feat].mean())
            if abs(feat_mean) < 1e-6:
                result[feat] = 0.0
                continue
            shocks       = [0.01, 0.05]
            elasticities = []
            for shock in shocks:
                df_p       = df.copy()
                df_p[feat] = df_p[feat] * (1 + shock)
                Q_p        = model.predict(df_p[exogenous]).sum()
                dQ         = (Q_p - Q_sum) / Q_sum
                elasticities.append(dQ / shock)
            e = float(np.median(elasticities))

        result[feat] = round(float(e), 4)

    return result



# def compute_elasticites(final_df, target, exogenous, model):
#     SKIP     = {'PARAM_ANNEE', 'PARAM_JOUR'}
#     BINAIRES = {'PARAM_VACANCE', 'PARAM_STOCK'}
#     ORDINAUX = {'PARAM_MOIS', 'PARAM_JOUR_SEMAIN'}

#     df     = final_df.copy()
#     Q_base = model.predict(df[exogenous])
#     Q_sum  = max(float(Q_base.sum()), 1.0)
#     result = {}

#     for feat in exogenous:
#         if feat in SKIP:
#             continue
#         if feat not in df.columns or df[feat].std() < 1e-6:
#             result[feat] = 0.0
#             continue

#         if feat in BINAIRES or feat.startswith('PARAM_JS_'):
#             mask_off = df[feat] == 0
#             if mask_off.sum() < 10:
#                 result[feat] = 0.0
#                 continue
#             df_0       = df[mask_off].copy()
#             Q_0        = model.predict(df_0[exogenous])
#             Q_sum_0    = max(float(Q_0.sum()), 1.0)
#             df_1       = df_0.copy()
#             df_1[feat] = 1
#             Q_1        = model.predict(df_1[exogenous])
#             e = float((Q_1.sum() - Q_0.sum()) / Q_sum_0)
#             e = max(min(e, 10.0), -1.0)

#         elif feat in ORDINAUX:
#             vals    = sorted(df[feat].unique())
#             impacts = [float(model.predict(df.assign(**{feat: v})[exogenous]).sum())
#                        for v in vals]
#             q_mean = np.mean(impacts)
#             q_max  = np.max(impacts)
#             e = (q_max - q_mean) / max(q_mean, 1.0)

#         elif feat == 'PARAM_PRIX':
#             df_p       = df.copy()
#             df_p[feat] = df_p[feat] * 0.99
#             Q_p        = model.predict(df_p[exogenous])
#             dQ         = (Q_p.sum() - Q_base.sum()) / Q_sum
#             e          = min(float(dQ) / (-0.01), 0.0)

#         elif 'PARAM_PROMO' in feat:
#             mask_promo = df[feat] > 0
#             if int(mask_promo.sum()) < 3:
#                 result[feat] = 0.0
#                 continue

#             # Signal 1 — perturbation modèle : promo=0 vs promo réelle sur jours promo
#             df_avec        = df[mask_promo].copy()
#             df_sans_       = df_avec.copy()
#             df_sans_[feat] = 0.0
#             Q_avec         = float(model.predict(df_avec[exogenous]).sum())
#             Q_sans         = float(model.predict(df_sans_[exogenous]).sum())
#             denom_p        = max(Q_sans, 1.0)
#             e_perturb      = max((Q_avec - Q_sans) / denom_p, 0.0)

#             # Signal 2 — OLS semi-log avec tous les controls
#             # log(Q+1) = α + β×PROMO + γ×PRIX + γ×MOIS + ...
#             # β = effet promo isolé de la saisonnalité et du prix
#             try:
#                 log_q  = np.log(df[target].clip(lower=0.01))
#                 X_ols  = df[exogenous].copy().astype(float)
#                 ols    = sm.OLS(log_q, sm.add_constant(X_ols)).fit()
#                 e_ols  = max(float(ols.params.get(feat, 0.0)), 0.0)
#             except Exception:
#                 e_ols = e_perturb

#             # Moyenne des deux signaux — fidèle au passé et au modèle
#             e = (e_perturb + e_ols) / 2.0

#         else:
#             feat_mean = float(df[feat].mean())
#             if abs(feat_mean) < 1e-6:
#                 result[feat] = 0.0
#                 continue
#             df_p       = df.copy()
#             df_p[feat] = df_p[feat] * 1.01
#             Q_p = model.predict(df_p[exogenous])
#             dQ  = (Q_p.sum() - Q_base.sum()) / Q_sum
#             e   = float(dQ) / 0.01
#             if 'PARAM_CONC' in feat: e = max(e, 0.0)

#         result[feat] = round(e, 4)

#     return result


# UTILITAIRES
def GET_Date_Product(id_produit, id_so, connection):
    with connection.cursor() as cursor:
        cursor.execute(f"""SELECT MAX(Date_Entrainement)
                           FROM TRAIN_HISTORIC
                           WHERE id_produit = {id_produit} AND id_so = {id_so}""")
        row = cursor.fetchone()
        return row[0] if row else None


def useHyperParamII(features_model, param_model):
    return ast.literal_eval(features_model), ast.literal_eval(param_model)


def ModelPrediction(model, X_test):
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0
    return y_pred


# DONNÉES
def process_data(Product_parametre_json, Product_features_json,
                 Product_quantity_json, Product_future_features_json):
    dfs = []
    if (Product_features_json and Product_quantity_json
            and len(Product_features_json) == len(Product_quantity_json)):
        for features, quantity in zip(Product_features_json, Product_quantity_json):
            try:
                df = pd.DataFrame([features])
                df['QUANTITE'] = quantity
                dfs.append(df)
            except Exception:
                print("Error processing row")
    else:
        print(f"Listes vides ou longueurs différentes : "
              f"{len(Product_features_json)} vs {len(Product_quantity_json)}")

    final_df  = pd.concat(dfs, ignore_index=True).set_index('DATE_IMPORT')
    target    = 'QUANTITE'
    exogenous = [x for x in Product_parametre_json if x != target]

    final_df[exogenous] = final_df[exogenous].apply(pd.to_numeric, errors='coerce')
    final_df[target]    = pd.to_numeric(final_df[target], errors='coerce').clip(lower=0)
    final_df.fillna(0, inplace=True)

    x_future = pd.DataFrame(Product_future_features_json)
    nb_jours = len(x_future)

    if nb_jours > 0:
        if 'DATE_TMP' in x_future.columns:
            x_future = x_future.set_index('DATE_TMP')
        for col in exogenous:
            if col in x_future.columns and x_future[col].dtype == 'object':
                x_future[col] = pd.to_numeric(x_future[col], errors='coerce')

    final_df = final_df.select_dtypes(exclude=['object'])
    return x_future, final_df, target, nb_jours, exogenous


# prédictions prix & promo (SET / GET )
def _predict_prix_promo(model, x_future, nb_jours, features, last_price):
    preds         = {}
    vec_prix_test = [last_price * f for f in VEC_PRIX_FACTORS]

    for cpt, prix in enumerate(vec_prix_test, start=1):
        x_future['PARAM_PRIX'] = [prix] * nb_jours
        preds[f'PRIX_{cpt}']   = ModelPrediction(model, x_future[features])
    x_future['PARAM_PRIX'] = [last_price] * nb_jours

    vec_promo_test  = np.arange(0, 0.95, 0.05).tolist()
    cols_type_promo = [c for c in features if c.startswith("PARAM_PROMO_")]

    if cols_type_promo:
        for col in cols_type_promo:
            for promo in vec_promo_test:
                xf      = x_future.copy()
                xf[col] = [promo] * nb_jours
                preds[f'{col}_POURCENT_{int(promo * 100)}'] = ModelPrediction(model, xf[features])
    else:
        for promo in vec_promo_test:
            xf = x_future.copy()
            xf['PARAM_PROMO'] = [promo] * nb_jours
            preds[f'PROMO_{int(promo * 100)}'] = ModelPrediction(model, xf[features])

    return preds, vec_prix_test


# GET
def GET_process_product_Version(x_future, final_df, target, nb_jours, exogenous,
                                id_produit, id_so, date_import,
                                model_name, feature_Model, parm_model):
    list_featureModel, dic_parm_model = useHyperParamII(feature_Model, parm_model)
    final_features = [i for i in list_featureModel if i != 'QUANTITE']

    # Reconstruction du modèle selon le type stocké en base
    if model_name == 'xgboost':
        model = GETXgboostRegressor(final_df[final_features], final_df[target], dic_parm_model)
    else:
        model = GETLightBMRegressor(final_df[final_features], final_df[target], dic_parm_model)

    last_price         = final_df['PARAM_PRIX'].iloc[-1]
    preds_pp, vec_prix = _predict_prix_promo(model, x_future, nb_jours, final_features, last_price)

    preds_converted = {
        'QUANTITE_AJUSTE': _to_list_dict(ModelPrediction(model, final_df[final_features]), final_df.index),
        'QUANTITE_0':      _to_list_dict(ModelPrediction(model, x_future[final_features]), x_future.index),
    }
    for key, values in preds_pp.items():
        preds_converted[key] = _to_list_dict(values, x_future.index)

    preds_converted['ELASTICITE']      = compute_elasticites(final_df, target, final_features, model)
    preds_converted['PRIX_INTERVAL']   = vec_prix
    preds_converted['PROFIL_PRODUIT']  = get_profil_produit(final_df, target)
    preds_converted['OK']              = str(1)

    return json.dumps(preds_converted)


# SET
def SET_process_product_Version(x_future, final_df, target, nb_jours,
                                exogenous, id_produit, id_so):
    (best_model, model, best_error, accuracy_wmape,
     accuracy_global, best_param, mae, rmse, error_booper) = ModelChoice(final_df, exogenous, target)

    last_price         = final_df['PARAM_PRIX'].iloc[-1]
    preds_pp, vec_prix = _predict_prix_promo(model, x_future, nb_jours, exogenous, last_price)

    preds_converted = {
        'QUANTITE_AJUSTE': _to_list_dict(ModelPrediction(model, final_df[exogenous]), final_df.index),
        'QUANTITE_0':      _to_list_dict(ModelPrediction(model, x_future[exogenous]), x_future.index),
    }
    for key, values in preds_pp.items():
        preds_converted[key] = _to_list_dict(values, x_future.index)

    preds_converted.update({
        "ERROR_GLOBAL":    error_booper,
        "ACCURACY_GLOBAL": accuracy_global,
        "ERROR_WMAPE":     best_error,
        "ACCURACY_WMAPE":  accuracy_wmape,
        "MAE_LAST_MONTH":  mae,
        "RMSE_TRAIN":      rmse,
        "FEATURES_MODEL":  str(list(exogenous)),
        "PARAM_MODEL":     str(best_param),
        "MODEL_NAME":      best_model,
        "MODEL":           best_model,
        "ELASTICITE":      compute_elasticites(final_df, target, exogenous, model),
        "PRIX_INTERVAL":   vec_prix,
        "PROFIL_PRODUIT":  get_profil_produit(final_df, target),
        "OK":              str(1),
    })

    return json.dumps(preds_converted)



# UTILITAIRE DATES
def format_date_in_predictions(preds_converted):
    for key, values in preds_converted.items():
        if isinstance(values, list) and values and isinstance(values[0], dict) and 'date' in values[0]:
            for pred in values:
                try:
                    pred['date'] = datetime.strptime(
                        pred['date'], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')
                except ValueError:
                    pass
    return preds_converted



# CHAIN
def GET_process_product_Version_chain(x_future, final_df, target, nb_jours,
                                      exogenous, id_produit, id_so, model_list):
    final_df.index = pd.to_datetime(final_df.index, format='%d/%m/%Y')

    feature_Model_list = model_list['FEATURES_MODEL'].tolist()
    model_name_list    = model_list['MODEL_NAME'].tolist()
    parm_model_list    = model_list['PARAM_MODEL'].tolist()
    date_import_list   = model_list['DATE_IMPORT'].tolist()

    sorted_indices     = sorted(range(len(date_import_list)),
                                key=lambda i: datetime.strptime(date_import_list[i], '%d/%m/%Y'))
    feature_Model_list = [feature_Model_list[i] for i in sorted_indices]
    model_name_list    = [model_name_list[i]    for i in sorted_indices]
    parm_model_list    = [parm_model_list[i]    for i in sorted_indices]
    date_import_list   = [date_import_list[i]   for i in sorted_indices]

    last_price     = final_df['PARAM_PRIX'].iloc[-1]
    vec_prix_test  = [last_price * f for f in VEC_PRIX_FACTORS]
    vec_promo_test = np.arange(0, 0.95, 0.05).tolist()

    combined_preds = {'QUANTITE_AJUSTE': [], 'QUANTITE_0': []}
    for cpt in range(1, len(vec_prix_test) + 1):
        combined_preds[f'PRIX_{cpt}'] = []
    for promo in vec_promo_test:
        combined_preds[f'PROMO_{int(promo * 100)}'] = []

    last_period_preds = {}

    for i in range(len(date_import_list)):
        start_date = datetime.strptime(date_import_list[i], '%d/%m/%Y')
        end_date   = (datetime.strptime(date_import_list[i + 1], '%d/%m/%Y')
                      if i < len(date_import_list) - 1
                      else final_df.index[-1])

        period_df     = final_df[final_df.index <= start_date].copy()
        period_future = final_df[(final_df.index >= start_date) &
                                 (final_df.index < end_date)].copy()

        if len(period_df) == 0 and len(period_future) == 0:
            continue

        period_preds_json = GET_process_product_Version(
            period_future, period_df, target, len(period_future),
            exogenous, id_produit, id_so, date_import_list[i],
            model_name_list[i], feature_Model_list[i], parm_model_list[i]
        )
        last_period_preds = json.loads(period_preds_json)

        for key in ['QUANTITE_AJUSTE', 'QUANTITE_0']:
            if key in last_period_preds:
                combined_preds[key].extend(last_period_preds[key])

        for cpt in range(1, len(vec_prix_test) + 1):
            key = f'PRIX_{cpt}'
            if key in last_period_preds:
                combined_preds[key].extend(last_period_preds[key])

        for promo in vec_promo_test:
            key = f'PROMO_{int(promo * 100)}'
            if key in last_period_preds:
                combined_preds[key].extend(last_period_preds[key])

    combined_preds['ELASTICITE']    = last_period_preds.get('ELASTICITE', {})
    combined_preds['PRIX_INTERVAL'] = vec_prix_test
    combined_preds['OK']            = str(1)

    return json.dumps(combined_preds)
