#!/usr/bin/env python3
"""
benchmark_per_product.py

Compare deux approches de forecasting :
  A) Per-product : 1 modele (XGB ou LightGBM) par couple (ID_PRODUIT, ID_SO),
     GridSearchCV, MSE, elasticite OLS — equivalent a la methode Booper API.
  B) Global V1 Optuna : 1 modele Tweedie par classe S-B, Optuna,
     elasticite par perturbation.

Selectionne 2 produits par classe S-B (fort/moyen volume), entraine les deux
pipelines sur les memes donnees, et compare MAE, RMSE, elasticites.

Usage :
  python benchmark_per_product.py                 # 2 produits par classe
  python benchmark_per_product.py --n-per-class 3 # 3 produits par classe
  python benchmark_per_product.py --products 123_4 456_7  # produits specifiques
"""
from __future__ import annotations

import os
import sys
import gc
import json
import time
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_lightgbm_sb import (
    build_features, get_feature_cols, optimize_dataframe,
    get_optimized_dtypes, OUTPUT_DIR, TARGET,
)

CLASSES = ["smooth", "intermittent", "erratic", "lumpy"]
MIN_HISTORY_DAYS = 180
MIN_TEST_OBS = 30

# Fenetre de test alignee avec le modele global V1 Optuna (test_days=30 dans
# metadata). Les deux approches sont evaluees sur les MEMES 30 derniers jours
# pour que la comparaison MAE/RMSE soit equitable.
# Avant ce fix : per-product testait sur les 218 derniers jours (split 70/30
# max 365j), ce qui penalisait fortement son MAE vs le global.
TEST_DAYS = 30
PURGE_DAYS = 1  # Gap train/test comme dans train_lightgbm_v1_optuna

# Features exclues de l'entrainement per-product (identifiants, non-actionnables)
EXCLUDE_TRAIN = {"ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE", "date"}

# Features exclues du calcul d'elasticite (calendrier, cycliques, IDs)
EXCLUDE_ELAST = {
    "ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE",
    "PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR", "PARAM_JOUR_SEMAIN",
    "dow", "month", "day_of_year",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "day_sin", "day_cos", "horizon",
    "sin_doy", "cos_doy", "sin_dow", "cos_dow", "sin_month", "cos_month",
}


# ======================================================================
# PRODUCT SELECTION
# ======================================================================

def select_products(n_per_class=2, explicit_products=None):
    """
    Selectionne des produits representatifs depuis les CSV splits.
    Retourne un dict {cls: [(prod, so, n_obs, total_qty), ...]}
    """
    if explicit_products:
        # Parse "123_4" -> (123, 4)
        result = {}
        for ps in explicit_products:
            parts = ps.split("_")
            prod, so = int(parts[0]), int(parts[1])
            # Trouver la classe
            for cls in CLASSES:
                csv_path = os.path.join(OUTPUT_DIR, f"data_{cls}.csv")
                if not os.path.exists(csv_path):
                    continue
                dtypes = get_optimized_dtypes()
                for chunk in pd.read_csv(csv_path, sep=";", chunksize=50000,
                                         encoding="utf-8", dtype=dtypes):
                    mask = (chunk["ID_PRODUIT"] == prod) & (chunk["ID_SO"] == so)
                    if mask.any():
                        n = int(mask.sum())
                        q = float(chunk.loc[mask, "QUANTITE"].sum())
                        result.setdefault(cls, []).append((prod, so, n, q))
                        break
                if cls in result and any(p == prod and s == so for p, s, _, _ in result[cls]):
                    break
        return result

    result = {}
    for cls in CLASSES:
        csv_path = os.path.join(OUTPUT_DIR, f"data_{cls}.csv")
        if not os.path.exists(csv_path):
            continue

        print(f"  [{cls}] Scan des couples...")
        dtypes = get_optimized_dtypes()
        couple_stats = {}

        for chunk in pd.read_csv(csv_path, sep=";", chunksize=100000,
                                 encoding="utf-8", dtype=dtypes):
            for (p, s), grp in chunk.groupby(["ID_PRODUIT", "ID_SO"]):
                key = (int(p), int(s))
                if key not in couple_stats:
                    couple_stats[key] = {"n": 0, "qty": 0.0}
                couple_stats[key]["n"] += len(grp)
                couple_stats[key]["qty"] += float(grp["QUANTITE"].sum())

        # Filtrer : min historique
        valid = [(p, s, v["n"], v["qty"])
                 for (p, s), v in couple_stats.items()
                 if v["n"] >= MIN_HISTORY_DAYS]

        if not valid:
            print(f"    Aucun couple avec >= {MIN_HISTORY_DAYS} jours")
            continue

        # Trier par volume total decroissant
        valid.sort(key=lambda x: -x[3])

        n_target = min(n_per_class, len(valid))
        if n_target <= 0:
            continue

        # Echantillonnage stratifie sur la distribution de volume :
        # on prend n_target indices repartis uniformement de 0 (top volume)
        # a len(valid)-1 (plus petit volume), pour avoir un mix
        # representatif (gros, moyens, petits).
        if n_target == 1:
            indices = [0]
        elif n_target >= len(valid):
            indices = list(range(len(valid)))
        else:
            indices = [int(round(i * (len(valid) - 1) / (n_target - 1)))
                       for i in range(n_target)]
            # deduplicate (peut arriver sur les bords)
            indices = sorted(set(indices))

        selected = [valid[i] for i in indices]
        result[cls] = selected
        print(f"    {len(selected)} produits selectionnes "
              f"(sur {len(valid)} disponibles)")
        for prod, so, n, qty in selected:
            print(f"    {prod}_{so} : {n:,} obs, total_qty={qty:,.0f}")

    return result


# ======================================================================
# DATA LOADING PER PRODUCT
# ======================================================================

def load_product_data(cls, prod, so):
    """Charge les donnees d'un produit depuis le CSV de sa classe S-B."""
    csv_path = os.path.join(OUTPUT_DIR, f"data_{cls}.csv")
    dtypes = get_optimized_dtypes()
    chunks = []
    for chunk in pd.read_csv(csv_path, sep=";", chunksize=100000,
                             encoding="utf-8", dtype=dtypes):
        mask = (chunk["ID_PRODUIT"] == prod) & (chunk["ID_SO"] == so)
        if mask.any():
            chunks.append(chunk[mask].copy())
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    df = build_features(df)
    df = optimize_dataframe(df)
    df = df.dropna(subset=[TARGET, "date"])
    df = df[df[TARGET] >= 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ======================================================================
# PER-PRODUCT TRAINING (BOOPER STYLE)
# ======================================================================

def get_perprod_features(df):
    """Retourne les features utilisables pour le training per-product."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols
                if c not in EXCLUDE_TRAIN and c != TARGET]
    # Retirer les colonnes constantes
    features = [c for c in features if df[c].nunique() > 1]
    return features


def split_df_aligned(df):
    """
    Split aligne avec le modele global : les 30 derniers jours = test.
    Avec un gap de PURGE_DAYS entre train et test (comme V1 Optuna).

    Avant, on utilisait le split Booper historique (70/30 max 365j) qui
    donnait 218 jours de test vs 30 pour le global -> comparaison biaisee.
    """
    if "date" not in df.columns:
        # Fallback : split par index si pas de date
        n = len(df)
        nb_test = max(MIN_TEST_OBS, min(TEST_DAYS, n // 3))
        return df.iloc[:-nb_test].copy(), df.iloc[-nb_test:].copy()

    date_max = df["date"].max()
    cutoff_test = date_max - pd.Timedelta(days=TEST_DAYS)
    cutoff_train = cutoff_test - pd.Timedelta(days=PURGE_DAYS)

    train_df = df[df["date"] <= cutoff_train].copy()
    test_df = df[df["date"] > cutoff_test].copy()

    # Securite : si trop peu d'obs test, fallback sur split index
    if len(test_df) < 5 or len(train_df) < 30:
        n = len(df)
        nb_test = max(MIN_TEST_OBS, min(TEST_DAYS, n // 3))
        return df.iloc[:-nb_test].copy(), df.iloc[-nb_test:].copy()

    return train_df, test_df


def train_per_product(df, features):
    """
    Entraine XGBoost + LightGBM via GridSearchCV, retourne le winner.
    Reproduit la logique ModelChoice de AllModels.py.

    XGBoost est skippe si l'env DISABLE_XGB=1 (utile sur vieux glibc).

    Returns: (winner_name, model, best_params, train_metrics)
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    DISABLE_XGB = os.environ.get("DISABLE_XGB", "0") == "1"
    xgb = None
    if not DISABLE_XGB:
        try:
            import xgboost as xgb
        except Exception as e:
            print(f"      [WARN] xgboost import failed ({e}), skip XGB")
            xgb = None

    train_df, test_df = split_df_aligned(df)

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.float32)
    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)

    tscv = TimeSeriesSplit(n_splits=3)

    # --- LightGBM ---
    lgb_grid = {
        "num_leaves": [8, 16, 32, 64],
        "max_depth": [3, 5, 7],
        "min_child_samples": [4, 6, 8],
    }
    lgb_fixed = dict(
        learning_rate=0.03, n_estimators=1000, reg_lambda=1,
        subsample=0.6, force_row_wise=True, verbose=-1, n_jobs=-1,
    )
    lgb_model = lgb.LGBMRegressor(**lgb_fixed)
    lgb_search = GridSearchCV(lgb_model, lgb_grid, cv=tscv, scoring="neg_mean_absolute_error",
                              n_jobs=1, refit=True)
    try:
        lgb_search.fit(X_train, y_train)
        lgb_pred = np.maximum(lgb_search.predict(X_test), 0)
        lgb_error = abs(float(np.sum(lgb_pred)) - float(np.sum(y_test)))
        lgb_mae = float(np.mean(np.abs(y_test - lgb_pred)))
        lgb_ok = True
    except Exception as e:
        print(f"      [WARN] LightGBM GridSearch failed: {e}")
        lgb_ok = False
        lgb_error = float("inf")
        lgb_mae = float("inf")

    # --- XGBoost (optionnel) ---
    xgb_ok = False
    xgb_error = float("inf")
    xgb_mae = float("inf")
    xgb_search = None
    xgb_pred = None
    if xgb is not None:
        xgb_grid = {
            "max_depth": [3, 5, 7],
            "min_child_weight": [6, 7, 8, 9],
        }
        xgb_fixed = dict(
            objective="reg:squarederror", learning_rate=0.03,
            n_estimators=1000, reg_lambda=1, colsample_bytree=0.8,
            gamma=0, n_jobs=-1, verbosity=0,
        )
        xgb_model = xgb.XGBRegressor(**xgb_fixed)
        xgb_search = GridSearchCV(xgb_model, xgb_grid, cv=tscv,
                                  scoring="neg_mean_absolute_error",
                                  n_jobs=1, refit=True)
        try:
            xgb_search.fit(X_train, y_train)
            xgb_pred = np.maximum(xgb_search.predict(X_test), 0)
            xgb_error = abs(float(np.sum(xgb_pred)) - float(np.sum(y_test)))
            xgb_mae = float(np.mean(np.abs(y_test - xgb_pred)))
            xgb_ok = True
        except Exception as e:
            print(f"      [WARN] XGBoost GridSearch failed: {e}")

    # --- Winner ---
    if not lgb_ok and not xgb_ok:
        return None, None, None, {}

    if lgb_error <= xgb_error:
        winner_name = "lightgbm"
        winner_model = lgb_search.best_estimator_
        winner_params = lgb_search.best_params_
        winner_pred = lgb_pred if lgb_ok else None
    else:
        winner_name = "xgboost"
        winner_model = xgb_search.best_estimator_
        winner_params = xgb_search.best_params_
        winner_pred = xgb_pred if xgb_ok else None

    # Metriques sur test
    if winner_pred is not None:
        mae = float(np.mean(np.abs(y_test - winner_pred)))
        rmse = float(np.sqrt(np.mean((y_test - winner_pred) ** 2)))
        mean_q = float(np.mean(y_test))
    else:
        mae, rmse, mean_q = float("inf"), float("inf"), 0

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_qty": round(mean_q, 4),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "winner": winner_name,
        "lgb_mae": round(lgb_mae, 4) if lgb_ok else None,
        "xgb_mae": round(xgb_mae, 4) if xgb_ok else None,
    }
    return winner_name, winner_model, winner_params, metrics


# ======================================================================
# ELASTICITE OLS (BOOPER STYLE)
# ======================================================================

def compute_elasticity_ols(model, df, features):
    """
    Elasticite par OLS + feature importance, comme get_feature_importance()
    de AllModels.py. Retourne {feature: elasticity_pct}.
    """
    import statsmodels.api as sm

    X = df[features].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)

    # Feature importance du modele (gain)
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
    elif hasattr(model, "get_booster"):
        scores = model.get_booster().get_score(importance_type="gain")
        raw_imp = np.array([scores.get(f, 0) for f in features])
    else:
        raw_imp = np.ones(len(features))

    total_imp = raw_imp.sum()
    if total_imp > 0:
        norm_imp = raw_imp / total_imp
    else:
        norm_imp = np.ones(len(features)) / len(features)

    # OLS regression
    X_ols = sm.add_constant(pd.DataFrame(X, columns=features))
    try:
        ols_result = sm.OLS(y, X_ols).fit()
    except Exception:
        return {}

    mean_y = float(np.mean(y))
    if abs(mean_y) < 1e-8:
        return {}

    result = {}
    for i, feat in enumerate(features):
        if feat in EXCLUDE_ELAST:
            continue
        coeff = ols_result.params.get(feat, 0)
        mean_x = float(np.mean(df[feat]))
        elast = (coeff * mean_x) / mean_y

        # Regles business Booper
        if "PRIX" in feat.upper() and elast > 0:
            elast = 0
        if "PROMO" in feat.upper() and elast < 0:
            elast = 0

        if abs(elast) > 1e-8:
            result[feat] = round(float(elast * 100), 2)

    return result


# ======================================================================
# ELASTICITE PERTURBATION (PER-PRODUCT, MEME METHODE QUE GLOBAL)
# ======================================================================

def compute_elasticity_perturbation(model, df, features):
    """
    Elasticite par perturbation +/-10% sur un seul produit.
    Retourne {feature: elasticity_pct}.
    """
    X = df[features].values.astype(np.float32)

    if hasattr(model, "predict"):
        y_base = np.maximum(model.predict(X), 0).astype(np.float64)
    else:
        return {}

    result = {}
    for i, feat in enumerate(features):
        if feat in EXCLUDE_ELAST:
            continue

        col = X[:, i].astype(np.float64)
        mean_x = float(np.mean(col))
        std_x = float(np.std(col))

        # Determiner le delta
        is_binary = set(np.unique(col[~np.isnan(col)])).issubset({0, 1})
        if is_binary:
            # Impact : predict(feat=1) - predict(feat=0) / predict(feat=0)
            X_0 = X.copy()
            X_0[:, i] = 0
            X_1 = X.copy()
            X_1[:, i] = 1
            y0 = np.maximum(model.predict(X_0), 0)
            y1 = np.maximum(model.predict(X_1), 0)
            mask = y0 > 0.5
            if mask.sum() > 0:
                impacts = (y1[mask] - y0[mask]) / y0[mask] * 100
                val = round(float(np.median(impacts)), 2)
            else:
                val = 0
            if abs(val) > 1e-8:
                result[feat] = val
            continue

        if abs(mean_x) > 1e-8:
            delta = mean_x * 0.10
        elif std_x > 1e-8:
            delta = std_x * 0.10
        else:
            continue

        X_up = X.copy()
        X_up[:, i] = col + delta
        y_up = np.maximum(model.predict(X_up), 0)

        X_down = X.copy()
        X_down[:, i] = col - delta
        y_down = np.maximum(model.predict(X_down), 0)

        mask = y_base > 1e-8
        if mask.sum() == 0:
            continue

        e_up = (y_up[mask] - y_base[mask]) / y_base[mask] / 0.10
        e_down = (y_base[mask] - y_down[mask]) / y_base[mask] / 0.10
        elast = (e_up + e_down) / 2
        val = round(float(np.mean(elast)), 2)

        if abs(val) > 1e-8:
            result[feat] = val

    return result


# ======================================================================
# GLOBAL MODEL EVALUATION (V1 OPTUNA)
# ======================================================================

def eval_global_direct(cls, df, prod, so, model_path, meta_path):
    """
    Evalue le modele v1_direct sur un produit.
    Construit les samples (origin, horizon, target) dont target tombe
    dans la fenetre test, predit, retourne metriques.
    """
    from train_lightgbm_v1_direct import (
        generate_samples_for_product, df_to_xy, HORIZONS_TRAIN,
    )

    booster = lgb.Booster(model_file=model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        return None, None

    # Calculer les cutoffs
    test_days = meta.get("test_days", 30)
    purge_days = meta.get("purge_days", 1)
    date_max = df["date"].max()
    cutoff_test = date_max - pd.Timedelta(days=test_days)
    cutoff_train = cutoff_test - pd.Timedelta(days=purge_days)

    # Detecter les target_cols (exclure IDs + calendrier)
    EXCLUDE_TARGET = {
        "ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE", "date",
        "QUANTITE", "PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR",
        "PARAM_JOUR_SEMAIN",
    }
    target_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in EXCLUDE_TARGET]

    samples = generate_samples_for_product(
        df, target_cols, cutoff_train, cutoff_test, include_test=True)
    if not samples:
        return None, None

    df_s = pd.DataFrame(samples)
    df_test = df_s[df_s["_is_test"]]
    if df_test.empty:
        return None, None

    X, y, _ = df_to_xy(df_test, feature_cols)
    pred = np.maximum(booster.predict(X), 0)

    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    # Par horizon (pour debug / info)
    horizons = df_test["horizon"].values
    per_h = {}
    for h in sorted(set(horizons.tolist())):
        mask = horizons == h
        if mask.sum() > 0:
            per_h[h] = {
                "mae": round(float(np.mean(np.abs(y[mask] - pred[mask]))), 4),
                "n": int(mask.sum()),
            }

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_test": len(y),
        "model_type": os.path.basename(model_path),
        "per_horizon": per_h,
    }

    # Elasticites : non calculees ici (couteux sur direct multi-horizon).
    # Si besoin, reutiliser compute_elasticity_perturbation mais adapter
    # au format de samples direct.
    return metrics, {}


def eval_global_model(cls, df, prod, so, model_kind="v1_optuna"):
    """
    Charge un modele global (V1 Optuna par defaut, ou v1_direct si specifie)
    et evalue sur le test set du produit.

    model_kind : 'v1_optuna' (defaut) ou 'v1_direct' pour le multi-horizon.
    """
    if model_kind == "v1_direct":
        model_path = os.path.join(OUTPUT_DIR, f"model_{cls}_v1_direct.txt")
        meta_path = os.path.join(OUTPUT_DIR, f"model_{cls}_v1_direct_metadata.json")
        if os.path.exists(model_path) and os.path.exists(meta_path):
            return eval_global_direct(cls, df, prod, so, model_path, meta_path)
        print(f"    [WARN] v1_direct manquant pour {cls}, fallback v1_optuna")

    model_path = os.path.join(OUTPUT_DIR, f"model_{cls}_v1_optuna.txt")
    meta_path = os.path.join(OUTPUT_DIR, f"model_{cls}_v1_optuna_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        # Fallback : modele V1 vanilla
        model_path = os.path.join(OUTPUT_DIR, f"model_{cls}.txt")
        meta_path = os.path.join(OUTPUT_DIR, f"model_{cls}_metadata.json")
        if not os.path.exists(model_path):
            return None, None

    booster = lgb.Booster(model_file=model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_cols") or meta.get("features") or []

    if not feature_cols:
        return None, None

    # Split : memes 30 derniers jours que global
    test_days = meta.get("test_days", 30)
    date_max = df["date"].max()
    cutoff_test = date_max - pd.Timedelta(days=test_days)
    test_df = df[df["date"] > cutoff_test].copy()

    if len(test_df) < 5:
        return None, None

    # Preparer les features
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)
    y_pred = np.maximum(booster.predict(X_test), 0)

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_test": len(test_df),
        "model_type": os.path.basename(model_path),
    }

    # Elasticite perturbation sur ce produit uniquement
    # Utiliser les dernieres 90 jours (ou tout si moins)
    cutoff_recent = date_max - pd.Timedelta(days=90)
    recent = df[df["date"] > cutoff_recent].copy()
    if len(recent) < 30:
        recent = df.copy()

    for c in feature_cols:
        if c not in recent.columns:
            recent[c] = 0.0

    X_recent = recent[feature_cols].values.astype(np.float32)
    y_base = np.maximum(booster.predict(X_recent), 0).astype(np.float64)

    elasticites = {}
    for fi, feat in enumerate(feature_cols):
        if feat in EXCLUDE_ELAST:
            continue

        col = X_recent[:, fi].astype(np.float64)
        mean_x = float(np.mean(col))
        std_x = float(np.std(col))

        is_binary = set(np.unique(col[~np.isnan(col)])).issubset({0, 1})
        if is_binary:
            X_0 = X_recent.copy()
            X_0[:, fi] = 0
            X_1 = X_recent.copy()
            X_1[:, fi] = 1
            y0 = np.maximum(booster.predict(X_0), 0)
            y1 = np.maximum(booster.predict(X_1), 0)
            mask = y0 > 0.5
            if mask.sum() > 0:
                impacts = (y1[mask] - y0[mask]) / y0[mask] * 100
                val = round(float(np.median(impacts)), 2)
            else:
                val = 0
            if abs(val) > 1e-8:
                elasticites[feat] = val
            continue

        if abs(mean_x) > 1e-8:
            delta = mean_x * 0.10
        elif std_x > 1e-8:
            delta = std_x * 0.10
        else:
            continue

        X_up = X_recent.copy()
        X_up[:, fi] = col + delta
        y_up = np.maximum(booster.predict(X_up), 0)

        X_down = X_recent.copy()
        X_down[:, fi] = col - delta
        y_down = np.maximum(booster.predict(X_down), 0)

        mask = y_base > 1e-8
        if mask.sum() == 0:
            continue

        e_up = (y_up[mask] - y_base[mask]) / y_base[mask] / 0.10
        e_down = (y_base[mask] - y_down[mask]) / y_base[mask] / 0.10
        elast = (e_up + e_down) / 2
        val = round(float(np.mean(elast)), 2)
        if abs(val) > 1e-8:
            elasticites[feat] = val

    return metrics, elasticites


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-product vs global V1 Optuna")
    parser.add_argument("--n-per-class", type=int, default=2,
                        help="Nombre de produits par classe S-B (defaut: 2)")
    parser.add_argument("--products", type=str, nargs="+", default=None,
                        help="Produits specifiques au format PROD_SO (ex: 123_4)")
    parser.add_argument("--global-model", type=str, default="v1_optuna",
                        choices=["v1_optuna", "v1_direct"],
                        help="Modele global a evaluer (defaut: v1_optuna)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  BENCHMARK PER-PRODUCT vs GLOBAL V1 OPTUNA")
    print(f"{'='*70}")

    # 1. Selection des produits
    print(f"\n  Selection des produits...")
    t0 = time.time()
    products = select_products(
        n_per_class=args.n_per_class,
        explicit_products=args.products,
    )

    total_products = sum(len(v) for v in products.values())
    print(f"\n  {total_products} produits selectionnes en {time.time()-t0:.1f}s")

    if total_products == 0:
        print("  [ERREUR] Aucun produit selectionne.")
        return

    # 2. Benchmark
    results = []

    for cls, prods in products.items():
        for prod, so, n_obs, total_qty in prods:
            print(f"\n{'='*70}")
            print(f"  PRODUIT {prod}_{so} (classe={cls}, n={n_obs:,}, qty={total_qty:,.0f})")
            print(f"{'='*70}")

            # Charger les donnees
            print(f"  Chargement...")
            df = load_product_data(cls, prod, so)
            if df is None or len(df) < 50:
                print(f"  [SKIP] Donnees insuffisantes")
                continue

            features = get_perprod_features(df)
            print(f"  {len(df):,} lignes, {len(features)} features")

            # --- A) Per-product (Booper style) ---
            print(f"\n  [PER-PRODUCT] Training XGB + LightGBM GridSearchCV...")
            t1 = time.time()
            winner_name, pp_model, pp_params, pp_metrics = train_per_product(df, features)
            elapsed_pp = time.time() - t1

            if winner_name is None:
                print(f"  [PER-PRODUCT] ECHEC")
                continue

            print(f"  [PER-PRODUCT] Winner: {winner_name} | MAE={pp_metrics['mae']:.4f} "
                  f"| RMSE={pp_metrics['rmse']:.4f} | {elapsed_pp:.1f}s")
            if pp_metrics.get("lgb_mae") is not None:
                print(f"    LightGBM MAE={pp_metrics['lgb_mae']:.4f} | "
                      f"XGBoost MAE={pp_metrics['xgb_mae']:.4f}")

            # Elasticite OLS (Booper)
            print(f"  [PER-PRODUCT] Elasticite OLS...")
            elast_ols = compute_elasticity_ols(pp_model, df, features)

            # Elasticite perturbation (per-product)
            print(f"  [PER-PRODUCT] Elasticite perturbation...")
            elast_pert_pp = compute_elasticity_perturbation(pp_model, df, features)

            # --- B) Global (V1 Optuna ou V1 Direct) ---
            print(f"\n  [GLOBAL] Evaluation modele {args.global_model} {cls}...")
            t2 = time.time()
            global_result = eval_global_model(cls, df, prod, so,
                                              model_kind=args.global_model)

            if global_result[0] is None:
                print(f"  [GLOBAL] Modele introuvable pour {cls}")
                gl_metrics = {"mae": None, "rmse": None}
                elast_global = {}
            else:
                gl_metrics, elast_global = global_result
                print(f"  [GLOBAL] MAE={gl_metrics['mae']:.4f} "
                      f"| RMSE={gl_metrics['rmse']:.4f} | {time.time()-t2:.1f}s")

            # --- Comparaison ---
            # Trouver les features prix/promo communes
            prix_feat = None
            promo_feat = None
            for f in features:
                if f.upper() == "PARAM_PRIX" and prix_feat is None:
                    prix_feat = f
                if "PROMO" in f.upper() and promo_feat is None:
                    promo_feat = f

            row = {
                "ID_PRODUIT": prod,
                "ID_SO": so,
                "classe": cls,
                "n_train": pp_metrics.get("n_train", 0),
                "n_test": pp_metrics.get("n_test", 0),
                "winner_model": winner_name,
                # MAE
                "MAE_perprod": pp_metrics.get("mae"),
                "MAE_global": gl_metrics.get("mae"),
                "delta_MAE": (round(pp_metrics["mae"] - gl_metrics["mae"], 4)
                              if gl_metrics.get("mae") is not None
                              and pp_metrics.get("mae") is not None
                              and np.isfinite(pp_metrics["mae"])
                              and np.isfinite(gl_metrics["mae"])
                              else None),
                # RMSE
                "RMSE_perprod": pp_metrics.get("rmse"),
                "RMSE_global": gl_metrics.get("rmse"),
                # Elasticites prix
                "ELAST_PRIX_ols": elast_ols.get(prix_feat) if prix_feat else None,
                "ELAST_PRIX_pert_pp": elast_pert_pp.get(prix_feat) if prix_feat else None,
                "ELAST_PRIX_global": elast_global.get(prix_feat) if prix_feat else None,
                # Elasticites promo
                "ELAST_PROMO_ols": elast_ols.get(promo_feat) if promo_feat else None,
                "ELAST_PROMO_pert_pp": elast_pert_pp.get(promo_feat) if promo_feat else None,
                "ELAST_PROMO_global": elast_global.get(promo_feat) if promo_feat else None,
                # Temps
                "time_perprod_sec": round(elapsed_pp, 1),
            }
            results.append(row)

            # Afficher les elasticites detaillees
            print(f"\n  Elasticites comparees :")
            print(f"  {'Feature':<25} {'OLS':>10} {'Pert_PP':>10} {'Global':>10}")
            all_feats = sorted(set(list(elast_ols.keys()) + list(elast_pert_pp.keys()) + list(elast_global.keys())))
            for feat in all_feats[:15]:  # Top 15
                v_ols = elast_ols.get(feat, "")
                v_pp = elast_pert_pp.get(feat, "")
                v_gl = elast_global.get(feat, "")
                print(f"  {feat:<25} {str(v_ols):>10} {str(v_pp):>10} {str(v_gl):>10}")

            gc.collect()

    # 3. Resume
    if not results:
        print("\n  Aucun resultat.")
        return

    print(f"\n\n{'='*70}")
    print(f"  RESUME BENCHMARK ({len(results)} produits)")
    print(f"{'='*70}")

    df_results = pd.DataFrame(results)

    # Tableau principal
    print(f"\n  {'Produit':<20} {'Classe':<12} {'MAE_PP':>8} {'MAE_GL':>8} {'delta':>8} "
          f"{'RMSE_PP':>8} {'RMSE_GL':>8} {'Winner':>8}")
    for _, r in df_results.iterrows():
        delta_str = f"{r['delta_MAE']:+.2f}" if r.get("delta_MAE") is not None else "n/a"
        mae_gl = f"{r['MAE_global']:.2f}" if r.get("MAE_global") is not None else "n/a"
        rmse_gl = f"{r['RMSE_global']:.2f}" if r.get("RMSE_global") is not None else "n/a"
        print(f"  {r['ID_PRODUIT']}_{r['ID_SO']:<10} {r['classe']:<12} "
              f"{r['MAE_perprod']:>8.2f} {mae_gl:>8} {delta_str:>8} "
              f"{r['RMSE_perprod']:>8.2f} {rmse_gl:>8} {r['winner_model']:>8}")

    # Stats aggregees
    valid = df_results.dropna(subset=["MAE_perprod", "MAE_global"])
    if len(valid) > 0:
        gl_wins = (valid["delta_MAE"] > 0).sum()   # delta>0 = MAE_pp > MAE_gl = global meilleur
        pp_wins = (valid["delta_MAE"] <= 0).sum()  # delta<=0 = per-product meilleur ou egal
        mean_delta = valid["delta_MAE"].mean()

        print(f"\n  Global gagne : {gl_wins}/{len(valid)} produits")
        print(f"  Per-prod gagne : {pp_wins}/{len(valid)} produits")
        print(f"  Delta MAE moyen : {mean_delta:+.4f} "
              f"({'global meilleur' if mean_delta > 0 else 'per-product meilleur'})")

    # Sauvegarder : suffixe automatique selon le global model evalue
    csv_path = os.path.join(
        OUTPUT_DIR, f"benchmark_per_product_{args.global_model}.csv")
    df_results.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
    print(f"\n  [SAVE] {csv_path}")


if __name__ == "__main__":
    main()
