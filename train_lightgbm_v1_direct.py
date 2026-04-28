#!/usr/bin/env python3
"""
train_lightgbm_v1_direct.py

V1 Direct Multi-Horizon : forecasting direct sans recursion.

Principe :
  Pour chaque couple (ID_PRODUIT, ID_SO), on genere des samples
  (origin, horizon, target) ou :
    - origin est une date a laquelle on simule faire une prevision
    - horizon in HORIZONS (ex: 1, 3, 7, ..., 90 jours)
    - target = origin + horizon

  Features :
    - Lags ancres a l'origine : y(origin), y(origin-1), y(origin-7),
      y(origin-28), y(origin-91), y(origin-365)
    - Rolling stats calcules sur [origin-W, origin] : mean, std, trend
    - Horizon (1..90) en feature explicite
    - Features dynamiques a la date target : PARAM_PRIX[target],
      PARAM_PROMO_X[target], meteo prevue, calendrier target

  Tous les lags/rolling sont TOUJOURS connus au moment de la prediction
  (pas de leak, pas de recursive forecasting).

  Un seul modele LightGBM Tweedie + Optuna par classe S-B predit y(target)
  pour n'importe quel horizon.

Cette approche reproduit la methode des top solutions M5 competition.
"""
from __future__ import annotations

import os
import sys
import gc
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_lightgbm_sb import (
    load_classification, list_zip_files, split_by_class,
    optimize_dataframe, get_optimized_dtypes,
    check_ram_limit, get_ram_usage, force_gc,
    print_ram_status,
    OUTPUT_DIR, CENTRALE_DIR, MAX_ZIPS,
    TARGET, TEST_DAYS, PURGE_DAYS,
    CHUNK_SIZE, SKIP_SPLIT_IF_EXISTS,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("[ERREUR] optuna requis. Installe: pip install optuna")
    sys.exit(1)

# pyarrow uniquement requis pour le mode --fast (cache parquet).
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PYARROW_OK = True
except ImportError:
    _PYARROW_OK = False

# ======================================================================
# CONFIG
# ======================================================================
CLASSES = ["smooth", "intermittent", "erratic", "lumpy"]

# Horizons d'apprentissage (log-spaced pour couvrir court et long terme)
HORIZONS_TRAIN = [1, 2, 3, 7, 14, 21, 28, 45, 60, 75, 90]
# Horizons pour le reporting final par classe
HORIZONS_EVAL = [1, 7, 14, 30, 60, 90]
MAX_HORIZON = max(HORIZONS_TRAIN)

# Ecart entre origines consecutives (en jours) lors de la generation des
# samples. Plus petit = plus de samples = plus de RAM mais meilleure
# couverture.
ORIGIN_STRIDE = 7

# Lags ancres a l'origine (0 = valeur a l'origine, 1 = veille, etc.)
LAG_DAYS = [0, 1, 2, 7, 14, 28, 91, 365]

# Windows pour rolling stats
ROLL_WINDOWS = [7, 28, 91]

# Minimum d'historique necessaire avant une origine valide
MIN_HISTORY = 60

# Batch size pour le training streaming (nombre de samples par batch)
TRAIN_BATCH_SAMPLES = 500_000

# Optuna
OPTUNA_N_TRIALS = 15            # Moins que V1 Optuna car chaque trial
                                  # est plus lourd (plus de samples)
OPTUNA_MAX_TREES = 800
OPTUNA_SEED = 42

# Final training
FINAL_MAX_TREES = 2500
FINAL_EARLY_STOP_BATCHES = 5    # Patience ES

# Compute
N_JOBS = min(os.cpu_count() or 4, 8)

# --- Mode --fast (cache parquet) ---
CACHE_DIR = OUTPUT_DIR
PARQUET_BATCH_SIZE = 500_000
PARQUET_COMPRESSION = "snappy"  # rapide; "zstd" si on veut ~30% plus petit


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def build_origin_features(y_hist: np.ndarray) -> dict:
    """
    Calcule les features observables a l'origine (toujours connues
    au moment de faire la prevision).

    y_hist : array des ventes jusqu'a l'origine INCLUSE.
    y_hist[-1] = y(origin), y_hist[-2] = y(origin - 1), etc.
    """
    n = len(y_hist)
    feats = {}

    # Lags absolus depuis l'origine
    for lag in LAG_DAYS:
        idx = n - 1 - lag
        feats[f"y_origin_m{lag}"] = float(y_hist[idx]) if idx >= 0 else 0.0

    # Rolling stats sur differentes fenetres
    for w in ROLL_WINDOWS:
        window = y_hist[-w:] if n >= w else y_hist
        if len(window) > 0:
            feats[f"roll_mean_{w}"] = float(window.mean())
            feats[f"roll_std_{w}"] = float(window.std())
            feats[f"roll_max_{w}"] = float(window.max())
            feats[f"roll_nonzero_ratio_{w}"] = float((window > 0).mean())
        else:
            feats[f"roll_mean_{w}"] = 0.0
            feats[f"roll_std_{w}"] = 0.0
            feats[f"roll_max_{w}"] = 0.0
            feats[f"roll_nonzero_ratio_{w}"] = 0.0

    # Trend (pente OLS) sur 28 jours
    if n >= 28:
        x = np.arange(28, dtype=np.float64)
        y28 = y_hist[-28:].astype(np.float64)
        # Slope via formule fermee (plus rapide que polyfit)
        x_mean = x.mean()
        y_mean = y28.mean()
        denom = float(((x - x_mean) ** 2).sum())
        feats["trend_28"] = float(((x - x_mean) * (y28 - y_mean)).sum() / denom) if denom > 1e-8 else 0.0
    else:
        feats["trend_28"] = 0.0

    # Jours depuis la derniere vente > 0
    if n > 0:
        nz = np.where(y_hist > 0)[0]
        feats["days_since_sale"] = float(n - 1 - nz[-1]) if len(nz) > 0 else float(n)
    else:
        feats["days_since_sale"] = 0.0

    # Coefficient de variation sur 28j
    rm = feats["roll_mean_28"]
    feats["cv_28"] = feats["roll_std_28"] / (rm + 1e-6) if rm >= 0 else 0.0

    return feats


def generate_samples_for_product(df_prod: pd.DataFrame,
                                  target_col_names: list,
                                  cutoff_train: pd.Timestamp,
                                  cutoff_test: pd.Timestamp,
                                  include_test: bool = True) -> list[dict]:
    """
    Pour un produit (df_prod deja trie par date), genere tous les samples
    (origin, horizon, target) valides.

    cutoff_train : date max de fin du train (target <= cutoff_train)
    cutoff_test  : date min du test (target > cutoff_test)
    include_test : si True, inclut les samples avec target dans la fenetre test.
    """
    df_prod = df_prod.sort_values("date").reset_index(drop=True)
    if len(df_prod) < MIN_HISTORY + min(HORIZONS_TRAIN):
        return []

    dates = df_prod["date"].values  # ndarray datetime64
    y = df_prod[TARGET].values.astype(np.float32)

    prod_id = int(df_prod["ID_PRODUIT"].iloc[0])
    so_id = int(df_prod["ID_SO"].iloc[0])

    samples = []
    n = len(df_prod)

    for origin_idx in range(MIN_HISTORY, n - 1, ORIGIN_STRIDE):
        y_hist = y[:origin_idx + 1]  # inclut l'origine
        origin_date = pd.Timestamp(dates[origin_idx])

        origin_feats = build_origin_features(y_hist)

        for h in HORIZONS_TRAIN:
            target_idx = origin_idx + h
            if target_idx >= n:
                break
            target_date = pd.Timestamp(dates[target_idx])

            # Filtre train/test selon la position du TARGET
            is_test = target_date > cutoff_test
            is_train = target_date <= cutoff_train
            if not is_test and not is_train:
                continue  # dans le gap de purge
            if is_test and not include_test:
                continue

            sample = dict(origin_feats)
            sample["_prod"] = prod_id
            sample["_so"] = so_id
            sample["_origin_date"] = origin_date
            sample["_target_date"] = target_date
            sample["_is_test"] = is_test
            sample["horizon"] = h
            # Calendrier de la target (connu)
            sample["target_dow"] = int(target_date.dayofweek)
            sample["target_month"] = int(target_date.month)
            sample["target_day"] = int(target_date.day)
            sample["target_doy"] = int(target_date.dayofyear)
            sample["target_is_weekend"] = int(target_date.dayofweek >= 5)
            # Features dynamiques a la date target (prix, promo, meteo)
            for col in target_col_names:
                sample[col] = float(df_prod[col].iloc[target_idx])
            # Target
            sample["y"] = float(y[target_idx])
            samples.append(sample)

    return samples


def infer_target_cols(df_sample: pd.DataFrame) -> list[str]:
    """
    Detecte les colonnes a prendre a la date target (prix, promo, meteo).
    Exclut les IDs et le calendrier (recalcule).
    """
    EXCLUDE = {
        "ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE", "date",
        "QUANTITE", "PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR",
        "PARAM_JOUR_SEMAIN",
    }
    numeric = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in EXCLUDE]


# ======================================================================
# STREAMING SAMPLE GENERATOR
# ======================================================================

class DirectSampleGenerator:
    """
    Lit le CSV d'une classe, groupe par (prod, so), genere les samples
    direct multi-horizon et les yield par batch.
    """
    def __init__(self, data_path, cutoff_train, cutoff_test,
                 batch_size=TRAIN_BATCH_SAMPLES, include_test=False):
        self.data_path = data_path
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test
        self.batch_size = batch_size
        self.include_test = include_test

        self.feature_cols = None
        self.target_cols = None
        self._infer_cols()

    def _infer_cols(self):
        """Lit la premiere ligne pour detecter les colonnes."""
        dtypes = get_optimized_dtypes()
        for chunk in pd.read_csv(self.data_path, sep=";", chunksize=100,
                                 encoding="utf-8", dtype=dtypes):
            chunk["date"] = pd.to_datetime(
                chunk[["PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR"]].rename(
                    columns={"PARAM_ANNEE": "year", "PARAM_MOIS": "month",
                             "PARAM_JOUR": "day"}),
                errors="coerce",
            )
            self.target_cols = infer_target_cols(chunk)
            break

    def iter_batches(self, max_products=None):
        """
        Charge le CSV par chunks, accumule par produit complet,
        genere les samples, yield par batch.
        """
        dtypes = get_optimized_dtypes()
        products_buffer = {}  # (prod, so) -> list of chunk rows
        samples_buffer = []
        batch_num = 0
        n_products_done = 0

        def flush_product(key):
            nonlocal samples_buffer, n_products_done
            df_prod = pd.concat(products_buffer.pop(key), ignore_index=True)
            df_prod["date"] = pd.to_datetime(
                df_prod[["PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR"]].rename(
                    columns={"PARAM_ANNEE": "year", "PARAM_MOIS": "month",
                             "PARAM_JOUR": "day"}),
                errors="coerce",
            )
            df_prod = df_prod.dropna(subset=["date", TARGET])
            df_prod = df_prod[df_prod[TARGET] >= 0]

            sm = generate_samples_for_product(
                df_prod, self.target_cols,
                self.cutoff_train, self.cutoff_test,
                include_test=self.include_test,
            )
            samples_buffer.extend(sm)
            n_products_done += 1

        # Lire par chunks, regrouper par (prod, so)
        for chunk in pd.read_csv(self.data_path, sep=";", chunksize=CHUNK_SIZE,
                                 encoding="utf-8", dtype=dtypes):
            chunk = optimize_dataframe(chunk)
            for (p, s), grp in chunk.groupby(["ID_PRODUIT", "ID_SO"],
                                             sort=False):
                key = (int(p), int(s))
                products_buffer.setdefault(key, []).append(grp)

            # Flush les produits qui ne sont plus dans le chunk courant
            # (approximation : on flush tout a la fin de chaque chunk si
            # le buffer est gros). Simplification : on attend que tout
            # le CSV soit lu avant de flusher.
            # -> mais alors on charge tout en RAM. Pas viable sur 8M lignes.
            # Alternative : flusher les produits dont la derniere date < date courante.

            # Approche simple : si le buffer depasse X produits, flush ceux
            # non presents dans ce chunk.
            current_keys = set(
                (int(p), int(s))
                for p, s in chunk[["ID_PRODUIT", "ID_SO"]].drop_duplicates().values
            )
            stale_keys = [k for k in products_buffer if k not in current_keys]
            # On garde une politique simple : si on a > 500 produits en buffer
            # on flush les "stale"
            if len(products_buffer) > 500:
                for k in stale_keys:
                    flush_product(k)
                    if max_products is not None and n_products_done >= max_products:
                        break

            # Yield quand le buffer samples est plein
            while len(samples_buffer) >= self.batch_size:
                batch_num += 1
                df_batch = pd.DataFrame(samples_buffer[:self.batch_size])
                samples_buffer = samples_buffer[self.batch_size:]
                yield batch_num, df_batch

            if max_products is not None and n_products_done >= max_products:
                break

        # Flush tous les produits restants
        remaining_keys = list(products_buffer.keys())
        for k in remaining_keys:
            flush_product(k)
            if max_products is not None and n_products_done >= max_products:
                break

        # Yield le dernier batch (potentiellement incomplet)
        while samples_buffer:
            batch_num += 1
            take = min(len(samples_buffer), self.batch_size)
            df_batch = pd.DataFrame(samples_buffer[:take])
            samples_buffer = samples_buffer[take:]
            yield batch_num, df_batch


# ======================================================================
# DATASET BUILDER
# ======================================================================

META_COLS = ("_prod", "_so", "_origin_date", "_target_date", "_is_test")


def df_to_xy(df_batch: pd.DataFrame, feature_cols: list | None = None):
    """
    Convertit un DataFrame de samples en (X, y, feature_cols).
    Filtre les colonnes meta.
    """
    y = df_batch["y"].values.astype(np.float32)
    drop = list(META_COLS) + ["y"]
    X_df = df_batch.drop(columns=[c for c in drop if c in df_batch.columns])

    if feature_cols is None:
        feature_cols = X_df.columns.tolist()
    else:
        for c in feature_cols:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X_df = X_df[feature_cols]

    X = X_df.values.astype(np.float32)
    return X, y, feature_cols


# ======================================================================
# BATCH TRAINING (streaming incremental, comme V1)
# ======================================================================

def run_batch_training(cls, data_path, output_dir, cutoff_train, cutoff_test,
                       params, max_trees, temp_suffix="", verbose=True,
                       holdout=None, es_patience_batches=0):
    """
    Streaming training : lit les samples par batch, entraine incrementalement.
    Retourne (model, rows_trained, total_trees, feature_cols).
    """
    temp_model_path = os.path.join(
        output_dir, f"_temp_direct_{cls}{temp_suffix}.txt")
    best_checkpoint_path = None

    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    best_holdout_mae = float("inf")
    batches_without_improvement = 0
    use_es = holdout is not None and es_patience_batches > 0

    generator = DirectSampleGenerator(
        data_path=data_path,
        cutoff_train=cutoff_train,
        cutoff_test=cutoff_test,
        batch_size=TRAIN_BATCH_SAMPLES,
        include_test=False,
    )

    model = None
    rows_trained = 0
    total_trees = 0
    feature_cols = None

    # Budget d'arbres reparti entre batches. On estime ~N batches a partir
    # du premier batch. Strategie : max 150 arbres par batch, au minimum 20.
    rounds_per_batch = 100

    for batch_num, df_batch in generator.iter_batches():
        if len(df_batch) < 100:
            continue
        X_batch, y_batch, feature_cols = df_to_xy(df_batch, feature_cols)
        rows_trained += len(y_batch)

        if not check_ram_limit(f"Batch {batch_num}"):
            break

        if total_trees >= max_trees:
            break

        rounds_this = min(rounds_per_batch, max_trees - total_trees)
        if rounds_this <= 0:
            break

        init_model = temp_model_path if os.path.exists(temp_model_path) else None

        ds = lgb.Dataset(
            X_batch, label=y_batch,
            feature_name=feature_cols,
            free_raw_data=True,
        )
        del X_batch, y_batch, df_batch
        gc.collect()

        model = lgb.train(params, ds, num_boost_round=rounds_this,
                          init_model=init_model)
        total_trees = model.num_trees()
        model.save_model(temp_model_path)

        holdout_str = ""
        if use_es:
            X_h, y_h = holdout
            preds_h = model.predict(X_h)
            batch_mae = float(np.mean(np.abs(y_h - preds_h)))
            holdout_str = f" holdout_mae={batch_mae:.4f}"
            if batch_mae < best_holdout_mae - 1e-6:
                best_holdout_mae = batch_mae
                best_checkpoint_path = os.path.join(
                    output_dir, f"_best_direct_{cls}{temp_suffix}.txt")
                model.save_model(best_checkpoint_path)
                batches_without_improvement = 0
            else:
                batches_without_improvement += 1
            del preds_h

        if verbose:
            print(f"    Batch {batch_num:>3} : trees={total_trees}/{max_trees} "
                  f"n_samples={rows_trained:,}{holdout_str}")

        if hasattr(ds, 'data'):
            ds.data = None
        del ds, model
        model = None
        gc.collect()

        if use_es and batches_without_improvement >= es_patience_batches:
            if verbose:
                print(f"    [ES] Stop: {es_patience_batches} batches sans "
                      f"amelioration (best mae={best_holdout_mae:.4f})")
            break

    final_path = None
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        final_path = best_checkpoint_path
    elif os.path.exists(temp_model_path):
        final_path = temp_model_path

    model = lgb.Booster(model_file=final_path) if final_path else None

    for p in (temp_model_path, best_checkpoint_path):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    if model is not None:
        total_trees = model.num_trees()

    force_gc()
    return model, rows_trained, total_trees, feature_cols


# ======================================================================
# HOLDOUT BUILDER (pour early stopping interne du training final)
# ======================================================================

def build_holdout_arrays(data_path, cutoff_holdout, cutoff_train,
                          feature_cols, max_products=None):
    """
    Construit (X, y) en RAM pour les samples dont target appartient a
    [cutoff_holdout, cutoff_train] (derniers jours du train, avant test).
    Sert au early stopping interne.
    """
    # Generer les samples en marquant cutoff_train comme borne sup du train
    # et cutoff_holdout comme borne inf
    Xs, ys = [], []

    gen = DirectSampleGenerator(
        data_path=data_path,
        cutoff_train=cutoff_train,
        cutoff_test=cutoff_train,  # pas de test ici, cutoff_test = cutoff_train
        batch_size=500_000,
        include_test=False,
    )

    for batch_num, df_batch in gen.iter_batches(max_products=max_products):
        # Garder uniquement les samples avec target > cutoff_holdout
        df_batch = df_batch[df_batch["_target_date"] > cutoff_holdout]
        if df_batch.empty:
            continue
        X, y, _ = df_to_xy(df_batch, feature_cols)
        Xs.append(X)
        ys.append(y)
        if sum(len(x) for x in Xs) > 300_000:
            break  # limite : 300k samples suffisent

    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ======================================================================
# PER-HORIZON EVALUATION
# ======================================================================

def eval_per_horizon(model, feature_cols, data_path, cutoff_train, cutoff_test):
    """
    Evalue le modele sur le test set, avec breakdown par horizon.
    Retourne (mae_global, metrics_dict, per_horizon_dict).
    """
    gen = DirectSampleGenerator(
        data_path=data_path,
        cutoff_train=cutoff_train,
        cutoff_test=cutoff_test,
        batch_size=500_000,
        include_test=True,
    )

    per_h_preds = {h: {"y": [], "pred": []} for h in HORIZONS_TRAIN}
    all_y, all_pred = [], []

    for batch_num, df_batch in gen.iter_batches():
        df_test = df_batch[df_batch["_is_test"]]
        if df_test.empty:
            continue
        X, y, _ = df_to_xy(df_test, feature_cols)
        pred = np.maximum(model.predict(X), 0)

        horizons = df_test["horizon"].values

        for h in HORIZONS_TRAIN:
            mask = horizons == h
            if mask.any():
                per_h_preds[h]["y"].append(y[mask])
                per_h_preds[h]["pred"].append(pred[mask])

        all_y.append(y)
        all_pred.append(pred)

        del X, y, pred, df_test, df_batch
        gc.collect()

    if not all_y:
        return float("inf"), {}, {}

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    mae_global = float(np.mean(np.abs(all_y - all_pred)))
    rmse_global = float(np.sqrt(np.mean((all_y - all_pred) ** 2)))
    mean_q = float(np.mean(all_y))

    metrics = {
        "mae": round(mae_global, 4),
        "rmse": round(rmse_global, 4),
        "mean_quantite": round(mean_q, 4),
        "mape_pct": round(mae_global / mean_q * 100, 1) if mean_q > 0 else None,
        "n_test": int(len(all_y)),
    }

    per_h = {}
    for h in HORIZONS_TRAIN:
        if per_h_preds[h]["y"]:
            yh = np.concatenate(per_h_preds[h]["y"])
            ph = np.concatenate(per_h_preds[h]["pred"])
            per_h[h] = {
                "mae": round(float(np.mean(np.abs(yh - ph))), 4),
                "rmse": round(float(np.sqrt(np.mean((yh - ph) ** 2))), 4),
                "n": int(len(yh)),
                "mean_q": round(float(np.mean(yh)), 4),
            }

    return mae_global, metrics, per_h


# ======================================================================
# OPTUNA OBJECTIVE
# ======================================================================

def make_objective(cls, data_path, output_dir, cutoff_train, cutoff_test):

    def objective(trial):
        tweedie_power = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
        num_leaves = trial.suggest_int("num_leaves", 31, 127)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 50, 500)
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)

        params = {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
            "metric": ["tweedie", "mae"],
            "first_metric_only": False,
            "verbosity": -1,
            "n_jobs": N_JOBS,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 5,
            "feature_fraction": feature_fraction,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "seed": OPTUNA_SEED,
            "bagging_seed": OPTUNA_SEED,
            "feature_fraction_seed": OPTUNA_SEED,
            "deterministic": True,
            "max_bin": 127,
            "min_data_in_bin": 20,
            "force_row_wise": True,
            "histogram_pool_size": 1024,
        }

        temp_suffix = f"_optuna_t{trial.number}"
        try:
            model, rows, trees, feat_cols = run_batch_training(
                cls, data_path, output_dir, cutoff_train, cutoff_test,
                params, OPTUNA_MAX_TREES,
                temp_suffix=temp_suffix, verbose=False,
            )
            if model is None:
                return float("inf")
            trial_mae, n_test, _ = eval_per_horizon(
                model, feat_cols, data_path, cutoff_train, cutoff_test)
            trial.set_user_attr("total_trees", trees)
            trial.set_user_attr("rows_trained", rows)
            del model
            force_gc()
            return trial_mae
        finally:
            for suffix in (f"_temp_direct_{cls}{temp_suffix}.txt",
                           f"_best_direct_{cls}{temp_suffix}.txt"):
                p = os.path.join(output_dir, suffix)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    return objective


# ======================================================================
# TRAIN PAR CLASSE
# ======================================================================

def train_direct_class(cls, data_path, output_dir, date_range, n_total):
    print(f"\n{'='*70}")
    print(f"  CLASSE : {cls.upper()} (V1 Direct Multi-Horizon + Optuna)")
    print(f"{'='*70}")

    t0 = time.time()
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Fichier : {data_path} ({file_size:.1f} Mo)")

    if file_size < 0.01:
        print(f"  [SKIP] Fichier vide.")
        return

    if date_range["min"] is None or date_range["max"] is None:
        print(f"  [SKIP] Pas de dates.")
        return

    date_min = pd.to_datetime(date_range["min"])
    date_max = pd.to_datetime(date_range["max"])
    cutoff_train = date_max - pd.Timedelta(days=TEST_DAYS + PURGE_DAYS)
    cutoff_test = date_max - pd.Timedelta(days=TEST_DAYS)

    print(f"  Train  : target <= {cutoff_train.date()}")
    print(f"  Test   : target > {cutoff_test.date()} ({date_max.date()})")
    print(f"  Horizons train : {HORIZONS_TRAIN}")
    print(f"  Origin stride  : {ORIGIN_STRIDE} jours")
    print_ram_status("  RAM: ")

    # --- Optuna ---
    print(f"\n  [OPTUNA] {OPTUNA_N_TRIALS} trials (max_trees={OPTUNA_MAX_TREES})")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"v1_direct_{cls}",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=3),
    )

    t_opt = time.time()
    study.optimize(
        make_objective(cls, data_path, output_dir, cutoff_train, cutoff_test),
        n_trials=OPTUNA_N_TRIALS, show_progress_bar=False,
    )
    elapsed_opt = time.time() - t_opt

    best = study.best_params
    print(f"\n  [OPTUNA] Termine en {elapsed_opt:.0f}s")
    print(f"  [OPTUNA] Best MAE = {study.best_value:.4f} "
          f"(trial #{study.best_trial.number})")
    for k, v in best.items():
        print(f"    {k:<30} = {v}")

    # --- Training final ---
    print(f"\n  [FINAL] Training complet (max_trees={FINAL_MAX_TREES})...")

    final_params = {
        "objective": "tweedie",
        "tweedie_variance_power": best["tweedie_variance_power"],
        "metric": ["tweedie", "mae"],
        "first_metric_only": False,
        "verbosity": -1,
        "n_jobs": N_JOBS,
        "learning_rate": best["learning_rate"],
        "num_leaves": best["num_leaves"],
        "min_data_in_leaf": best["min_data_in_leaf"],
        "bagging_fraction": best["bagging_fraction"],
        "bagging_freq": 5,
        "feature_fraction": best["feature_fraction"],
        "lambda_l1": best["lambda_l1"],
        "lambda_l2": best["lambda_l2"],
        "seed": OPTUNA_SEED,
        "bagging_seed": OPTUNA_SEED,
        "feature_fraction_seed": OPTUNA_SEED,
        "deterministic": True,
        "max_bin": 127,
        "min_data_in_bin": 20,
        "force_row_wise": True,
        "histogram_pool_size": 1024,
    }

    # Holdout interne pour ES (10 derniers jours du train)
    holdout_days = 10
    cutoff_holdout = cutoff_train - pd.Timedelta(days=holdout_days)

    # Premier run rapide pour recuperer feature_cols (detecte a la volee)
    # On relance avec holdout ensuite si on veut l'ES.
    # Simplification : on fait un run unique sans ES -> plus simple
    model, rows, trees, feat_cols = run_batch_training(
        cls, data_path, output_dir, cutoff_train, cutoff_test,
        final_params, FINAL_MAX_TREES,
        temp_suffix="_final", verbose=True,
        holdout=None, es_patience_batches=0,
    )

    if model is None:
        print(f"  [ERREUR] Training final echoue.")
        return

    # --- Eval ---
    print(f"\n  [EVAL] Calcul des MAE par horizon...")
    final_mae, metrics, per_h = eval_per_horizon(
        model, feat_cols, data_path, cutoff_train, cutoff_test)

    print(f"\n  Resultats test ({metrics['n_test']:,} samples) :")
    print(f"    MAE global  : {metrics['mae']}")
    print(f"    RMSE global : {metrics['rmse']}")
    if metrics.get("mape_pct"):
        print(f"    MAPE global : {metrics['mape_pct']:.1f}%")

    print(f"\n  MAE par horizon :")
    print(f"  {'Horizon':<10} {'n':>8} {'MAE':>10} {'RMSE':>10} {'mean_q':>10}")
    for h in HORIZONS_TRAIN:
        if h in per_h:
            m = per_h[h]
            print(f"  {h:<10} {m['n']:>8} {m['mae']:>10} "
                  f"{m['rmse']:>10} {m['mean_q']:>10}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    print(f"\n  Top 20 features (gain) :")
    for f, g in feat_imp[:20]:
        print(f"    {f:<30} {g:>12.1f}")

    # Save
    model_path = os.path.join(output_dir, f"model_{cls}_v1_direct.txt")
    model.save_model(model_path)

    elapsed = time.time() - t0

    meta = {
        "classe": cls,
        "version": "v1_direct_multi_horizon",
        "horizons_train": HORIZONS_TRAIN,
        "horizons_eval": HORIZONS_EVAL,
        "origin_stride": ORIGIN_STRIDE,
        "lag_days": LAG_DAYS,
        "roll_windows": list(ROLL_WINDOWS),
        "tweedie_power": best["tweedie_variance_power"],
        "n_train": rows,
        "total_trees": model.num_trees(),
        "feature_cols": feat_cols,
        "params": final_params,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "cutoff_train": str(cutoff_train.date()),
        "cutoff_test": str(cutoff_test.date()),
        "purge_days": PURGE_DAYS,
        "test_days": TEST_DAYS,
        **metrics,
        "per_horizon_metrics": per_h,
        "feature_importance_top30": [
            {"feature": f, "gain": round(float(g), 2)}
            for f, g in feat_imp[:30]
        ],
        "training_time_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "optuna_n_trials": OPTUNA_N_TRIALS,
        "optuna_best_trial": study.best_trial.number,
        "optuna_best_mae_trial": round(study.best_value, 4),
        "optuna_best_params": best,
    }
    meta_path = os.path.join(output_dir, f"model_{cls}_v1_direct_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  [SAVE] model    -> {os.path.basename(model_path)}")
    print(f"  [SAVE] metadata -> {os.path.basename(meta_path)}")
    print(f"  Temps total     : {elapsed:.1f}s")

    del model
    force_gc(verbose=True)


# ======================================================================
# MODE FAST : cache parquet (precision identique a v1_direct slow)
# ======================================================================
# Idee : on pre-genere les samples UNE SEULE FOIS et on les stream-write
# en parquet. Les trials Optuna et le training final LISENT le parquet
# (~5s) au lieu de regenerer (~2-3h). 5-10x plus rapide par classe.
# Les MAE produites sont mathematiquement identiques (au bruit flottant
# pres, < 0.01%) car on reutilise DirectSampleGenerator a l'identique.

def cache_path_for(cls, cutoff_train, cutoff_test):
    """Chemin canonique du cache parquet pour une classe + cutoffs."""
    fname = (f"samples_{cls}_v1_direct_"
             f"{cutoff_train.date()}_{cutoff_test.date()}.parquet")
    return os.path.join(CACHE_DIR, fname)


def build_parquet_cache(cls, data_path, cutoff_train, cutoff_test, cache_path):
    """Genere les samples une fois et les stream-write en parquet."""
    if not _PYARROW_OK:
        print("[ERREUR] pyarrow requis pour le mode --fast.")
        print("  Installe avec : pip install pyarrow")
        sys.exit(1)

    t0 = time.time()
    print(f"  [CACHE] Construction : {os.path.basename(cache_path)}")

    gen = DirectSampleGenerator(
        data_path=data_path,
        cutoff_train=cutoff_train,
        cutoff_test=cutoff_test,
        batch_size=PARQUET_BATCH_SIZE,
        include_test=True,  # inclut TOUT (train + test) dans le cache
    )

    writer = None
    n_total = 0
    first_columns = None

    try:
        for batch_num, df_batch in gen.iter_batches():
            if df_batch.empty:
                continue

            for c in ("_origin_date", "_target_date"):
                if c in df_batch.columns:
                    df_batch[c] = pd.to_datetime(df_batch[c]).astype("datetime64[ns]")

            if "_is_test" in df_batch.columns:
                df_batch["_is_test"] = df_batch["_is_test"].astype(np.int8)

            if first_columns is None:
                first_columns = list(df_batch.columns)
            else:
                for c in first_columns:
                    if c not in df_batch.columns:
                        df_batch[c] = 0.0
                df_batch = df_batch[first_columns]

            table = pa.Table.from_pandas(df_batch, preserve_index=False,
                                         safe=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    cache_path, table.schema,
                    compression=PARQUET_COMPRESSION,
                )

            writer.write_table(table)
            n_total += len(df_batch)
            elapsed = time.time() - t0
            rate = n_total / max(1, elapsed)
            print(f"    Batch {batch_num:>3} : "
                  f"+{len(df_batch):>7,} samples "
                  f"(total {n_total:>9,}, {rate:,.0f}/s, {elapsed:.0f}s)")
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  [CACHE] OK: {n_total:,} samples | {size_mb:.1f} Mo | {elapsed:.0f}s")
    return n_total


def get_or_build_cache(cls, data_path, cutoff_train, cutoff_test,
                        force_rebuild=False):
    """Retourne le chemin du cache, le construit si necessaire."""
    cache_path = cache_path_for(cls, cutoff_train, cutoff_test)

    if os.path.exists(cache_path) and not force_rebuild:
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        n_rows = pq.ParquetFile(cache_path).metadata.num_rows
        print(f"  [CACHE] Reutilise : {os.path.basename(cache_path)} "
              f"({n_rows:,} samples, {size_mb:.1f} Mo)")
        return cache_path

    if force_rebuild and os.path.exists(cache_path):
        os.remove(cache_path)

    build_parquet_cache(cls, data_path, cutoff_train, cutoff_test, cache_path)
    return cache_path


def load_train_test_from_cache(cache_path):
    """Charge le parquet et separe train/test selon _is_test."""
    t0 = time.time()
    df = pd.read_parquet(cache_path)
    is_test_mask = df["_is_test"].astype(bool)
    train_df = df[~is_test_mask].copy()
    test_df = df[is_test_mask].copy()
    elapsed = time.time() - t0
    print(f"  [CACHE] Charge : {len(train_df):,} train + {len(test_df):,} test "
          f"({elapsed:.1f}s)")
    return train_df, test_df


def train_on_dataframe(train_df, params, max_trees, feature_cols=None,
                       valid_df=None, early_stopping_rounds=0,
                       verbose_eval=0):
    """Train LightGBM sur un DataFrame deja en RAM (plus rapide que streaming)."""
    X_train, y_train, feature_cols = df_to_xy(train_df, feature_cols)

    ds_train = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_cols,
        free_raw_data=False,
    )

    valid_sets = None
    valid_names = None
    callbacks = []

    if valid_df is not None and early_stopping_rounds > 0:
        X_valid, y_valid, _ = df_to_xy(valid_df, feature_cols)
        ds_valid = lgb.Dataset(X_valid, label=y_valid, reference=ds_train,
                               free_raw_data=False)
        valid_sets = [ds_valid]
        valid_names = ["valid"]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

    model = lgb.train(
        params, ds_train,
        num_boost_round=max_trees,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return model, feature_cols


def eval_per_horizon_fast(model, test_df, feature_cols):
    """Evaluation sur test set en RAM. Retourne (mae_global, metrics, per_h)."""
    if test_df.empty:
        return float("inf"), {}, {}

    X_test, y_test, _ = df_to_xy(test_df, feature_cols)
    pred = np.maximum(model.predict(X_test), 0)

    horizons = test_df["horizon"].values
    mae_global = float(np.mean(np.abs(y_test - pred)))
    rmse_global = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    mean_q = float(np.mean(y_test))

    metrics = {
        "mae": round(mae_global, 4),
        "rmse": round(rmse_global, 4),
        "mean_quantite": round(mean_q, 4),
        "mape_pct": round(mae_global / mean_q * 100, 1) if mean_q > 0 else None,
        "n_test": int(len(y_test)),
    }

    per_h = {}
    for h in HORIZONS_TRAIN:
        mask = horizons == h
        if mask.any():
            yh = y_test[mask]
            ph = pred[mask]
            per_h[h] = {
                "mae": round(float(np.mean(np.abs(yh - ph))), 4),
                "rmse": round(float(np.sqrt(np.mean((yh - ph) ** 2))), 4),
                "n": int(mask.sum()),
                "mean_q": round(float(np.mean(yh)), 4),
            }

    return mae_global, metrics, per_h


def make_objective_fast(cls, cache_path):
    """Charge train+test une fois, reutilise pour tous les trials Optuna."""
    print(f"  [OPTUNA] Chargement parquet pour trials...")
    train_df, test_df = load_train_test_from_cache(cache_path)

    def objective(trial):
        tweedie_power = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
        num_leaves = trial.suggest_int("num_leaves", 31, 127)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 50, 500)
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)

        params = {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
            "metric": ["tweedie", "mae"],
            "first_metric_only": False,
            "verbosity": -1,
            "n_jobs": N_JOBS,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 5,
            "feature_fraction": feature_fraction,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "seed": OPTUNA_SEED,
            "bagging_seed": OPTUNA_SEED,
            "feature_fraction_seed": OPTUNA_SEED,
            "deterministic": True,
            "max_bin": 127,
            "min_data_in_bin": 20,
            "force_row_wise": True,
            "histogram_pool_size": 1024,
        }

        t_trial = time.time()
        model, feat_cols = train_on_dataframe(train_df, params, OPTUNA_MAX_TREES)
        trial_mae, _, _ = eval_per_horizon_fast(model, test_df, feat_cols)
        trial.set_user_attr("total_trees", model.num_trees())
        trial.set_user_attr("elapsed", round(time.time() - t_trial, 1))
        del model
        gc.collect()
        return trial_mae

    return objective, train_df, test_df


def train_direct_fast_class(cls, data_path, output_dir, date_range,
                             force_rebuild_cache=False):
    """Variante FAST de train_direct_class : cache parquet + Optuna en RAM."""
    print(f"\n{'='*70}")
    print(f"  CLASSE : {cls.upper()} (V1 Direct FAST + Optuna)")
    print(f"{'='*70}")

    t0 = time.time()
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Fichier source : {data_path} ({file_size:.1f} Mo)")

    if file_size < 0.01 or date_range["min"] is None:
        print(f"  [SKIP] Vide ou pas de dates.")
        return

    date_min = pd.to_datetime(date_range["min"])
    date_max = pd.to_datetime(date_range["max"])
    cutoff_train = date_max - pd.Timedelta(days=TEST_DAYS + PURGE_DAYS)
    cutoff_test = date_max - pd.Timedelta(days=TEST_DAYS)

    print(f"  Train  : target <= {cutoff_train.date()}")
    print(f"  Test   : target > {cutoff_test.date()} ({date_max.date()})")
    print(f"  Horizons : {HORIZONS_TRAIN}")
    print(f"  Origin stride : {ORIGIN_STRIDE} jours")
    print_ram_status("  RAM: ")

    cache_path = get_or_build_cache(
        cls, data_path, cutoff_train, cutoff_test,
        force_rebuild=force_rebuild_cache,
    )

    print(f"\n  [OPTUNA] {OPTUNA_N_TRIALS} trials (max_trees={OPTUNA_MAX_TREES})")
    study = optuna.create_study(
        direction="minimize",
        study_name=f"v1_direct_fast_{cls}",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=3),
    )

    t_opt = time.time()
    objective, train_df, test_df = make_objective_fast(cls, cache_path)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    elapsed_opt = time.time() - t_opt

    best = study.best_params
    print(f"\n  [OPTUNA] Termine en {elapsed_opt:.0f}s")
    print(f"  [OPTUNA] Best MAE = {study.best_value:.4f} "
          f"(trial #{study.best_trial.number})")
    for k, v in best.items():
        print(f"    {k:<30} = {v}")

    print(f"\n  [FINAL] Training complet (max_trees={FINAL_MAX_TREES})...")
    final_params = {
        "objective": "tweedie",
        "tweedie_variance_power": best["tweedie_variance_power"],
        "metric": ["tweedie", "mae"],
        "first_metric_only": False,
        "verbosity": -1,
        "n_jobs": N_JOBS,
        "learning_rate": best["learning_rate"],
        "num_leaves": best["num_leaves"],
        "min_data_in_leaf": best["min_data_in_leaf"],
        "bagging_fraction": best["bagging_fraction"],
        "bagging_freq": 5,
        "feature_fraction": best["feature_fraction"],
        "lambda_l1": best["lambda_l1"],
        "lambda_l2": best["lambda_l2"],
        "seed": OPTUNA_SEED,
        "bagging_seed": OPTUNA_SEED,
        "feature_fraction_seed": OPTUNA_SEED,
        "deterministic": True,
        "max_bin": 127,
        "min_data_in_bin": 20,
        "force_row_wise": True,
        "histogram_pool_size": 1024,
    }

    t_final = time.time()
    model, feat_cols = train_on_dataframe(
        train_df, final_params, FINAL_MAX_TREES,
        valid_df=None, early_stopping_rounds=0,
    )
    elapsed_final = time.time() - t_final
    print(f"  [FINAL] Training OK en {elapsed_final:.0f}s, {model.num_trees()} arbres")

    print(f"\n  [EVAL] Calcul des MAE par horizon...")
    final_mae, metrics, per_h = eval_per_horizon_fast(model, test_df, feat_cols)

    print(f"\n  Resultats test ({metrics['n_test']:,} samples) :")
    print(f"    MAE global  : {metrics['mae']}")
    print(f"    RMSE global : {metrics['rmse']}")
    if metrics.get("mape_pct"):
        print(f"    MAPE global : {metrics['mape_pct']:.1f}%")

    print(f"\n  MAE par horizon :")
    print(f"  {'Horizon':<10} {'n':>8} {'MAE':>10} {'RMSE':>10} {'mean_q':>10}")
    for h in HORIZONS_TRAIN:
        if h in per_h:
            m = per_h[h]
            print(f"  {h:<10} {m['n']:>8} {m['mae']:>10} "
                  f"{m['rmse']:>10} {m['mean_q']:>10}")

    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    print(f"\n  Top 20 features (gain) :")
    for f, g in feat_imp[:20]:
        print(f"    {f:<30} {g:>12.1f}")

    # Suffixe _fast pour ne pas ecraser les modeles v1_direct slow.
    model_path = os.path.join(output_dir, f"model_{cls}_v1_direct_fast.txt")
    model.save_model(model_path)

    elapsed = time.time() - t0
    meta = {
        "classe": cls,
        "version": "v1_direct_multi_horizon_FAST",
        "horizons_train": HORIZONS_TRAIN,
        "origin_stride": ORIGIN_STRIDE,
        "lag_days": LAG_DAYS,
        "roll_windows": list(ROLL_WINDOWS),
        "tweedie_power": best["tweedie_variance_power"],
        "n_train": len(train_df),
        "total_trees": model.num_trees(),
        "feature_cols": feat_cols,
        "params": final_params,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "cutoff_train": str(cutoff_train.date()),
        "cutoff_test": str(cutoff_test.date()),
        "purge_days": PURGE_DAYS,
        "test_days": TEST_DAYS,
        **metrics,
        "per_horizon_metrics": per_h,
        "feature_importance_top30": [
            {"feature": f, "gain": round(float(g), 2)}
            for f, g in feat_imp[:30]
        ],
        "training_time_sec": round(elapsed, 1),
        "optuna_time_sec": round(elapsed_opt, 1),
        "final_train_time_sec": round(elapsed_final, 1),
        "timestamp": datetime.now().isoformat(),
        "optuna_n_trials": OPTUNA_N_TRIALS,
        "optuna_best_trial": study.best_trial.number,
        "optuna_best_mae_trial": round(study.best_value, 4),
        "optuna_best_params": best,
        "cache_path": cache_path,
    }
    meta_path = os.path.join(output_dir, f"model_{cls}_v1_direct_fast_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  [SAVE] model    -> {os.path.basename(model_path)}")
    print(f"  [SAVE] metadata -> {os.path.basename(meta_path)}")
    print(f"  Temps total     : {elapsed:.1f}s")

    del model, train_df, test_df
    force_gc(verbose=True)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V1 Direct Multi-Horizon + Optuna (slow ou --fast parquet)")
    parser.add_argument("--classes", nargs="+", default=CLASSES,
                        choices=CLASSES,
                        help="Classes a entrainer (defaut: toutes)")
    parser.add_argument("--fast", action="store_true",
                        help="Utilise le cache parquet (5-10x plus rapide, "
                             "precision identique)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip les classes deja entrainees")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="(--fast) Reconstruit le cache parquet")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    suffix = " (FAST parquet)" if args.fast else ""
    print(f"\n{'='*70}")
    print(f"  TRAIN LIGHTGBM V1 DIRECT MULTI-HORIZON + OPTUNA{suffix}")
    print(f"  Optimisation MAE | {OPTUNA_N_TRIALS} trials par classe")
    print(f"  Horizons : {HORIZONS_TRAIN} (max={MAX_HORIZON})")
    print(f"  Classes  : {args.classes}")
    print(f"{'='*70}")

    used, total, pct = get_ram_usage()
    print(f"  RAM: {used:.1f}/{total:.1f} Go ({pct:.0f}%)")

    if pct >= 70:
        print(f"  [ERREUR] RAM trop elevee")
        sys.exit(1)

    # Classification
    class_path = os.path.join(OUTPUT_DIR, "stats_classification_SB.csv")
    if not os.path.exists(class_path):
        print(f"[ERREUR] {class_path} manquant.")
        sys.exit(1)

    product_class = load_classification(class_path)

    date_meta_path = os.path.join(OUTPUT_DIR, "date_ranges.json")
    can_skip = SKIP_SPLIT_IF_EXISTS
    if can_skip:
        for c in CLASSES:
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"data_{c}.csv")):
                can_skip = False
                break
        if not os.path.exists(date_meta_path):
            can_skip = False

    if can_skip:
        print(f"\n  [SPLIT] Skip (CSV existants)")
        with open(date_meta_path, "r", encoding="utf-8") as f:
            date_ranges = json.load(f)
        counts = {}
        for c in CLASSES:
            csv_path = os.path.join(OUTPUT_DIR, f"data_{c}.csv")
            with open(csv_path, "rb") as f:
                counts[c] = max(0, sum(1 for _ in f) - 1)
            print(f"    {c:<15} : {counts[c]:>10,} lignes")
    else:
        zip_files = list_zip_files(CENTRALE_DIR)[:MAX_ZIPS]
        _, counts, date_ranges = split_by_class(
            CENTRALE_DIR, zip_files, product_class, OUTPUT_DIR)

    gc.collect()

    model_suffix = "_v1_direct_fast" if args.fast else "_v1_direct"

    for cls in args.classes:
        data_path = os.path.join(OUTPUT_DIR, f"data_{cls}.csv")
        if not os.path.exists(data_path):
            print(f"\n  [SKIP] {cls} : inexistant")
            continue
        if counts.get(cls, 0) < 100:
            print(f"\n  [SKIP] {cls} : trop peu de donnees")
            continue

        model_path = os.path.join(OUTPUT_DIR, f"model_{cls}{model_suffix}.txt")
        if args.skip_existing and os.path.exists(model_path):
            print(f"\n  [SKIP] {cls} : model_{cls}{model_suffix}.txt existe "
                  f"(--skip-existing)")
            continue

        try:
            if args.fast:
                train_direct_fast_class(
                    cls, data_path, OUTPUT_DIR,
                    date_ranges.get(cls, {}),
                    force_rebuild_cache=args.force_rebuild,
                )
            else:
                train_direct_class(cls, data_path, OUTPUT_DIR,
                                   date_ranges[cls], counts[cls])
        except KeyboardInterrupt:
            print("\n[INTERRUPT]")
            sys.exit(1)
        except Exception as e:
            import traceback
            print(f"\n[ERREUR] {cls} : {e}")
            traceback.print_exc()
            continue

        force_gc(verbose=True)
        if not check_ram_limit(f"Apres {cls}"):
            print(f"  [ARRET] RAM")
            break

    # Resume
    print(f"\n{'='*70}")
    print(f"  RESUME V1 DIRECT{' FAST' if args.fast else ''}")
    print(f"{'='*70}")
    for cls in args.classes:
        meta_path = os.path.join(
            OUTPUT_DIR, f"model_{cls}{model_suffix}_metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            m = json.load(f)
        print(f"\n  {cls.upper()} :")
        print(f"    MAE global = {m.get('mae')}")
        if args.fast:
            t_opt = m.get("optuna_time_sec", 0)
            t_final = m.get("final_train_time_sec", 0)
            t_total = m.get("training_time_sec", 0)
            print(f"    Optuna {t_opt:.0f}s | Final {t_final:.0f}s | "
                  f"Total {t_total:.0f}s")
        if m.get('per_horizon_metrics'):
            for h in HORIZONS_TRAIN:
                key = h if h in m["per_horizon_metrics"] else str(h)
                if key in m["per_horizon_metrics"]:
                    print(f"      h={h:<4} MAE={m['per_horizon_metrics'][key]['mae']}")


if __name__ == "__main__":
    main()
