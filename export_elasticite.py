#!/usr/bin/env python3
"""
export_elasticite.py

Calcule les elasticites par couple (ID_PRODUIT, ID_SO) a partir des modeles
LightGBM sauvegardes (V2, V2_optuna, V1, global...) et exporte en .txt
au format {KEY=VALUE} compatible avec l'API existante.

Trois sorties par couple :
  1. Metriques de qualite : MAE, RMSE, wMAPE (prediction vs reel)
  2. TreeSHAP  : contribution relative de chaque feature (%) — par couple
  3. Perturbation : elasticite reelle apprise par le modele (%)
     -> BATCHED : toutes les observations de la classe en une passe

Performance :
  Le batching evite 200k petits predict() et les remplace par ~20 gros.
  LightGBM parallelise nativement sur les gros arrays (SIMD, multi-thread).
  Pour limiter la RAM, on perturbe feature par feature au lieu de tout empiler.
  Resultat : ~2-5 min par classe pour la perturbation, meme avec >10 000 couples.
  SHAP par couple est plus lent (~100-500ms/couple) : utiliser --no-shap pour un 1er run.

Sortie :
  CENTRAL_ELASTICITE_{label}.txt  (une ligne par couple)

Usage :
  python export_elasticite.py                   # meilleur modele auto-detecte
  python export_elasticite.py --model v2        # forcer V2 vanilla
  python export_elasticite.py --model v2_optuna # forcer V2 Optuna
  python export_elasticite.py --no-shap         # perturbation seulement (plus rapide)
"""
from __future__ import annotations

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# ======================================================================
# PATH SETUP
# ======================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

OUTPUT_DIR = os.path.expanduser("~/output_centrale")
CLASSES = ["smooth", "intermittent", "erratic", "lumpy"]
TARGET = "QUANTITE"

# Client par defaut pour le prefixe du fichier.
# Doit matcher EXPORT_CLIENT cote Java (CopierElasticteFromSftp_SO).
# Le Java filtre par entry.getFilename().startsWith(client.toUpperCase()+"_ELASTICITE").
DEFAULT_CLIENT = "CENTRALE"

# Caracteres interdits dans les valeurs (cassent le parser Java tres naif)
_FORBIDDEN_VALUE_CHARS = (',', '=', '"', '{', '}')

# =====================================================================
# DEUX LISTES D'EXCLUSION (split depuis 2026-04-22) :
#
# EXCLUDE_ALL / EXCLUDE_ALL_PREFIXES
#   -> Features totalement ignorees (ni en ELAST, ni en SHAP).
#      IDs, cyclicals, calendrier brut : aucune valeur business ni diagnostique.
#
# EXCLUDE_ELAST_ONLY / EXCLUDE_ELAST_ONLY_PREFIXES
#   -> Features exclues UNIQUEMENT de l'elasticite (perturbation) mais gardees
#      en SHAP_PARAM_* pour diagnostic. "Changer le passe" n'a pas de sens
#      business (pas un levier actionnable), mais savoir que le modele est
#      driven par roll_mean_28 vs roll_mean_91 est une info business precieuse.
#
# Le parser Java CopierElasticteFromSftp_SO() ne lit que les ELAST_PARAM_*,
# donc la base ne sera polluee par aucune "elasticite" semantiquement fausse.
# Les SHAP_PARAM_* restent dans le .txt pour consultation humaine / outils.
# =====================================================================

EXCLUDE_ALL = {
    # --- Identifiants ---
    "ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE", "sb_class",
    # --- Calendrier brut / redondant ---
    "PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR", "PARAM_JOUR_SEMAIN",
    "dow", "month", "day_of_year",
    "mois", "semaine", "jour_annee",
    # --- Horizon (parametre du modele, pas une feature produit) ---
    "horizon",
    # --- Cycliques sin/cos (redondance avec target_dow/month en V1 Direct) ---
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "day_sin", "day_cos",
    "sin_doy", "cos_doy", "sin_dow", "cos_dow", "sin_month", "cos_month",
    "sin_jour", "cos_jour", "sin_semaine", "cos_semaine",
    "sin_mois", "cos_mois",
    # --- Prix d'achat brut (doublon de PARAM_PRIX_ACHAT, scale aberrant) ---
    "PRIX_ACHAT01",
}
EXCLUDE_ALL_PREFIXES = ()

# Features historiques / calcules : pas des leviers mais info diagnostique utile.
# -> gardees en SHAP, exclues de ELAST.
EXCLUDE_ELAST_ONLY = {
    # V1 Direct : calendrier target (fait de la date, pas actionnable)
    "target_dow", "target_month", "target_day", "target_doy",
    "target_is_weekend", "target_week",
    # V1 Direct : etats calcules
    "days_since_sale", "cv_28",
    # V1 Direct : trend (pente OLS), pas actionnable
    "trend_28",
}
EXCLUDE_ELAST_ONLY_PREFIXES = (
    # V1 Direct : lags ancres origine (y(origin), y(origin-1), y(origin-7), ...)
    "y_origin_m", "y_origin",
    # V1 Direct : rolling stats calcules sur [origin-W, origin]
    "roll_mean_", "roll_std_", "roll_max_", "roll_min_",
    "roll_nonzero_ratio_", "roll_quantile_",
    # V1 Direct : autres derivees
    "trend_", "momentum_",
    # V2 : ratios dynamiques
    "ratio_dow", "ratio_month", "cv_month", "interaction_dow",
    "mean_dow_today", "mean_month_today", "std_month_today",
)

# Compat : EXCLUDE_FEATURES et EXCLUDE_PREFIXES sont desormais l'UNION
# (tout ce qui est exclu de l'elasticite). Utilisees par _classify_feature.
EXCLUDE_FEATURES = EXCLUDE_ALL | EXCLUDE_ELAST_ONLY
EXCLUDE_PREFIXES = EXCLUDE_ALL_PREFIXES + EXCLUDE_ELAST_ONLY_PREFIXES

# Classification des features pour la perturbation
# Binaires : impact quand 0->1
FEAT_BINAIRES_PREFIXES = ('PARAM_JS_',)
FEAT_BINAIRES = {'PARAM_VACANCE', 'est_promo'}
# Promo : perturbation conditionnelle
FEAT_PROMO_PREFIXES = ('PARAM_PROMO_',)
# Calendrier unitaire : perturbation +/-1 unite
FEAT_CALENDRIER_UNIT = {'PARAM_ANNEE'}
MIN_PRED_THRESHOLD = 0.5
MAX_PROMO_IMPACT = 150.0
MIN_PROMO_OBS = 1  # Abaisse : on veut preserver CHAQUE PARAM_PROMO_*
                    # meme pour les couples avec peu d'historique promo

# Garde-fou numerique : clip dur pour les elasticites continues. Au-dela,
# c'est un artefact (division par y_base proche de 0 sur quelques rows).
# Cf. ancien `compute_elasticites` qui se reposait sur Q_sum global pour
# eviter ce probleme.
MAX_ABS_ELASTICITY = 300.0

# Pre-check empirique pour les promos : ratio minimal de variation reelle
# de la quantite vendue entre lignes promo vs lignes hors promo, en dessous
# duquel on considere que la promo n'a pas d'effet detectable.
MIN_PROMO_RATIO_EMP = 0.15

# Pre-check de range pour les features continues : si la plage observee
# (q90 - q10) / |q10| est < ce seuil, la feature ne varie pas assez pour
# tirer un signal d'elasticite fiable.
MIN_CONTINUOUS_RANGE = 0.10

# Contraintes de signe par feature business. Force le respect des regles
# economiques de base : prix monte -> ventes baissent, promo monte -> ventes
# montent. L'absence d'entree = pas de contrainte (le modele decide librement).
#   ("max", 0.0)  -> elasticite plafonnee a 0   (signe negatif force)
#   ("min", 0.0)  -> elasticite plancher a 0    (signe positif force)
SIGN_CONSTRAINTS = {
    # Prix : hausse prix -> baisse ventes (elasticite negative)
    "PARAM_PRIX":             ("max", 0.0),
    "PARAM_LOG_PRIX":         ("max", 0.0),
    "PARAM_PRIX_ACHAT":       ("max", 0.0),
    "PARAM_MARKUP_PCT":       ("max", 0.0),
    # Promo / discount : effet positif sur ventes
    "PARAM_PRIX_VENTE_PROMO": ("min", 0.0),
    "PARAM_DISCOUNT_PCT":     ("min", 0.0),
    "PARAM_RATIO_PROMO":      ("min", 0.0),
    "PARAM_EST_PROMO":        ("min", 0.0),
}


def _apply_sign_constraint(value, feat):
    """
    Applique la contrainte de signe metier pour `feat` si elle existe.
    Renvoie value clipe ou inchange.
    """
    rule = SIGN_CONSTRAINTS.get(feat)
    if rule is None:
        return value
    op, threshold = rule
    if op == "max":
        return min(value, threshold)
    return max(value, threshold)



# ======================================================================
# DATA LOADING (reutilise compare_v1_v2)
# ======================================================================
from compare_v1_v2 import load_data


# ======================================================================
# MODEL DISCOVERY
# ======================================================================

MODEL_PRIORITY = ["v1_direct", "v2_optuna", "v2", "global_optuna", "global", "v1_optuna", "v1"]

MODEL_CONFIGS = {
    "v1": {
        "model_file": "model_{cls}.txt",
        "meta_file": "model_{cls}_metadata.json",
        "profiles_file": None,
        "predict_type": "v1",
    },
    "v1_optuna": {
        "model_file": "model_{cls}_v1_optuna.txt",
        "meta_file": "model_{cls}_v1_optuna_metadata.json",
        "profiles_file": None,
        "predict_type": "v1",
    },
    "v1_direct": {
        "model_file": "model_{cls}_v1_direct.txt",
        "meta_file": "model_{cls}_v1_direct_metadata.json",
        "profiles_file": None,
        "predict_type": "v1_direct",
    },
    "v1_direct_fast": {
        "model_file": "model_{cls}_v1_direct_fast.txt",
        "meta_file": "model_{cls}_v1_direct_fast_metadata.json",
        "profiles_file": None,
        "predict_type": "v1_direct",
    },
    "v2": {
        "model_file": "model_{cls}_v2.txt",
        "meta_file": "model_{cls}_v2_metadata.json",
        "profiles_file": "profiles_{cls}.parquet",
        "predict_type": "v2",
    },
    "v2_optuna": {
        "model_file": "model_{cls}_v2_optuna.txt",
        "meta_file": "model_{cls}_v2_optuna_metadata.json",
        "profiles_file": "profiles_{cls}_optuna.parquet",
        "predict_type": "v2",
    },
    "global": {
        "model_file": "model_global.txt",
        "meta_file": "model_global_metadata.json",
        "profiles_file": "profiles_global_{cls}.parquet",
        "predict_type": "global",
    },
    "global_optuna": {
        "model_file": "model_global_optuna.txt",
        "meta_file": "model_global_optuna_metadata.json",
        "profiles_file": "profiles_global_optuna_{cls}.parquet",
        "predict_type": "global",
    },
}


def find_best_model(cls, forced_model=None):
    """
    Trouve le meilleur modele disponible pour une classe SB.
    Retourne (model_type, booster, metadata, features, profiles_df) ou None.
    """
    priority = [forced_model] if forced_model else MODEL_PRIORITY

    for mtype in priority:
        cfg = MODEL_CONFIGS.get(mtype)
        if cfg is None:
            continue

        model_path = os.path.join(OUTPUT_DIR, cfg["model_file"].format(cls=cls))
        meta_path = os.path.join(OUTPUT_DIR, cfg["meta_file"].format(cls=cls))

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            continue

        profiles = None
        if cfg["profiles_file"]:
            prof_path = os.path.join(OUTPUT_DIR, cfg["profiles_file"].format(cls=cls))
            if not os.path.exists(prof_path):
                continue
            profiles = pd.read_parquet(prof_path)

        booster = lgb.Booster(model_file=model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        features = meta.get("features") or meta.get("feature_cols") or []

        print(f"  [{cls}] Modele selectionne : {mtype} "
              f"({len(features)} features, {booster.num_trees()} arbres)")
        return mtype, booster, meta, features, profiles

    return None


# ======================================================================
# BUILD ALL OBSERVATIONS FOR A CLASS (BATCHED)
# ======================================================================

def build_all_observations(df, features, profiles, predict_type, cls=None):
    """
    Construit le DataFrame d'observations pour TOUS les couples d'une classe.
    Retourne (obs_df, couple_ids) ou obs_df contient toutes les lignes
    et couple_ids est un array (n_rows,) avec l'index du couple pour chaque ligne.

    On ajoute des colonnes _prod et _so pour pouvoir groupby apres.
    """
    if predict_type == "v1_direct":
        # V1 Direct : pour chaque couple, on genere des samples (origin, horizon, target)
        # avec origin = derniere date disponible et horizons 1..90.
        # Les features d'origine (lags + rolling stats) sont calculees une fois
        # par produit, et les features target (prix, promo) sont prises a la
        # date target = origin + horizon.
        from train_lightgbm_v1_direct import (
            generate_samples_for_product, HORIZONS_TRAIN, MIN_HISTORY,
        )

        # Detecter les target_cols (exclure IDs + calendrier)
        EXCLUDE_TARGET = {
            "ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE", "date",
            "QUANTITE", "PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR",
            "PARAM_JOUR_SEMAIN",
        }
        target_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in EXCLUDE_TARGET]

        # Cutoffs : pour l'elasticite, on genere des samples sur les 90 derniers
        # jours (multiples origines) et TOUS les horizons. On utilise
        # include_test=True en positionnant cutoff_train=cutoff_test=date_max
        # de sorte que tout soit marque "test" (pour que generate_samples les
        # inclue tous).
        date_max = df["date"].max()
        cutoff_train = date_max - pd.Timedelta(days=MIN_HISTORY + max(HORIZONS_TRAIN))
        cutoff_test = cutoff_train

        all_samples = []
        couples = df.groupby(["ID_PRODUIT", "ID_SO"]).size().reset_index(name="n_obs")
        couples = couples[couples["n_obs"] >= MIN_HISTORY + min(HORIZONS_TRAIN)]

        for _, row in couples.iterrows():
            prod, so = int(row["ID_PRODUIT"]), int(row["ID_SO"])
            mask = (df["ID_PRODUIT"] == prod) & (df["ID_SO"] == so)
            df_c = df[mask]

            samples = generate_samples_for_product(
                df_c, target_cols, cutoff_train, cutoff_test, include_test=True
            )
            if not samples:
                continue

            df_samples = pd.DataFrame(samples)
            df_samples["_prod"] = prod
            df_samples["_so"] = so
            # Uniformiser avec le pipeline d'elasticite (V1/V2 utilisent
            # "date" et "QUANTITE"). v1_direct stocke le target sous "y"
            # et la date cible sous "_target_date" : on les renomme ici.
            df_samples["date"] = df_samples["_target_date"]
            if "y" in df_samples.columns:
                df_samples[TARGET] = df_samples["y"]  # TARGET = "QUANTITE"
            all_samples.append(df_samples)

        if not all_samples:
            return pd.DataFrame(), pd.DataFrame()

        big_df = pd.concat(all_samples, ignore_index=True)

        for c in features:
            if c not in big_df.columns:
                big_df[c] = 0.0

        return big_df, couples

    elif predict_type in ("v2", "global"):
        from train_lightgbm_v2 import build_sample as build_sample_v2, HORIZON
        CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

        date_max = df["date"].max()
        origin = date_max - pd.Timedelta(days=HORIZON)

        all_samples = []
        couples = df.groupby(["ID_PRODUIT", "ID_SO"]).size().reset_index(name="n_obs")
        couples = couples[couples["n_obs"] >= 30]

        for _, row in couples.iterrows():
            prod, so = int(row["ID_PRODUIT"]), int(row["ID_SO"])
            mask = (df["ID_PRODUIT"] == prod) & (df["ID_SO"] == so)
            df_c = df[mask]

            sample = build_sample_v2(df_c, origin, profiles, HORIZON)
            if sample.empty or len(sample) < 5:
                continue

            if predict_type == "global":
                sample["sb_class"] = CLASS_TO_INT.get(cls, 0)

            sample["_prod"] = prod
            sample["_so"] = so
            all_samples.append(sample)

        if not all_samples:
            return pd.DataFrame(), pd.DataFrame()

        big_df = pd.concat(all_samples, ignore_index=True)

        for c in features:
            if c not in big_df.columns:
                big_df[c] = 0.0

        return big_df, couples

    else:  # v1 / v1_optuna
        from train_lightgbm_sb import build_features as build_features_v1

        date_max = df["date"].max()
        cutoff = date_max - pd.Timedelta(days=90)
        recent = df[df["date"] > cutoff].copy()

        if recent.empty:
            return pd.DataFrame(), pd.DataFrame()

        couples = recent.groupby(["ID_PRODUIT", "ID_SO"]).size().reset_index(name="n_obs")
        couples = couples[couples["n_obs"] >= 30]

        # Filtrer avant build_features.
        # Remplace le recent.apply(lambda r: ..., axis=1) (row-wise, tres lent sur
        # gros df) par un merge semi-join : O(N log N) au lieu de O(N*K).
        recent = recent.merge(
            couples[["ID_PRODUIT", "ID_SO"]],
            on=["ID_PRODUIT", "ID_SO"],
            how="inner",
        )

        if recent.empty:
            return pd.DataFrame(), pd.DataFrame()

        recent["_prod"] = recent["ID_PRODUIT"].astype(int)
        recent["_so"] = recent["ID_SO"].astype(int)

        recent = build_features_v1(recent)
        for c in features:
            if c not in recent.columns:
                recent[c] = 0.0

        return recent, couples


# ======================================================================
# BATCHED PERTURBATION ELASTICITY
# ======================================================================

def _classify_feature(feat):
    """
    Retourne le type de feature : 'binary', 'promo', 'cal_unit', 'continuous', ou None.
    Fonctionne pour les features V1 (PARAM_*) et V2 (log_prix, est_promo, etc.).
    """
    if feat in EXCLUDE_FEATURES:
        return None
    if any(feat.startswith(p) for p in EXCLUDE_PREFIXES):
        return None

    if feat in FEAT_BINAIRES or any(feat.startswith(p) for p in FEAT_BINAIRES_PREFIXES):
        return 'binary'
    if any(feat.startswith(p) for p in FEAT_PROMO_PREFIXES):
        return 'promo'
    if feat in FEAT_CALENDRIER_UNIT:
        return 'cal_unit'
    return 'continuous'


# ----------------------------------------------------------------------
# HELPERS D'AGREGATION (vectorises avec pandas.groupby)
#
# Remplace l'ancien pattern :
#     for ci, ck in enumerate(unique_keys):
#         rows_mask = inverse == ci          # O(N) par couple
#         vals = impacts[rows_mask]          # O(N) par couple
#         ...
# Complexite : O(N * K) -> O(N log N)  (gain 50-200x sur l'agregation)
# ----------------------------------------------------------------------

def _cv(value, std_val):
    """Coefficient de variation, avec sentinel 99.0 quand |value| est negligeable."""
    return round(abs(std_val / value), 2) if abs(value) > 1e-8 else 99.0


def _groupby_mean_std(values, inverse):
    """
    Retourne (mean_series, std_series) indexes par couple (0..K-1).
    NaN sont droppes avant agregation. ddof=0 pour matcher np.std().
    """
    df = pd.DataFrame({"c": inverse, "v": values}).dropna(subset=["v"])
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = df.groupby("c")["v"]
    return g.mean(), g.std(ddof=0)


def _groupby_median_iqr(values, inverse):
    """Retourne (median_series, iqr_series) indexes par couple."""
    df = pd.DataFrame({"c": inverse, "v": values}).dropna(subset=["v"])
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = df.groupby("c")["v"]
    return g.median(), g.quantile(0.75) - g.quantile(0.25)


def _predict_with_col(booster, X, idx, new_col):
    """
    Predict apres remplacement in-place de la colonne idx par new_col.
    La colonne originale est restauree en sortie.

    Economise ~35x de RAM vs X.copy() (une colonne vs toute la matrice)
    et supprime les gros np.ndarray de 300+ Mo alloues a chaque iteration.
    """
    orig = X[:, idx].copy()
    try:
        X[:, idx] = new_col
        y = np.maximum(booster.predict(X), 0)
    finally:
        X[:, idx] = orig
    return y


# ----------------------------------------------------------------------
# HANDLERS PAR TYPE DE FEATURE
# Chacun retourne {couple_idx: {value, std, cv, type}} — couple_idx est
# l'indice 0..K-1 dans unique_keys. Le dict peut etre vide (feature sans
# impact agregable, ex. constante).
# ----------------------------------------------------------------------

def _perturb_binary(booster, X, idx, inverse, _col_vals, _y_base, feat, _y_true):
    """Feature binaire : impact (%) de la transition 0 -> 1.

    Clip per-row a [-MAX_ABS_ELASTICITY, +MAX_ABS_ELASTICITY] pour eviter
    qu'une poignee de rows avec y_0 quasi-nul ne produise une moyenne aberrante
    cote couple. Applique la contrainte de signe metier si definie.
    """
    y_0 = _predict_with_col(booster, X, idx, 0.0)
    y_1 = _predict_with_col(booster, X, idx, 1.0)

    mask = y_0 > MIN_PRED_THRESHOLD  # seuil plus strict que 1e-8
    impacts = np.where(mask, (y_1 - y_0) / y_0 * 100, np.nan)
    impacts = np.clip(impacts, -MAX_ABS_ELASTICITY, MAX_ABS_ELASTICITY)
    means, stds = _groupby_mean_std(impacts, inverse)

    out = {}
    for ci, m_raw in means.items():
        m = _apply_sign_constraint(float(m_raw), feat)
        m = round(m, 2)
        if m == 0.0:
            continue
        s = round(float(stds.get(ci, 0.0) or 0.0), 2)
        out[int(ci)] = {"value": m, "std": s, "cv": _cv(m, s),
                        "type": "impact_binaire"}
    return out


def _perturb_promo(booster, X, idx, inverse, col_vals, y_base, feat, y_true):
    """
    Promo (ex. PARAM_PROMO_27) : strategie A (historique) ou B (simulation).

    - A : impact reel = (y_base - y_off) / y_off sur les lignes ou la promo
          etait active. Utilise des que le couple a >= MIN_PROMO_OBS lignes.
    - B : impact simule = (y_on - y_base) / y_base ou y_on est obtenu en
          forcant la promo a sa valeur mediane historique. Fallback pour
          les couples sans historique sur cette promo.

    Pre-check empirique (si y_true dispo) : pour chaque couple ayant des
    rows promo ET non-promo, on calcule
        ratio_emp = |Q_avec - Q_sans| / max(Q_sans, 0.01)
    sur les QUANTITE reellement observees. Si ratio_emp < MIN_PROMO_RATIO_EMP
    pour un couple, la promo n'a pas de signal historique detectable -> on
    skip ce couple pour cette feature (evite de produire du bruit).
    """
    promo_active = col_vals > 1e-8

    # ---- Pre-check empirique par couple (s'appuie sur la QUANTITE reelle) ----
    couples_no_signal = set()
    if y_true is not None:
        df_emp = pd.DataFrame({
            "c": inverse, "promo": promo_active, "q": np.asarray(y_true),
        })
        q_avec = df_emp[df_emp["promo"]].groupby("c")["q"].mean()
        q_sans = df_emp[~df_emp["promo"]].groupby("c")["q"].mean()
        common = q_avec.index.intersection(q_sans.index)
        if len(common) > 0:
            qa = q_avec.loc[common]
            qs = q_sans.loc[common]
            ratio = (qa - qs).abs() / qs.clip(lower=0.01)
            couples_no_signal = set(
                ratio[ratio < MIN_PROMO_RATIO_EMP].index.tolist())

    col_nonzero = col_vals[promo_active]
    median_promo_val = (float(np.median(col_nonzero))
                        if len(col_nonzero) > 0 else 1.0)

    y_off = _predict_with_col(booster, X, idx, 0.0)
    y_on  = _predict_with_col(booster, X, idx, median_promo_val)

    mask_denom = y_off > MIN_PRED_THRESHOLD
    mask_A = promo_active & mask_denom
    impacts_A = np.where(mask_A, (y_base - y_off) / y_off * 100, np.nan)

    mask_B = y_base > MIN_PRED_THRESHOLD
    impacts_B = np.where(mask_B, (y_on - y_base) / y_base * 100, np.nan)

    df_A = pd.DataFrame({"c": inverse, "v": impacts_A}).dropna(subset=["v"])
    df_B = pd.DataFrame({"c": inverse, "v": impacts_B}).dropna(subset=["v"])

    n_A = df_A.groupby("c").size()
    couples_A = set(n_A[n_A >= MIN_PROMO_OBS].index.tolist())

    df_A_sel = df_A[df_A["c"].isin(couples_A)].assign(type_promo="impact_promo")
    df_B_sel = df_B[~df_B["c"].isin(couples_A)].assign(type_promo="impact_promo_simule")
    df_all = pd.concat([df_A_sel, df_B_sel], ignore_index=True)
    if df_all.empty:
        return {}

    # Eliminer les couples sans signal empirique
    if couples_no_signal:
        df_all = df_all[~df_all["c"].isin(couples_no_signal)]
        if df_all.empty:
            return {}

    # Clip a [p5, p95] par couple uniquement si >= 20 obs (robustesse).
    def _clip_if_enough(s):
        if len(s) >= 20:
            p5, p95 = np.percentile(s, [5, 95])
            return s.clip(p5, p95)
        return s
    df_all["v"] = df_all.groupby("c")["v"].transform(_clip_if_enough)

    g = df_all.groupby("c")["v"]
    med = g.median().clip(-MAX_PROMO_IMPACT, MAX_PROMO_IMPACT)
    iqr = g.quantile(0.75) - g.quantile(0.25)
    type_per_c = df_all.groupby("c")["type_promo"].first()

    out = {}
    for ci, m_raw in med.items():
        m = _apply_sign_constraint(float(m_raw), feat)
        m = round(m, 2)
        iqr_v = iqr.loc[ci]
        s = round(float((iqr_v / 1.349) if not np.isnan(iqr_v) else 0.0), 2)
        # On garde meme si m == 0.0 pour preserver la trace de chaque PROMO.
        out[int(ci)] = {"value": m, "std": s, "cv": _cv(m, s),
                        "type": str(type_per_c.loc[ci])}
    return out


def _perturb_cal_unit(booster, X, idx, inverse, col_vals, y_base, feat, _y_true):
    """Calendrier unitaire (ex. PARAM_ANNEE) : perturbation +/-1 symetrisee."""
    y_up   = _predict_with_col(booster, X, idx, col_vals + 1.0)
    y_down = _predict_with_col(booster, X, idx, col_vals - 1.0)

    mask = y_base > MIN_PRED_THRESHOLD
    impact_up   = np.where(mask, (y_up   - y_base) / y_base * 100, np.nan)
    impact_down = np.where(mask, (y_base - y_down) / y_base * 100, np.nan)
    impact_sym  = (impact_up + impact_down) / 2
    impact_sym  = np.clip(impact_sym, -MAX_ABS_ELASTICITY, MAX_ABS_ELASTICITY)

    med, iqr = _groupby_median_iqr(impact_sym, inverse)

    out = {}
    for ci, m_raw in med.items():
        m = _apply_sign_constraint(float(m_raw), feat)
        m = round(m, 2)
        if m == 0.0:
            continue
        iqr_v = iqr.loc[ci]
        s = round(float((iqr_v / 1.349) if not np.isnan(iqr_v) else 0.0), 2)
        out[int(ci)] = {"value": m, "std": s, "cv": _cv(m, s),
                        "type": "impact_annuel"}
    return out


def _perturb_continuous(booster, X, idx, inverse, col_vals, y_base, feat, _y_true):
    """
    Feature continue : elasticite via perturbation symetrique +/-10%.

    Garde-fous (sur le modele de l'ancien `compute_elasticites` qui
    travaillait par sums globaux) :
      1. Skip si la feature est ~constante (std < 1e-6)
      2. Skip si la plage observee est trop etroite : (q90-q10)/|q10| < 10%
      3. Mask plus strict y_base > MIN_PRED_THRESHOLD (au lieu de 1e-8)
      4. Clip per-row a [-MAX_ABS_ELASTICITY, +MAX_ABS_ELASTICITY] avant
         agregation, pour empecher quelques rows avec y_base proche de
         MIN_PRED_THRESHOLD de produire une moyenne aberrante (-1.3M%
         observe avec PARAM_PRIX_ACHAT)
      5. Sign constraint metier (PARAM_PRIX <= 0, PARAM_DISCOUNT >= 0...)
    """
    # 1. Pre-check : feature ~constante ?
    std_x  = float(np.std(col_vals))
    mean_x = float(np.mean(col_vals))
    if std_x < 1e-6:
        return {}

    # 2. Pre-check : plage observee suffisante ?
    q10, q90 = np.quantile(col_vals, [0.10, 0.90])
    if abs(q10) > 1e-6 and (q90 - q10) / abs(q10) < MIN_CONTINUOUS_RANGE:
        return {}

    # Delta de perturbation : 10% de la moyenne (ou std si moyenne nulle)
    if abs(mean_x) > 1e-8:
        delta = mean_x * 0.10
    elif std_x > 1e-8:
        delta = std_x * 0.10
    else:
        return {}

    y_up   = _predict_with_col(booster, X, idx, col_vals + delta)
    y_down = _predict_with_col(booster, X, idx, col_vals - delta)

    # 3. Mask strict pour eviter les explosions y_base ~ 0
    mask = y_base > MIN_PRED_THRESHOLD
    elast_up   = np.where(mask, (y_up   - y_base) / y_base / 0.10, np.nan)
    elast_down = np.where(mask, (y_base - y_down) / y_base / 0.10, np.nan)
    elast_sym  = (elast_up + elast_down) / 2

    # 4. Clip per-row : empeche les outliers de dominer la moyenne du couple
    elast_sym = np.clip(elast_sym, -MAX_ABS_ELASTICITY, MAX_ABS_ELASTICITY)

    means, stds = _groupby_mean_std(elast_sym, inverse)

    out = {}
    for ci, m_raw in means.items():
        # 5. Sign constraint metier
        m = _apply_sign_constraint(float(m_raw), feat)
        m = round(m, 2)
        if m == 0.0:
            continue
        s = round(float(stds.get(ci, 0.0) or 0.0), 2)
        out[int(ci)] = {"value": m, "std": s, "cv": _cv(m, s),
                        "type": "elasticite"}
    return out


_HANDLERS_BY_FTYPE = {
    "binary":     _perturb_binary,
    "promo":      _perturb_promo,
    "cal_unit":   _perturb_cal_unit,
    "continuous": _perturb_continuous,
}


# ----------------------------------------------------------------------
# ORCHESTRATEUR
# ----------------------------------------------------------------------

def compute_elasticity_batched(booster, big_df, features,
                                X=None, y_base=None):
    """
    Calcule l'elasticite par perturbation pour TOUTES les observations d'une classe.
    Perturbe feature par feature pour limiter la RAM (~1 Go max au lieu de ~10 Go).

    Args:
        booster: lgb.Booster
        big_df: DataFrame avec toutes les obs de la classe + colonnes _prod, _so
        features: liste des features du modele
        X: array (n_total, n_features) float32 pre-calcule (optionnel, gagne du temps)
        y_base: array (n_total,) float32 des predictions de base (optionnel,
            evite un predict() redondant cote appelant)

    Returns:
        (couple_results, unique_keys, X, y_base) — X et y_base sont renvoyes
        pour reutilisation par le code appelant (metriques, SHAP).
    """
    t0 = time.time()

    if X is None:
        X = big_df[features].values.astype(np.float32)
    prods = big_df["_prod"].values
    sos = big_df["_so"].values
    n_total = len(X)

    if y_base is None:
        print(f"    Prediction de base ({n_total:,} lignes)...")
        y_base = np.maximum(booster.predict(X), 0).astype(np.float32)
    else:
        print(f"    Prediction de base reutilisee ({n_total:,} lignes)")

    # QUANTITE reelle : utilisee par _perturb_promo pour le pre-check empirique
    # par couple. None si la colonne TARGET n'est pas dans big_df (peut arriver
    # sur certains generateurs de samples, cf. v1_direct early state).
    y_true = (big_df[TARGET].values.astype(np.float32)
              if TARGET in big_df.columns else None)

    # Index unique des couples : inverse[i] = indice 0..K-1 du couple de la ligne i
    couple_keys = np.array([f"{p}_{s}" for p, s in zip(prods, sos)])
    unique_keys, inverse = np.unique(couple_keys, return_inverse=True)
    n_couples = len(unique_keys)
    print(f"    {n_couples:,} couples uniques, {n_total:,} observations totales")

    # Filtrer les features pertinentes + recuperer leurs indices de colonne
    col_indices = {col: idx for idx, col in enumerate(features)}
    features_to_process = [(f, _classify_feature(f)) for f in features
                           if _classify_feature(f) is not None]
    print(f"    {len(features_to_process)} features a perturber")

    couple_results = {k: {} for k in unique_keys}

    for fi, (feat, ftype) in enumerate(features_to_process):
        idx = col_indices[feat]
        col_vals = X[:, idx].astype(np.float64)
        handler = _HANDLERS_BY_FTYPE[ftype]
        per_couple = handler(booster, X, idx, inverse, col_vals, y_base,
                              feat, y_true)

        # Merger dans le dict global (cle = string "prod_so")
        for ci, vals in per_couple.items():
            couple_results[unique_keys[ci]][feat] = vals

        if (fi + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"      feature {fi+1}/{len(features_to_process)} "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"    Perturbation terminee en {elapsed:.1f}s")

    return couple_results, unique_keys, X, y_base


# ======================================================================
# SHAP PAR COUPLE
# ======================================================================

def compute_shap_per_couple(booster, big_df, features, unique_keys, inverse):
    """
    Calcule les contributions SHAP (%) par couple (ID_PRODUIT, ID_SO).
    Utilise un seul TreeExplainer puis agrege par couple.

    Returns:
        dict {couple_key: {feature: contribution_pct}}
    """
    try:
        import shap
    except ImportError:
        print("  [WARN] shap non installe (pip install shap), SHAP desactive")
        return {}

    X = big_df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.fillna(0, inplace=True)

    print(f"    SHAP sur {len(X):,} lignes ({len(unique_keys):,} couples)...")
    t0 = time.time()

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X.values)
    # shap_values shape: (n_total, n_features)

    # Features a garder pour SHAP.
    # IMPORTANT : on exclut UNIQUEMENT EXCLUDE_ALL / EXCLUDE_ALL_PREFIXES
    # (IDs, cyclicals, calendrier brut). Les features historiques de V1 Direct
    # (lags, rolling stats, trend) sont GARDEES car leur SHAP a une valeur
    # diagnostique : savoir que le produit est drive par roll_mean_28 vs
    # roll_mean_91 est une info business utile.
    # Ces memes features sont exclues de l'elasticite (voir _classify_feature)
    # car on ne peut pas "changer le passe" -> ELAST_PARAM_* pas semantique.
    feat_mask = []
    for i, feat in enumerate(features):
        keep = (feat not in EXCLUDE_ALL and
                not any(feat.startswith(p) for p in EXCLUDE_ALL_PREFIXES))
        feat_mask.append(keep)

    # Agreger par couple : mean(|SHAP|) normalise en %
    result = {}
    for ci, ck in enumerate(unique_keys):
        rows_mask = inverse == ci
        sv = shap_values[rows_mask]  # (n_obs_couple, n_features)

        mean_abs = np.mean(np.abs(sv), axis=0)
        total_abs = np.sum(mean_abs)
        if total_abs == 0:
            result[ck] = {}
            continue

        couple_shap = {}
        for i, feat in enumerate(features):
            if not feat_mask[i]:
                continue
            pct = round(float(mean_abs[i] / total_abs * 100), 2)
            if pct > 0.01:
                couple_shap[feat] = pct
        result[ck] = couple_shap

        if (ci + 1) % 2000 == 0:
            print(f"      SHAP agrege {ci+1}/{len(unique_keys)} couples")

    elapsed = time.time() - t0
    print(f"    SHAP termine en {elapsed:.1f}s")

    return result


# ======================================================================
# METRIQUES PAR COUPLE (MAE, RMSE, wMAPE)
# ======================================================================

def compute_metrics_per_couple(y_true, y_pred, unique_keys, inverse):
    """
    Calcule MAE, RMSE et wMAPE par couple a partir des arrays globaux.

    Args:
        y_true: array (n_total,) valeurs reelles
        y_pred: array (n_total,) predictions du modele
        unique_keys: array des cles de couples uniques
        inverse: array (n_total,) mapping ligne -> index couple

    Returns:
        dict {couple_key: {mae, rmse, wmape, n_obs, mean_actual,
                           sum_actual, sum_pred, diff_pct}}
    """
    result = {}
    for ci, ck in enumerate(unique_keys):
        rows_mask = inverse == ci
        yt = y_true[rows_mask]
        yp = y_pred[rows_mask]

        # Filtrer NaN
        valid = ~(np.isnan(yt) | np.isnan(yp))
        yt = yt[valid]
        yp = yp[valid]
        n = len(yt)

        if n < 2:
            result[ck] = {"mae": None, "rmse": None,
                          "wmape": None, "n_obs": n, "mean_actual": None,
                          "sum_actual": None, "sum_pred": None,
                          "diff_pct": None}
            continue

        errors = yt - yp
        mae_val = float(np.mean(np.abs(errors)))
        rmse_val = float(np.sqrt(np.mean(errors ** 2)))
        mean_yt = float(np.mean(yt))

        # Sommes sur la fenetre de test : volume total reel vs predit
        sum_actual = float(np.sum(yt))
        sum_pred = float(np.sum(yp))
        # Biais en %: +ve = surpredit, -ve = sous-predit
        diff_pct = ((sum_pred - sum_actual) / sum_actual * 100) if sum_actual > 1e-8 else None

        # wMAPE = MAE / mean(y_true) * 100
        # Pertinent surtout pour smooth/erratic (volumes significatifs)
        wmape_val = (mae_val / mean_yt * 100) if mean_yt > 1e-8 else None

        result[ck] = {
            "mae": round(mae_val, 4),
            "rmse": round(rmse_val, 4),
            "wmape": round(wmape_val, 2) if wmape_val is not None else None,
            "n_obs": n,
            "mean_actual": round(mean_yt, 4),
            "sum_actual": round(sum_actual, 2),
            "sum_pred": round(sum_pred, 2),
            "diff_pct": round(diff_pct, 2) if diff_pct is not None else None,
        }

    return result


# ======================================================================
# EXPORT FORMAT {KEY=VALUE}
# ======================================================================

def normalize_feat_name(feat):
    """
    Uniformise le nom de feature pour la sortie.
    Toutes les covariables du modele sont prefixees PARAM_ pour
    homogeneiser le format (ELAST_PARAM_*, TYPE_PARAM_*, CV_PARAM_*).

    Exemples :
        PARAM_PRIX    -> PARAM_PRIX       (inchange)
        log_prix      -> PARAM_LOG_PRIX
        est_promo     -> PARAM_EST_PROMO
        markup_pct    -> PARAM_MARKUP_PCT
        PARAM_PROMO_1 -> PARAM_PROMO_1    (inchange, chaque promo preservee)
    """
    if feat.startswith("PARAM_"):
        return feat
    return f"PARAM_{feat.upper()}"


def _is_safe_value(v):
    """Verifie qu'une valeur est serialisable dans le format {K=V, K=V}.
    Interdit : NaN/Inf, None, caracteres qui cassent le parser Java."""
    if v is None:
        return False
    if isinstance(v, float):
        if not np.isfinite(v):
            return False
    sv = str(v)
    for c in _FORBIDDEN_VALUE_CHARS:
        if c in sv:
            return False
    return True


def dict_to_line(d):
    """Convertit un dict en ligne {KEY=VALUE, KEY=VALUE, ...}.

    Filtre silencieusement les valeurs non-serialisables (NaN, Inf, None,
    strings contenant ',', '=', '"', '{', '}') pour ne pas casser le parser
    Java CopierElasticteFromSftp_SO() qui fait des string-replace fragiles.
    """
    parts = [f"{k}={v}" for k, v in d.items() if _is_safe_value(v)]
    return "{" + ", ".join(parts) + "}"


# ======================================================================
# MAIN PROCESSING (BATCHED)
# ======================================================================

def _build_couple_metadata(df_chunk):
    """
    Extrait dates historiques et nomenclature par couple depuis le DataFrame brut
    du chunk. Retourne (couple_dates, couple_nomenclature) tous deux indexes
    par (prod, so).
    """
    couple_dates = {}
    couple_nomenclature = {}
    for (p, s), grp in df_chunk.groupby(["ID_PRODUIT", "ID_SO"]):
        key = (int(p), int(s))
        couple_dates[key] = (
            str(grp["date"].min().date()),
            str(grp["date"].max().date()),
        )
        if "ID_NOMENCLATURE" in grp.columns:
            nom_values = grp["ID_NOMENCLATURE"].dropna().unique()
            if len(nom_values) > 0:
                couple_nomenclature[key] = int(nom_values[0])
    return couple_dates, couple_nomenclature


def _build_rows(unique_keys, key_to_ids, couple_results, couple_metrics,
                 couple_dates, couple_nomenclature, shap_per_couple,
                 cls, model_type):
    """Assemble les dicts {K=V} qui seront serialises en une ligne par couple."""
    rows = []
    for ck in unique_keys:
        prod, so = key_to_ids[ck]
        row = {"ID_PRODUIT": prod, "ID_SO": so}

        nom = couple_nomenclature.get((prod, so))
        if nom is not None:
            row["ID_NOMENCLATURE"] = nom

        m = couple_metrics.get(ck, {})
        if m.get("mae") is not None:
            row["MAE"] = m["mae"]
            row["RMSE"] = m["rmse"]
            if m.get("wmape") is not None:
                row["WMAPE"] = m["wmape"]
            row["N_OBS"] = m["n_obs"]
            row["MEAN_ACTUAL"] = m["mean_actual"]
            row["SUM_ACTUAL"] = m["sum_actual"]
            row["SUM_PRED"] = m["sum_pred"]
            if m.get("diff_pct") is not None:
                row["DIFF_PCT"] = m["diff_pct"]

        for feat, contrib in shap_per_couple.get(ck, {}).items():
            norm = normalize_feat_name(feat)
            row[f"SHAP_{norm}"] = contrib

        # Perturbation : chaque PARAM_PROMO_X est preservee individuellement,
        # meme a 0, pour tracer la feature cote Java.
        for feat, vals in couple_results.get(ck, {}).items():
            val = vals.get("value", 0)
            is_promo = feat.startswith("PARAM_PROMO_")
            if val == 0.0 and not is_promo:
                continue
            norm = normalize_feat_name(feat)
            row[f"ELAST_{norm}"] = val
            row[f"TYPE_{norm}"] = vals.get("type", "elasticite")
            row[f"CV_{norm}"] = vals.get("cv", 99.0)

        dmin, dmax = couple_dates.get((prod, so), ("", ""))
        row["DATE_HIST_MIN"] = dmin
        row["DATE_HIST_MAX"] = dmax
        row["CLASSE_SB"] = cls
        row["MODELE"] = model_type
        row["OK"] = 1
        rows.append(row)
    return rows


def _process_chunk(df_chunk, booster, features, profiles, predict_type,
                    cls, model_type, use_shap):
    """
    Pipeline complet sur un sous-ensemble de couples :
      build_all_observations -> compute_elasticity_batched -> metrics -> rows.
    Toute la memoire lourde (big_df, X, y_pred) est liberee avant retour.

    Retourne la liste des lignes pretes a ecrire. Peut etre vide si aucun
    couple du chunk ne passe les filtres de build_all_observations.
    """
    t0 = time.time()
    big_df, _couples_info = build_all_observations(
        df_chunk, features, profiles, predict_type, cls)
    if big_df.empty:
        return []
    n_couples_chunk = big_df.groupby(["_prod", "_so"]).ngroups
    print(f"    build: {n_couples_chunk:,} couples, {len(big_df):,} obs "
          f"({time.time()-t0:.1f}s)")

    couple_dates, couple_nomenclature = _build_couple_metadata(df_chunk)

    couple_results, unique_keys, X_shared, y_pred = compute_elasticity_batched(
        booster, big_df, features)

    # Reconstruire inverse + mapping couple_key -> (prod, so) pour metriques/SHAP.
    prods = big_df["_prod"].values
    sos = big_df["_so"].values
    couple_keys_str = np.array([f"{p}_{s}" for p, s in zip(prods, sos)])
    _uk, first_idx, inverse = np.unique(couple_keys_str,
                                         return_index=True,
                                         return_inverse=True)
    prod_per_key = prods[first_idx]
    so_per_key = sos[first_idx]
    key_to_ids = {ck: (int(prod_per_key[i]), int(so_per_key[i]))
                  for i, ck in enumerate(_uk)}

    # Metriques par couple (reutilise y_pred, evite un predict() redondant).
    y_true = (big_df[TARGET].values.astype(np.float32)
              if TARGET in big_df.columns else None)
    couple_metrics = (compute_metrics_per_couple(y_true, y_pred, unique_keys, inverse)
                      if y_true is not None else {})

    # Liberer X le plus tot possible (le predict() SHAP n'en a pas besoin).
    del X_shared, y_pred
    gc.collect()

    shap_per_couple = (compute_shap_per_couple(booster, big_df, features,
                                                unique_keys, inverse)
                       if use_shap else {})

    rows = _build_rows(unique_keys, key_to_ids, couple_results, couple_metrics,
                        couple_dates, couple_nomenclature, shap_per_couple,
                        cls, model_type)

    del big_df, couple_results, couple_metrics, shap_per_couple
    gc.collect()
    return rows


def process_class(cls, forced_model=None, use_shap=True,
                   chunk_size=5000, write_rows_callback=None):
    """
    Traite une classe SB par CHUNKS de `chunk_size` couples pour borner la RAM.

    Sur gros volumes (v1_direct + classe intermittent : ~56k couples x ~875 rows
    par couple = 32M lignes), charger big_df + X en une seule passe consomme
    ~35 Go de RAM et provoque un OOM kill. Le chunking borne le peak memory
    a ~3-5 Go par chunk (ordre de grandeur : chunk_size * rows_par_couple
    * n_features * 4 octets).

    Si `write_rows_callback` est fourni, il est appele APRES CHAQUE CHUNK avec
    la liste des lignes produites → ecriture progressive sur disque. Kill-
    resilient : si le process meurt au chunk N, les N-1 chunks precedents
    sont deja sauves.

    Returns:
        Liste cumulee des lignes de la classe (pour les stats finales).
    """
    print(f"\n{'='*70}")
    print(f"  CLASSE {cls.upper()}")
    print(f"{'='*70}")

    result = find_best_model(cls, forced_model)
    if result is None:
        print(f"  [SKIP] Aucun modele trouve pour {cls}")
        return []
    model_type, booster, meta, features, profiles = result
    predict_type = MODEL_CONFIGS[model_type]["predict_type"]

    df = load_data(cls)
    if df is None or df.empty:
        print(f"  [SKIP] Pas de donnees pour {cls}")
        return []

    # Liste des couples disponibles dans la classe (le filtre n_obs est
    # applique a l'interieur de build_all_observations).
    couples_all = (df.groupby(["ID_PRODUIT", "ID_SO"])
                     .size().reset_index(name="n_obs")
                     [["ID_PRODUIT", "ID_SO"]]
                     .reset_index(drop=True))
    n_total = len(couples_all)
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    print(f"  {n_total:,} couples dans la classe "
          f"(chunk_size={chunk_size} -> {n_chunks} chunks)")

    all_rows = []
    for ci in range(n_chunks):
        chunk_keys = couples_all.iloc[ci * chunk_size:(ci + 1) * chunk_size]
        t_chunk = time.time()
        print(f"\n  [CHUNK {ci+1}/{n_chunks}] filtrage sur {len(chunk_keys):,} couples...")

        # Semi-join vectorise : df limite aux couples du chunk.
        df_chunk = df.merge(chunk_keys, on=["ID_PRODUIT", "ID_SO"], how="inner")

        rows = _process_chunk(df_chunk, booster, features, profiles,
                               predict_type, cls, model_type, use_shap)

        if rows and write_rows_callback is not None:
            write_rows_callback(rows)

        all_rows.extend(rows)
        del df_chunk, rows
        gc.collect()
        print(f"  [CHUNK {ci+1}/{n_chunks}] OK en {time.time()-t_chunk:.1f}s "
              f"(cumul classe: {len(all_rows):,} couples)")

    print(f"\n  [{cls}] {len(all_rows):,} couples exportes au total")

    del booster, df
    if profiles is not None:
        del profiles
    gc.collect()
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Export elasticites par produit")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Forcer un type de modele (defaut: meilleur dispo)")
    parser.add_argument("--no-shap", action="store_true",
                        help="Desactiver SHAP (plus rapide)")
    parser.add_argument("--classes", type=str, nargs="+", default=CLASSES,
                        help="Classes a traiter (defaut: toutes)")
    parser.add_argument("--label", type=str, default=None,
                        help="Label pour le fichier de sortie")
    parser.add_argument("--split-by-nomenclature", action="store_true",
                        help="Genere un .txt par ID_NOMENCLATURE (plus un index)")
    parser.add_argument("--client", type=str,
                        default=os.environ.get("EXPORT_CLIENT", DEFAULT_CLIENT),
                        help=f"Nom du client (prefixe fichier). Defaut: "
                             f"env EXPORT_CLIENT ou {DEFAULT_CLIENT}. "
                             f"Doit matcher EXPORT_CLIENT cote Java.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dossier de sortie (defaut: ~/output_centrale). "
                             "En prod, utiliser /upload/{client}/")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Nombre de couples traites par batch pour borner "
                             "la RAM (defaut: 5000). Reduire sur machine <16 Go, "
                             "augmenter sur machine >= 64 Go pour legerement plus "
                             "de throughput.")
    args = parser.parse_args()

    use_shap = not args.no_shap
    label = args.label or datetime.now().strftime("%Y%m%d")
    client = args.client.upper()  # Java fait client.toUpperCase()
    out_dir = args.output_dir if args.output_dir else OUTPUT_DIR

    print(f"\n{'='*70}")
    print(f"  EXPORT ELASTICITES (mode BATCHED)")
    print(f"  Classes : {args.classes}")
    print(f"  Modele  : {args.model or 'auto (meilleur disponible)'}")
    print(f"  SHAP    : {'oui (par couple)' if use_shap else 'non'}")
    print(f"  Client  : {client}")
    print(f"  Label   : {label}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*70}")

    # Prefixe compatible avec le pipeline Java CopierElasticteFromSftp_SO :
    # filtre entry.getFilename().startsWith(client.toUpperCase()+"_ELASTICITE")
    FILE_PREFIX = f"{client}_ELASTICITE"
    os.makedirs(out_dir, exist_ok=True)

    if args.split_by_nomenclature:
        # Mode split : un .txt par ID_NOMENCLATURE dans un sous-dossier
        # Format identique au mode flat : une ligne {K=V} par couple
        # Compatible direct avec CopierElasticteFromSftp_SO() cote Java
        elast_dir = os.path.join(out_dir, f"{FILE_PREFIX}_{label}")
        os.makedirs(elast_dir, exist_ok=True)
        print(f"  Mode split : un .txt par nomenclature dans {elast_dir}")
        # On truncate les fichiers existants de nomenclatures qu'on va re-ecrire
        # mais on ne peut pas savoir lesquels a l'avance, donc on truncate
        # a la premiere ecriture de chaque nomenclature (via un set).
        elast_path = None
    else:
        elast_path = os.path.join(out_dir, f"{FILE_PREFIX}_{label}.txt")
        # Vider le fichier si il existe deja (nouveau run)
        with open(elast_path, 'w', encoding='utf-8') as f:
            pass  # truncate

    all_elasticites = []
    # Compteur mutable (dict) car les closures Python ne peuvent pas muter
    # un int du scope enclosant. Mis a jour par write_rows a chaque chunk.
    state = {"total_written": 0}
    # Index (prod, so) -> nomenclature
    nomenclature_index = {}
    # Tracker les nomenclatures deja truncatees dans ce run
    nomenclatures_truncated = set()

    def write_rows(rows):
        """
        Mode flat  : append en .txt format {K=V} dans un seul fichier
        Mode split : append en .txt format {K=V}, un fichier par nomenclature
                     Chaque fichier est truncate a la 1ere ecriture de ce run.

        Appele par process_class APRES CHAQUE CHUNK (ecriture progressive,
        kill-resilient).
        """
        if args.split_by_nomenclature:
            # Grouper les lignes par nomenclature
            by_nom = {}
            for row in rows:
                nom = row.get("ID_NOMENCLATURE", "unknown")
                by_nom.setdefault(nom, []).append(row)
                nomenclature_index[(row["ID_PRODUIT"], row["ID_SO"])] = nom

            for nom, nom_rows in by_nom.items():
                fpath = os.path.join(elast_dir, f"{FILE_PREFIX}_{nom}.txt")
                # Truncate a la 1ere ecriture (classe precedente deja ecrite
                # ou pas du tout — on commence propre)
                mode = 'a' if nom in nomenclatures_truncated else 'w'
                with open(fpath, mode, encoding='utf-8') as f:
                    for row in nom_rows:
                        f.write(dict_to_line(row) + "\n")
                nomenclatures_truncated.add(nom)
        else:
            with open(elast_path, 'a', encoding='utf-8') as f:
                for row in rows:
                    f.write(dict_to_line(row) + "\n")
        state["total_written"] += len(rows)

    for cls in args.classes:
        if cls not in CLASSES:
            print(f"  [WARN] Classe inconnue : {cls}")
            continue
        try:
            # process_class ecrit lui-meme chaque chunk via write_rows (callback).
            # Il retourne la liste cumulee de la classe pour le resume final.
            rows = process_class(cls, forced_model=args.model,
                                  use_shap=use_shap,
                                  chunk_size=args.chunk_size,
                                  write_rows_callback=write_rows)
            if rows:
                all_elasticites.extend(rows)
                total_written = state["total_written"]
                if args.split_by_nomenclature:
                    n_nom = len(set(r.get("ID_NOMENCLATURE") for r in rows))
                    print(f"  [SAVE] {cls} : {len(rows):,} lignes ecrites "
                          f"dans {n_nom} fichiers de nomenclature "
                          f"(total: {total_written:,})")
                else:
                    size_mb = os.path.getsize(elast_path) / 1024 / 1024
                    print(f"  [SAVE] {cls} : {len(rows):,} lignes ecrites "
                          f"(total: {total_written:,}, {size_mb:.1f} Mo)")
        except Exception as e:
            import traceback
            print(f"\n[ERREUR] classe {cls}: {e}")
            traceback.print_exc()
            if elast_path:
                print(f"  [INFO] Les classes precedentes sont deja sauvees dans {elast_path}")

    # Ecrire l'index (prod, so) -> nomenclature en mode split
    if args.split_by_nomenclature and nomenclature_index:
        index_path = os.path.join(elast_dir, "INDEX_nomenclature.csv")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("ID_PRODUIT;ID_SO;ID_NOMENCLATURE;FICHIER\n")
            for (prod, so), nom in sorted(nomenclature_index.items()):
                f.write(f"{prod};{so};{nom};CENTRAL_ELASTICITE_{nom}.json\n")
        print(f"\n  [INDEX] {index_path} "
              f"({len(nomenclature_index):,} couples, "
              f"{len(set(nomenclature_index.values()))} nomenclatures)")

    if not all_elasticites:
        print("\n  Aucune elasticite calculee (pas de modeles ou pas de donnees).")
        return

    total_written = state["total_written"]
    print(f"\n{'='*70}")
    if args.split_by_nomenclature:
        n_files = len(set(nomenclature_index.values()))
        dir_size = sum(os.path.getsize(os.path.join(elast_dir, f))
                       for f in os.listdir(elast_dir)) / 1024 / 1024
        print(f"  [EXPORT] {elast_dir}/")
        print(f"  [EXPORT] {total_written:,} couples, {n_files} fichiers, "
              f"{dir_size:.1f} Mo")
    else:
        size_mb = os.path.getsize(elast_path) / 1024 / 1024
        print(f"  [EXPORT] {elast_path}")
        print(f"  [EXPORT] {total_written:,} couples, {size_mb:.1f} Mo")
    print(f"{'='*70}")

    # ==================================================================
    # RESUME & AUDIT QUALITE
    # ==================================================================

    # --- 1. Metriques par classe ---
    print(f"\n  Resume des metriques par classe :")
    print(f"  {'Classe':<14} {'n':>6} {'MAE_moy':>10} {'RMSE_moy':>10} "
          f"{'wMAPE_med':>10} {'wMAPE>100%':>10}")
    for cls in CLASSES:
        cls_rows = [r for r in all_elasticites if r.get("CLASSE_SB") == cls]
        if not cls_rows:
            continue
        maes = [r["MAE"] for r in cls_rows if r.get("MAE") is not None]
        rmses = [r["RMSE"] for r in cls_rows if r.get("RMSE") is not None]
        wmapes = [r["WMAPE"] for r in cls_rows if r.get("WMAPE") is not None]
        n_wmape_bad = sum(1 for w in wmapes if w > 100)
        if maes:
            wmape_str = f"{np.median(wmapes):>9.1f}%" if wmapes else "      n/a"
            pct_bad = f"{n_wmape_bad:>5} ({n_wmape_bad/len(wmapes)*100:.0f}%)" if wmapes else "  n/a"
            print(f"  {cls:<14} {len(maes):>6} {np.mean(maes):>10.4f} "
                  f"{np.mean(rmses):>10.4f} {wmape_str} {pct_bad}")

    # --- 2. Resume des elasticites par feature ---
    print(f"\n  Resume des elasticites par feature :")
    feat_counts = {}
    feat_values = {}
    for row in all_elasticites:
        for k, v in row.items():
            if k.startswith("ELAST_"):
                feat = k.replace("ELAST_", "")
                feat_counts[feat] = feat_counts.get(feat, 0) + 1
                feat_values.setdefault(feat, []).append(v)

    if feat_counts:
        print(f"  {'Feature':<30} {'n_produits':>10} {'median':>10} {'mean':>10}")
        for feat in sorted(feat_counts, key=lambda x: -feat_counts[x]):
            vals = np.array(feat_values[feat])
            print(f"  {feat:<30} {feat_counts[feat]:>10} "
                  f"{np.median(vals):>10.2f} {np.mean(vals):>10.2f}")

    # --- 3. Audit coherence des elasticites ---
    # Regles metier : certaines elasticites ont un signe attendu.
    # Noms normalises (PARAM_* en majuscule pour les features derivees).
    SIGN_RULES = {
        # Prix : hausse prix -> baisse ventes (elasticite negative)
        "PARAM_PRIX":             ("<=", 0, "prix + -> ventes -"),
        "PARAM_LOG_PRIX":         ("<=", 0, "prix + -> ventes -"),
        # Promo : promo -> hausse ventes (impact positif)
        "PARAM_EST_PROMO":        (">=", 0, "promo -> ventes +"),
        "PARAM_PRIX_VENTE_PROMO": (">=", 0, "promo -> ventes +"),
        "PARAM_DISCOUNT_PCT":     (">=", 0, "remise -> ventes +"),
    }

    print(f"\n  Audit coherence des elasticites (regles metier) :")
    print(f"  {'Feature':<30} {'Regle':<25} {'n_total':>8} {'n_incoher':>10} {'%':>6}")
    for feat, (op, threshold, desc) in SIGN_RULES.items():
        if feat not in feat_values:
            continue
        vals = np.array(feat_values[feat])
        n_total = len(vals)
        if op == "<=":
            n_incoherent = int(np.sum(vals > threshold))
        else:  # >=
            n_incoherent = int(np.sum(vals < threshold))
        pct = n_incoherent / n_total * 100 if n_total > 0 else 0
        status = "OK" if pct < 10 else "ATTENTION" if pct < 30 else "PROBLEME"
        print(f"  {feat:<30} {desc:<25} {n_total:>8} {n_incoherent:>10} "
              f"{pct:>5.1f}%  {status}")

    # --- 4. Resume global ---
    total = len(all_elasticites)
    wmapes_all = [r["WMAPE"] for r in all_elasticites if r.get("WMAPE") is not None]
    n_wmape_good = sum(1 for w in wmapes_all if w <= 50)
    n_wmape_ok = sum(1 for w in wmapes_all if 50 < w <= 100)
    n_wmape_bad = sum(1 for w in wmapes_all if w > 100)

    print(f"\n  Qualite globale ({total:,} couples) :")
    print(f"    wMAPE <= 50%  (bon)      : {n_wmape_good:>8,} ({n_wmape_good/total*100:.1f}%)")
    print(f"    wMAPE 50-100% (moyen)    : {n_wmape_ok:>8,} ({n_wmape_ok/total*100:.1f}%)")
    print(f"    wMAPE > 100%  (faible)   : {n_wmape_bad:>8,} ({n_wmape_bad/total*100:.1f}%)")


if __name__ == "__main__":
    main()

