#!/usr/bin/env python3
"""
Entrainement LightGBM Tweedie INCREMENTAL par classe Syntetos-Boylan.

Principe : on n'accumule jamais plus de TRAIN_BATCH lignes en RAM.
  - On lit le CSV par chunks
  - On separe train (avant cutoff) et test (apres cutoff)
  - Le train est decoupe en batches de TRAIN_BATCH lignes
  - Chaque batch entraine le modele via init_model (continuation)
  - Le test est evalue dans une passe separee (pas d'accumulation en RAM)

Etapes :
  0. Charger la classification S-B
  1. Split les ZIP en un CSV par classe
  2. Entrainement incremental par classe (Passe 1: train, Passe 2: eval)
"""
from __future__ import annotations

import os
import sys
import re
import csv
import json
import zipfile
import math
import gc
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from datetime import datetime

# === CONFIGURATION ===
# Chemins auto-detectes selon la plateforme ; override via variables d'environnement
# TESTLIGHTGBM_CENTRALE_DIR et TESTLIGHTGBM_OUTPUT_DIR.
if sys.platform.startswith("win"):
    _DEFAULT_CENTRALE = r"C:\Users\jcric\API_model_multiVar\API_GPU_022026\json_output\CENTRALE"
    _DEFAULT_OUTPUT = r"C:\Users\jcric\TESTLIGHTGBMGLOB\output_centrale"
else:
    _DEFAULT_CENTRALE = "/home/sftpuser/upload/CENTRALE"
    _DEFAULT_OUTPUT = os.path.expanduser("~/output_centrale")
CENTRALE_DIR = os.environ.get("TESTLIGHTGBM_CENTRALE_DIR", _DEFAULT_CENTRALE)
OUTPUT_DIR = os.environ.get("TESTLIGHTGBM_OUTPUT_DIR", _DEFAULT_OUTPUT)
MAX_ZIPS = 50
CHUNK_SIZE = 50_000     # Lignes par chunk lecture CSV (reduit)
TEST_DAYS = 30          # Derniers N jours pour validation
PURGE_DAYS = 1          # Gap de purge entre train et test
TRAIN_BATCH = 100_000   # Lignes par batch (reduit pour eviter OOM)
MAX_TOTAL_TREES = 2000  # Limite totale d'arbres par classe (modele chaine via disque)

# CONTROLE RAM
RAM_WARN_PCT = 50       # Alerte si RAM > 50%
RAM_STOP_PCT = 70       # Arret si RAM > 70% (eviter OOM - marge de securite)
RAM_CHECK_INTERVAL = 1  # Verifier RAM apres chaque batch

# SKIP ETAPE 1 si fichiers existent deja
SKIP_SPLIT_IF_EXISTS = True  # Mettre False pour forcer le re-split

TWEEDIE_POWER = 1.5

# Nombre estime de features apres build_features (~30 colonnes float32)
ESTIMATED_N_FEATURES = 30
BYTES_PER_ROW = ESTIMATED_N_FEATURES * 4  # float32 = 4 bytes

ZIP_PATTERN = re.compile(r"data_CENTRALE_(\d+)_(\d+)\.zip")

TARGET = "QUANTITE"
DATE_COLS = ["PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR", "PARAM_JOUR_SEMAIN"]
ID_COLS = ["ID_PRODUIT", "ID_SO"]
PRICE_COLS = ["PRIX_ACHAT01", "PARAM_PRIX_ACHAT", "PARAM_PRIX", "PRIX_VENTE_PROMO"]
CONTEXT_COLS = ["PARAM_STOCK", "PARAM_VACANCE", "PARAM_TEMPERATURE", "PARAM_PLUIE"]


# =====================================================================
#  MONITORING RAM
# =====================================================================

def get_ram_usage():
    """Retourne (used_gb, total_gb, percent) via /proc/meminfo ou psutil."""
    try:
        # Linux: lecture directe /proc/meminfo (plus fiable)
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    meminfo[key] = int(parts[1])  # en kB

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used_kb = total_kb - available_kb

            total_gb = total_kb / (1024 * 1024)
            used_gb = used_kb / (1024 * 1024)
            pct = (used_kb / total_kb * 100) if total_kb > 0 else 0
            return used_gb, total_gb, pct
    except:
        pass

    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3), mem.percent
    except:
        return 0, 0, 0


def get_process_ram():
    """RAM utilisee par ce processus (en Go)."""
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)  # Go
    except:
        pass

    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)
    except:
        return 0


def print_ram_status(prefix=""):
    """Affiche l'etat de la RAM."""
    used, total, pct = get_ram_usage()
    proc_ram = get_process_ram()

    status = "OK"
    if pct >= RAM_STOP_PCT:
        status = "CRITIQUE"
    elif pct >= RAM_WARN_PCT:
        status = "ATTENTION"

    print(f"    {prefix}RAM: {used:.1f}/{total:.1f} Go ({pct:.0f}%) "
          f"| Process: {proc_ram:.1f} Go | [{status}]")

    return pct


def check_ram_limit(context=""):
    """Verifie si on approche de la limite RAM. Retourne True si OK."""
    used, total, pct = get_ram_usage()

    if pct >= RAM_STOP_PCT:
        print(f"\n  [STOP RAM] {context}")
        print(f"    RAM: {used:.1f}/{total:.1f} Go ({pct:.0f}%)")
        print(f"    Seuil STOP = {RAM_STOP_PCT}%")
        print(f"    Arret pour eviter OOM killer.")
        return False

    if pct >= RAM_WARN_PCT:
        print(f"    [WARN RAM] {pct:.0f}% utilise ({context})")

    return True


def force_gc(verbose=False):
    """Force le garbage collector."""
    gc.collect()
    gc.collect()  # 2x pour les cycles
    if verbose:
        proc_ram = get_process_ram()
        print(f"    [GC] RAM process: {proc_ram:.1f} Go")


def compute_adaptive_batch_size():
    """Calcule un batch size adapte a la RAM disponible."""
    used, total, pct = get_ram_usage()
    available_gb = total * (1 - pct / 100)

    # Utiliser max 20% de la RAM disponible pour un batch
    # Un batch occupe ~8x bytes_per_row en pic (raw data + Dataset + histograms + overhead)
    usable_bytes = available_gb * 0.20 * (1024 ** 3)
    max_rows = int(usable_bytes / (BYTES_PER_ROW * 8))

    adaptive = max(50_000, min(TRAIN_BATCH, max_rows))
    print(f"    [ADAPTIVE] RAM dispo: {available_gb:.1f} Go -> batch_size={adaptive:,} "
          f"(config max={TRAIN_BATCH:,})")
    return adaptive


def debug_memory_usage(context=""):
    """Affiche un rapport detaille de l'utilisation memoire (une seule passe gc)."""
    import sys

    proc_ram = get_process_ram()
    used, total, pct = get_ram_usage()

    print(f"\n    === DEBUG RAM: {context} ===")
    print(f"    Systeme : {used:.1f}/{total:.1f} Go ({pct:.0f}%)")
    print(f"    Process : {proc_ram:.1f} Go")

    # Une seule iteration sur gc.get_objects()
    type_counts = {}
    type_sizes = {}
    df_count = 0
    df_total_mem = 0
    np_count = 0
    np_total_mem = 0
    lgb_count = 0

    for obj in gc.get_objects():
        obj_type = type(obj).__name__

        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        try:
            type_sizes[obj_type] = type_sizes.get(obj_type, 0) + sys.getsizeof(obj)
        except:
            pass

        if isinstance(obj, pd.DataFrame):
            df_count += 1
            try:
                df_total_mem += obj.memory_usage(deep=True).sum()
            except:
                pass
        elif isinstance(obj, np.ndarray):
            np_count += 1
            try:
                np_total_mem += obj.nbytes
            except:
                pass
        elif obj_type == 'Booster':
            lgb_count += 1

    sorted_types = sorted(type_sizes.items(), key=lambda x: -x[1])[:10]
    print(f"    Top objets Python (par taille):")
    for t, s in sorted_types:
        count = type_counts.get(t, 0)
        print(f"      {t:<25} : {count:>8,} objets | {s / (1024*1024):>8.1f} Mo")

    if df_count > 0:
        print(f"    DataFrames en memoire: {df_count} | {df_total_mem / (1024**3):.2f} Go")
    if np_count > 0:
        print(f"    Arrays numpy: {np_count} | {np_total_mem / (1024**3):.2f} Go")
    if lgb_count > 0:
        print(f"    LightGBM Boosters: {lgb_count}")

    python_tracked = sum(type_sizes.values()) / (1024**3)
    untracked = proc_ram - python_tracked
    print(f"    Memoire Python trackee: {python_tracked:.2f} Go")
    print(f"    Memoire non-trackee (C/LightGBM): {untracked:.2f} Go")
    print(f"    ===========================\n")

    return proc_ram


# =====================================================================
#  UTILS
# =====================================================================

def list_zip_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".zip"))

def parse_zip_ids(filename):
    match = ZIP_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def parse_kv_line(line):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        line = line[1:-1]
    else:
        return None
    result = {}
    for pair in re.split(r",\s+", line):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
        result[key] = value
    return result if result else None

def iter_zip_rows(filepath):
    """Yield rows. Prefere CENTRALE_HIST_*.txt (avec QUANTITE) si plusieurs fichiers."""
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            txt_files = [f for f in zf.namelist()
                         if f.endswith(".txt") and not f.startswith("__")]
            if not txt_files:
                return
            hist_files = [f for f in txt_files if "HIST" in f.upper()]
            target = hist_files[0] if hist_files else txt_files[0]
            with zf.open(target) as f:
                for raw in f:
                    parsed = parse_kv_line(raw.decode("utf-8", errors="replace"))
                    if parsed:
                        yield parsed
    except Exception as e:
        print(f"    [ERREUR] {e}")


# =====================================================================
#  ETAPE 0 : Classification S-B
# =====================================================================

def load_classification(path):
    """Charge la classification S-B par couple (ID_PRODUIT, ID_SO).

    Retourne dict[(prod_int, so_int) -> classe] (smooth/intermittent/erratic/lumpy).
    Le CSV doit contenir les colonnes ID_PRODUIT, ID_SO, classification.
    """
    print(f"\n{'='*70}")
    print(f"  ETAPE 0 : CHARGEMENT CLASSIFICATION S-B")
    print(f"{'='*70}\n")
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    if "ID_SO" not in df.columns:
        print(f"  [ERREUR] Colonne ID_SO absente. Le pipeline_centrale.py "
              f"doit etre execute en version par (ID_PRODUIT, ID_SO).")
        sys.exit(1)
    df["ID_PRODUIT"] = df["ID_PRODUIT"].astype(int)
    df["ID_SO"] = df["ID_SO"].astype(int)
    mapping = {(row.ID_PRODUIT, row.ID_SO): row.classification
               for row in df.itertuples(index=False)}
    counts = df["classification"].value_counts()
    for cls, n in counts.items():
        print(f"    {cls:<15} : {n:>6} couples (prod, magasin)")
    print(f"    {'TOTAL':<15} : {len(mapping):>6}")
    print(f"    Produits distincts : {df['ID_PRODUIT'].nunique():,}")
    print(f"    Magasins distincts : {df['ID_SO'].nunique():,}")
    return mapping


# =====================================================================
#  ETAPE 1 : Split par classe S-B
# =====================================================================

def split_by_class(centrale_dir, zip_files, product_class, output_dir):
    print(f"\n{'='*70}")
    print(f"  ETAPE 1 : SPLIT PAR CLASSE S-B + DETECTION DATES")
    print(f"{'='*70}\n")

    classes = ["smooth", "intermittent", "erratic", "lumpy"]
    keep_cols = (ID_COLS + DATE_COLS + [TARGET] + PRICE_COLS + CONTEXT_COLS +
                 ["ID_NOMENCLATURE"])

    writers = {}
    files = {}
    counts = defaultdict(int)
    header_written = False
    final_cols = None

    # Tracking des dates par classe (pour eviter Passe 1)
    date_ranges = {c: {"min": None, "max": None} for c in classes}

    for cls in classes:
        path = os.path.join(output_dir, f"data_{cls}.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        files[cls] = f
        writers[cls] = None

    for i, zf_name in enumerate(zip_files, 1):
        filepath = os.path.join(centrale_dir, zf_name)
        id_nom, id_fich = parse_zip_ids(zf_name)
        n = 0
        for row in iter_zip_rows(filepath):
            prod = row.get("ID_PRODUIT")
            so = row.get("ID_SO")
            q = row.get("QUANTITE")
            if prod is None or so is None or q is None:
                continue
            try:
                prod_int = int(prod)
                so_int = int(so)
            except (ValueError, TypeError):
                continue
            cls = product_class.get((prod_int, so_int))
            if cls is None:
                continue
            row["ID_NOMENCLATURE"] = id_nom
            if not header_written:
                # Afficher TOUTES les cles trouvees dans le 1er row source
                all_keys = sorted(row.keys())
                print(f"\n    [DIAGNOSTIC] {len(all_keys)} colonnes trouvees dans le ZIP source :")
                for k in all_keys:
                    print(f"      - {k}")

                # Detection dynamique des features a inclure
                js_found = sorted([k for k in row.keys() if k.startswith("PARAM_JS_")])
                promo_found = sorted([k for k in row.keys() if k.startswith("PARAM_PROMO_")])
                final_cols = keep_cols + js_found + promo_found

                print(f"\n    [SPLIT] Colonnes conservees dans data_*.csv ({len(final_cols)}) :")
                print(f"      IDs + dates + target   : {len(ID_COLS + DATE_COLS) + 1} colonnes")
                print(f"      Prix                   : {len(PRICE_COLS)} colonnes -> {PRICE_COLS}")
                print(f"      Contexte               : {len(CONTEXT_COLS)} colonnes -> {CONTEXT_COLS}")
                print(f"      PARAM_JS_* (detect.)   : {len(js_found)} colonnes -> {js_found}")
                print(f"      PARAM_PROMO_* (detect.): {len(promo_found)} colonnes -> {promo_found}")
                print(f"      ID_NOMENCLATURE        : 1 colonne")

                if not promo_found:
                    print(f"\n    [WARN] Aucune colonne PARAM_PROMO_* detectee dans le ZIP source.")
                    print(f"           Les elasticites par type de promo ne seront pas disponibles.")

                for c in classes:
                    writers[c] = csv.DictWriter(
                        files[c], fieldnames=final_cols,
                        delimiter=";", extrasaction="ignore", restval="")
                    writers[c].writeheader()
                header_written = True

            # Tracker les dates pendant le split
            try:
                y = int(row.get("PARAM_ANNEE", 0))
                m = int(row.get("PARAM_MOIS", 0))
                d = int(row.get("PARAM_JOUR", 0))
                if y > 2000 and 1 <= m <= 12 and 1 <= d <= 31:
                    date_str = f"{y:04d}-{m:02d}-{d:02d}"
                    if date_ranges[cls]["min"] is None or date_str < date_ranges[cls]["min"]:
                        date_ranges[cls]["min"] = date_str
                    if date_ranges[cls]["max"] is None or date_str > date_ranges[cls]["max"]:
                        date_ranges[cls]["max"] = date_str
            except (ValueError, TypeError):
                pass

            writers[cls].writerow(row)
            counts[cls] += 1
            n += 1
        for c in classes:
            files[c].flush()
        print(f"    [{i:3d}/{len(zip_files)}] {zf_name} : {n:>10,} lignes")

    for c in classes:
        files[c].close()

    # Sauvegarder les metadonnees de dates
    date_meta_path = os.path.join(output_dir, "date_ranges.json")
    with open(date_meta_path, "w", encoding="utf-8") as f:
        json.dump(date_ranges, f, indent=2)
    print(f"\n  [SAVE] Date ranges -> {date_meta_path}")

    print(f"\n  Resume du split :")
    for c in classes:
        path = os.path.join(output_dir, f"data_{c}.csv")
        size = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        dr = date_ranges[c]
        date_info = f"  [{dr['min']} -> {dr['max']}]" if dr['min'] else ""
        print(f"    {c:<15} : {counts[c]:>10,} lignes  ({size:.1f} Mo){date_info}")

    return final_cols, counts, date_ranges


# =====================================================================
#  ETAPE 2 : Features + Entrainement incremental
# =====================================================================

def get_optimized_dtypes():
    """Retourne les dtypes optimises pour la lecture CSV (float32 au lieu de float64)."""
    return {
        "ID_PRODUIT": "int32",
        "ID_SO": "int32",
        "ID_NOMENCLATURE": "int32",
        "PARAM_ANNEE": "int16",
        "PARAM_MOIS": "int8",
        "PARAM_JOUR": "int8",
        "PARAM_JOUR_SEMAIN": "int8",
        "QUANTITE": "float32",
        "PRIX_ACHAT01": "float32",
        "PARAM_PRIX_ACHAT": "float32",
        "PARAM_PRIX": "float32",
        "PRIX_VENTE_PROMO": "float32",
        "PARAM_STOCK": "float32",
        "PARAM_VACANCE": "int8",
        "PARAM_TEMPERATURE": "float32",
        "PARAM_PLUIE": "float32",
    }


def optimize_dataframe(df):
    """Optimise les types du DataFrame pour reduire la RAM."""
    # Colonnes numeriques -> float32/int32
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            # Verifier si on peut reduire
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            else:
                df[col] = df[col].astype("int32")
    return df


def build_features(df):
    if all(c in df.columns for c in ["PARAM_ANNEE", "PARAM_MOIS", "PARAM_JOUR"]):
        df["date"] = pd.to_datetime(
            pd.DataFrame({"year": df["PARAM_ANNEE"].astype(int),
                           "month": df["PARAM_MOIS"].astype(int),
                           "day": df["PARAM_JOUR"].astype(int)}),
            errors="coerce")
    else:
        df["date"] = pd.NaT

    df["jour_annee"] = df["date"].dt.dayofyear
    df["semaine"] = df["date"].dt.isocalendar().week.astype(int)
    df["mois"] = df["date"].dt.month
    df["sin_jour"] = np.sin(2 * np.pi * df["jour_annee"] / 365.25)
    df["cos_jour"] = np.cos(2 * np.pi * df["jour_annee"] / 365.25)
    df["sin_semaine"] = np.sin(2 * np.pi * df["semaine"] / 52)
    df["cos_semaine"] = np.cos(2 * np.pi * df["semaine"] / 52)
    df["sin_mois"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["cos_mois"] = np.cos(2 * np.pi * df["mois"] / 12)

    if "PARAM_PRIX" in df.columns and "PRIX_VENTE_PROMO" in df.columns:
        prix = df["PARAM_PRIX"].fillna(0)
        promo = df["PRIX_VENTE_PROMO"].fillna(0)
        df["ratio_promo"] = np.where(prix > 0, promo / prix, 0)
        df["est_promo"] = (promo > 0).astype(int)
        df["discount_pct"] = np.where(
            (prix > 0) & (promo > 0), (prix - promo) / prix, 0)
    else:
        df["ratio_promo"] = 0
        df["est_promo"] = 0
        df["discount_pct"] = 0

    if "PARAM_PRIX" in df.columns and "PARAM_PRIX_ACHAT" in df.columns:
        prix = df["PARAM_PRIX"].fillna(0)
        pa = df["PARAM_PRIX_ACHAT"].fillna(0)
        df["markup_pct"] = np.where(prix > 0, (prix - pa) / prix, 0)
    else:
        df["markup_pct"] = 0

    if "PARAM_PRIX" in df.columns:
        df["log_prix"] = np.log1p(df["PARAM_PRIX"].fillna(0).clip(lower=0))

    for col in ["ID_PRODUIT", "ID_SO", "ID_NOMENCLATURE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def get_feature_cols(df):
    exclude = {TARGET, "date", "__source_zip__", "jour_annee"}
    return [c for c in df.columns if c not in exclude]


# =====================================================================
#  GENERATEUR DE BATCHES pour entrainement incremental
# =====================================================================

class TrainDataGenerator:
    """
    Generateur de batches pour entrainement incremental LightGBM.

    Charge les donnees batch par batch depuis le CSV.
    Utilise yield pour ne jamais accumuler en RAM.
    """

    def __init__(self, data_path, cutoff_train, batch_size=TRAIN_BATCH, chunk_size=CHUNK_SIZE):
        self.data_path = data_path
        self.cutoff_train = cutoff_train
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dtypes = get_optimized_dtypes()

        # Decouvrir features depuis le 1er chunk seulement (pas de scan complet)
        print(f"    [Generator] Decouverte features (1er chunk)...")
        self.feature_cols = None
        self.cat_cols = None

        for chunk in pd.read_csv(data_path, sep=";", chunksize=chunk_size,
                                  encoding="utf-8", dtype=self.dtypes):
            chunk = build_features(chunk)
            chunk = chunk.dropna(subset=[TARGET, "date"])
            chunk = chunk[chunk[TARGET] >= 0]

            mask_train = chunk["date"] <= cutoff_train
            if mask_train.any():
                train_sample = chunk[mask_train].head(1)
                self.feature_cols = get_feature_cols(train_sample)
                # NOTE: On ne declare PAS ID_PRODUIT/ID_SO/ID_NOMENCLATURE comme
                # categorical_feature car ils ont trop de valeurs uniques (1000+).
                # LightGBM cree une bin par valeur unique -> explosion memoire.
                # En mode numerique (int32), LightGBM les bin automatiquement (max_bin=63).
                self.cat_cols = []
                del chunk, train_sample
                gc.collect()
                break

            del chunk
            gc.collect()

        # Estimer nb lignes via taille fichier (rapide, pas de lecture)
        file_size = os.path.getsize(data_path)
        estimated_rows = max(1000, file_size // 120)  # ~120 bytes/ligne en CSV
        self.n_train_rows = estimated_rows  # Estimation, sera affine dans iter_batches
        self.n_batches = max(1, (estimated_rows + batch_size - 1) // batch_size)
        print(f"    [Generator] ~{estimated_rows:,} lignes estimees -> ~{self.n_batches} batches")

    def iter_batches(self):
        """Genere les batches (X, y) un par un."""
        buffer_X = []
        buffer_y = []
        buffer_size = 0
        batch_num = 0

        for chunk in pd.read_csv(self.data_path, sep=";", chunksize=self.chunk_size,
                                  encoding="utf-8", dtype=self.dtypes):
            chunk = build_features(chunk)
            chunk = optimize_dataframe(chunk)
            chunk = chunk.dropna(subset=[TARGET, "date"])
            chunk = chunk[chunk[TARGET] >= 0]

            mask_train = chunk["date"] <= self.cutoff_train
            train_chunk = chunk[mask_train]

            if len(train_chunk) > 0:
                X_chunk = train_chunk[self.feature_cols].values.astype(np.float32)
                y_chunk = train_chunk[TARGET].values.astype(np.float32)

                buffer_X.append(X_chunk)
                buffer_y.append(y_chunk)
                buffer_size += len(y_chunk)

            del chunk, train_chunk
            gc.collect()

            # Yield batch quand buffer plein
            while buffer_size >= self.batch_size:
                # Concatener et couper
                X_all = np.vstack(buffer_X)
                y_all = np.concatenate(buffer_y)

                X_batch = X_all[:self.batch_size]
                y_batch = y_all[:self.batch_size]

                # Garder le reste
                if len(X_all) > self.batch_size:
                    buffer_X = [X_all[self.batch_size:]]
                    buffer_y = [y_all[self.batch_size:]]
                    buffer_size = len(buffer_X[0])
                else:
                    buffer_X = []
                    buffer_y = []
                    buffer_size = 0

                batch_num += 1
                yield batch_num, X_batch, y_batch

                del X_all, y_all, X_batch, y_batch
                gc.collect()

        # Dernier batch (reste du buffer)
        if buffer_size > 0:
            X_batch = np.vstack(buffer_X)
            y_batch = np.concatenate(buffer_y)
            batch_num += 1
            yield batch_num, X_batch, y_batch
            del buffer_X, buffer_y
            gc.collect()


def train_class(cls, data_path, output_dir, date_range, n_total, chunk_size=CHUNK_SIZE):
    """
    Entrainement LightGBM Tweedie avec GENERATEUR DE BATCHES.

    Utilise un generateur pour streamer les donnees:
    - Ne charge qu'un batch a la fois en RAM
    - Utilise init_model pour continuer l'entrainement
    - Liberation memoire apres chaque batch
    """
    print(f"\n  {'-'*50}")
    print(f"  ENTRAINEMENT AVEC GENERATEUR : {cls.upper()}")
    print(f"  Tweedie p={TWEEDIE_POWER} | batch={TRAIN_BATCH:,} | max {MAX_TOTAL_TREES} arbres")
    print(f"  Mode: Streaming par batch")
    print(f"  {'-'*50}")

    t0 = time.time()
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Fichier : {data_path} ({file_size:.1f} Mo)")

    if file_size < 0.01:
        print(f"  [SKIP] Fichier vide.")
        return None

    # --- Dates pre-calculees ---
    if date_range["min"] is None or date_range["max"] is None:
        print(f"  [SKIP] Pas de dates.")
        return None

    date_min = pd.to_datetime(date_range["min"])
    date_max = pd.to_datetime(date_range["max"])
    cutoff_train = date_max - pd.Timedelta(days=TEST_DAYS + PURGE_DAYS)
    cutoff_test = date_max - pd.Timedelta(days=TEST_DAYS)

    print(f"  Plage  : {date_min.date()} -> {date_max.date()}")
    print(f"  Train  : ... -> {cutoff_train.date()}")
    print(f"  Purge  : {PURGE_DAYS}j")
    print(f"  Test   : {cutoff_test.date()} -> {date_max.date()}")
    print(f"  Lignes : {n_total:,}")

    # Verification RAM initiale
    print(f"\n  Verification RAM initiale:")
    print_ram_status("  ")
    if not check_ram_limit("Avant entrainement"):
        return None

    # --- Batch adaptatif ---
    adaptive_batch = compute_adaptive_batch_size()

    # --- Creer le generateur ---
    print(f"\n  Creation du generateur de batches...")
    try:
        generator = TrainDataGenerator(
            data_path=data_path,
            cutoff_train=cutoff_train,
            batch_size=adaptive_batch,
            chunk_size=chunk_size
        )
    except Exception as e:
        print(f"  [ERREUR] Impossible de creer le generateur: {e}")
        return None

    if generator.n_train_rows == 0:
        print(f"  [SKIP] Aucune donnee d'entrainement.")
        return None

    feature_cols = generator.feature_cols
    cat_cols = generator.cat_cols
    cat_indices = [feature_cols.index(c) for c in cat_cols] if cat_cols else []
    n_train_total = generator.n_train_rows
    n_batches = generator.n_batches

    # Calculer rounds par batch :
    # - Floor=2  : au moins 2 arbres par batch (sinon apprentissage trop faible)
    # - Cap=100  : borne superieure pour eviter qu'un batch avec peu d'autres
    #              batches n'en fasse trop. La regularisation LightGBM
    #              (subsample=0.8, reg_alpha=0.1, reg_lambda=1.0) limite l'overfit.
    # Objectif : couvrir 100% des batches en utilisant au mieux MAX_TOTAL_TREES.
    rounds_per_batch = max(2, min(100, MAX_TOTAL_TREES // n_batches))

    print(f"  Features      : {len(feature_cols)}")
    print(f"  Categoriques  : {cat_cols}")
    print(f"  Rounds/batch  : {rounds_per_batch}")

    # --- Parametres LightGBM ---
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": TWEEDIE_POWER,
        "metric": "tweedie",
        "verbosity": -1,
        "n_jobs": 4,             # Limiter threads (chaque thread alloue ses histogrammes)
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        # Optimisations memoire
        "max_bin": 63,
        "min_data_in_bin": 10,
        "feature_pre_filter": True,
        "is_enable_sparse": True,
        "force_row_wise": True,
        "histogram_pool_size": 512,  # Limiter a 512 Mo (au lieu de illimite)
    }

    if n_total < 500_000:
        params["num_leaves"] = 31
        params["min_child_samples"] = 30

    # --- Entrainement batch par batch ---
    print(f"\n  Entrainement par batches...")
    print_ram_status("  Avant train: ")

    model = None
    rows_trained = 0
    total_trees = 0  # Compteur d'arbres (car on libere le modele entre batches)
    # Fichier temporaire pour forcer la liberation memoire entre batches
    temp_model_path = os.path.join(output_dir, f"_temp_model_{cls}.txt")

    for batch_num, X_batch, y_batch in generator.iter_batches():
        batch_size = len(y_batch)
        rows_trained += batch_size

        # Verifier RAM
        if not check_ram_limit(f"Batch {batch_num}"):
            print(f"  [ARRET] Limite RAM atteinte")
            del X_batch, y_batch
            break

        # Verifier limite arbres
        if total_trees >= MAX_TOTAL_TREES:
            print(f"  [INFO] Limite {MAX_TOTAL_TREES} arbres atteinte")
            del X_batch, y_batch
            break

        # Calculer rounds restants
        remaining = MAX_TOTAL_TREES - total_trees
        rounds_this_batch = min(rounds_per_batch, remaining)

        if rounds_this_batch <= 0:
            del X_batch, y_batch
            break

        # Charger modele precedent depuis disque (si existe)
        init_model = None
        if os.path.exists(temp_model_path):
            init_model = temp_model_path  # LightGBM accepte un chemin fichier

        # Liberer le modele en memoire AVANT de creer le Dataset
        if model is not None:
            del model
            model = None

        # GC avant creation Dataset (pic memoire)
        gc.collect()

        # Creer Dataset
        ds = lgb.Dataset(
            X_batch, label=y_batch,
            feature_name=feature_cols,
            categorical_feature=cat_indices,
            free_raw_data=True
        )

        # Liberer les arrays sources AVANT l'entrainement
        del X_batch, y_batch
        gc.collect()

        # Entrainer (init_model est un chemin fichier, pas un objet en memoire)
        model = lgb.train(
            params, ds,
            num_boost_round=rounds_this_batch,
            init_model=init_model
        )

        total_trees = model.num_trees()

        # Sauvegarder sur disque AVANT de liberer
        model.save_model(temp_model_path)

        print(f"    Batch {batch_num:>3}/{n_batches} : {batch_size:>10,} lignes | "
              f"cumul={rows_trained:>12,} | trees={total_trees}/{MAX_TOTAL_TREES}")

        # Liberer dataset et modele pour le prochain batch
        if hasattr(ds, 'data'):
            ds.data = None
        del ds
        del model
        model = None

        gc.collect()

        print_ram_status("      ")

    # Recharger le modele final depuis le disque
    if os.path.exists(temp_model_path):
        model = lgb.Booster(model_file=temp_model_path)
        os.remove(temp_model_path)
    else:
        model = None

    force_gc(verbose=True)

    if model is None:
        print(f"  [SKIP] Aucune donnee d'entrainement.")
        return None

    print(f"\n  Entrainement termine :")
    print(f"    Batches    : {n_batches}")
    print(f"    Lignes     : {rows_trained:,}")
    print(f"    Trees      : {model.num_trees()}")

    # --- Passe 2 : Evaluation sur test (sans accumulation) ---
    print(f"\n  Passe 2 : evaluation sur test...")

    metrics = {
        "classe": cls,
        "tweedie_power": TWEEDIE_POWER,
        "n_train": rows_trained,
        "n_batches": n_batches,
        "train_batch_size": TRAIN_BATCH,
        "mode": "generator_streaming",
        "max_total_trees": MAX_TOTAL_TREES,
        "total_trees": model.num_trees(),
        "n_test": 0,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "params": params,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "cutoff_train": str(cutoff_train.date()),
        "cutoff_test": str(cutoff_test.date()),
        "purge_days": PURGE_DAYS,
        "test_days": TEST_DAYS,
    }

    # Metriques incrementales
    n_test = 0
    sum_abs_error = 0.0
    sum_sq_error = 0.0
    sum_actual = 0.0
    sum_tweedie_dev = 0.0
    sample_actuals = []  # Echantillon pour median (1 sur 100)

    print_ram_status("  Avant eval: ")
    n_eval_chunks = 0
    dtypes = get_optimized_dtypes()

    for chunk in pd.read_csv(data_path, sep=";", chunksize=chunk_size,
                              encoding="utf-8", dtype=dtypes):
        n_eval_chunks += 1
        chunk = chunk.reset_index(drop=True)
        chunk = build_features(chunk)
        chunk = optimize_dataframe(chunk)
        chunk = chunk.dropna(subset=[TARGET, "date"])
        chunk = chunk[chunk[TARGET] >= 0]

        if chunk.empty:
            del chunk
            continue

        # Filtrer test seulement
        mask_test = chunk["date"] > cutoff_test
        if not mask_test.any():
            del chunk
            force_gc()
            continue

        df_test_chunk = chunk[mask_test].copy()
        del chunk
        force_gc()

        if len(df_test_chunk) == 0 or feature_cols is None:
            del df_test_chunk
            continue

        X_test = df_test_chunk[feature_cols].values.astype(np.float32)
        actuals = df_test_chunk[TARGET].values.astype(np.float32)
        preds = model.predict(X_test)
        preds_clipped = np.clip(preds, 1e-8, None)

        del X_test, df_test_chunk

        # Accumuler metriques
        n_test += len(actuals)
        sum_abs_error += float(np.sum(np.abs(actuals - preds)))
        sum_sq_error += float(np.sum((actuals - preds) ** 2))
        sum_actual += float(np.sum(actuals))

        # Tweedie deviance incrementale (power=1.5)
        # d = 2 * (y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
        p = TWEEDIE_POWER
        y = actuals
        mu = preds_clipped
        dev = 2 * (
            np.power(y, 2-p) / ((1-p) * (2-p))
            - y * np.power(mu, 1-p) / (1-p)
            + np.power(mu, 2-p) / (2-p)
        )
        sum_tweedie_dev += float(np.sum(dev))

        # Echantillon pour median (1 sur 100)
        sample_actuals.extend(actuals[::100].tolist())

        del actuals, preds, preds_clipped, dev
        force_gc()

        # Verifier RAM tous les 20 chunks
        if n_eval_chunks % 20 == 0:
            if not check_ram_limit(f"Eval chunk {n_eval_chunks}"):
                print(f"  [WARN] Evaluation interrompue pour limiter RAM")
                break

    print(f"    Evaluation: {n_eval_chunks} chunks traites")

    if n_test > 10:
        mae = sum_abs_error / n_test
        rmse = math.sqrt(sum_sq_error / n_test)
        mean_q = sum_actual / n_test
        tw_dev = sum_tweedie_dev / n_test
        median_q = float(np.median(sample_actuals)) if sample_actuals else mean_q

        metrics["n_test"] = n_test
        metrics["mae"] = round(mae, 4)
        metrics["rmse"] = round(rmse, 4)
        metrics["mean_quantite"] = round(mean_q, 4)
        metrics["median_quantite"] = round(median_q, 4)
        metrics["tweedie_deviance"] = round(tw_dev, 6)
        if mean_q > 0:
            metrics["mape_pct"] = round(mae / mean_q * 100, 1)

        print(f"\n  Resultats test ({n_test:,} lignes) :")
        print(f"    MAE        : {mae:.4f}")
        print(f"    RMSE       : {rmse:.4f}")
        print(f"    Tweedie    : {tw_dev:.6f}")
        print(f"    Moy reelle : {mean_q:.4f}")
        print(f"    Med reelle : {median_q:.4f} (echantillon)")
        if mean_q > 0:
            print(f"    MAPE       : {mae/mean_q*100:.1f}%")
    else:
        print(f"  [SKIP] Pas assez de donnees test ({n_test} lignes)")

    del sample_actuals
    gc.collect()

    # --- Feature importance ---
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    metrics["feature_importance_top20"] = [
        {"feature": f, "gain": round(float(g), 2)} for f, g in feat_imp[:20]
    ]

    print(f"\n  Top 15 features (gain) :")
    for feat, imp in feat_imp[:15]:
        print(f"    {feat:<30} {imp:>12.1f}")

    # --- Sauvegarder ---
    model_path = os.path.join(output_dir, f"model_{cls}.txt")
    model.save_model(model_path)

    meta_path = os.path.join(output_dir, f"model_{cls}_metadata.json")
    metrics["training_time_sec"] = round(time.time() - t0, 1)
    metrics["timestamp"] = datetime.now().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  [SAVE] Modele   -> {model_path}")
    print(f"  [SAVE] Metadata -> {meta_path}")
    print(f"  Temps           : {metrics['training_time_sec']:.1f}s")
    print_ram_status("  Final: ")

    # Nettoyage final - liberer le modele AVANT de retourner
    del model
    force_gc(verbose=True)

    return True  # Succes (modele sauvegarde sur disque)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === VERIFICATION RAM INITIALE ===
    print(f"\n{'='*70}")
    print(f"  VERIFICATION SYSTEME")
    print(f"{'='*70}")
    used, total, pct = get_ram_usage()
    print(f"  RAM systeme : {used:.1f} / {total:.1f} Go ({pct:.0f}%)")
    print(f"  Seuils      : WARN={RAM_WARN_PCT}% | STOP={RAM_STOP_PCT}%")
    print(f"  Batch size  : {TRAIN_BATCH:,} lignes")
    print(f"  Chunk size  : {CHUNK_SIZE:,} lignes")

    if pct >= RAM_STOP_PCT:
        print(f"\n  [ERREUR] RAM deja a {pct:.0f}% - trop elevee pour demarrer")
        print(f"  Liberez de la memoire avant de relancer.")
        sys.exit(1)

    if pct >= RAM_WARN_PCT:
        print(f"\n  [ATTENTION] RAM deja a {pct:.0f}% - risque d'OOM")

    class_path = os.path.join(OUTPUT_DIR, "stats_classification_SB.csv")
    if not os.path.exists(class_path):
        print(f"[ERREUR] Fichier manquant : {class_path}")
        print(f"  Lancez d'abord pipeline_centrale.py")
        sys.exit(1)

    # === ETAPE 0 ===
    product_class = load_classification(class_path)

    # === ETAPE 1 ===
    classes = ["smooth", "intermittent", "erratic", "lumpy"]
    date_meta_path = os.path.join(OUTPUT_DIR, "date_ranges.json")

    # Verifier si on peut sauter l'etape 1
    can_skip = SKIP_SPLIT_IF_EXISTS
    if can_skip:
        for c in classes:
            csv_path = os.path.join(OUTPUT_DIR, f"data_{c}.csv")
            if not os.path.exists(csv_path):
                can_skip = False
                break
        if not os.path.exists(date_meta_path):
            can_skip = False

    if can_skip:
        print(f"\n{'='*70}")
        print(f"  ETAPE 1 : SKIP (fichiers CSV existants)")
        print(f"{'='*70}\n")

        # Charger les date_ranges depuis le fichier JSON
        with open(date_meta_path, "r", encoding="utf-8") as f:
            date_ranges = json.load(f)

        # Compter les lignes des fichiers existants (rapide avec wc -l style)
        counts = {}
        for c in classes:
            csv_path = os.path.join(OUTPUT_DIR, f"data_{c}.csv")
            size = os.path.getsize(csv_path) / (1024 * 1024)
            # Compter lignes rapidement
            with open(csv_path, "rb") as f:
                line_count = sum(1 for _ in f) - 1  # -1 pour header
                line_count = max(0, line_count)
            counts[c] = line_count
            dr = date_ranges.get(c, {})
            date_info = f"  [{dr.get('min')} -> {dr.get('max')}]" if dr.get('min') else ""
            print(f"    {c:<15} : {counts[c]:>10,} lignes  ({size:.1f} Mo){date_info}")

        print(f"\n  Pour forcer le re-split, mettez SKIP_SPLIT_IF_EXISTS = False")
        final_cols = None  # Sera detecte automatiquement
    else:
        zip_files = list_zip_files(CENTRALE_DIR)[:MAX_ZIPS]
        final_cols, counts, date_ranges = split_by_class(
            CENTRALE_DIR, zip_files, product_class, OUTPUT_DIR)

    gc.collect()

    # === ETAPE 2 ===
    print(f"\n{'='*70}")
    print(f"  ETAPE 2 : ENTRAINEMENT INCREMENTAL PAR CLASSE")
    print(f"{'='*70}")

    trained_classes = []  # Liste des classes entrainees (modeles sauvegardes sur disque)

    for cls in classes:
        data_path = os.path.join(OUTPUT_DIR, f"data_{cls}.csv")
        if not os.path.exists(data_path):
            print(f"\n  [SKIP] {cls} : fichier inexistant")
            continue
        if counts.get(cls, 0) < 100:
            print(f"\n  [SKIP] {cls} : trop peu de donnees ({counts.get(cls, 0)})")
            continue

        m = train_class(cls, data_path, OUTPUT_DIR, date_ranges[cls], counts[cls])
        if m:
            trained_classes.append(cls)
            # IMPORTANT: Liberer le modele de la RAM (deja sauvegarde sur disque)
            del m
            print(f"    [RAM] Modele {cls} libere de la memoire (sauvegarde sur disque)")

        # Nettoyage agressif entre les classes
        force_gc(verbose=True)
        print_ram_status(f"  Apres {cls}: ")

        # Verifier si on peut continuer
        if not check_ram_limit(f"Apres entrainement {cls}"):
            print(f"  [ARRET] Limite RAM atteinte, arret des entrainements restants")
            break

    # === Resume ===
    print(f"\n{'='*70}")
    print(f"  RESUME FINAL")
    print(f"{'='*70}")

    # RAM finale
    print_ram_status("  RAM finale: ")
    print()

    for cls in classes:
        meta_path = os.path.join(OUTPUT_DIR, f"model_{cls}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            mae = meta.get("mae", "?")
            n_train = meta.get("n_train", "?")
            n_test = meta.get("n_test", "?")
            n_b = meta.get("n_batches", "?")
            t = meta.get("training_time_sec", "?")
            print(f"  {cls:<15} : train={n_train}  batches={n_b}  test={n_test}  MAE={mae}  ({t}s)")
        else:
            print(f"  {cls:<15} : pas de modele")

    print(f"\n  Fichiers dans : {OUTPUT_DIR}")
    print(f"\n  Configuration utilisee :")
    print(f"    CHUNK_SIZE   = {CHUNK_SIZE:,}")
    print(f"    TRAIN_BATCH  = {TRAIN_BATCH:,}")
    print(f"    RAM_WARN_PCT = {RAM_WARN_PCT}%")
    print(f"    RAM_STOP_PCT = {RAM_STOP_PCT}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()