from __future__ import annotations

import os
import sys
import re
import csv
import math
import pickle
import zipfile
from datetime import date

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

# Plage de dates couverte par le bitset (1 bit par jour, optimise RAM)
EPOCH = date(2018, 1, 1)
HORIZON_DAYS = 366 * 10   # 10 ans -> 3660 jours -> ~458 bytes/serie

# Checkpoint : sauvegarde l'etat tous les N zips (reprise auto si crash)
CHECKPOINT_INTERVAL = 5
CHECKPOINT_FILE = "_checkpoint.pkl"
CHECKPOINT_VERSION = 3   # bump si la structure des dicts change (invalide ancien ck)

# Seuils Syntetos-Boylan
SB_ADI_CUTOFF = 1.32   # Demand interval (jours moyens entre 2 ventes)
SB_CV2_CUTOFF = 0.49   # CV au carre


# =====================================================================
#  UTILS
# =====================================================================

def list_zip_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".zip"))


def parse_kv_line(line):
    """Parse '{KEY1=VAL1, KEY2=VAL2, ...}' -> dict (cle/valeur en str)."""
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    line = line[1:-1]
    result = {}
    for pair in re.split(r",\s+", line):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        result[key.strip()] = value.strip()
    return result if result else None


def iter_zip_rows(filepath):
    """Yield les rows d'un ZIP, ligne par ligne.
    Si plusieurs .txt, prefere CENTRALE_HIST_* (historique avec QUANTITE)
    plutot que CENTRALE_FUTURE_* (donnees futures sans QUANTITE)."""
    with zipfile.ZipFile(filepath, "r") as zf:
        txt_files = [f for f in zf.namelist()
                     if f.endswith(".txt") and not f.startswith("__")]
        if not txt_files:
            return
        # Prioriser HIST si present
        hist_files = [f for f in txt_files if "HIST" in f.upper()]
        target = hist_files[0] if hist_files else txt_files[0]
        with zf.open(target) as f:
            for raw in f:
                row = parse_kv_line(raw.decode("utf-8", errors="replace"))
                if row:
                    yield row


def parse_date_from_row(row):
    """Retourne (days_since_epoch, date_obj) ou (None, None).
    Gere les floats en string : '2024.0' -> 2024."""
    try:
        y = int(float(row.get("PARAM_ANNEE", 0)))
        m = int(float(row.get("PARAM_MOIS", 0)))
        d = int(float(row.get("PARAM_JOUR", 0)))
        if y < 2000 or not (1 <= m <= 12) or not (1 <= d <= 31):
            return None, None
        dt = date(y, m, d)
        days = (dt - EPOCH).days
        if 0 <= days < HORIZON_DAYS:
            return days, dt
    except (ValueError, TypeError):
        pass
    return None, None


def parse_int(v):
    """Parse en int, gere les floats en string ('1.0', '4.0', etc)."""
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def parse_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def save_checkpoint(checkpoint_path, serie_stats, serie_dates, serie_dates_demand, done_zips):
    """Ecriture atomique : .tmp puis rename, evite la corruption en cas de crash."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({
            "version": CHECKPOINT_VERSION,
            "serie_stats": serie_stats,
            "serie_dates": serie_dates,
            "serie_dates_demand": serie_dates_demand,
            "done_zips": done_zips,
            "epoch": EPOCH.isoformat(),
            "horizon_days": HORIZON_DAYS,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path):
    """Charge un checkpoint existant. Retourne (serie_stats, serie_dates, serie_dates_demand, done_zips)."""
    if not os.path.exists(checkpoint_path):
        return {}, {}, {}, set()
    try:
        with open(checkpoint_path, "rb") as f:
            ck = pickle.load(f)
        # Verifier compat (version / epoch / horizon)
        if (ck.get("version") != CHECKPOINT_VERSION
                or ck.get("epoch") != EPOCH.isoformat()
                or ck.get("horizon_days") != HORIZON_DAYS):
            print(f"  [WARN] Checkpoint incompatible (version/EPOCH/HORIZON change). Reset.")
            return {}, {}, {}, set()
        print(f"  [CHECKPOINT] Reprise : {len(ck['done_zips'])} ZIPs deja traites, "
              f"{len(ck['serie_stats']):,} series (prod x so)")
        return (ck["serie_stats"], ck["serie_dates"],
                ck.get("serie_dates_demand", {}), set(ck["done_zips"]))
    except Exception as e:
        print(f"  [WARN] Checkpoint corrompu ({e}). Reset.")
        return {}, {}, {}, set()


# =====================================================================
#  STREAMING : une passe, agregations incrementales
# =====================================================================

def stream_process(centrale_dir, output_dir, max_zips=MAX_ZIPS):
    """
    Une passe sur les ZIPs avec checkpoint periodique.
    Reprend automatiquement depuis le dernier checkpoint si crash.
    """
    print(f"\n{'='*70}")
    print(f"  STREAMING : agregation des ventes par produit")
    print(f"{'='*70}\n")

    zip_files = list_zip_files(centrale_dir)
    selected = zip_files[:max_zips]
    print(f"  ZIPs disponibles : {len(zip_files)}")
    print(f"  ZIPs cibles      : {len(selected)}")
    print(f"  Plage temporelle : {EPOCH} -> {EPOCH.fromordinal(EPOCH.toordinal() + HORIZON_DAYS - 1)}")
    print(f"  Checkpoint tous les {CHECKPOINT_INTERVAL} ZIPs\n")

    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    serie_stats, serie_dates, serie_dates_demand, done_zips = load_checkpoint(checkpoint_path)

    bitset_bytes = (HORIZON_DAYS + 7) // 8
    n_bad_dates = 0
    n_no_so = 0
    n_processed_this_run = 0
    # Plage de dates reellement observee (en days_since_epoch)
    global_day_min = None
    global_day_max = None

    for i, zf_name in enumerate(selected, 1):
        if zf_name in done_zips:
            print(f"  [{i:3d}/{len(selected)}] {zf_name} : SKIP (deja traite)")
            continue

        filepath = os.path.join(centrale_dir, zf_name)
        n_zip = 0
        n_kept_zip = 0
        try:
            for row in iter_zip_rows(filepath):
                n_zip += 1
                prod = parse_int(row.get("ID_PRODUIT"))
                so = parse_int(row.get("ID_SO"))
                q = parse_float(row.get("QUANTITE"))
                if prod is None or q is None:
                    continue
                if so is None:
                    n_no_so += 1
                    continue

                key = (prod, so)

                # Stats par (produit, magasin)
                # Layout: [n_records, sum_all, sumsq_all, nz_records, sum_nz, sumsq_nz]
                # - sum_all/sumsq_all : utiles pour stats descriptives globales
                # - sum_nz/sumsq_nz   : utilises pour CV2 Syntetos-Boylan (sur non-zeros)
                stats = serie_stats.get(key)
                if stats is None:
                    stats = [0, 0.0, 0.0, 0, 0.0, 0.0]
                    serie_stats[key] = stats
                stats[0] += 1
                stats[1] += q
                stats[2] += q * q
                if q > 0:
                    stats[3] += 1
                    stats[4] += q
                    stats[5] += q * q

                # Couverture temporelle (meme cle) - DEUX bitsets
                # serie_dates        : tous les jours observes (data presente, q quelconque)
                # serie_dates_demand : jours avec q > 0 (utilise pour ADI Syntetos-Boylan)
                days, _ = parse_date_from_row(row)
                if days is None:
                    n_bad_dates += 1
                else:
                    bs = serie_dates.get(key)
                    if bs is None:
                        bs = bytearray(bitset_bytes)
                        serie_dates[key] = bs
                    bs[days >> 3] |= 1 << (days & 7)
                    if q > 0:
                        bsd = serie_dates_demand.get(key)
                        if bsd is None:
                            bsd = bytearray(bitset_bytes)
                            serie_dates_demand[key] = bsd
                        bsd[days >> 3] |= 1 << (days & 7)
                    # Tracking plage globale
                    if global_day_min is None or days < global_day_min:
                        global_day_min = days
                    if global_day_max is None or days > global_day_max:
                        global_day_max = days

                n_kept_zip += 1
        except Exception as e:
            print(f"  [{i:3d}/{len(selected)}] {zf_name} : ERREUR -> {e}")
            save_checkpoint(checkpoint_path, serie_stats, serie_dates, serie_dates_demand, done_zips)
            continue

        done_zips.add(zf_name)
        n_processed_this_run += 1
        # Compter produits distincts (pour info)
        n_prod = len({k[0] for k in serie_stats})
        print(f"  [{i:3d}/{len(selected)}] {zf_name} : "
              f"{n_zip:>10,} lignes ({n_kept_zip:>10,} retenues) | "
              f"prod={n_prod:>5,} series={len(serie_stats):>7,}")

        if n_processed_this_run % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(checkpoint_path, serie_stats, serie_dates, serie_dates_demand, done_zips)
            print(f"      [CHECKPOINT sauvegarde]")

    save_checkpoint(checkpoint_path, serie_stats, serie_dates, serie_dates_demand, done_zips)

    n_prod = len({k[0] for k in serie_stats})
    n_so = len({k[1] for k in serie_stats})
    print(f"\n  ZIPs traites cette session : {n_processed_this_run}")
    print(f"  Produits uniques     : {n_prod:,}")
    print(f"  Magasins uniques     : {n_so:,}")
    print(f"  Series (prod x so)   : {len(serie_stats):,}")
    if n_no_so:
        print(f"  Lignes sans ID_SO    : {n_no_so:,}")
    if n_bad_dates:
        print(f"  Dates non parsables  : {n_bad_dates:,}")

    # Plage de dates reelle (vs capacite du bitset)
    if global_day_min is not None:
        dt_min = date.fromordinal(EPOCH.toordinal() + global_day_min)
        dt_max = date.fromordinal(EPOCH.toordinal() + global_day_max)
        span_days = global_day_max - global_day_min + 1
        used_pct = span_days / HORIZON_DAYS * 100
        print(f"  Plage observee       : {dt_min} -> {dt_max} ({span_days:,} jours)")
        print(f"  Capacite bitset      : {EPOCH} -> "
              f"{date.fromordinal(EPOCH.toordinal() + HORIZON_DAYS - 1)} "
              f"({HORIZON_DAYS:,} jours, {used_pct:.1f}% utilise)")
        if used_pct < 50:
            new_horizon = ((span_days + 365) // 366 + 1) * 366
            print(f"  [TIP] Tu peux reduire HORIZON_DAYS a ~{new_horizon:,} "
                  f"et EPOCH a {dt_min.replace(month=1, day=1)} pour gagner de la RAM")

    ram_dates_mb = (len(serie_dates) + len(serie_dates_demand)) * bitset_bytes / (1024 * 1024)
    print(f"  RAM bitsets dates    : ~{ram_dates_mb:.1f} Mo (obs + demand)")

    return serie_stats, serie_dates, serie_dates_demand




def _popcount_bytes(b):
    """Compte le nombre de bits a 1 dans un bytearray."""
    if b is None:
        return 0
    # int.from_bytes + bit_count() est tres rapide en Python 3.10+
    try:
        return int.from_bytes(b, "little").bit_count()
    except AttributeError:
        # Fallback Python < 3.10
        return bin(int.from_bytes(b, "little")).count("1")


def write_stats_ventes(serie_stats, serie_dates, serie_dates_demand, output_path):
    """Calcule les stats Syntetos-Boylan correctes par (ID_PRODUIT, ID_SO).

    - CV2 est calcule sur les VALEURS NON NULLES uniquement (cv2_nz).
    - ADI = nb_jours_observes / nb_jours_avec_demande (via popcount des bitsets).

    Sortie CSV : conserve les anciennes colonnes pour compatibilite et ajoute :
      - nb_jours_obs        : popcount(serie_dates)        - jours distincts observes
      - nb_jours_demand     : popcount(serie_dates_demand) - jours distincts avec q>0
      - mean_nz, cv_nz, cv2_nz : moments sur les non-zeros (Syntetos-Boylan)
    """
    print(f"\n  Ecriture stats ventes par (produit, magasin)...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ID_PRODUIT", "ID_SO",
                    "nb_records", "nb_records_avec_vente",
                    "nb_jours_obs", "nb_jours_demand",
                    "moyenne_all", "ecart_type_all", "cv_all",
                    "moyenne_nz", "ecart_type_nz", "cv_nz",
                    "somme",
                    "adi", "cv2", "profil_demande"])

        for (prod, so) in sorted(serie_stats.keys()):
            stats = serie_stats[(prod, so)]
            # Compat ancien checkpoint (4 colonnes au lieu de 6)
            if len(stats) == 4:
                n, s, ss, nz = stats
                sum_nz, sumsq_nz = float("nan"), float("nan")
            else:
                n, s, ss, nz, sum_nz, sumsq_nz = stats
            if n == 0:
                continue

            # Stats globales (pour info, incluent les zeros)
            mean_all = s / n
            var_all = max(0.0, ss / n - mean_all * mean_all)
            std_all = math.sqrt(var_all)
            cv_all = std_all / mean_all if mean_all > 0 else float("nan")

            # Stats Syntetos-Boylan : sur les non-zeros uniquement
            if nz > 0 and not math.isnan(sum_nz):
                mean_nz = sum_nz / nz
                var_nz = max(0.0, sumsq_nz / nz - mean_nz * mean_nz)
                std_nz = math.sqrt(var_nz)
                cv_nz = std_nz / mean_nz if mean_nz > 0 else float("nan")
                cv2 = cv_nz * cv_nz if not math.isnan(cv_nz) else float("nan")
            else:
                mean_nz = float("nan")
                std_nz = float("nan")
                cv_nz = float("nan")
                cv2 = float("nan")

            # ADI : inter-demand interval calcule sur les jours distincts.
            # ADI = jours_observes / jours_avec_demande
            n_obs = _popcount_bytes(serie_dates.get((prod, so)))
            n_dem = _popcount_bytes(serie_dates_demand.get((prod, so)))
            if n_dem > 0:
                adi = n_obs / n_dem
            else:
                adi = float("inf")

            profil = classify_sb(adi, cv2)
            w.writerow([prod, so,
                        n, nz,
                        n_obs, n_dem,
                        round(mean_all, 4), round(std_all, 4),
                        round(cv_all, 4) if not math.isnan(cv_all) else "",
                        round(mean_nz, 4) if not math.isnan(mean_nz) else "",
                        round(std_nz, 4) if not math.isnan(std_nz) else "",
                        round(cv_nz, 4) if not math.isnan(cv_nz) else "",
                        round(s, 2),
                        round(adi, 4) if math.isfinite(adi) else "",
                        round(cv2, 4) if not math.isnan(cv2) else "",
                        profil])
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [SAVE] {output_path} ({size:.2f} Mo)")


def classify_sb(adi, cv2):
    """Classification Syntetos-Boylan."""
    if math.isnan(cv2) or not math.isfinite(adi):
        return "indetermine"
    if adi < SB_ADI_CUTOFF and cv2 < SB_CV2_CUTOFF:
        return "smooth"
    if adi >= SB_ADI_CUTOFF and cv2 < SB_CV2_CUTOFF:
        return "intermittent"
    if adi < SB_ADI_CUTOFF and cv2 >= SB_CV2_CUTOFF:
        return "erratic"
    return "lumpy"


def write_classification_sb(stats_csv_path, output_path):
    """Extrait (ID_PRODUIT, ID_SO, classification) depuis stats_ventes."""
    print(f"\n  Ecriture classification S-B par (produit, magasin)...")
    counts = {"smooth": 0, "intermittent": 0, "erratic": 0, "lumpy": 0, "indetermine": 0}
    with open(stats_csv_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter=";")
        writer = csv.writer(fout, delimiter=";")
        writer.writerow(["ID_PRODUIT", "ID_SO", "classification"])
        for row in reader:
            cls = row["profil_demande"]
            counts[cls] = counts.get(cls, 0) + 1
            writer.writerow([row["ID_PRODUIT"], row["ID_SO"], cls])
    print(f"  [SAVE] {output_path}")
    print(f"  Repartition S-B (par couple produit x magasin) :")
    total = sum(counts.values())
    for cls in ["smooth", "intermittent", "erratic", "lumpy", "indetermine"]:
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"    {cls:<15} : {n:>6,} ({pct:.1f}%)")


def write_gaps(serie_dates, output_path):
    """Pour chaque (prod, so), calcule [min, max, present, manquants] depuis le bitset."""
    print(f"\n  Ecriture rapport trous dates...")
    n_with_gaps = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ID_PRODUIT", "ID_SO", "date_min", "date_max",
                    "nb_jours_attendus", "nb_jours_presents",
                    "nb_jours_manquants", "taux_couverture_pct"])
        for (prod, so), bs in serie_dates.items():
            # Trouver premier et dernier bit a 1
            day_min = None
            day_max = None
            count = 0
            for byte_idx, byte_val in enumerate(bs):
                if byte_val == 0:
                    continue
                for bit in range(8):
                    if byte_val & (1 << bit):
                        d = byte_idx * 8 + bit
                        if day_min is None:
                            day_min = d
                        day_max = d
                        count += 1
            if day_min is None or count < 2:
                continue
            expected = day_max - day_min + 1
            missing = expected - count
            if missing > 0:
                n_with_gaps += 1
            dt_min = date.fromordinal(EPOCH.toordinal() + day_min)
            dt_max = date.fromordinal(EPOCH.toordinal() + day_max)
            w.writerow([prod, so, dt_min, dt_max,
                        expected, count, missing,
                        round(count / expected * 100, 1)])
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [SAVE] {output_path} ({size:.2f} Mo)")
    print(f"  Series avec trous : {n_with_gaps:,} / {len(serie_dates):,}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(CENTRALE_DIR):
        print(f"[ERREUR] CENTRALE_DIR introuvable : {CENTRALE_DIR}")
        sys.exit(1)

    serie_stats, serie_dates, serie_dates_demand = stream_process(
        CENTRALE_DIR, OUTPUT_DIR, max_zips=MAX_ZIPS)

    if not serie_stats:
        print("Aucune donnee. Arret.")
        sys.exit(1)

    stats_path = os.path.join(OUTPUT_DIR, "stats_ventes_produits.csv")
    write_stats_ventes(serie_stats, serie_dates, serie_dates_demand, stats_path)

    sb_path = os.path.join(OUTPUT_DIR, "stats_classification_SB.csv")
    write_classification_sb(stats_path, sb_path)

    gaps_path = os.path.join(OUTPUT_DIR, "rapport_trous_dates.csv")
    write_gaps(serie_dates, gaps_path)

    print(f"\n{'='*70}")
    print(f"  Pipeline termine. Fichiers dans : {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
