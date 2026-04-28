#!/usr/bin/env python3
"""
main_pipeline.py — Pipeline minimal de bout en bout.

Etapes :
  1. PREPARATION       (PreparePourGlobal)      -> stats_classification_SB.csv
  2. TRAIN SB          (train_lightgbm_sb)      -> data_{cls}.csv + model_{cls}.txt
  3. TRAIN DIRECT      (train_lightgbm_v1_direct ou _fast)
                                                -> model_{cls}_v1_direct[_fast].txt
  4. BENCHMARK         (benchmark_per_product)  -> benchmark_per_product_*.csv
  5. EXPORT ELASTICITE (export_elasticite)      -> {CLIENT}_ELASTICITE_{date}.txt

Usage :
  python main_pipeline.py                       # pipeline complet
  python main_pipeline.py --skip-prep           # saute la preparation
  python main_pipeline.py --only train_sb       # une seule etape
  python main_pipeline.py --steps train_sb,export
  python main_pipeline.py --fast                # utilise la variante _fast (parquet)
  python main_pipeline.py --client BOOPER       # client pour l'export
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ALL_STEPS = ["prep", "train_sb", "train_direct", "benchmark", "export"]


def _banner(title: str) -> None:
    print(f"\n{'#' * 72}")
    print(f"#  {title}")
    print(f"{'#' * 72}\n")


def _run(name: str, fn) -> bool:
    t0 = time.time()
    try:
        fn()
        print(f"\n[OK] {name} en {time.time() - t0:.0f}s")
        return True
    except SystemExit as e:
        if e.code in (None, 0):
            print(f"\n[OK] {name} en {time.time() - t0:.0f}s")
            return True
        print(f"\n[ECHEC] {name} : sys.exit({e.code})")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] {name}")
        raise
    except Exception as exc:
        import traceback
        print(f"\n[ERREUR] {name} : {exc}")
        traceback.print_exc()
        return False


def _with_argv(argv: list[str], fn) -> None:
    """Exécute `fn` en mockant temporairement sys.argv (modules argparse)."""
    saved = sys.argv
    sys.argv = argv
    try:
        fn()
    finally:
        sys.argv = saved


# --- ETAPE 1 : PREPARATION -------------------------------------------------
def step_prep() -> None:
    _banner("ETAPE 1 / PREPARATION (classification S-B + split ZIP)")
    import PreparePourGlobal
    if hasattr(PreparePourGlobal, "main"):
        PreparePourGlobal.main()
    else:
        print("[INFO] PreparePourGlobal sans main(); execution module-level")
        # Le module fait son boulot a l'import si pas de main()


# --- ETAPE 2 : TRAIN LightGBM Tweedie incremental --------------------------
def step_train_sb() -> None:
    _banner("ETAPE 2 / TRAIN LightGBM Tweedie incremental (1 modele/classe)")
    import train_lightgbm_sb
    train_lightgbm_sb.main()


# --- ETAPE 3 : TRAIN LightGBM Direct multi-horizon -------------------------
def step_train_direct(fast: bool = False) -> None:
    label = "FAST parquet" if fast else "slow streaming"
    _banner(f"ETAPE 3 / TRAIN Direct multi-horizon + Optuna ({label})")
    import train_lightgbm_v1_direct
    argv = ["train_lightgbm_v1_direct.py"]
    if fast:
        argv.append("--fast")
    _with_argv(argv, train_lightgbm_v1_direct.main)


# --- ETAPE 4 : BENCHMARK per-product vs global -----------------------------
def step_benchmark(global_model: str = "v1_direct", n_per_class: int = 2) -> None:
    _banner(f"ETAPE 4 / BENCHMARK per-product vs {global_model}")
    import benchmark_per_product
    _with_argv(
        [
            "benchmark_per_product.py",
            "--global-model", global_model,
            "--n-per-class", str(n_per_class),
        ],
        benchmark_per_product.main,
    )


# --- ETAPE 5 : EXPORT ELASTICITE -------------------------------------------
def step_export(client: str = "BOOPER", model: str | None = None,
                use_shap: bool = True) -> None:
    _banner(f"ETAPE 5 / EXPORT ELASTICITE (client={client})")
    import export_elasticite
    argv = ["export_elasticite.py", "--client", client]
    if model:
        argv += ["--model", model]
    if not use_shap:
        argv += ["--no-shap"]
    _with_argv(argv, export_elasticite.main)


# --- ORCHESTRATEUR ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline minimal Booper / LightGBM")
    p.add_argument("--steps", type=str, default=None,
                   help=f"Etapes a executer separees par virgule. "
                        f"Disponibles : {','.join(ALL_STEPS)}")
    p.add_argument("--only", type=str, default=None, choices=ALL_STEPS,
                   help="Executer une seule etape")
    p.add_argument("--skip", type=str, default=None,
                   help="Etapes a sauter, separees par virgule")
    p.add_argument("--fast", action="store_true",
                   help="Utilise la variante FAST (parquet) pour train_direct")
    p.add_argument("--global-model", default="v1_direct",
                   choices=["v1_optuna", "v1_direct"],
                   help="Modele global a evaluer dans le benchmark")
    p.add_argument("--n-per-class", type=int, default=2,
                   help="Produits par classe pour le benchmark")
    p.add_argument("--client", default=os.environ.get("EXPORT_CLIENT", "BOOPER"),
                   help="Nom du client pour l'export elasticite")
    p.add_argument("--no-shap", action="store_true",
                   help="Desactive SHAP dans l'export elasticite")
    p.add_argument("--export-model", default=None,
                   help="Forcer un type de modele pour l'export "
                        "(v1_direct, v1_direct_fast, v1_optuna, v1)")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continuer meme si une etape echoue")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.only:
        steps = [args.only]
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    else:
        steps = list(ALL_STEPS)

    if args.skip:
        skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
        steps = [s for s in steps if s not in skipped]

    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        print(f"[ERREUR] Etapes inconnues : {unknown}")
        print(f"  Disponibles : {ALL_STEPS}")
        return 2

    print(f"\n>>> Pipeline : {' -> '.join(steps)}\n")
    t_start = time.time()

    runners = {
        "prep":         lambda: step_prep(),
        "train_sb":     lambda: step_train_sb(),
        "train_direct": lambda: step_train_direct(fast=args.fast),
        "benchmark":    lambda: step_benchmark(
            global_model=args.global_model,
            n_per_class=args.n_per_class,
        ),
        "export":       lambda: step_export(
            client=args.client,
            model=args.export_model,
            use_shap=not args.no_shap,
        ),
    }

    failures: list[str] = []
    for s in steps:
        ok = _run(s, runners[s])
        if not ok:
            failures.append(s)
            if not args.continue_on_error:
                print(f"\n[ARRET] Etape '{s}' a echoue (--continue-on-error pour ignorer)")
                break

    _banner("RESUME PIPELINE")
    print(f"  Duree totale : {time.time() - t_start:.0f}s")
    print(f"  Etapes OK    : {len(steps) - len(failures)}/{len(steps)}")
    if failures:
        print(f"  Etapes KO    : {failures}")
        return 1
    print(f"  Statut       : SUCCES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
