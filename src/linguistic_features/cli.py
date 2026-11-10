"""Comandos para preparar los datos, ajustar el modelo y abrir resultados."""

import argparse
import json
import shutil
import webbrowser

import pandas as pd

from .config import PROCESSED, RESULTS_ROOT, ROOT, dataset_results, load_settings
from .data import prepare_data
from .pipeline import run_pipeline
from .posterior import build_posterior_explorer


def main():
    parser = argparse.ArgumentParser(
        description="Imputación bayesiana multirrespuesta de rasgos lingüísticos"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepara una fuente y la matriz MCC")
    prepare.add_argument("--model-config", default="config/model.json")
    run = commands.add_parser("run", help="Evalúa y luego imputa los rasgos binarios faltantes")
    run.add_argument("--model-config", default=None)
    run.add_argument("--inference-config", default="config/inference.json")
    run.add_argument(
        "--all-datasets", action="store_true",
        help="Ejecuta tanto el panel WALS como el panel de Grambank",
    )
    map_command = commands.add_parser("map", help="Abre el mapa de un conjunto")
    map_command.add_argument("--dataset", choices=["wals", "gbi_statistical"], default="wals")
    explore = commands.add_parser(
        "explore", help="Abre las distribuciones posteriores y las etapas del análisis"
    )
    explore.add_argument("--dataset", choices=["wals", "gbi_statistical"], default="wals")
    audit_command = commands.add_parser("audit", help="Muestra la cobertura de una fuente")
    audit_command.add_argument("--dataset", choices=["wals", "gbi_statistical"], default="wals")
    commands.add_parser("site", help="Abre el portal para alternar entre conjuntos")
    commands.add_parser(
        "clean-results",
        help="Borra solo los resultados generados anteriormente",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        model, _ = load_settings(args.model_config, "config/inference.json")
        prepare_data(model)
    elif args.command == "run":
        if args.model_config:
            run_pipeline(args.model_config, args.inference_config)
        elif args.all_datasets:
            for config_path in ("config/model.json", "config/model_gbi.json"):
                run_pipeline(config_path, args.inference_config)
        else:
            run_pipeline("config/model.json", args.inference_config)
    elif args.command == "map":
        folder = dataset_results(args.dataset)
        path = folder / "map.html"
        run_path, predictions_path = folder / "run.json", folder / "predictions.csv"
        if not run_path.exists() or not predictions_path.exists():
            raise SystemExit(
                f"Todavía no hay resultados para {args.dataset}. "
                "Ejecuta `uv run lingfeat run --all-datasets`."
            )
        run = json.loads(run_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(
            predictions_path, dtype={"ID": str}, keep_default_na=False
        )
        from .visualization import build_map

        build_map(
            predictions, path, run["configuration"]["features"],
            run["configuration"].get("dataset_label", args.dataset),
        )
        webbrowser.open(path.resolve().as_uri())
    elif args.command == "explore":
        folder = dataset_results(args.dataset)
        required = [
            folder / "run.json", folder / "predictions.csv",
            folder / "evaluation.npz", folder / "imputation.npz",
        ]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise SystemExit(
                "Faltan resultados: " + ", ".join(missing)
                + ". Ejecuta `uv run lingfeat run --all-datasets`."
            )
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(
            folder / "predictions.csv", dtype={"ID": str}, keep_default_na=False
        )
        page = build_posterior_explorer(folder, run, predictions)
        webbrowser.open((folder / "posterior.html").resolve().as_uri())
    elif args.command == "audit":
        path = PROCESSED / args.dataset / "audit.json"
        if not path.exists():
            raise SystemExit(
                f"Todavía no hay auditoría para {args.dataset}. "
                "Ejecuta `uv run lingfeat prepare --model-config "
                f"{'config/model_gbi.json' if args.dataset == 'gbi_statistical' else 'config/model.json'}`."
            )
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=True))
    elif args.command == "site":
        path = RESULTS_ROOT / "index.html"
        if not path.exists():
            raise SystemExit("Todavía no hay resultados. Ejecuta `uv run lingfeat run --all-datasets`.")
        webbrowser.open(path.resolve().as_uri())
    elif args.command == "clean-results":
        _clean_generated_results()


def _clean_generated_results():
    """Elimina únicamente ``results/current`` tras validar su ruta absoluta."""
    if RESULTS_ROOT.is_symlink():
        raise SystemExit("No se limpia results/current si es un enlace simbólico.")
    target = RESULTS_ROOT.resolve(strict=False)
    allowed = (ROOT / "results" / "current").resolve(strict=False)
    if target != allowed or target.parent != (ROOT / "results").resolve():
        raise SystemExit(f"Ruta fuera del destino permitido: {target}")
    if target.exists():
        if not target.is_dir():
            raise SystemExit(f"El destino no es una carpeta: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Resultados previos eliminados de {target}")


if __name__ == "__main__":
    main()
