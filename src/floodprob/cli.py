from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from floodprob.config import load_config
from floodprob.evaluate import run_evaluate
from floodprob.logging_utils import configure_logging
from floodprob.predict import run_predict
from floodprob.preprocess import run_ingest, run_preprocess, run_stac_check
from floodprob.train import run_train
from floodprob.visualize import run_folium_preview
from floodprob.compare import run_compare

app = typer.Typer(help="Flood probability pipeline (STAC-first).")
LOGGER = logging.getLogger(__name__)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level)


@app.command("stac-check")
def stac_check(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    report = run_stac_check(cfg)
    typer.echo(json.dumps(report, indent=2))


@app.command("ingest")
def ingest(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    manifest = run_ingest(cfg)
    LOGGER.info("Ingest complete | targets=%s", len(manifest["targets"]))


@app.command("preprocess")
def preprocess(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    manifest = run_preprocess(cfg)
    LOGGER.info("Preprocess complete | targets=%s", len(manifest["targets"]))


@app.command("train")
def train(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    report = run_train(cfg)
    LOGGER.info(
        "Train complete | water_model=%s flood_model=%s",
        report["water_model_type"],
        report["flood_model_type"],
    )


@app.command("predict")
def predict(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    report = run_predict(cfg)
    LOGGER.info("Predict complete | windows=%s", len(report["predictions"]))


@app.command("evaluate")
def evaluate(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    cfg = load_config(config)
    report = run_evaluate(cfg)
    LOGGER.info("Evaluate complete | overall=%s", report["overall"])


@app.command("folium")
def folium_preview(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    key: str = typer.Option("", "--key", help="Prediction window key, e.g. 20241028_20241102"),
) -> None:
    cfg = load_config(config)
    out = run_folium_preview(cfg, key=key or None)
    LOGGER.info("Folium preview generated: %s", out)


@app.command("compare")
def compare(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="Ignore cached bundles and retrain all models."),
) -> None:
    """Train & evaluate all configured models and generate comparison figures."""
    cfg = load_config(config)
    report = run_compare(cfg, force_retrain=force_retrain)
    LOGGER.info(
        "Compare complete | water_models=%s flood_models=%s",
        report["models_water_trained"],
        report["models_flood_trained"],
    )


if __name__ == "__main__":
    app()
