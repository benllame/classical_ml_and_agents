"""MLflow tracking setup and utilities.

Centralizes MLflow configuration to enforce consistency.  All modules import
from here rather than configuring MLflow independently.  This prevents
tracking URI mismatches that could silently send metrics to different servers
or create duplicate experiments.  A single source of truth for the experiment
name and model registry name also simplifies refactoring and environment
promotion (dev → staging → prod).
"""

from __future__ import annotations

import mlflow
from loguru import logger

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI


def init_mlflow() -> str:
    """Initialize MLflow tracking URI and experiment.

    Sets both tracking URI and experiment name so all runs land in the
    same place and are easy to compare in the MLflow UI.

    Returns
    -------
    str
        The experiment ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(
        f"MLflow initialized — URI: {MLFLOW_TRACKING_URI} | "
        f"Experiment: {MLFLOW_EXPERIMENT_NAME} (id={experiment.experiment_id})"
    )
    return experiment.experiment_id


def get_production_model_uri() -> str:
    """Get the URI for the latest Production model from the registry.

    Tries Production first, then Staging as a fallback. Raises if neither
    stage has a registered model.

    Returns
    -------
    str
        MLflow model URI like ``'models:/<name>/<version>'``.
    """
    client = mlflow.MlflowClient()
    # Search for the latest version with alias "Production" or "Staging"
    try:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if versions:
            uri = f"models:/{MLFLOW_MODEL_NAME}/{versions[0].version}"
            logger.info(f"Production model: {uri}")
            return uri
    except Exception:
        pass

    # Fallback to Staging
    try:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Staging"])
        if versions:
            uri = f"models:/{MLFLOW_MODEL_NAME}/{versions[0].version}"
            logger.info(f"Staging model (fallback): {uri}")
            return uri
    except Exception:
        pass

    raise RuntimeError(
        f"No model found in registry '{MLFLOW_MODEL_NAME}' with stage Production or Staging. "
        "Run the training pipeline first (src/train.py)."
    )


def log_figure(fig, name: str) -> None:
    """Log a matplotlib figure as an MLflow artifact.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to log.
    name : str
        Artifact filename (e.g. 'roc_curve.png').
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f18")
        mlflow.log_artifact(str(path), artifact_path="figures")
        logger.debug(f"Logged figure artifact: figures/{name}")
