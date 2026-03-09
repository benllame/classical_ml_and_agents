"""SHAP-based model explainability.

For each customer, SHAP tells us which features pushed their churn
prediction up or down and by how much. That turns a black-box probability
into something a retention manager can actually act on.

Provides:
- Global explanations (summary plot, feature importance ranking)
- Per-customer explanations (waterfall plots)
- `get_shap_explanation(customer_id)` → structured dict for the ReAct agent
- MI vs SHAP comparison: checks whether model importance aligns with
  what information theory says is informative

Usage:
    python src/explainer.py                      # generate all SHAP artifacts
    python src/explainer.py --customer-id 0001   # single customer explanation
    python src/explainer.py --mi-comparison       # MI vs SHAP analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

from src.config import (
    FIGURES_DIR,
    ID_COL,
    MODELS_DIR,
    RAW_CSV,
    SHAP_EXPLAINER_PATH,
    TARGET_COL,
)
from src.eda import TEXT_COLOR
from src.preprocessing import get_feature_names, load_pipeline, prepare_data

matplotlib.use("Agg")  # non-interactive backend


# ── Explainer Builder ────────────────────────────────────────────────────────


def build_shap_explainer(
    model,
    X_train: np.ndarray,
    feature_names: list[str] | None = None,
) -> shap.TreeExplainer:
    """Build and save a SHAP TreeExplainer.

    Parameters
    ----------
    model
        Fitted tree-based model (RF, GBM, XGBoost).
    X_train : np.ndarray
        Training data (subsample for background).
    feature_names : list[str]
        Feature names for labeling.

    Returns
    -------
    shap.TreeExplainer
    """
    # We use 200 background samples to compute baseline values.
    # More samples is more accurate but also slower — 200 is a good
    # balance for this dataset size.
    n_background = min(200, X_train.shape[0])
    background = shap.sample(X_train, n_background)

    try:
        # TreeExplainer is fast and exact for tree-based models.
        explainer = shap.TreeExplainer(model)
        logger.info("Built TreeExplainer (native tree SHAP)")
    except Exception:
        # KernelExplainer works for any model but is much slower.
        explainer = shap.KernelExplainer(model.predict_proba, background)
        logger.info("Built KernelExplainer (fallback)")

    return explainer


def save_explainer(explainer, path: Path | None = None) -> None:
    """Serialize the SHAP explainer to disk."""
    path = path or SHAP_EXPLAINER_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, path)
    logger.success(f"SHAP explainer saved to {path}")


def load_explainer(path: Path | None = None) -> shap.TreeExplainer:
    """Load the SHAP explainer from disk."""
    path = path or SHAP_EXPLAINER_PATH
    explainer = joblib.load(path)
    logger.info(f"SHAP explainer loaded from {path}")
    return explainer


# ── SHAP Computation ─────────────────────────────────────────────────────────


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute SHAP values for the given data.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    shap_values = explainer.shap_values(X)

    # For binary classifiers, shap_values is a list of two arrays.
    # We take index [1] for the positive class (Churn=1), so a positive
    # SHAP value means the feature increased churn probability.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        data=X,
        feature_names=feature_names,
    )

    return explanation


# ── Agent Tool: get_shap_explanation ─────────────────────────────────────────


def get_shap_explanation(
    customer_id: str,
    df: pd.DataFrame | None = None,
    pipeline=None,
    explainer=None,
    model=None,
    top_n: int = 3,
) -> dict[str, Any]:
    """Get a structured SHAP explanation for a single customer.

    This is designed to be called by the ReAct agent's explain_prediction tool.

    Parameters
    ----------
    customer_id : str
        The customer ID to explain.
    df : pd.DataFrame | None
        Full dataset. Loaded from RAW_CSV if None.
    pipeline : Pipeline | None
        Preprocessing pipeline. Loaded from disk if None.
    explainer : shap explainer | None
        SHAP explainer. Loaded from disk if None.
    model : estimator | None
        The prediction model (for probability).
    top_n : int
        Number of top contributing features to return.

    Returns
    -------
    dict
        {
            "customer_id": str,
            "churn_probability": float,
            "risk_segment": str,
            "base_value": float,
            "top_factors": [
                {"feature": str, "shap_value": float, "direction": "increases_risk"|"decreases_risk", "magnitude": str},
                ...
            ]
        }
    """
    # Load resources if not provided
    if df is None:
        df = pd.read_csv(RAW_CSV)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    if pipeline is None:
        pipeline = load_pipeline()

    if explainer is None:
        explainer = load_explainer()

    # Find the customer
    customer_row = df[df[ID_COL] == customer_id]
    if customer_row.empty:
        return {"error": f"Customer {customer_id} not found"}

    # Transform
    X_customer, _, _ = prepare_data(customer_row, fit_pipeline=False, pipeline=pipeline)

    # SHAP values
    shap_vals = explainer.shap_values(X_customer)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]

    shap_vals = shap_vals.flatten()
    base_value = (
        float(explainer.expected_value[1])
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )

    # Feature names
    feature_names = get_feature_names(pipeline)
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

    # Prediction
    churn_prob = None
    if model is not None:
        churn_prob = float(model.predict_proba(X_customer)[0, 1])

    # Top factors — buckets by impact magnitude.
    # These thresholds are rough empirical guides:
    #   |SHAP| > 0.5 → very high impact (top ~5% of features)
    #   0.2 - 0.5   → high impact
    #   0.1 - 0.2   → moderate
    #   < 0.1       → low
    abs_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_vals)[-top_n:][::-1]

    top_factors = []
    for idx in top_indices:
        val = float(shap_vals[idx])
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"

        magnitude = "low"
        if abs(val) > 0.5:
            magnitude = "very_high"
        elif abs(val) > 0.2:
            magnitude = "high"
        elif abs(val) > 0.1:
            magnitude = "medium"

        top_factors.append(
            {
                "feature": name,
                "shap_value": round(val, 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "magnitude": magnitude,
            }
        )

    risk_segment = "low"
    if churn_prob is not None:
        if churn_prob > 0.6:
            risk_segment = "high"
        elif churn_prob > 0.3:
            risk_segment = "medium"

    # This function is the bridge between the ML pipeline and the agent.
    # Returning a structured dict lets the LLM translate numbers into
    # plain-language explanations for retention managers.
    return {
        "customer_id": customer_id,
        "churn_probability": churn_prob,
        "risk_segment": risk_segment,
        "base_value": round(base_value, 4),
        "top_factors": top_factors,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_summary(explanation: shap.Explanation, save: bool = True) -> plt.Figure:
    """Global SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        show=False,
        plot_size=None,
        max_display=15,
    )
    plt.title("SHAP Summary — Global Feature Impact on Churn", fontsize=13, color=TEXT_COLOR)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    return fig


def plot_waterfall(
    explanation: shap.Explanation, idx: int = 0, save: bool = True, label: str = ""
) -> plt.Figure:
    """Waterfall plot for a single observation."""
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation[idx], show=False, max_display=10)
    plt.title(f"SHAP Waterfall — {label}", fontsize=12)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").lower()
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{safe_label}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_dependence(
    explanation: shap.Explanation,
    feature: str,
    interaction_feature: str | None = None,
    save: bool = True,
) -> plt.Figure:
    """SHAP dependence plot for a feature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature,
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    plt.title(f"SHAP Dependence — {feature}", fontsize=12)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            FIGURES_DIR / f"shap_dependence_{feature}.png",
            dpi=150,
            bbox_inches="tight",
        )
    return fig


# ── MI vs SHAP Comparison ───────────────────────────────────────────────────


def run_mi_vs_shap_comparison(
    explanation: shap.Explanation | None = None,
    shap_values: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    save_plots: bool = True,
) -> dict:
    """Compare MI-based and SHAP-based feature importance.

    MI measures how much raw information a feature carries about churn
    (model-agnostic). SHAP measures how much the trained model actually
    uses each feature (model-specific).

    When they agree, that's a good sign. When MI rank is much higher
    than SHAP rank, the model might be underusing a useful feature.
    When SHAP rank is much higher than MI rank, the model may be picking
    up interaction effects that marginal MI misses.

    Parameters
    ----------
    explanation : shap.Explanation | None
        Full SHAP explanation object.
    shap_values : np.ndarray | None
        SHAP values matrix (n_samples, n_features). Used if explanation is None.
    feature_names : list[str] | None
        Feature names matching SHAP columns. Used if explanation is None.
    save_plots : bool
        Save comparison plot to FIGURES_DIR.

    Returns
    -------
    dict
        {
            "comparison_df": pd.DataFrame,
            "rank_correlation": dict,
            "mi_scores": pd.Series,
            "shap_importance": pd.Series,
            "key_findings": list[str],
        }
    """
    from src.eda import load_raw_data
    from src.information_theory import (
        compare_mi_vs_shap,
        compute_mi_scores,
        compute_rank_correlation,
        plot_mi_vs_shap,
    )

    logger.info("── MI vs SHAP Comparison ──")

    # Extract SHAP values and feature names
    if explanation is not None:
        shap_vals = explanation.values
        feat_names = explanation.feature_names or [f"f_{i}" for i in range(shap_vals.shape[1])]
    elif shap_values is not None:
        shap_vals = shap_values
        feat_names = feature_names or [f"f_{i}" for i in range(shap_vals.shape[1])]
    else:
        raise ValueError("Provide either `explanation` or `shap_values` + `feature_names`")

    # SHAP importance: mean |SHAP| per feature
    shap_importance = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feat_names,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    # MI scores on raw features
    df = load_raw_data()
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    mi_scores = compute_mi_scores(X, y)

    # Compare
    comparison_df = compare_mi_vs_shap(mi_scores, shap_importance)
    rank_corr = compute_rank_correlation(comparison_df)

    # Plot
    fig = plot_mi_vs_shap(comparison_df, rank_corr, save=save_plots)
    plt.close(fig)

    # Derive key findings
    key_findings = _derive_mi_shap_findings(comparison_df, rank_corr)

    logger.info(f"Spearman ρ = {rank_corr['spearman_rho']:.3f}")
    logger.info(f"Kendall  τ = {rank_corr['kendall_tau']:.3f}")
    for finding in key_findings:
        logger.info(f"  → {finding}")

    return {
        "comparison_df": comparison_df,
        "rank_correlation": rank_corr,
        "mi_scores": mi_scores,
        "shap_importance": shap_importance,
        "key_findings": key_findings,
    }


def _derive_mi_shap_findings(
    comparison_df: pd.DataFrame,
    rank_corr: dict,
) -> list[str]:
    """Turn the MI vs SHAP numbers into readable findings.

    Returns
    -------
    list[str]
        Key insights in Spanish for business-facing reports.
    """
    findings = []

    rho = rank_corr["spearman_rho"]
    if rho > 0.8:
        findings.append(
            f"Alta concordancia (ρ={rho:.3f}): el modelo captura fielmente "
            "las dependencias no lineales detectadas por MI."
        )
    elif rho > 0.5:
        findings.append(
            f"Concordancia moderada (ρ={rho:.3f}): la mayoría de features "
            "coinciden, pero hay divergencias significativas."
        )
    else:
        findings.append(
            f"Baja concordancia (ρ={rho:.3f}): MI y SHAP dan rankings muy "
            "diferentes — investigar causas."
        )

    # Find biggest divergences
    df = comparison_df.copy()
    if "rank_difference" in df.columns:
        # Features where MI rank >> SHAP rank (model underweights)
        underweighted = df[df["rank_difference"] > 3].head(3)
        for _, row in underweighted.iterrows():
            findings.append(
                f"'{row['feature']}' tiene MI rank {row['MI_rank']} pero SHAP rank "
                f"{row['SHAP_rank']}: el modelo podría beneficiarse de darle más peso."
            )

        # Features where SHAP rank >> MI rank (model overweights)
        overweighted = df[df["rank_difference"] < -3].head(3)
        for _, row in overweighted.iterrows():
            findings.append(
                f"'{row['feature']}' tiene MI rank {row['MI_rank']} pero SHAP rank "
                f"{row['SHAP_rank']}: el modelo le da más importancia que su MI "
                "marginal — posible efecto de interacciones."
            )

    return findings


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--customer-id", type=str, default=None, help="Explain a single customer")
    parser.add_argument(
        "--mi-comparison",
        action="store_true",
        help="Run MI vs SHAP comparison (requires trained model + SHAP explainer)",
    )
    args = parser.parse_args()

    if args.customer_id:
        result = get_shap_explanation(args.customer_id)
        print(json.dumps(result, indent=2))
    elif args.mi_comparison:
        # Load pre-computed SHAP explainer and run comparison
        logger.info("Loading SHAP explainer and model for MI comparison...")
        try:
            shap_explainer = load_explainer()
            pipeline = load_pipeline()
            df = pd.read_csv(RAW_CSV)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["Churn"] = (df["Churn"] == "Yes").astype(int)

            X_transformed, y, _ = prepare_data(df, fit_pipeline=False, pipeline=pipeline)
            feature_names = get_feature_names(pipeline)

            explanation = compute_shap_values(shap_explainer, X_transformed, feature_names)
            results = run_mi_vs_shap_comparison(explanation=explanation)

            print("\n── MI vs SHAP Key Findings ──")
            for finding in results["key_findings"]:
                print(f"  • {finding}")
            print(f"\nSpearman ρ = {results['rank_correlation']['spearman_rho']:.4f}")
            print(f"Kendall  τ = {results['rank_correlation']['kendall_tau']:.4f}")
        except FileNotFoundError as e:
            logger.error(f"Missing artifact: {e}")
            logger.info("Run the training pipeline first to generate model + SHAP explainer.")
    else:
        # Default: build SHAP explainer from best_model.joblib + preprocessor.joblib
        logger.info("Building SHAP explainer from saved model and pipeline...")
        try:
            from src.preprocessing import get_feature_names, load_pipeline, prepare_data

            pipeline = load_pipeline()
            model = joblib.load(MODELS_DIR / "best_model.joblib")
            logger.info(f"Model loaded: {type(model).__name__}")

            df = pd.read_csv(RAW_CSV)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["Churn"] = (df["Churn"] == "Yes").astype(int)

            X_transformed, y, _ = prepare_data(df, fit_pipeline=False, pipeline=pipeline)
            feature_names = get_feature_names(pipeline)

            # Build and save explainer
            explainer = build_shap_explainer(model, X_transformed, feature_names)
            save_explainer(explainer)

            # Compute SHAP values and generate plots
            explanation = compute_shap_values(explainer, X_transformed, feature_names)
            plot_summary(explanation, save=True)
            logger.success("SHAP summary plot saved to figures/shap_summary.png")

            # Waterfall plots for a high-risk and a low-risk sample
            churn_idx = int(y.argmax())
            no_churn_idx = int((1 - y).argmax())
            plot_waterfall(explanation, idx=churn_idx, save=True, label="high_risk_example")
            plot_waterfall(explanation, idx=no_churn_idx, save=True, label="low_risk_example")
            logger.success("SHAP waterfall plots saved to figures/")

        except FileNotFoundError as e:
            logger.error(f"Missing artifact: {e}")
            logger.info(
                "Make sure you have run the training pipeline first: python -m src.train --quick"
            )
