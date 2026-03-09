"""False Negative (FN) analysis for churn prediction.

False Negatives are customers predicted 'No Churn' who actually churned.
This is the most costly error in churn prediction: FN customers receive NO
retention intervention, so their revenue is lost entirely.

In contrast, False Positives (customers flagged but who wouldn't have churned)
waste intervention budget — but they stay. The asymmetry between FN cost and
FP cost is the business case for threshold calibration.

This script:
    1. Loads a trained model + preprocessor (from joblib path or MLflow run,
       or auto-discovers the best model from the benchmark summary CSV)
    2. Evaluates on a held-out test split using the OOF-optimised threshold
    3. Segments predictions into TP / TN / FP / FN
    4. Profiles FN customers vs other segments (contract type, tenure,
       monthly charges, service count, payment method)
    5. Computes revenue at risk from missed churners (MonthlyCharges × months)
    6. Generates a PaCMAP embedding of the model's feature space coloured by
       segment (TP/TN/FP/FN) — reveals structure of errors in feature space
    7. Generates a predicted-probability boxplot per segment with the operating
       threshold annotated — reveals where FP/FN cluster on the score axis
    8. Generates dark-themed publication-quality plots
    9. Prints a business-oriented summary to the console

Methodology:
    Segmentation follows the 2×2 confusion matrix convention:
        TP — predicted churn, actually churned (correctly flagged)
        TN — predicted no churn, actually no churn (correctly cleared)
        FP — predicted churn, actually no churn (over-flagged, wasted budget)
        FN — predicted no churn, actually churned (missed — highest cost)

    Revenue at risk per FN = MonthlyCharges × `months` (default 3).
    The 3-month window is a conservative estimate of revenue lost before
    a churning customer is detected through natural attrition signals.
    Ref: Berry & Linoff (2004) "Data Mining Techniques", Ch. 9.

Usage:
    python -m src.fn_analysis                              # auto-discovers model
    python -m src.fn_analysis --model-path models/best_model.joblib
    python -m src.fn_analysis --run-id <mlflow-run-id>
    python -m src.fn_analysis --best                       # auto from benchmark CSV
    python -m src.fn_analysis --feature-set hill_climbing  # specify feature set
    python -m src.fn_analysis --threshold 0.35
    python -m src.fn_analysis --months 6                   # 6-month revenue window
    python -m src.fn_analysis --no-pacmap                  # skip PaCMAP embedding
    python -m src.fn_analysis --no-plots                   # skip all plots
"""

from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from joblib import load
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    ID_COL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RAW_CSV,
    TARGET_COL,
)
from src.eda import (
    AMBER,
    CORAL,
    CYAN,
    DARK_BG,
    DARK_CARD,
    GRID_COLOR,
    GREEN,
    MUTED_COLOR,
    TEXT_COLOR,
    VIOLET,
    set_dark_style,
)
from src.preprocessing import get_feature_names, prepare_data

# ── PaCMAP availability guard ──────────────────────────────────────────────────
try:
    import pacmap  # type: ignore
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False

# ── Constants ─────────────────────────────────────────────────────────────────

# Segment colour mapping: semantically meaningful colours.
# FN = coral (alarm — missed revenue), FP = amber (caution — wasted spend)
# TP = green (correct alarm), TN = cyan (correct clearance)
SEGMENT_COLORS = {
    "TP": GREEN,
    "TN": CYAN,
    "FP": AMBER,
    "FN": CORAL,
}

SEGMENT_LABELS = {
    "TP": "True Positive\n(flagged, churned)",
    "TN": "True Negative\n(cleared, stayed)",
    "FP": "False Positive\n(flagged, stayed)",
    "FN": "False Negative\n(missed, churned)",
}


# ── Best-model auto-discovery ─────────────────────────────────────────────────


def discover_best_model(
    summary_csv: Optional[Path] = None,
) -> tuple[str, str, str]:
    """Auto-discover (model_name, feature_set, run_id) from benchmark summary.

    Reads ``models/comprehensive_benchmark_summary.csv`` (falls back to
    ``models/benchmark_results.csv`` if the comprehensive one is absent),
    picks the row with the highest ``roc_auc_mean``, then queries MLflow for
    the corresponding run.

    Parameters
    ----------
    summary_csv : Path, optional
        Override path to summary CSV. Uses project default when None.

    Returns
    -------
    tuple[str, str, str]
        (model_name, feature_set, mlflow_run_id)

    Raises
    ------
    FileNotFoundError
        If no benchmark summary CSV is found.
    RuntimeError
        If no matching MLflow run is found for the best (model, feature_set).
    """
    # Locate summary CSV
    if summary_csv is None:
        comprehensive = MODELS_DIR / "comprehensive_benchmark_summary.csv"
        quick = MODELS_DIR / "benchmark_results.csv"
        if comprehensive.exists():
            summary_csv = comprehensive
            logger.info(f"Using comprehensive benchmark summary: {comprehensive}")
        elif quick.exists():
            summary_csv = quick
            logger.warning(
                f"Comprehensive benchmark not found; falling back to {quick}. "
                "Run the full benchmark for more reliable model selection."
            )
        else:
            raise FileNotFoundError(
                "No benchmark summary CSV found. Run the training pipeline first:\n"
                "  python -m src.train"
            )

    df = pd.read_csv(summary_csv)

    # Normalise column names — quick benchmark uses slightly different names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find AUC column (comprehensive uses roc_auc_mean; quick uses roc_auc)
    auc_col = next(
        (c for c in ["roc_auc_mean", "roc_auc", "test_roc_auc"] if c in df.columns),
        None,
    )
    if auc_col is None:
        raise ValueError(
            f"Cannot find AUC column in {summary_csv}. Columns: {list(df.columns)}"
        )

    # Identify model-name and feature-set columns
    model_col = next(
        (c for c in ["model_name", "model", "model_type"] if c in df.columns), None
    )
    feat_col = next(
        (c for c in ["feature_set", "features", "feature_set_name"] if c in df.columns),
        None,
    )

    # Filter out dummy baseline
    mask_non_dummy = ~df[model_col].str.lower().str.contains("dummy") if model_col else slice(None)
    df_models = df[mask_non_dummy].copy()

    best_row = df_models.loc[df_models[auc_col].idxmax()]
    model_name = str(best_row[model_col]) if model_col else "unknown"
    feature_set = str(best_row[feat_col]) if feat_col else "all"

    logger.info(
        f"Best model from benchmark: {model_name!r} / feature_set={feature_set!r} "
        f"({auc_col}={best_row[auc_col]:.4f})"
    )

    # Query MLflow for matching run
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found at "
            f"{MLFLOW_TRACKING_URI}. Run the training pipeline first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.model_type = '{model_name}' "
            f"AND tags.feature_set = '{feature_set}'"
        ),
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            f"No MLflow run found for model_type='{model_name}', "
            f"feature_set='{feature_set}'. "
            "Check that training completed successfully."
        )

    run_id = runs[0].info.run_id
    logger.info(f"MLflow run_id resolved: {run_id}")
    return model_name, feature_set, run_id


# ── Model Loading ─────────────────────────────────────────────────────────────


def load_model_and_pipeline(
    model_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple:
    """Load a trained classifier and the fitted preprocessing pipeline.

    Parameters
    ----------
    model_path : str, optional
        Path to a joblib-serialised sklearn estimator.  If None, attempts
        to load from MLflow using ``run_id``.
    run_id : str, optional
        MLflow run ID.  Used only when ``model_path`` is None.

    Returns
    -------
    tuple[estimator, Pipeline]
        Fitted classifier and preprocessing pipeline.

    Raises
    ------
    FileNotFoundError
        If neither a valid model_path nor a valid MLflow run can be resolved.
    """
    # Load preprocessing pipeline
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. "
            "Run `python -m src.train` first."
        )
    pipeline = load(PREPROCESSOR_PATH)
    logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")

    # Load classifier
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model, pipeline

    if run_id is not None:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from MLflow run {run_id}")
        return model, pipeline

    # Fall back: look for any saved model file in MODELS_DIR
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    # Exclude known non-classifier artifacts
    model_files = [
        f for f in model_files
        if "preprocessor" not in f.name and "shap" not in f.name
    ]
    if model_files:
        chosen = model_files[-1]  # most recently modified
        model = load(chosen)
        logger.info(f"Model auto-loaded from {chosen}")
        return model, pipeline

    raise FileNotFoundError(
        "No model found. Provide --model-path, --run-id, or --best, "
        "or run `python -m src.train` first."
    )


# ── Feature Selection (re-computation for non-'all' feature sets) ─────────────


def get_model_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Return the boolean mask and names for the features the model was trained on.

    For feature_set='all', returns a full-True mask.
    For 'mi' or 'hill_climbing', re-runs the selection on X_train.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features (n_train × n_features).
    y_train : np.ndarray
        Binary targets for training rows.
    feature_names : list[str]
        Names for each column of X_train.
    feature_set : str
        One of 'all', 'mi', 'hill_climbing'.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[mask, selected_names]
    """
    if feature_set == "all":
        mask = np.ones(len(feature_names), dtype=bool)
        return mask, feature_names

    from src.train import select_features_mi, select_features_hill_climbing

    if feature_set == "mi":
        _, mask, names = select_features_mi(
            X_train, y_train, feature_names, random_state=random_state
        )
        return mask, names

    if feature_set == "hill_climbing":
        _, mask, names = select_features_hill_climbing(
            X_train, y_train, feature_names, random_state=random_state
        )
        return mask, names

    logger.warning(
        f"Unknown feature_set '{feature_set}'; falling back to 'all'."
    )
    mask = np.ones(len(feature_names), dtype=bool)
    return mask, feature_names


# ── Prediction & Segmentation ─────────────────────────────────────────────────


def segment_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    X_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split predictions into TP, TN, FP, FN segments.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        True binary labels (1 = churn).
    y_pred : np.ndarray of shape (n,)
        Predicted binary labels at the operating threshold.
    y_proba : np.ndarray of shape (n,)
        Predicted churn probabilities (class 1).
    X_raw : pd.DataFrame
        Raw (un-preprocessed) feature DataFrame aligned with y_true.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'TP', 'TN', 'FP', 'FN'.
        Each value is a subset of X_raw with 'y_true', 'y_pred', 'y_proba'
        columns appended.
    """
    df = X_raw.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    tp_mask = (df["y_true"] == 1) & (df["y_pred"] == 1)
    tn_mask = (df["y_true"] == 0) & (df["y_pred"] == 0)
    fp_mask = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fn_mask = (df["y_true"] == 1) & (df["y_pred"] == 0)

    segments = {
        "TP": df[tp_mask].copy(),
        "TN": df[tn_mask].copy(),
        "FP": df[fp_mask].copy(),
        "FN": df[fn_mask].copy(),
    }

    for seg_name, seg_df in segments.items():
        logger.info(f"{seg_name}: {len(seg_df):,} customers")

    return segments


# ── Revenue at Risk ───────────────────────────────────────────────────────────


def compute_revenue_at_risk(
    fn_df: pd.DataFrame,
    months: int = 3,
) -> pd.DataFrame:
    """Estimate revenue at risk for each False Negative customer.

    Revenue at risk = MonthlyCharges × months.  This is a conservative lower
    bound — it ignores customer lifetime value beyond the window, which would
    further increase the true cost of a miss.

    Parameters
    ----------
    fn_df : pd.DataFrame
        FN segment DataFrame (from segment_predictions).
    months : int
        Revenue window in months.  Default 3.

    Returns
    -------
    pd.DataFrame
        fn_df with an additional 'revenue_at_risk' column.
    """
    if "MonthlyCharges" not in fn_df.columns:
        logger.warning(
            "'MonthlyCharges' column not found. Revenue at risk set to 0."
        )
        fn_df = fn_df.copy()
        fn_df["revenue_at_risk"] = 0.0
        return fn_df

    fn_df = fn_df.copy()
    fn_df["revenue_at_risk"] = fn_df["MonthlyCharges"] * months
    total = fn_df["revenue_at_risk"].sum()
    avg = fn_df["revenue_at_risk"].mean()
    logger.info(
        f"Revenue at risk ({months} months): "
        f"total = ${total:,.0f}, avg per FN = ${avg:,.0f}"
    )
    return fn_df


# ── Profiling ─────────────────────────────────────────────────────────────────


def profile_false_negatives(segments: dict[str, pd.DataFrame]) -> dict:
    """Statistical profile of FN customers vs other segments.

    Computes mean/median/std for numeric features and value counts for
    categorical features, across all four segments.

    Parameters
    ----------
    segments : dict[str, pd.DataFrame]

    Returns
    -------
    dict
        'numeric': pd.DataFrame (metric × segment)
        'contract': pd.DataFrame (contract type counts, FN only)
        'payment': pd.DataFrame (payment method counts, FN only)
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [c for c in numeric_cols if c in segments["FN"].columns]

    rows = []
    for seg_name, seg_df in segments.items():
        for col in numeric_cols:
            rows.append(
                {
                    "segment": seg_name,
                    "feature": col,
                    "mean": seg_df[col].mean(),
                    "median": seg_df[col].median(),
                    "std": seg_df[col].std(),
                }
            )
    numeric_profile = pd.DataFrame(rows)

    fn_df = segments["FN"]

    contract_counts = pd.Series(dtype=int)
    if "Contract" in fn_df.columns:
        contract_counts = fn_df["Contract"].value_counts()

    payment_counts = pd.Series(dtype=int)
    if "PaymentMethod" in fn_df.columns:
        payment_counts = fn_df["PaymentMethod"].value_counts()

    # Log key stats
    logger.info("=== FN Profile ===")
    fn_stats = numeric_profile[numeric_profile["segment"] == "FN"].set_index("feature")
    for feat, row in fn_stats.iterrows():
        logger.info(
            f"  {feat}: mean={row['mean']:.1f}, median={row['median']:.1f}, "
            f"std={row['std']:.1f}"
        )
    if len(contract_counts):
        logger.info(f"  Contract breakdown: {contract_counts.to_dict()}")

    return {
        "numeric": numeric_profile,
        "contract": contract_counts,
        "payment": payment_counts,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_fn_profile(
    segments: dict[str, pd.DataFrame],
    save: bool = True,
) -> plt.Figure:
    """Dark-themed 2×2 panel profiling the four prediction segments.

    Panel layout:
        (0,0) — Segment size bar chart
        (0,1) — Tenure distribution by segment (box plot)
        (1,0) — Monthly charges distribution by segment (box plot)
        (1,1) — Contract type breakdown within FN customers (bar chart)

    Parameters
    ----------
    segments : dict[str, pd.DataFrame]
    save : bool
        If True, saves figure to FIGURES_DIR/fn_profile.png.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle(
        "False Negative Analysis — Prediction Segment Profiles",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel (0,0): Segment sizes ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    seg_names = ["TP", "TN", "FP", "FN"]
    counts = [len(segments[s]) for s in seg_names]
    colors = [SEGMENT_COLORS[s] for s in seg_names]
    bars = ax0.bar(seg_names, counts, color=colors, edgecolor=DARK_BG, linewidth=1.2)
    ax0.set_title("Prediction Segment Sizes", color=TEXT_COLOR)
    ax0.set_ylabel("Count", color=TEXT_COLOR)
    ax0.set_xlabel("Segment", color=TEXT_COLOR)
    for bar, count in zip(bars, counts):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=10,
        )
    ax0.set_facecolor(DARK_CARD)
    ax0.tick_params(colors=MUTED_COLOR)

    # ── Panel (0,1): Tenure distribution ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    if "tenure" in segments["FN"].columns:
        tenure_data = [segments[s]["tenure"].dropna().values for s in seg_names]
        bp = ax1.boxplot(
            tenure_data,
            patch_artist=True,
            medianprops={"color": TEXT_COLOR, "linewidth": 2},
            whiskerprops={"color": MUTED_COLOR},
            capprops={"color": MUTED_COLOR},
            flierprops={
                "marker": "o",
                "markerfacecolor": MUTED_COLOR,
                "alpha": 0.3,
                "markersize": 3,
            },
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax1.set_xticklabels(seg_names, color=MUTED_COLOR)
        ax1.set_title("Tenure Distribution by Segment (months)", color=TEXT_COLOR)
        ax1.set_ylabel("Tenure (months)", color=TEXT_COLOR)
    else:
        ax1.text(
            0.5, 0.5, "tenure not available",
            ha="center", va="center", color=MUTED_COLOR, transform=ax1.transAxes,
        )
    ax1.set_facecolor(DARK_CARD)
    ax1.tick_params(colors=MUTED_COLOR)

    # ── Panel (1,0): Monthly charges distribution ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if "MonthlyCharges" in segments["FN"].columns:
        charges_data = [
            segments[s]["MonthlyCharges"].dropna().values for s in seg_names
        ]
        bp2 = ax2.boxplot(
            charges_data,
            patch_artist=True,
            medianprops={"color": TEXT_COLOR, "linewidth": 2},
            whiskerprops={"color": MUTED_COLOR},
            capprops={"color": MUTED_COLOR},
            flierprops={
                "marker": "o",
                "markerfacecolor": MUTED_COLOR,
                "alpha": 0.3,
                "markersize": 3,
            },
        )
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax2.set_xticklabels(seg_names, color=MUTED_COLOR)
        ax2.set_title("Monthly Charges by Segment ($)", color=TEXT_COLOR)
        ax2.set_ylabel("Monthly Charges ($)", color=TEXT_COLOR)
    else:
        ax2.text(
            0.5, 0.5, "MonthlyCharges not available",
            ha="center", va="center", color=MUTED_COLOR, transform=ax2.transAxes,
        )
    ax2.set_facecolor(DARK_CARD)
    ax2.tick_params(colors=MUTED_COLOR)

    # ── Panel (1,1): Contract type in FN customers ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    fn_df = segments["FN"]
    if "Contract" in fn_df.columns:
        contract_counts = fn_df["Contract"].value_counts()
        contract_colors = [CORAL, AMBER, VIOLET][: len(contract_counts)]
        contract_bars = ax3.bar(
            contract_counts.index,
            contract_counts.values,
            color=contract_colors,
            edgecolor=DARK_BG,
            linewidth=1.2,
        )
        ax3.set_title("Contract Type Breakdown — FN Customers", color=TEXT_COLOR)
        ax3.set_ylabel("Count", color=TEXT_COLOR)
        ax3.set_xlabel("Contract Type", color=TEXT_COLOR)
        for bar, count in zip(contract_bars, contract_counts.values):
            pct = count / len(fn_df) * 100
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(fn_df) * 0.005,
                f"{count:,}\n({pct:.0f}%)",
                ha="center",
                va="bottom",
                color=TEXT_COLOR,
                fontsize=9,
            )
    else:
        ax3.text(
            0.5, 0.5, "Contract column not available",
            ha="center", va="center", color=MUTED_COLOR, transform=ax3.transAxes,
        )
    ax3.set_facecolor(DARK_CARD)
    ax3.tick_params(colors=MUTED_COLOR)

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fn_profile.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.success(f"FN profile plot saved to {out_path}")

    return fig


def plot_revenue_at_risk(
    fn_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Dark-themed histogram of per-customer revenue at risk.

    Shows the distribution of MonthlyCharges × months across FN customers,
    annotated with the total revenue at risk.

    Parameters
    ----------
    fn_df : pd.DataFrame
        FN segment with 'revenue_at_risk' column (from compute_revenue_at_risk).
    save : bool
        If True, saves figure to FIGURES_DIR/fn_revenue_at_risk.png.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
    fig.suptitle(
        "False Negative Revenue at Risk",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
    )

    ax0 = axes[0]
    ax0.set_facecolor(DARK_CARD)
    if "revenue_at_risk" in fn_df.columns and len(fn_df) > 0:
        rar = fn_df["revenue_at_risk"].dropna()
        ax0.hist(
            rar,
            bins=30,
            color=CORAL,
            edgecolor=DARK_BG,
            linewidth=0.8,
            alpha=0.85,
        )
        total = rar.sum()
        median = rar.median()
        ax0.axvline(
            median,
            color=AMBER,
            linewidth=2,
            linestyle="--",
            label=f"Median: ${median:,.0f}",
        )
        ax0.set_title(
            f"Revenue at Risk per Missed Churner\nTotal: ${total:,.0f}",
            color=TEXT_COLOR,
        )
        ax0.set_xlabel("Revenue at Risk ($)", color=TEXT_COLOR)
        ax0.set_ylabel("# FN Customers", color=TEXT_COLOR)
        ax0.legend(facecolor=DARK_CARD, edgecolor=DARK_CARD, labelcolor=TEXT_COLOR)
        ax0.tick_params(colors=MUTED_COLOR)

        ax1 = axes[1]
        ax1.set_facecolor(DARK_CARD)
        sorted_rar = np.sort(rar.values)
        cumsum = np.cumsum(sorted_rar) / total * 100
        pct_customers = np.arange(1, len(sorted_rar) + 1) / len(sorted_rar) * 100
        ax1.plot(pct_customers, cumsum, color=CORAL, linewidth=2)
        ax1.plot(
            [0, 100], [0, 100],
            color=MUTED_COLOR, linewidth=1, linestyle="--", label="Perfect equality"
        )
        idx_80 = np.searchsorted(pct_customers, 80)
        if idx_80 < len(cumsum):
            ax1.axhline(
                cumsum[idx_80],
                color=AMBER,
                linewidth=1.5,
                linestyle=":",
                label=f"Top 20% → {100 - cumsum[idx_80]:.0f}% of revenue",
            )
            ax1.axvline(80, color=AMBER, linewidth=1.5, linestyle=":")
        ax1.set_title(
            "Cumulative Revenue at Risk\n(% customers vs % revenue)", color=TEXT_COLOR
        )
        ax1.set_xlabel("% of FN Customers (sorted by charges)", color=TEXT_COLOR)
        ax1.set_ylabel("% of Total Revenue at Risk", color=TEXT_COLOR)
        ax1.legend(facecolor=DARK_CARD, edgecolor=DARK_CARD, labelcolor=TEXT_COLOR)
        ax1.tick_params(colors=MUTED_COLOR)
    else:
        for ax in axes:
            ax.text(
                0.5, 0.5, "revenue_at_risk not available",
                ha="center", va="center", color=MUTED_COLOR, transform=ax.transAxes,
            )

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fn_revenue_at_risk.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.success(f"Revenue at risk plot saved to {out_path}")

    return fig


def plot_probability_boxplot(
    segments: dict[str, pd.DataFrame],
    threshold: float,
    save: bool = True,
) -> plt.Figure:
    """Predicted-probability boxplot per segment with threshold line.

    Shows 4 boxplots (TP / TN / FP / FN), y-axis = predicted churn
    probability.  The operating threshold is drawn as a horizontal dashed
    line.  Jittered scatter points are overlaid for sample-level visibility.

    Interpretation:
        - FP cluster near (but above) the threshold — marginal positives.
        - FN cluster near (but below) the threshold — near-misses.
        - Wider FN/FP boxes indicate the model is uncertain in those segments.

    Parameters
    ----------
    segments : dict[str, pd.DataFrame]
        Must contain a 'y_proba' column in each segment DataFrame.
    threshold : float
        Operating decision threshold (annotated as a dashed line).
    save : bool
        If True, saves figure to FIGURES_DIR/fn_proba_boxplot.png.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_CARD)

    seg_names = ["TP", "TN", "FP", "FN"]
    proba_data = [segments[s]["y_proba"].values for s in seg_names]
    colors = [SEGMENT_COLORS[s] for s in seg_names]
    labels = [SEGMENT_LABELS[s] for s in seg_names]

    bp = ax.boxplot(
        proba_data,
        patch_artist=True,
        medianprops={"color": TEXT_COLOR, "linewidth": 2.5},
        whiskerprops={"color": MUTED_COLOR, "linewidth": 1.2},
        capprops={"color": MUTED_COLOR, "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": MUTED_COLOR,
            "alpha": 0.2,
            "markersize": 3,
        },
        widths=0.5,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Jittered scatter overlay for sample-level visibility
    rng = np.random.default_rng(42)
    for i, (data, color) in enumerate(zip(proba_data, colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(data))
        ax.scatter(
            np.full(len(data), i) + jitter,
            data,
            color=color,
            alpha=0.25,
            s=8,
            zorder=3,
            linewidths=0,
        )

    # Threshold line
    ax.axhline(
        threshold,
        color=AMBER,
        linewidth=2,
        linestyle="--",
        zorder=5,
        label=f"Decision threshold = {threshold:.2f}",
    )

    ax.set_xticks(range(1, len(seg_names) + 1))
    ax.set_xticklabels(labels, color=MUTED_COLOR, fontsize=10)
    ax.set_ylabel("Predicted Churn Probability", color=TEXT_COLOR, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Predicted Probability Distribution by Segment\n"
        "(FP/FN cluster near the threshold — the model's decision boundary)",
        color=TEXT_COLOR,
        fontsize=13,
    )
    ax.legend(
        facecolor=DARK_CARD,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=10,
    )
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fn_proba_boxplot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.success(f"Probability boxplot saved to {out_path}")

    return fig


def plot_pacmap_embedding(
    X_model: np.ndarray,
    segments: dict[str, pd.DataFrame],
    feature_names: list[str],
    save: bool = True,
) -> Optional[plt.Figure]:
    """PaCMAP embedding of the model's feature space coloured by segment.

    Uses PaCMAP (Pairwise Controlled Manifold Approximation) to embed
    X_model (the exact feature matrix fed to predict_proba) into 2D.
    Points are coloured by their prediction segment (TP/TN/FP/FN).

    FN points appearing in regions occupied by TN (low-risk cluster) signal
    that those customers genuinely look like non-churners — the model may
    be inherently limited for that sub-population.

    FN points in high-churn regions suggest the threshold is too conservative
    and can be lowered to recover those misses without hurting precision.

    Ref: Wang et al. (2021) "Understanding How Dimension Reduction Tools Work:
    An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for
    Data Visualization." JMLR 22(201): 1-73.

    Parameters
    ----------
    X_model : np.ndarray of shape (n_test, n_model_features)
        Feature matrix fed to the model's predict_proba (after feature selection).
    segments : dict[str, pd.DataFrame]
        Segment DataFrames from segment_predictions.  Their index alignment
        with X_model rows is used to assign colours.
    feature_names : list[str]
        Names of the model features (used in subtitle).
    save : bool
        If True, saves figure to FIGURES_DIR/fn_pacmap.png.

    Returns
    -------
    plt.Figure or None
        None if pacmap is not installed.
    """
    if not HAS_PACMAP:
        logger.warning(
            "pacmap is not installed. Skipping PaCMAP embedding.\n"
            "  Install with:  pip install pacmap"
        )
        return None

    n_samples = X_model.shape[0]
    logger.info(f"Running PaCMAP on {n_samples:,} samples × {X_model.shape[1]} features...")

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
        verbose=False,
    )
    embedding = reducer.fit_transform(X_model)
    logger.info("PaCMAP embedding complete.")

    # Build a segment-label array aligned with X_model rows
    n_test = n_samples
    seg_labels = np.full(n_test, "OTHER", dtype=object)
    # Rebuild per-row indices from segments (they store original aligned position)
    seg_names = ["TP", "TN", "FP", "FN"]
    seg_idx: dict[str, np.ndarray] = {}

    # We need index positions. The segments were built from a reset_index copy,
    # so we use the integer index of each segment's DataFrame row.
    offset = 0
    for seg in segments.values():
        pass  # just to confirm structure

    # Reconstruct index from cumulative segment sizes in canonical order
    # (TP, TN, FP, FN all came from the same X_profile reset_index copy)
    all_segs_combined = pd.concat(
        [s.assign(_seg=name) for name, s in segments.items()], ignore_index=False
    )
    for seg_name in seg_names:
        idx = all_segs_combined[all_segs_combined["_seg"] == seg_name].index.values
        seg_labels[idx] = seg_name

    # Plot
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_CARD)

    # Draw TN/TP first (background), FP/FN on top (foreground)
    draw_order = ["TN", "TP", "FP", "FN"]
    alphas = {"TN": 0.25, "TP": 0.35, "FP": 0.65, "FN": 0.80}
    sizes = {"TN": 6, "TP": 8, "FP": 14, "FN": 16}

    for seg in draw_order:
        mask = seg_labels == seg
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=SEGMENT_COLORS[seg],
            alpha=alphas[seg],
            s=sizes[seg],
            linewidths=0,
            label=f"{seg} ({mask.sum():,})",
            zorder={"TN": 1, "TP": 2, "FP": 3, "FN": 4}[seg],
        )

    ax.set_title(
        f"PaCMAP Embedding — Model Feature Space\n"
        f"({X_model.shape[1]} features: {', '.join(feature_names[:5])}"
        + (f" … +{len(feature_names)-5} more" if len(feature_names) > 5 else "")
        + ")",
        color=TEXT_COLOR,
        fontsize=13,
    )
    ax.set_xlabel("PaCMAP-1", color=MUTED_COLOR)
    ax.set_ylabel("PaCMAP-2", color=MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    ax.legend(
        facecolor=DARK_CARD,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=10,
        markerscale=2,
    )

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fn_pacmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.success(f"PaCMAP plot saved to {out_path}")

    return fig


# ── Business Summary ──────────────────────────────────────────────────────────


def print_business_summary(
    segments: dict[str, pd.DataFrame],
    fn_df: pd.DataFrame,
    threshold: float,
    months: int,
) -> None:
    """Print a concise business-oriented summary to the console.

    Parameters
    ----------
    segments : dict
    fn_df : pd.DataFrame
        FN segment with 'revenue_at_risk' column.
    threshold : float
        Operating threshold used.
    months : int
        Revenue window in months.
    """
    n_total = sum(len(v) for v in segments.values())
    n_tp = len(segments["TP"])
    n_tn = len(segments["TN"])
    n_fp = len(segments["FP"])
    n_fn = len(segments["FN"])

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    fn_rate = n_fn / (n_fn + n_tp) if (n_fn + n_tp) > 0 else 0

    rar_total = fn_df["revenue_at_risk"].sum() if "revenue_at_risk" in fn_df.columns else 0
    rar_avg = fn_df["revenue_at_risk"].mean() if "revenue_at_risk" in fn_df.columns else 0

    logger.info("=" * 60)
    logger.info("FALSE NEGATIVE ANALYSIS — BUSINESS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Operating threshold  : {threshold:.2f}")
    logger.info(f"  Test set size        : {n_total:,} customers")
    logger.info(f"  TP (correctly caught): {n_tp:,} ({n_tp/n_total*100:.1f}%)")
    logger.info(f"  TN (correctly cleared): {n_tn:,} ({n_tn/n_total*100:.1f}%)")
    logger.info(f"  FP (false alarms)    : {n_fp:,} ({n_fp/n_total*100:.1f}%)")
    logger.info(f"  FN (missed churners) : {n_fn:,} ({n_fn/n_total*100:.1f}%)")
    logger.info(f"  Precision (of flags) : {precision:.3f}")
    logger.info(f"  Recall (churn caught): {recall:.3f}")
    logger.info(f"  FN rate (miss rate)  : {fn_rate:.3f}")
    logger.info("-" * 60)
    logger.info(f"  Revenue at risk ({months} months)")
    logger.info(f"    Total              : ${rar_total:,.0f}")
    logger.info(f"    Avg per FN customer: ${rar_avg:,.0f}")
    logger.info("=" * 60)

    if "Contract" in fn_df.columns:
        logger.info("FN customers by contract type:")
        for ctype, cnt in fn_df["Contract"].value_counts().items():
            logger.info(f"  {ctype}: {cnt:,} ({cnt/n_fn*100:.1f}%)")

    if "tenure" in fn_df.columns:
        logger.info(
            f"FN tenure: median={fn_df['tenure'].median():.0f} months, "
            f"mean={fn_df['tenure'].mean():.1f} months"
        )

    logger.info(
        "\nRecommendation: Lower the threshold to increase recall and reduce FN count, "
        "at the cost of more FP (wasted interventions). Use the profit curve in "
        "train.py to find the optimal operating point given your intervention budget."
    )


# ── High-Level Runner ─────────────────────────────────────────────────────────


def run_fn_analysis(
    model_path: Optional[str] = None,
    run_id: Optional[str] = None,
    threshold: Optional[float] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    months: int = 3,
    save_plots: bool = True,
    use_best: bool = False,
    feature_set: str = "all",
    no_pacmap: bool = False,
) -> dict:
    """End-to-end False Negative analysis pipeline.

    Parameters
    ----------
    model_path : str, optional
        Path to a joblib-serialised sklearn classifier.
    run_id : str, optional
        MLflow run ID to load model from.
    threshold : float, optional
        Decision threshold.  If None, uses 0.5.
    test_size : float
        Fraction of data to hold out as test set.
    random_state : int
        Random seed for reproducibility.
    months : int
        Revenue window in months for revenue-at-risk calculation.
    save_plots : bool
        If True, save plots to FIGURES_DIR.
    use_best : bool
        If True, auto-discover the best model from the benchmark summary CSV
        and load it from MLflow (overrides model_path/run_id).
    feature_set : str
        Feature set the model was trained on ('all', 'mi', 'hill_climbing').
        Used to reconstruct the feature mask for PaCMAP embedding.
    no_pacmap : bool
        If True, skip PaCMAP embedding even if pacmap is installed.

    Returns
    -------
    dict
        Keys: 'segments', 'fn_df', 'profile', 'threshold'
    """
    # ── 1. Load raw data ──────────────────────────────────────────────────────
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_CSV}. "
            "Run `python -m src.download_data` first."
        )
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    df = df.dropna(subset=[TARGET_COL])
    logger.info(f"Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    logger.info(f"Split: train={len(train_df):,}, test={len(test_df):,}")

    # Keep raw test features before preprocessing (for profiling)
    X_raw_test = test_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore").reset_index(drop=True)
    y_test = test_df[TARGET_COL].values

    # ── 3. Auto-discover best model if requested ──────────────────────────────
    if use_best and run_id is None and model_path is None:
        try:
            _model_name, feature_set, run_id = discover_best_model()
            logger.info(
                f"--best resolved: model={_model_name}, "
                f"feature_set={feature_set}, run_id={run_id}"
            )
        except Exception as exc:
            logger.warning(
                f"Best-model auto-discovery failed ({exc}). "
                "Falling back to local model auto-discovery."
            )

    # ── 4. Load model + pipeline ──────────────────────────────────────────────
    model, pipeline = load_model_and_pipeline(model_path=model_path, run_id=run_id)

    # ── 5. Preprocess ─────────────────────────────────────────────────────────
    X_train_proc, y_train_proc, _ = prepare_data(train_df, fit_pipeline=False, pipeline=pipeline)
    X_test_proc, _, _ = prepare_data(test_df, fit_pipeline=False, pipeline=pipeline)
    feature_names_all = get_feature_names(pipeline)

    # ── 6. Feature selection mask (for non-'all' feature sets) ────────────────
    feat_mask, model_feature_names = get_model_features(
        X_train_proc,
        y_train_proc,
        feature_names_all,
        feature_set=feature_set,
        random_state=random_state,
    )
    # X_model: the exact features fed to predict_proba
    X_model_test = X_test_proc[:, feat_mask]
    model_feature_names_masked = [n for n, m in zip(feature_names_all, feat_mask) if m]

    # Reconcile feature count: if the model was trained on fewer features than
    # the current preprocessor produces (e.g. a feature was added after training),
    # silently trim to the first n_features_in_ columns so predict_proba succeeds.
    # Note: CatBoost reports n_features_in_=0 (doesn't follow sklearn convention),
    # so we only trim when expected_n is a positive integer.
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None and expected_n > 0 and X_model_test.shape[1] != expected_n:
        logger.warning(
            f"Feature count mismatch: model expects {expected_n}, "
            f"preprocessor produced {X_model_test.shape[1]}. "
            f"Trimming to first {expected_n} features."
        )
        X_model_test = X_model_test[:, :expected_n]
        model_feature_names = model_feature_names_masked[:expected_n]
    else:
        model_feature_names = model_feature_names_masked

    logger.info(
        f"Feature set '{feature_set}': "
        f"{X_model_test.shape[1]} / {len(feat_mask)} features used by model"
    )

    # ── 7. Predict ────────────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_model_test)[:, 1]

    if threshold is None:
        threshold = 0.5
        logger.info("No threshold provided — using default 0.5")
    y_pred = (y_proba >= threshold).astype(int)

    # ── 8. Segment ────────────────────────────────────────────────────────────
    raw_cols_for_profiling = [
        c for c in ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod"]
        if c in test_df.columns
    ]
    X_profile = test_df[raw_cols_for_profiling].reset_index(drop=True)

    segments = segment_predictions(y_test, y_pred, y_proba, X_profile)

    # ── 9. Revenue at risk ────────────────────────────────────────────────────
    fn_df = compute_revenue_at_risk(segments["FN"], months=months)
    segments["FN"] = fn_df

    # ── 10. Profile ───────────────────────────────────────────────────────────
    profile = profile_false_negatives(segments)

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    if save_plots:
        plot_fn_profile(segments, save=True)
        plot_revenue_at_risk(fn_df, save=True)
        plot_probability_boxplot(segments, threshold=threshold, save=True)

        if not no_pacmap:
            plot_pacmap_embedding(
                X_model_test,
                segments,
                model_feature_names,
                save=True,
            )
        else:
            logger.info("PaCMAP embedding skipped (--no-pacmap).")

    # ── 12. Business summary ──────────────────────────────────────────────────
    print_business_summary(segments, fn_df, threshold=threshold, months=months)

    return {
        "segments": segments,
        "fn_df": fn_df,
        "profile": profile,
        "threshold": threshold,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="False Negative analysis for churn prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a joblib-serialised sklearn classifier (e.g. models/best_model.joblib). "
             "If omitted, auto-discovers the latest .joblib in models/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to load model from. Used only when --model-path is not set.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help=(
            "Auto-discover the best model from models/comprehensive_benchmark_summary.csv "
            "(falls back to models/benchmark_results.csv) and load it from MLflow."
        ),
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=["all", "mi", "hill_climbing"],
        help=(
            "Feature set the model was trained on. Used to reconstruct the feature "
            "mask for PaCMAP embedding and predict_proba input."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for binary predictions. Default: 0.5.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out as test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split reproducibility.",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Revenue window (months) for revenue-at-risk calculation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots to figures/.",
    )
    parser.add_argument(
        "--no-pacmap",
        action="store_true",
        help="Skip PaCMAP embedding plot (useful when pacmap is not installed).",
    )

    args = parser.parse_args()

    run_fn_analysis(
        model_path=args.model_path,
        run_id=args.run_id,
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.seed,
        months=args.months,
        save_plots=not args.no_plots,
        use_best=args.best,
        feature_set=args.feature_set,
        no_pacmap=args.no_pacmap,
    )
