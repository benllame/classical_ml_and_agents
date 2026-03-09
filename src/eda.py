"""EDA utilities — business-oriented exploration of Telco Churn dataset.

Functions used by notebook_02_eda.ipynb and reusable across the project.

Methodology: this module follows the "business understanding first" principle
from the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework.
Every visualization and metric is chosen to answer a specific business question
(e.g., "What is the revenue impact of churn?", "Which contract types are most
at risk?") before diving into statistical or ML modelling. This ensures EDA
outputs are directly actionable by stakeholders.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.config import FIGURES_DIR, RAW_CSV

# ── Dark theme for publication-quality plots ──────────────────────────────────
# Publication-quality dark palette designed for data science portfolio
# presentations. Dark backgrounds reduce visual fatigue during extended
# analysis sessions and provide better contrast for colored data points.
# Color choices follow colorblind-safe principles: cyan/coral as primary
# diverging pair, avoiding pure red/green.
DARK_BG = "#0a0f18"
DARK_CARD = "#0f1621"
GRID_COLOR = "#172030"
TEXT_COLOR = "#d8e8f5"
MUTED_COLOR = "#527a96"
CYAN = "#00e5ff"
GREEN = "#2dff7f"
AMBER = "#ffbe3d"
CORAL = "#ff4f6b"
VIOLET = "#a374ff"
PINK = "#ff6eb4"

# Index 0=No Churn (cyan, cool tone -> stable),
# 1=Churn (coral, warm tone -> alarm).
# Color semantics help viewers intuitively associate churn with warning.
PALETTE_CHURN = [CYAN, CORAL]


def set_dark_style() -> None:
    """Apply custom dark theme to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_CARD,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.3,
            "legend.facecolor": DARK_CARD,
            "legend.edgecolor": GRID_COLOR,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )


def load_raw_data() -> pd.DataFrame:
    """Load the raw Telco Churn CSV with basic type fixes.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with TotalCharges coerced to numeric and Churn as binary.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_CSV}. " "Run `python src/download_data.py` first."
        )

    df = pd.read_csv(RAW_CSV)

    # TotalCharges has whitespace strings for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns from {RAW_CSV.name}")
    return df


def churn_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute churn rate and counts.

    Returns
    -------
    pd.DataFrame
        Summary with columns: Churn, Count, Percentage.
    """
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)
    counts["Churn"] = counts["Churn"].map({0: "No Churn", 1: "Churn"})
    return counts


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate bias-corrected Cramer's V for two categorical series.

    Uses the bias correction from Bergsma (2013), "A bias-correction for
    Cramer's V and Tschuprow's T", *Journal of the Korean Statistical
    Society*, 42(3), 323-328.

    Formula
    -------
    V = sqrt(phi2_corrected / min(k_corr - 1, r_corr - 1))

    where:
        phi2_corrected = max(0, phi2 - (k-1)(r-1) / (n-1))
        phi2           = chi2 / n
        r_corr         = r - (r-1)^2 / (n-1)
        k_corr         = k - (k-1)^2 / (n-1)

    The bias correction adjusts for the positive bias in chi-squared that
    inflates V for small samples. Without correction, V would be
    systematically overestimated, especially for contingency tables with
    many categories.

    V is in [0, 1] where 0 = no association, 1 = perfect association.
    """
    from scipy.stats import chi2_contingency

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


def compute_cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute a pairwise Cramer's V matrix for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list[str]
        Categorical column names.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of Cramer's V values.
    """
    n = len(cols)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = cramers_v(df[cols[i]], df[cols[j]])
            matrix[i, j] = v
            matrix[j, i] = v
    return pd.DataFrame(matrix, index=cols, columns=cols)


def estimate_ltv(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate Customer Lifetime Value per segment.

    Simplified LTV = MonthlyCharges * E[tenure]. See intervention_engine.py
    for the full decision-theoretic LTV used in ROI calculations. This
    version is for EDA summary — grouping by Contract x Churn shows how
    LTV differs across segments, giving stakeholders an intuitive sense
    of the monetary value at risk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain MonthlyCharges, tenure, Contract, Churn columns.

    Returns
    -------
    pd.DataFrame
        LTV summary grouped by Contract type and Churn status.
    """
    ltv = (
        df.groupby(["Contract", "Churn"])
        .agg(
            avg_monthly=("MonthlyCharges", "mean"),
            avg_tenure=("tenure", "mean"),
            count=("customerID", "count"),
        )
        .reset_index()
    )
    ltv["estimated_ltv"] = (ltv["avg_monthly"] * ltv["avg_tenure"]).round(2)
    ltv["Churn"] = ltv["Churn"].map({0: "No Churn", 1: "Churn"})
    return ltv


def monthly_churn_loss(df: pd.DataFrame) -> dict:
    """Estimate monthly revenue loss attributable to churn.

    Annual projection assumes constant churn rate — a conservative lower
    bound since churn often accelerates (e.g., seasonal effects, competitor
    promotions). This metric is key for C-level reporting: it translates
    churn % into revenue impact, making the business case for retention
    investment concrete.

    Returns
    -------
    dict
        Keys: churned_customers, monthly_loss, annual_loss_projected.
    """
    churned = df[df["Churn"] == 1]
    monthly_loss = churned["MonthlyCharges"].sum()
    return {
        "churned_customers": len(churned),
        "monthly_loss": round(monthly_loss, 2),
        "annual_loss_projected": round(monthly_loss * 12, 2),
        "avg_monthly_charge_churned": round(churned["MonthlyCharges"].mean(), 2),
    }


def plot_churn_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart of churn distribution."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts().sort_index()
    bars = ax.bar(
        ["No Churn", "Churn"],
        counts.values,
        color=PALETTE_CHURN,
        edgecolor="none",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
    ax.set_title("Distribución de Churn")
    ax.set_ylabel("Clientes")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_tenure_survival(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Pseudo-Kaplan-Meier survival curve showing churn by tenure.

    Not a true survival analysis (would require time-to-event data with
    censoring), but approximates retention curves by plotting the empirical
    CDF complement for each churn group. The 12-month vertical line
    highlights the critical first-year retention period — telco industry
    data shows most churn occurs within the first 12 months (the
    "onboarding danger zone").
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color, name in [(0, CYAN, "No Churn"), (1, CORAL, "Churn")]:
        subset = df[df["Churn"] == label]["tenure"].sort_values()
        survival = 1 - np.arange(1, len(subset) + 1) / len(subset)
        ax.step(subset.values, survival, where="post", color=color, label=name, linewidth=1.5)

    ax.set_title("Curva de Supervivencia por Tenure")
    ax.set_xlabel("Tenure (meses)")
    ax.set_ylabel("Proporción restante")
    ax.legend(framealpha=0.8)
    ax.grid(alpha=0.2)
    ax.axvline(x=12, color=AMBER, linestyle="--", alpha=0.5, label="12 meses")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "tenure_survival.png", dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_charges_by_churn(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """KDE + histogram overlay of MonthlyCharges by churn status.

    Reveals the bimodal distribution typical of telco: low-spend customers
    (basic plans) and high-spend customers (premium bundles). Churners
    skew toward high monthly charges, suggesting price sensitivity as a
    churn driver — a pattern consistently reported in telco literature.
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color, name in [(0, CYAN, "No Churn"), (1, CORAL, "Churn")]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        ax.hist(
            subset, bins=40, alpha=0.5, color=color, label=name, density=True, edgecolor="none"
        )
        subset.plot.kde(ax=ax, color=color, linewidth=1.5)

    ax.set_title("Distribución de MonthlyCharges por Churn")
    ax.set_xlabel("MonthlyCharges ($)")
    ax.set_ylabel("Densidad")
    ax.legend(framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "monthly_charges_by_churn.png", dpi=150, bbox_inches="tight")
    return fig


def plot_contract_churn_rate(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Churn rate by contract type.

    Contract type is consistently the #1 predictor across telco churn
    studies (Ahn et al. 2006, Vafeiadis et al. 2015). Month-to-month
    contracts have dramatically higher churn because they impose no
    switching cost — the customer can leave at any billing cycle without
    penalty, making retention entirely dependent on perceived value.
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    rates = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
    colors = [CORAL, AMBER, GREEN]
    bars = ax.barh(
        rates.index, rates.values, color=colors[: len(rates)], height=0.5, edgecolor="none"
    )

    for bar, val in zip(bars, rates.values, strict=False):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.set_title("Tasa de Churn por Tipo de Contrato")
    ax.set_xlabel("Churn Rate (%)")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "contract_churn_rate.png", dpi=150, bbox_inches="tight")
    return fig


def plot_cramers_heatmap(df: pd.DataFrame, cols: list[str], save: bool = True) -> plt.Figure:
    """Heatmap of Cramer's V associations."""
    set_dark_style()
    matrix = compute_cramers_v_matrix(df, cols)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=1,
    )
    ax.set_title("Cramer's V — Asociación entre Variables Categóricas")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "cramers_v_heatmap.png", dpi=150, bbox_inches="tight")
    return fig
