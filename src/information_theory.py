"""Information-theoretic analysis for churn prediction.

We use information theory as a model-agnostic way to measure how much
each feature tells us about churn. This runs independently of any ML
model and helps validate feature choices.

Provides:
- Shannon entropy, joint entropy, conditional entropy
- Mutual Information (MI) and conditional MI
- Interaction Information (synergy / redundancy detection)
- MI-based feature selection
- MI vs SHAP ranking comparison
- Dark-themed visualizations

Usage:
    python src/information_theory.py            # full analysis with plots
    python src/information_theory.py --mi-only  # MI matrix only
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer

from src.config import (
    BINARY_FEATURES,
    FIGURES_DIR,
    ID_COL,
    MI_N_BINS,
    MI_N_NEIGHBORS,
    MI_THRESHOLD,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_CSV,
    TARGET_COL,
)
from src.eda import (
    AMBER,
    CORAL,
    CYAN,
    DARK_BG,
    DARK_CARD,
    GREEN,
    GRID_COLOR,
    MUTED_COLOR,
    TEXT_COLOR,
    VIOLET,
    set_dark_style,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
# Imported from config.py: MI_N_BINS, MI_THRESHOLD, MI_N_NEIGHBORS
# Re-exported here for backward compatibility with direct module usage
N_BINS_DISCRETIZE = MI_N_BINS


# ── Core Entropy Functions ───────────────────────────────────────────────────


def shannon_entropy(x: np.ndarray, base: float = 2.0) -> float:
    """Compute Shannon entropy H(X) of a discrete variable.

    H(X) = -sum p(x) log p(x). Returns bits when base=2.
    H(X)=0 means one outcome is certain; H(X)=log(|X|) means all
    outcomes are equally likely.

    Parameters
    ----------
    x : np.ndarray
        1D array of discrete values (categorical or pre-binned).
    base : float
        Logarithm base. 2.0 -> bits, e -> nats.

    Returns
    -------
    float
        H(X) in the specified base.
    """
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))


def joint_entropy(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    """Compute joint entropy H(X, Y).

    H(X,Y) <= H(X) + H(Y) with equality iff X,Y are independent.
    The gap H(X) + H(Y) - H(X,Y) = I(X;Y) is the mutual information.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of discrete values (same length).
    base : float
        Logarithm base.

    Returns
    -------
    float
        H(X, Y)
    """
    xy = np.column_stack([x, y])
    _, counts = np.unique(xy, axis=0, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))


def conditional_entropy(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    """Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).

    Chain rule of entropy. Measures residual uncertainty about X after
    observing Y. H(X|Y) = 0 means Y completely determines X.

    Parameters
    ----------
    x : np.ndarray
        Variable whose conditional entropy we compute.
    y : np.ndarray
        Conditioning variable.
    base : float
        Logarithm base.

    Returns
    -------
    float
        H(X|Y)
    """
    return joint_entropy(x, y, base) - shannon_entropy(y, base)


def mutual_information_discrete(
    x: np.ndarray, y: np.ndarray, base: float = 2.0
) -> float:
    """Compute Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y).

    Measures how much knowing Y reduces uncertainty about X (and vice
    versa). Unlike correlation, MI captures non-linear dependencies.
    The max(0, ...) clamp handles floating-point rounding errors.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of discrete values.
    base : float
        Logarithm base.

    Returns
    -------
    float
        I(X;Y) — always non-negative.
    """
    mi = shannon_entropy(x, base) + shannon_entropy(y, base) - joint_entropy(x, y, base)
    return max(0.0, mi)  # clamp rounding errors


def conditional_mutual_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, base: float = 2.0
) -> float:
    """Compute Conditional Mutual Information I(X;Y|Z).

    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Measures how much X tells us about Y beyond what Z already tells us.
    If I(X;Y|Z) ≈ 0, then X is redundant given Z.

    Parameters
    ----------
    x, y, z : np.ndarray
        1D arrays of discrete values (same length).
    base : float
        Logarithm base.

    Returns
    -------
    float
        I(X;Y|Z)
    """
    xz = np.column_stack([x, z])
    yz = np.column_stack([y, z])
    xyz = np.column_stack([x, y, z])

    # Convert to string tuples for unique counting
    def _hash_rows(arr):
        if arr.ndim == 1:
            return arr.astype(str)
        return np.array(["_".join(map(str, row)) for row in arr])

    h_xz = shannon_entropy(_hash_rows(xz), base)
    h_yz = shannon_entropy(_hash_rows(yz), base)
    h_xyz = shannon_entropy(_hash_rows(xyz), base)
    h_z = shannon_entropy(z, base)

    cmi = h_xz + h_yz - h_xyz - h_z
    return max(0.0, cmi)


def interaction_information(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, base: float = 2.0
) -> float:
    """Compute Interaction Information II(X1; X2; Y).

    II = I(X1,X2; Y) - I(X1; Y) - I(X2; Y)

    This is the only information-theoretic measure that can be negative.

    Interpretation:
        II > 0 -> synergy: X1 and X2 together tell us more about Y than
                  they do individually
        II < 0 -> redundancy: X1 and X2 share overlapping information
                  about Y
        II ~ 0 -> independence: X1 and X2 contribute independently

    Parameters
    ----------
    x1, x2, y : np.ndarray
        1D arrays of discrete values.
    base : float
        Logarithm base.

    Returns
    -------
    float
        II(X1; X2; Y) — can be negative (unlike MI).
    """
    x1x2 = np.array(["_".join(map(str, pair)) for pair in zip(x1, x2)])
    mi_joint = mutual_information_discrete(x1x2, y, base)
    mi_x1 = mutual_information_discrete(x1, y, base)
    mi_x2 = mutual_information_discrete(x2, y, base)
    return mi_joint - mi_x1 - mi_x2


# ── Discretization Helper ───────────────────────────────────────────────────


def discretize_continuous(
    X: pd.DataFrame,
    numeric_cols: list[str],
    n_bins: int = N_BINS_DISCRETIZE,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Discretize continuous columns for entropy-based computations.

    Entropy estimators need discrete data. Quantile strategy gives roughly
    equal bin counts, which avoids empty bins.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with at least the `numeric_cols`.
    numeric_cols : list[str]
        Columns to discretize.
    n_bins : int
        Number of bins.
    strategy : str
        'quantile', 'uniform', or 'kmeans'.

    Returns
    -------
    pd.DataFrame
        Copy with discretized numeric columns.
    """
    X_disc = X.copy()
    for col in numeric_cols:
        if col not in X_disc.columns:
            continue
        vals = X_disc[col].values.reshape(-1, 1)
        # Handle NaN: fill with median before binning
        mask = np.isnan(vals.ravel())
        if mask.any():
            median_val = np.nanmedian(vals)
            vals[mask.reshape(-1, 1)] = median_val

        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        X_disc[col] = binner.fit_transform(vals).ravel().astype(int)

    return X_disc


# ── MI-Based Feature Selection ──────────────────────────────────────────────


def compute_mi_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    n_neighbors: int = MI_N_NEIGHBORS,
    random_state: int = 42,
) -> pd.Series:
    """Compute Mutual Information between each feature and the target.

    Uses sklearn's mutual_info_classif, which handles continuous features
    with k-NN estimation and discrete features with counting.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (pre-engineering, raw categorical + numeric).
    y : np.ndarray
        Binary target.
    n_neighbors : int
        k for k-NN MI estimation.
    random_state : int
        For reproducibility.

    Returns
    -------
    pd.Series
        MI scores indexed by feature name, sorted descending.
    """
    # Tell sklearn which features are categorical so it uses counting
    # for those and k-NN density estimation for continuous ones.
    discrete_mask = [
        col in BINARY_FEATURES + NOMINAL_FEATURES for col in X.columns
    ]

    # Encode categoricals as integers for sklearn
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            X_encoded[col] = X_encoded[col].astype("category").cat.codes

    # Fill NaN
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

    mi_scores = mutual_info_classif(
        X_encoded,
        y,
        discrete_features=discrete_mask,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    result = pd.Series(mi_scores, index=X.columns, name="MI_score").sort_values(ascending=False)
    logger.info(f"MI scores computed for {len(result)} features")
    return result


def compute_conditional_mi_matrix(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 10,
    n_bins: int = N_BINS_DISCRETIZE,
) -> pd.DataFrame:
    """Compute conditional MI: I(Xi; Y | Xj) for top-k features.

    The diagonal stores marginal MI I(Xi;Y). Off-diagonal (i,j) stores
    I(Xi;Y|Xj) = 'how much does Xi tell about Y if we already know
    Xj?'. If off-diagonal << diagonal, the feature is largely
    redundant given the other.

    This reveals which features become redundant once another feature
    is already known. A low I(Xi; Y | Xj) means Xi is redundant given Xj.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary target.
    top_k : int
        Number of top MI features to include (limits computation).
    n_bins : int
        Bins for discretizing continuous features.

    Returns
    -------
    pd.DataFrame
        Square matrix where entry (i, j) = I(Xi; Y | Xj).
    """
    # First get top features by MI
    mi_scores = compute_mi_scores(X, y)
    top_features = mi_scores.head(top_k).index.tolist()

    # Discretize everything
    numeric_cols = [c for c in top_features if c in NUMERIC_FEATURES]
    X_sub = X[top_features].copy()

    # Encode categoricals
    for col in X_sub.columns:
        if X_sub[col].dtype == "object":
            X_sub[col] = X_sub[col].astype("category").cat.codes

    X_disc = discretize_continuous(X_sub, numeric_cols, n_bins)
    X_disc = X_disc.fillna(0).astype(int)

    # Compute CMI matrix
    cmi_matrix = pd.DataFrame(
        np.zeros((len(top_features), len(top_features))),
        index=top_features,
        columns=top_features,
    )

    y_arr = y.astype(int)

    for i, fi in enumerate(top_features):
        for j, fj in enumerate(top_features):
            if i == j:
                # Diagonal: standard MI
                cmi_matrix.loc[fi, fj] = mutual_information_discrete(
                    X_disc[fi].values, y_arr
                )
            else:
                cmi_matrix.loc[fi, fj] = conditional_mutual_information(
                    X_disc[fi].values, y_arr, X_disc[fj].values
                )

    logger.info(f"Conditional MI matrix computed for top {top_k} features")
    return cmi_matrix


def compute_interaction_matrix(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 8,
    n_bins: int = N_BINS_DISCRETIZE,
) -> pd.DataFrame:
    """Compute Interaction Information II(Xi; Xj; Y) for top-k features.

    Symmetric matrix: II(Xi;Xj;Y). We only compute the upper triangle
    (i < j) and mirror. Diagonal is 0 by definition (a feature can't
    synergize with itself).

    Reveals synergistic pairs (II > 0) vs redundant pairs (II < 0).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary target.
    top_k : int
        Number of top MI features to analyze.
    n_bins : int
        Bins for discretizing continuous features.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix where entry (i, j) = II(Xi; Xj; Y).
    """
    mi_scores = compute_mi_scores(X, y)
    top_features = mi_scores.head(top_k).index.tolist()

    numeric_cols = [c for c in top_features if c in NUMERIC_FEATURES]
    X_sub = X[top_features].copy()

    for col in X_sub.columns:
        if X_sub[col].dtype == "object":
            X_sub[col] = X_sub[col].astype("category").cat.codes

    X_disc = discretize_continuous(X_sub, numeric_cols, n_bins)
    X_disc = X_disc.fillna(0).astype(int)

    ii_matrix = pd.DataFrame(
        np.zeros((len(top_features), len(top_features))),
        index=top_features,
        columns=top_features,
    )

    y_arr = y.astype(int)

    for i, fi in enumerate(top_features):
        for j, fj in enumerate(top_features):
            if i >= j:
                continue  # symmetric + diagonal is 0
            ii_val = interaction_information(
                X_disc[fi].values, X_disc[fj].values, y_arr
            )
            ii_matrix.loc[fi, fj] = ii_val
            ii_matrix.loc[fj, fi] = ii_val

    logger.info(f"Interaction Information matrix computed for top {top_k} features")
    return ii_matrix


def select_features_mi(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = MI_THRESHOLD,
    n_neighbors: int = MI_N_NEIGHBORS,
) -> dict[str, Any]:
    """MI-based feature selection with redundancy filtering.

    Two-stage selection:
    1. Remove features below the MI threshold (noise filter).
    2. Flag pairs where one feature becomes largely redundant given
       another (redundancy ratio > 50%). We report rather than auto-drop
       these — human review is better for marginal cases.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary target.
    threshold : float
        Minimum MI to keep a feature.
    n_neighbors : int
        k for k-NN MI estimation.

    Returns
    -------
    dict
        {
            "mi_scores": pd.Series,
            "selected_features": list[str],
            "dropped_features": list[str],
            "redundancy_report": list[dict],
        }
    """
    mi_scores = compute_mi_scores(X, y, n_neighbors)

    # Split by threshold
    selected = mi_scores[mi_scores >= threshold].index.tolist()
    dropped = mi_scores[mi_scores < threshold].index.tolist()

    if dropped:
        logger.info(
            f"Dropped {len(dropped)} features below MI threshold ({threshold}): {dropped}"
        )

    # Redundancy analysis on selected features
    redundancy_report = []
    if len(selected) > 1:
        numeric_in_selected = [c for c in selected if c in NUMERIC_FEATURES]
        X_sel = X[selected].copy()
        for col in X_sel.columns:
            if X_sel[col].dtype == "object":
                X_sel[col] = X_sel[col].astype("category").cat.codes

        X_disc = discretize_continuous(X_sel, numeric_in_selected)
        X_disc = X_disc.fillna(0).astype(int)
        y_arr = y.astype(int)

        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                fi, fj = selected[i], selected[j]
                # MI between the two features
                mi_pair = mutual_information_discrete(
                    X_disc[fi].values, X_disc[fj].values
                )
                # CMI: how much does fi tell about Y beyond fj?
                cmi_val = conditional_mutual_information(
                    X_disc[fi].values, y_arr, X_disc[fj].values
                )
                # If CMI is much lower than marginal MI, fi is partially redundant
                marginal_mi = float(mi_scores[fi])
                if marginal_mi > 0:
                    redundancy_ratio = 1.0 - (cmi_val / marginal_mi) if marginal_mi > 0 else 0.0
                else:
                    redundancy_ratio = 0.0

                if redundancy_ratio > 0.5:
                    redundancy_report.append(
                        {
                            "feature_a": fi,
                            "feature_b": fj,
                            "MI(a,b)": round(mi_pair, 4),
                            "I(a;Y)": round(marginal_mi, 4),
                            "I(a;Y|b)": round(cmi_val, 4),
                            "redundancy_ratio": round(redundancy_ratio, 3),
                            "interpretation": (
                                f"{fi} pierde {redundancy_ratio:.0%} de su información "
                                f"sobre Churn cuando ya se conoce {fj}"
                            ),
                        }
                    )

    logger.info(
        f"Feature selection: {len(selected)} selected, {len(dropped)} dropped, "
        f"{len(redundancy_report)} redundant pairs flagged"
    )

    return {
        "mi_scores": mi_scores,
        "selected_features": selected,
        "dropped_features": dropped,
        "redundancy_report": redundancy_report,
    }


# ── MI vs SHAP Comparison ───────────────────────────────────────────────────


def compare_mi_vs_shap(
    mi_scores: pd.Series,
    shap_importance: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """Compare MI-based and SHAP-based feature importance rankings.

    When rankings diverge it can reveal features the model underweights
    despite high MI, or features the model overweights despite low MI.
    Agreement between the two is strong evidence of importance.

    Parameters
    ----------
    mi_scores : pd.Series
        MI scores indexed by feature name.
    shap_importance : pd.Series
        Mean |SHAP| values indexed by feature name.
    top_n : int
        Top features to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table with ranks, scores, and divergence analysis.
    """
    # MI uses raw feature names. SHAP names include pipeline prefixes
    # (num__, bin__, nom__) and OHE suffixes. We strip prefixes and
    # aggregate OHE levels back to the original feature name.
    common_features = mi_scores.index.intersection(shap_importance.index)

    if len(common_features) < 3:
        # Try prefix-based mapping: SHAP names may have prefixes like
        # "num__", "bin__", "nom__". We strip them and aggregate OHE
        # levels back to the original feature name.
        logger.info(
            "Direct name match found < 3 features; "
            "attempting prefix-based mapping"
        )
        shap_to_raw = {}
        for shap_name in shap_importance.index:
            # Strip prefixes like num__, bin__, nom__
            raw = shap_name
            for prefix in ["num__", "bin__", "nom__"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                    break
            # For OHE features like nom__Contract_Two year, map to Contract
            if "_" in raw:
                base = raw.split("_")[0]
                if base in mi_scores.index:
                    # Aggregate: sum SHAP for all OHE levels of the same original feature
                    if base not in shap_to_raw:
                        shap_to_raw[base] = 0.0
                    shap_to_raw[base] += shap_importance[shap_name]
                    continue
            if raw in mi_scores.index:
                if raw not in shap_to_raw:
                    shap_to_raw[raw] = 0.0
                shap_to_raw[raw] += shap_importance[shap_name]

        if shap_to_raw:
            shap_importance_mapped = pd.Series(shap_to_raw, name="mean_abs_shap").sort_values(
                ascending=False
            )
        else:
            logger.warning("Could not map SHAP feature names to MI feature names")
            shap_importance_mapped = shap_importance
    else:
        shap_importance_mapped = shap_importance.reindex(common_features).dropna()

    # Now build the comparison
    mi_top = mi_scores.head(top_n)
    shap_top = shap_importance_mapped.head(top_n)

    all_features = list(set(mi_top.index) | set(shap_top.index))

    rows = []
    for feat in all_features:
        mi_val = float(mi_scores.get(feat, 0.0))
        shap_val = float(shap_importance_mapped.get(feat, 0.0))

        # Rank within each system (1 = most important)
        mi_ranked = mi_scores.rank(ascending=False)
        shap_ranked = shap_importance_mapped.rank(ascending=False)

        mi_rank = int(mi_ranked.get(feat, len(mi_scores)))
        shap_rank = int(shap_ranked.get(feat, len(shap_importance_mapped)))

        rank_diff = mi_rank - shap_rank

        if abs(rank_diff) <= 2:
            interpretation = "Concordancia — importancia consistente"
        elif rank_diff > 2:
            interpretation = (
                f"MI rank {mi_rank} vs SHAP rank {shap_rank}: "
                "modelo subestima esta feature respecto a su contenido informativo"
            )
        else:
            interpretation = (
                f"MI rank {mi_rank} vs SHAP rank {shap_rank}: "
                "modelo sobrepondera esta feature respecto a su MI marginal"
            )

        rows.append(
            {
                "feature": feat,
                "MI_score": round(mi_val, 4),
                "MI_rank": mi_rank,
                "mean_abs_SHAP": round(shap_val, 4),
                "SHAP_rank": shap_rank,
                "rank_difference": rank_diff,
                "interpretation": interpretation,
            }
        )

    df_cmp = pd.DataFrame(rows).sort_values("MI_rank")
    logger.info(
        f"MI vs SHAP comparison: {len(df_cmp)} features, "
        f"mean |rank_diff| = {df_cmp['rank_difference'].abs().mean():.1f}"
    )
    return df_cmp


def compute_rank_correlation(comparison_df: pd.DataFrame) -> dict[str, float]:
    """Compute Spearman and Kendall rank correlations between MI and SHAP rankings.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of compare_mi_vs_shap().

    Returns
    -------
    dict
        {"spearman_rho": float, "kendall_tau": float, "interpretation": str}
    """
    from scipy.stats import kendalltau, spearmanr

    rho, p_rho = spearmanr(comparison_df["MI_rank"], comparison_df["SHAP_rank"])
    tau, p_tau = kendalltau(comparison_df["MI_rank"], comparison_df["SHAP_rank"])

    if rho > 0.8:
        interp = (
            "Alta concordancia: el modelo captura las dependencias no lineales "
            "que la MI detecta. Rankings consistentes."
        )
    elif rho > 0.5:
        interp = (
            "Concordancia moderada: el modelo captura la mayoría de las relaciones, "
            "pero hay features donde MI y SHAP divergen — investigar esas divergencias."
        )
    else:
        interp = (
            "Baja concordancia: el modelo y la MI dan rankings muy diferentes. "
            "Posibles causas: interacciones complejas que el modelo captura pero "
            "la MI marginal no, o features con alta MI que el modelo ignora."
        )

    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p_value": round(float(p_rho), 4),
        "kendall_tau": round(float(tau), 4),
        "kendall_p_value": round(float(p_tau), 4),
        "interpretation": interp,
    }


# ── Visualizations ──────────────────────────────────────────────────────────


def plot_mi_scores(mi_scores: pd.Series, top_n: int = 15, save: bool = True) -> plt.Figure:
    """Bar plot of MI scores I(Xi; Churn).

    Parameters
    ----------
    mi_scores : pd.Series
        MI scores sorted descending.
    top_n : int
        Number of features to display.
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    top = mi_scores.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    colors = [CYAN if v >= MI_THRESHOLD else MUTED_COLOR for v in top.values]
    bars = ax.barh(range(len(top)), top.values, color=colors, edgecolor="none", alpha=0.85)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information I(X; Churn)  [nats]", fontsize=11)
    ax.set_title("Información Mutua — Features vs Churn", fontsize=13, color=TEXT_COLOR)
    ax.axvline(MI_THRESHOLD, color=CORAL, linestyle="--", alpha=0.7, linewidth=1,
               label=f"Umbral = {MI_THRESHOLD}")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "mi_scores_bar.png", dpi=150, bbox_inches="tight")
        logger.info("Saved mi_scores_bar.png")
    return fig


def plot_conditional_mi_heatmap(
    cmi_matrix: pd.DataFrame, save: bool = True
) -> plt.Figure:
    """Heatmap of Conditional MI: I(Xi; Churn | Xj).

    Diagonal = marginal MI. Off-diagonal = how much MI remains after
    conditioning on another feature.

    Parameters
    ----------
    cmi_matrix : pd.DataFrame
        Output of compute_conditional_mi_matrix().
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.zeros_like(cmi_matrix.values, dtype=bool)  # no mask — show all
    sns.heatmap(
        cmi_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={"label": "I(Xi; Churn | Xj)  [bits]"},
        annot_kws={"fontsize": 8},
    )

    ax.set_title(
        "MI Condicional — I(Feature; Churn | Otra Feature)\n"
        "Diagonal = MI marginal · Off-diagonal = MI residual",
        fontsize=12,
        color=TEXT_COLOR,
    )
    ax.set_xlabel("Feature condicionante (Xj)", fontsize=10)
    ax.set_ylabel("Feature evaluada (Xi)", fontsize=10)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "conditional_mi_heatmap.png", dpi=150, bbox_inches="tight")
        logger.info("Saved conditional_mi_heatmap.png")
    return fig


def plot_interaction_information(
    ii_matrix: pd.DataFrame, save: bool = True
) -> plt.Figure:
    """Heatmap of Interaction Information II(Xi; Xj; Churn).

    Red = redundancy (II < 0). Blue = synergy (II > 0).

    Parameters
    ----------
    ii_matrix : pd.DataFrame
        Output of compute_interaction_matrix().
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    # Diverging colormap centered at 0
    vmax = max(abs(ii_matrix.values.min()), abs(ii_matrix.values.max()))
    if vmax == 0:
        vmax = 0.01

    sns.heatmap(
        ii_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={"label": "II(Xi; Xj; Churn)  [bits]"},
        annot_kws={"fontsize": 8},
    )

    ax.set_title(
        "Información de Interacción — II(Xi; Xj; Churn)\n"
        "Azul = Sinergia · Rojo = Redundancia",
        fontsize=12,
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "interaction_information_heatmap.png", dpi=150, bbox_inches="tight")
        logger.info("Saved interaction_information_heatmap.png")
    return fig


def plot_mi_vs_shap(
    comparison_df: pd.DataFrame,
    rank_corr: dict[str, float] | None = None,
    save: bool = True,
) -> plt.Figure:
    """Scatter plot comparing MI rank vs SHAP rank.

    Points on the diagonal → agreement. Points far off → divergence.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of compare_mi_vs_shap().
    rank_corr : dict
        Output of compute_rank_correlation() (for annotation).
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left panel: rank scatter ──
    ax = axes[0]
    df = comparison_df.copy()

    ax.scatter(
        df["MI_rank"], df["SHAP_rank"],
        c=CYAN, s=70, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3,
    )

    # Label each point
    for _, row in df.iterrows():
        ax.annotate(
            row["feature"],
            (row["MI_rank"], row["SHAP_rank"]),
            fontsize=7.5,
            color=TEXT_COLOR,
            alpha=0.85,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Diagonal (perfect agreement)
    max_rank = max(df["MI_rank"].max(), df["SHAP_rank"].max()) + 1
    ax.plot([1, max_rank], [1, max_rank], color=CORAL, linestyle="--", alpha=0.5,
            label="Concordancia perfecta")
    ax.set_xlabel("MI Rank (1 = más informativo)", fontsize=11)
    ax.set_ylabel("SHAP Rank (1 = más importante para el modelo)", fontsize=11)
    ax.set_title("MI vs SHAP — Comparación de Rankings", fontsize=13, color=TEXT_COLOR)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(alpha=0.2)

    if rank_corr:
        corr_text = (
            f"Spearman ρ = {rank_corr['spearman_rho']:.3f}\n"
            f"Kendall τ = {rank_corr['kendall_tau']:.3f}"
        )
        ax.text(
            0.95, 0.05, corr_text,
            transform=ax.transAxes, fontsize=10, color=AMBER,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_CARD, edgecolor=GRID_COLOR),
        )

    # ── Right panel: side-by-side bar ──
    # Both scores are normalized to [0,1] since MI (nats) and SHAP
    # (log-odds) are on different scales.
    ax2 = axes[1]
    df_sorted = df.sort_values("MI_rank").head(12)
    y_pos = np.arange(len(df_sorted))
    bar_height = 0.35

    # Normalize both scores to [0, 1] for visual comparison
    mi_norm = df_sorted["MI_score"] / df_sorted["MI_score"].max() if df_sorted["MI_score"].max() > 0 else df_sorted["MI_score"]
    shap_norm = df_sorted["mean_abs_SHAP"] / df_sorted["mean_abs_SHAP"].max() if df_sorted["mean_abs_SHAP"].max() > 0 else df_sorted["mean_abs_SHAP"]

    ax2.barh(y_pos - bar_height / 2, mi_norm, bar_height, color=CYAN, alpha=0.8, label="MI (normalizada)")
    ax2.barh(y_pos + bar_height / 2, shap_norm, bar_height, color=VIOLET, alpha=0.8, label="SHAP (normalizada)")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted["feature"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia Normalizada", fontsize=11)
    ax2.set_title("MI vs SHAP — Importancia Normalizada", fontsize=13, color=TEXT_COLOR)
    ax2.legend(loc="lower right", framealpha=0.8)
    ax2.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "mi_vs_shap_comparison.png", dpi=150, bbox_inches="tight")
        logger.info("Saved mi_vs_shap_comparison.png")
    return fig


def plot_entropy_profile(X: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar plot of per-feature Shannon entropy.

    Useful for understanding the information content of each variable.
    Low entropy → little variability. High entropy → high information potential.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw, before transformation).
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()

    entropies = {}
    for col in X.columns:
        vals = X[col].copy()
        if vals.dtype == "object":
            vals = vals.astype("category").cat.codes
        elif vals.dtype in ["float64", "float32"]:
            # Discretize for entropy calculation
            valid = vals.dropna()
            if len(valid) > 0:
                binner = KBinsDiscretizer(n_bins=N_BINS_DISCRETIZE, encode="ordinal", strategy="quantile")
                vals = pd.Series(
                    binner.fit_transform(valid.values.reshape(-1, 1)).ravel().astype(int),
                    index=valid.index,
                )
        entropies[col] = shannon_entropy(vals.dropna().values)

    ent_series = pd.Series(entropies).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, len(ent_series) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ent_series)))
    ax.barh(range(len(ent_series)), ent_series.values, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(ent_series)))
    ax.set_yticklabels(ent_series.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Entropía de Shannon H(X)  [bits]", fontsize=11)
    ax.set_title("Perfil de Entropía — Contenido Informativo por Feature", fontsize=13, color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "entropy_profile.png", dpi=150, bbox_inches="tight")
        logger.info("Saved entropy_profile.png")
    return fig


# ── High-Level Analysis Runner ──────────────────────────────────────────────


def run_full_analysis(save_plots: bool = True) -> dict[str, Any]:
    """Run the complete information-theoretic analysis.

    Steps:
    1. Load raw data
    2. Compute entropy profile
    3. Compute MI scores and feature selection
    4. Compute conditional MI matrix
    5. Compute interaction information matrix
    6. Generate all plots

    Returns
    -------
    dict
        All computed artifacts.
    """
    from src.eda import load_raw_data

    logger.info("=" * 60)
    logger.info("Information Theory Analysis — Starting")
    logger.info("=" * 60)

    df = load_raw_data()
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    # 1. Entropy profile
    logger.info("── Step 1: Entropy Profile ──")
    fig_entropy = plot_entropy_profile(X, save=save_plots)
    plt.close(fig_entropy)

    # 2. MI scores + feature selection
    logger.info("── Step 2: MI-Based Feature Selection ──")
    selection_result = select_features_mi(X, y)
    mi_scores = selection_result["mi_scores"]

    fig_mi = plot_mi_scores(mi_scores, save=save_plots)
    plt.close(fig_mi)

    # 3. Conditional MI
    logger.info("── Step 3: Conditional Mutual Information ──")
    cmi_matrix = compute_conditional_mi_matrix(X, y, top_k=10)
    fig_cmi = plot_conditional_mi_heatmap(cmi_matrix, save=save_plots)
    plt.close(fig_cmi)

    # 4. Interaction Information
    logger.info("── Step 4: Interaction Information ──")
    ii_matrix = compute_interaction_matrix(X, y, top_k=8)
    fig_ii = plot_interaction_information(ii_matrix, save=save_plots)
    plt.close(fig_ii)

    logger.info("=" * 60)
    logger.info("Information Theory Analysis — Complete")
    logger.info("=" * 60)

    return {
        "mi_scores": mi_scores,
        "selection_result": selection_result,
        "cmi_matrix": cmi_matrix,
        "ii_matrix": ii_matrix,
    }


def run_mi_vs_shap_analysis(
    shap_values: np.ndarray,
    feature_names: list[str],
    save_plots: bool = True,
) -> dict[str, Any]:
    """Run MI vs SHAP comparison analysis.

    Must be called after training (needs SHAP values).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix (n_samples, n_features).
    feature_names : list[str]
        Feature names matching SHAP columns.
    save_plots : bool
        Save plots to FIGURES_DIR.

    Returns
    -------
    dict
        Comparison table, rank correlations, and figures.
    """
    from src.eda import load_raw_data

    logger.info("── MI vs SHAP Comparison ──")

    df = load_raw_data()
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    # MI scores
    mi_scores = compute_mi_scores(X, y)

    # SHAP importance: mean |SHAP| per feature
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    # Compare
    comparison_df = compare_mi_vs_shap(mi_scores, shap_importance)
    rank_corr = compute_rank_correlation(comparison_df)

    # Plot
    fig = plot_mi_vs_shap(comparison_df, rank_corr, save=save_plots)
    plt.close(fig)

    logger.info(f"Spearman ρ = {rank_corr['spearman_rho']:.3f}")
    logger.info(f"Kendall  τ = {rank_corr['kendall_tau']:.3f}")
    logger.info(rank_corr["interpretation"])

    return {
        "comparison_df": comparison_df,
        "rank_correlation": rank_corr,
        "mi_scores": mi_scores,
        "shap_importance": shap_importance,
    }


# ── Hill-Climbing Feature Selection ─────────────────────────────────────────


def greedy_forward_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator=None,
    max_features: int | None = None,
    min_improvement: float = 0.001,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    mi_seed: bool = True,
) -> dict[str, Any]:
    """Sequential Forward Selection (hill climbing) on transformed features.

    Algorithm:
    1. Start with an empty selected set.
    2. At each step, evaluate every candidate feature by computing
       cross-validated `scoring` with it added to the current set.
    3. Add the feature that gives the greatest improvement.
    4. Stop if the best gain < min_improvement, or we hit max_features.

    When mi_seed=True (default), candidates are evaluated in descending
    MI order first. This is a heuristic that speeds up convergence by
    trying the most informative features first.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw, before pipeline transformation).
    y : np.ndarray
        Binary target.
    estimator : sklearn estimator or None
        Classifier to use for CV evaluation. Defaults to XGBoost if
        installed, else RandomForest.
    max_features : int or None
        Maximum number of features to select. None = no limit.
    min_improvement : float
        Stop if the best single-feature gain is below this threshold.
    cv_folds : int
        Number of CV folds for evaluation.
    scoring : str
        Scoring metric for sklearn cross_val_score.
    random_state : int
        Seed for CV splits.
    mi_seed : bool
        If True, sort candidates by MI score (highest first) before
        each greedy step.

    Returns
    -------
    dict
        {
            "selected_features": list[str],  # in selection order
            "cv_scores": list[float],        # AUC after each addition
            "score_history": list[dict],     # detailed per-step log
            "final_score": float,            # AUC with full selected set
            "baseline_score": float,         # AUC using ALL features
        }
    """
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.preprocessing import LabelEncoder

    # ── Default estimator ──
    if estimator is None:
        try:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
                n_estimators=100,
                max_depth=4,
                scale_pos_weight=3,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(
                random_state=random_state, n_estimators=100, class_weight="balanced"
            )

    # ── Encode X for CV ──
    X_enc = X.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == "object":
            X_enc[col] = X_enc[col].astype("category").cat.codes
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True)).fillna(0)

    all_features = list(X_enc.columns)
    max_f = max_features or len(all_features)

    # ── MI ordering for candidate evaluation ──
    if mi_seed:
        mi_scores = compute_mi_scores(X, y, random_state=random_state)
        # Order: highest MI first; unknown features go to the end
        feature_order = [
            f for f in mi_scores.index if f in all_features
        ] + [f for f in all_features if f not in mi_scores.index]
    else:
        feature_order = list(all_features)

    # ── Baseline: all features ──
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    baseline_score = float(
        _cvs(estimator, X_enc[all_features], y, cv=cv, scoring=scoring, n_jobs=-1).mean()
    )
    logger.info(f"[hill-climbing] Baseline (all {len(all_features)} features): "
                f"{scoring}={baseline_score:.4f}")

    selected: list[str] = []
    candidates: list[str] = list(feature_order)
    cv_scores: list[float] = []
    score_history: list[dict] = []
    current_score = 0.0

    while candidates and len(selected) < max_f:
        best_gain = -np.inf
        best_feature = None
        best_score = current_score

        for feat in candidates:
            trial_set = selected + [feat]
            score = float(
                _cvs(
                    estimator, X_enc[trial_set], y,
                    cv=cv, scoring=scoring, n_jobs=-1,
                ).mean()
            )
            gain = score - current_score
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_score = score

        if best_gain < min_improvement:
            logger.info(
                f"[hill-climbing] Stopping: best gain {best_gain:.5f} < "
                f"min_improvement {min_improvement}"
            )
            break

        selected.append(best_feature)
        candidates.remove(best_feature)
        current_score = best_score
        cv_scores.append(best_score)
        score_history.append({
            "step": len(selected),
            "added_feature": best_feature,
            "gain": round(best_gain, 5),
            "cumulative_score": round(best_score, 5),
        })

        logger.info(
            f"[hill-climbing] Step {len(selected):2d}: +{best_feature} "
            f"→ {scoring}={best_score:.4f} (gain={best_gain:+.5f})"
        )

    logger.info(
        f"[hill-climbing] Selected {len(selected)} / {len(all_features)} features. "
        f"Final {scoring}={current_score:.4f} vs baseline={baseline_score:.4f} "
        f"(Δ={current_score - baseline_score:+.4f})"
    )

    return {
        "selected_features": selected,
        "cv_scores": cv_scores,
        "score_history": score_history,
        "final_score": current_score,
        "baseline_score": baseline_score,
    }


def plot_hill_climbing_curve(result: dict, save: bool = True) -> plt.Figure:
    """Plot the hill-climbing AUC curve vs number of selected features.

    Parameters
    ----------
    result : dict
        Output of greedy_forward_selection().
    save : bool
        Save to FIGURES_DIR.

    Returns
    -------
    plt.Figure
    """
    set_dark_style()
    scores = result["cv_scores"]
    baseline = result["baseline_score"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(scores) + 1), scores, color=CYAN, linewidth=2,
            marker="o", markersize=4, label="SFS (greedy)")
    ax.axhline(baseline, color=CORAL, linestyle="--", linewidth=1.5,
               label=f"Baseline (todas las features) = {baseline:.4f}")

    best_idx = int(np.argmax(scores))
    ax.axvline(best_idx + 1, color=AMBER, linestyle=":", linewidth=1.5,
               label=f"Mejor subset: {best_idx + 1} features = {scores[best_idx]:.4f}")

    ax.set_xlabel("Número de features seleccionadas", fontsize=11)
    ax.set_ylabel("CV ROC-AUC (media)", fontsize=11)
    ax.set_title("Hill Climbing (SFS) — Curva de Selección de Features", fontsize=13)
    ax.legend(framealpha=0.8, fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "hill_climbing_curve.png", dpi=150, bbox_inches="tight")
        logger.info("Saved hill_climbing_curve.png")
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Information Theory analysis for churn prediction")
    parser.add_argument("--mi-only", action="store_true", help="Only compute MI scores (skip CMI/II)")
    parser.add_argument(
        "--hill-climbing", action="store_true",
        help="Run greedy forward selection (hill climbing) and save curve plot"
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help="Maximum features to select in hill climbing (default: auto-stop)"
    )
    parser.add_argument(
        "--compare-shap", action="store_true",
        help="Compare MI scores vs SHAP importances (requires a trained model with SHAP values)"
    )
    args = parser.parse_args()

    if args.mi_only:
        from src.eda import load_raw_data

        df = load_raw_data()
        y = df[TARGET_COL].values
        X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
        mi_scores = compute_mi_scores(X, y)
        print("\n── Mutual Information Scores (nats) ──")
        print(mi_scores.to_string())

    elif args.hill_climbing:
        from src.eda import load_raw_data

        df = load_raw_data()
        y = df[TARGET_COL].values
        X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
        result = greedy_forward_selection(
            X, y, max_features=args.max_features, random_state=42
        )
        fig = plot_hill_climbing_curve(result, save=True)
        plt.close(fig)
        print("\n── Selected Features (in order of addition) ──")
        for step in result["score_history"]:
            print(f"  Step {step['step']:2d}: {step['added_feature']:<30} "
                  f"AUC={step['cumulative_score']:.4f}  gain={step['gain']:+.5f}")
        print(f"\nFinal AUC : {result['final_score']:.4f}")
        print(f"Baseline  : {result['baseline_score']:.4f}")
        print(f"Saved curve to figures/hill_climbing_curve.png")

    elif args.compare_shap:
        from src.explainer import (
            build_shap_explainer,
            compute_shap_values,
            run_mi_vs_shap_comparison,
        )
        from src.preprocessing import get_feature_names, load_pipeline, prepare_data
        from xgboost import XGBClassifier

        logger.info("Loading pipeline for MI vs SHAP comparison...")
        pipeline = load_pipeline()

        from src.eda import load_raw_data

        df = load_raw_data()
        X_transformed, y, _ = prepare_data(df, fit_pipeline=False, pipeline=pipeline)
        feature_names = get_feature_names(pipeline)

        # Train a fresh model on the current preprocessor output so feature
        # counts always match — avoids stale artifact mismatch errors.
        logger.info(
            f"Training in-memory XGBoost on {X_transformed.shape[1]} features "
            "for SHAP comparison (not saved to disk)..."
        )
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_transformed, y)

        shap_explainer = build_shap_explainer(model, X_transformed, feature_names)
        explanation = compute_shap_values(shap_explainer, X_transformed, feature_names)
        results = run_mi_vs_shap_comparison(explanation=explanation, save_plots=True)

        print("\n── MI vs SHAP Key Findings ──")
        for finding in results["key_findings"]:
            print(f"  • {finding}")
        print(f"\nSpearman ρ = {results['rank_correlation']['spearman_rho']:.4f}")
        print(f"Kendall  τ = {results['rank_correlation']['kendall_tau']:.4f}")
        print(f"Interpretation: {results['rank_correlation']['interpretation']}")
        print("\nSaved figure to figures/mi_vs_shap.png")

    else:
        results = run_full_analysis()
        print("\n── MI Scores ──")
        print(results["mi_scores"].to_string())
        print("\n── Selected Features ──")
        print(results["selection_result"]["selected_features"])
        if results["selection_result"]["redundancy_report"]:
            print("\n── Redundancy Report ──")
            for r in results["selection_result"]["redundancy_report"]:
                print(f"  {r['feature_a']} ↔ {r['feature_b']}: "
                      f"redundancy={r['redundancy_ratio']:.0%} — {r['interpretation']}")
