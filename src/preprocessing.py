"""Feature engineering and sklearn preprocessing pipeline.

Builds a single ColumnTransformer pipeline that:
- Fixes TotalCharges (whitespace → NaN → median imputation)
- Creates engineered features (charges_per_tenure, service_count, etc.)
- Encodes categoricals (one-hot for nominal, ordinal for binary)
- Scales numeric features

The serialized pipeline is the single artifact connecting training → API → agent.
Saving everything in one pipeline means the same transformations are applied
at training time and at serving time — no risk of them drifting apart.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    ID_COL,
    MODELS_DIR,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_CSV,
    TARGET_COL,
)


# ── Custom Transformers ──────────────────────────────────────────────────────


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features from the raw Telco columns.

    New features and why they exist:

        charges_per_tenure
            Monthly charge divided by how long the customer has been around.
            New customers paying a lot relative to their tenure are more likely
            to feel they haven't gotten enough value yet.

        service_count
            How many services the customer has subscribed to. More services
            means more switching cost, which tends to reduce churn.

        is_new_customer
            Binary flag for tenure ≤ 6 months. The first six months are the
            danger zone — churn is highest here, so it's worth calling out
            explicitly.

        high_value_flag
            Binary flag for MonthlyCharges > $70. High-value customers are
            priority retention targets because losing them hurts more.

        total_charges_ratio
            Ratio of actual TotalCharges to the expected cumulative spend
            (MonthlyCharges × tenure). Values below 1 suggest discounts or
            credits were applied; above 1 suggests upgrades.
    """

    SERVICE_COLS = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    # Services where "No" (not "No internet service") signals low stickiness
    STICKY_SERVICES = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # charges_per_tenure
        X["charges_per_tenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)

        # service_count — count how many services have "Yes"
        service_cols_present = [c for c in self.SERVICE_COLS if c in X.columns]
        X["service_count"] = X[service_cols_present].apply(
            lambda row: sum(1 for v in row if str(v).strip().lower() == "yes"), axis=1
        )

        # is_new_customer
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)

        # high_value_flag
        X["high_value_flag"] = (X["MonthlyCharges"] > 70).astype(int)

        # total_charges_ratio
        denom = X["MonthlyCharges"] * X["tenure"] + 1
        X["total_charges_ratio"] = X["TotalCharges"].fillna(0) / denom

        # ── New interaction features ──────────────────────────────────────

        # monthly_contract: explicit binary flag for month-to-month contract.
        # The #1 churn predictor in this dataset — worth having as its own
        # feature so it can gate the interaction features below.
        if "Contract" in X.columns:
            X["monthly_contract"] = (
                X["Contract"].str.strip() == "Month-to-month"
            ).astype(int)
        else:
            X["monthly_contract"] = 0

        # fiber_unprotected: Fiber optic customer with NO OnlineSecurity AND
        # NO TechSupport. This group has ~50% churn — they pay premium prices
        # but feel exposed, which drives dissatisfaction. A tree needs three
        # separate splits to find this; an explicit feature makes it instant.
        if all(c in X.columns for c in ["InternetService", "OnlineSecurity", "TechSupport"]):
            X["fiber_unprotected"] = (
                (X["InternetService"].str.strip() == "Fiber optic")
                & (X["OnlineSecurity"].str.strip() == "No")
                & (X["TechSupport"].str.strip() == "No")
            ).astype(int)
        else:
            X["fiber_unprotected"] = 0

        # early_monthly: month-to-month AND tenure ≤ 12.
        # The single highest-risk segment (~55% churn). Two individually
        # important features combine into something even stronger.
        X["early_monthly"] = (
            (X["monthly_contract"] == 1) & (X["tenure"] <= 12)
        ).astype(int)

        # no_sticky_count: how many of the four value-add services
        # (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport)
        # the customer explicitly skipped. More skipped services = less
        # locked in = more likely to leave.
        sticky_present = [c for c in self.STICKY_SERVICES if c in X.columns]
        X["no_sticky_count"] = X[sticky_present].apply(
            lambda row: sum(1 for v in row if str(v).strip() == "No"), axis=1
        )

        # charge_risk_score: high monthly charges × short tenure × month-to-month.
        # Combines three risk signals into one number. A high score means the
        # customer is paying a lot, hasn't been around long, and has no commitment.
        X["charge_risk_score"] = (
            X["MonthlyCharges"] * (1.0 / (X["tenure"] + 1)) * X["monthly_contract"]
        )

        # ── Interaction features (v3) ────────────────────────────────────

        # echeck_monthly: Electronic check + month-to-month.
        # Electronic check is the payment method with the highest churn (~45%).
        # Combined with month-to-month flexibility, this segment reaches ~52%.
        if all(c in X.columns for c in ["PaymentMethod", "Contract"]):
            X["echeck_monthly"] = (
                (X["PaymentMethod"].str.strip() == "Electronic check")
                & (X["Contract"].str.strip() == "Month-to-month")
            ).astype(int)
        else:
            X["echeck_monthly"] = 0

        # senior_monthly: Senior citizen + month-to-month contract.
        # Seniors have ~42% churn overall; on month-to-month it rises to ~48%.
        # They're also less likely to call support to resolve issues, so problems
        # tend to quietly compound until they cancel.
        if all(c in X.columns for c in ["SeniorCitizen", "Contract"]):
            X["senior_monthly"] = (
                (X["SeniorCitizen"] == 1)
                & (X["Contract"].str.strip() == "Month-to-month")
            ).astype(int)
        else:
            X["senior_monthly"] = 0

        # digital_risk: Paperless billing + month-to-month + tenure ≤ 24.
        # Paperless billing customers tend to comparison-shop more actively.
        # Under 2 years of tenure means they're still in the high-churn window.
        # Together this group hits ~38% churn — hard to catch without the flag.
        if all(c in X.columns for c in ["PaperlessBilling", "Contract"]):
            X["digital_risk"] = (
                (X["PaperlessBilling"].str.strip() == "Yes")
                & (X["Contract"].str.strip() == "Month-to-month")
                & (X["tenure"] <= 24)
            ).astype(int)
        else:
            X["digital_risk"] = 0

        # ── Additional interaction features (v2) ─────────────────────────

        # charges_efficiency: how much the customer has actually paid vs.
        # what we'd expect from their current plan × tenure. Below 1 suggests
        # discounts or plan downgrades; close to 1 means stable billing.
        X["charges_efficiency"] = X["TotalCharges"].fillna(0) / (
            X["MonthlyCharges"] * (X["tenure"] + 1) + 1e-6
        )

        # overloaded_monthly: customer has lots of services but still no
        # long-term contract. That paradox — high stickiness signals but
        # no commitment — often means they're comparison-shopping. ~42% churn.
        X["overloaded_monthly"] = (
            (X["service_count"] >= 4) & (X["monthly_contract"] == 1)
        ).astype(int)

        # highest_risk: fiber_unprotected AND early_monthly at the same time.
        # The intersection of these two high-risk groups is the most at-risk
        # segment in the dataset (~60% churn). Worth calling out explicitly.
        X["highest_risk"] = (
            (X["fiber_unprotected"] == 1) & (X["early_monthly"] == 1)
        ).astype(int)

        return X


class TotalChargesFixer(BaseEstimator, TransformerMixin):
    """Fix TotalCharges before anything else touches it.

    The raw CSV has whitespace strings where TotalCharges should be 0
    (new customers with tenure=0). We convert those to NaN, then impute
    them as MonthlyCharges (best guess for a customer with no billing history).
    Any remaining NaN gets handled by the downstream median imputer.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        # Deterministic imputation for tenure == 0 rows
        mask_new = (X["tenure"] == 0) & X["TotalCharges"].isna()
        if mask_new.any():
            X.loc[mask_new, "TotalCharges"] = X.loc[mask_new, "MonthlyCharges"]
        return X


# ── Engineered feature lists ─────────────────────────────────────────────────

ENGINEERED_NUMERIC = [
    "charges_per_tenure",
    "service_count",
    "total_charges_ratio",
    "no_sticky_count",
    "charge_risk_score",
    "charges_efficiency",
]

ENGINEERED_BINARY = [
    "is_new_customer",
    "high_value_flag",
    "monthly_contract",
    "fiber_unprotected",
    "early_monthly",
    "overloaded_monthly",
    "highest_risk",
    "echeck_monthly",
    "senior_monthly",
    "digital_risk",
]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_BINARY = BINARY_FEATURES + ENGINEERED_BINARY


# ── Pipeline Builder ─────────────────────────────────────────────────────────


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for the churn prediction pipeline.

    Returns
    -------
    ColumnTransformer
        Fitted-ready transformer that handles numeric, binary, and nominal features.
    """

    # Numeric pipeline: median imputation (robust to TotalCharges skew)
    # then StandardScaler so all features are on the same scale.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Binary pipeline: ordinal encoding (0/1) for Yes/No columns.
    # handle_unknown=-1 means unseen values at serving time get a sentinel
    # rather than blowing up.
    binary_pipeline = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ]
    )

    # Nominal pipeline: one-hot encoding for multi-level categoricals.
    # drop='if_binary' removes the redundant column for 2-level features.
    # handle_unknown='ignore' means unseen categories become all-zeros rows.
    nominal_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            )),
        ]
    )

    # Contract gets its own ordinal encoder because it has a clear ordering:
    #   Month-to-month (0) → highest churn (~43%)
    #   One year       (1) → medium churn  (~11%)
    #   Two year       (2) → lowest churn  (~3%)
    # Preserving that order in a single column is more useful than
    # one-hot encoding it into 3 separate columns.
    contract_ordinal_pipeline = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(
                categories=[["Month-to-month", "One year", "Two year"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]
    )

    # Nominal features excluding Contract (handled as ordinal above)
    nominal_features_no_contract = [f for f in NOMINAL_FEATURES if f != "Contract"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC),
            ("bin", binary_pipeline, ALL_BINARY),
            ("nom", nominal_pipeline, nominal_features_no_contract),
            ("ord_contract", contract_ordinal_pipeline, ["Contract"]),
        ],
        remainder="drop",
        # verbose_feature_names_out=True keeps prefixes like "num__tenure"
        # so SHAP feature names can be traced back to original columns.
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_full_pipeline() -> Pipeline:
    """Build the complete preprocessing pipeline including feature engineering.

    Returns
    -------
    Pipeline
        [TotalChargesFixer → FeatureEngineer → ColumnTransformer]
    """
    return Pipeline(
        steps=[
            ("fix_charges", TotalChargesFixer()),
            ("engineer", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def prepare_data(
    df: pd.DataFrame,
    fit_pipeline: bool = True,
    pipeline: Pipeline | None = None,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Prepare features and target from raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with all original columns.
    fit_pipeline : bool
        If True, fit the pipeline on this data. If False, only transform.
    pipeline : Pipeline | None
        Pre-fitted pipeline. Required if fit_pipeline is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Pipeline]
        X_transformed, y, fitted_pipeline
    """
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    if fit_pipeline:
        pipeline = build_full_pipeline()
        X_transformed = pipeline.fit_transform(X)
        logger.info(f"Pipeline fitted — output shape: {X_transformed.shape}")
    else:
        if pipeline is None:
            raise ValueError("Must provide a fitted pipeline when fit_pipeline=False")
        X_transformed = pipeline.transform(X)
        logger.info(f"Pipeline transformed — output shape: {X_transformed.shape}")

    return X_transformed, y, pipeline


def save_pipeline(pipeline: Pipeline, path=None) -> None:
    """Serialize the fitted pipeline to disk."""
    path = path or PREPROCESSOR_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, path)
    logger.success(f"Pipeline saved to {path}")


def load_pipeline(path=None) -> Pipeline:
    """Load a serialized pipeline from disk."""
    path = path or PREPROCESSOR_PATH
    pipeline = load(path)
    logger.info(f"Pipeline loaded from {path}")
    return pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names from the fitted pipeline.

    Returns
    -------
    list[str]
        List of output feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        logger.warning("Could not extract feature names from pipeline.")
        return []


# ── MI-Based Feature Diagnostics ─────────────────────────────────────────────


def run_mi_feature_diagnostics(
    df: pd.DataFrame | None = None,
    save_plots: bool = True,
) -> dict:
    """Run Mutual Information diagnostics on the raw feature set.

    Computes MI scores, conditional MI, and interaction information to
    validate feature engineering choices. The results are advisory — they
    inform decisions but don't change the pipeline itself. This keeps the
    pipeline stable between retraining cycles.

    Parameters
    ----------
    df : pd.DataFrame | None
        Raw data. Loaded from RAW_CSV if None.
    save_plots : bool
        Whether to save plots to FIGURES_DIR.

    Returns
    -------
    dict
        {
            "mi_scores": pd.Series,
            "selected_features": list[str],
            "dropped_features": list[str],
            "redundancy_report": list[dict],
            "cmi_matrix": pd.DataFrame,
            "ii_matrix": pd.DataFrame,
            "engineering_validation": dict,
        }
    """
    from src.information_theory import (
        compute_conditional_mi_matrix,
        compute_interaction_matrix,
        plot_conditional_mi_heatmap,
        plot_entropy_profile,
        plot_interaction_information,
        plot_mi_scores,
        select_features_mi,
    )

    if df is None:
        from src.eda import load_raw_data

        df = load_raw_data()

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    logger.info("=" * 60)
    logger.info("MI Feature Diagnostics — Starting")
    logger.info("=" * 60)

    # 1. Entropy profile
    fig_ent = plot_entropy_profile(X, save=save_plots)
    plt.close(fig_ent)

    # 2. MI selection
    selection = select_features_mi(X, y)
    mi_scores = selection["mi_scores"]
    fig_mi = plot_mi_scores(mi_scores, save=save_plots)
    plt.close(fig_mi)

    # 3. Conditional MI matrix
    cmi_matrix = compute_conditional_mi_matrix(X, y, top_k=10)
    fig_cmi = plot_conditional_mi_heatmap(cmi_matrix, save=save_plots)
    plt.close(fig_cmi)

    # 4. Interaction Information
    ii_matrix = compute_interaction_matrix(X, y, top_k=8)
    fig_ii = plot_interaction_information(ii_matrix, save=save_plots)
    plt.close(fig_ii)

    # 5. Validate engineered features against MI insights
    engineering_validation = _validate_engineering_with_mi(mi_scores, ii_matrix)

    logger.info("=" * 60)
    logger.info("MI Feature Diagnostics — Complete")
    logger.info("=" * 60)

    return {
        "mi_scores": mi_scores,
        "selected_features": selection["selected_features"],
        "dropped_features": selection["dropped_features"],
        "redundancy_report": selection["redundancy_report"],
        "cmi_matrix": cmi_matrix,
        "ii_matrix": ii_matrix,
        "engineering_validation": engineering_validation,
    }


def _validate_engineering_with_mi(
    mi_scores: pd.Series,
    ii_matrix: pd.DataFrame,
) -> dict:
    """Cross-reference engineered features with MI findings.

    Synergistic feature pairs (II > 0) justify interaction features —
    combining them reveals more about churn than either alone.
    Redundant pairs (II < 0) might be candidates for dropping one.

    We use a soft cutoff of 0.005 to avoid flagging noise-level interactions.

    Returns
    -------
    dict
        Validation summary per engineered feature.
    """
    validations = {}

    # Check if key raw features have high MI (justifying their inclusion)
    for feat in NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES:
        if feat in mi_scores.index:
            mi_val = float(mi_scores[feat])
            validations[feat] = {
                "MI_score": round(mi_val, 4),
                "status": "justified" if mi_val >= 0.01 else "low_MI — revisit",
            }

    # Check for synergistic pairs that justify interaction features
    if not ii_matrix.empty:
        synergies = []
        for i in ii_matrix.index:
            for j in ii_matrix.columns:
                if i < j and ii_matrix.loc[i, j] > 0.005:
                    synergies.append((i, j, round(float(ii_matrix.loc[i, j]), 4)))

        if synergies:
            validations["_synergistic_pairs"] = synergies
            logger.info(f"Found {len(synergies)} synergistic feature pairs (II > 0)")

        redundancies = []
        for i in ii_matrix.index:
            for j in ii_matrix.columns:
                if i < j and ii_matrix.loc[i, j] < -0.005:
                    redundancies.append((i, j, round(float(ii_matrix.loc[i, j]), 4)))

        if redundancies:
            validations["_redundant_pairs"] = redundancies
            logger.info(f"Found {len(redundancies)} redundant feature pairs (II < 0)")

    return validations
