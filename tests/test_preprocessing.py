"""Tests for preprocessing pipeline — Fase 03.

Unit tests for the preprocessing pipeline. These tests validate the contract
between raw data and transformed features. They catch regressions in:
  (1) type coercion (TotalCharges whitespace → NaN),
  (2) feature engineering formulas,
  (3) pipeline output shape and NaN-free guarantee.

Test fixture uses a minimal 5-row sample that covers edge cases: whitespace
TotalCharges, tenure=0, senior citizens, all contract types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FeatureEngineer,
    TotalChargesFixer,
    build_full_pipeline,
    prepare_data,
)


# Minimal but representative fixture. Row 2 has TotalCharges=' ' (whitespace)
# — this is the actual format in the raw Telco Churn CSV for new customers
# with tenure=0. Row 4 has tenure=0 to test division-by-zero safety in
# charges_per_tenure. All 3 contract types are represented. The fixture is
# small enough for fast tests but large enough to cover the pipeline's code
# paths.
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal sample dataframe mimicking Telco Churn structure."""
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004", "0005"],
            "gender": ["Female", "Male", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 1, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes", "No"],
            "tenure": [1, 34, 2, 45, 0],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": [
                "No phone service",
                "No",
                "Yes",
                "Yes",
                "No",
            ],
            "InternetService": ["DSL", "DSL", "Fiber optic", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "Yes", "No", "Yes", "No"],
            "StreamingTV": ["No", "No", "Yes", "Yes", "No"],
            "StreamingMovies": ["No", "No", "Yes", "Yes", "No"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
            "TotalCharges": ["29.85", "1889.50", " ", "1840.75", "151.65"],
            "Churn": [1, 0, 1, 0, 1],
        }
    )


# Tests the first stage of the pipeline. The Telco dataset stores TotalCharges
# as strings with whitespace for new customers (an IBM data quality artifact).
# This must be handled before any numeric computation.
class TestTotalChargesFixer:
    """Tests for TotalChargesFixer transformer."""

    def test_converts_whitespace_to_nan(self, sample_df):
        fixer = TotalChargesFixer()
        result = fixer.transform(sample_df)
        assert pd.isna(result.loc[2, "TotalCharges"])

    def test_preserves_valid_values(self, sample_df):
        fixer = TotalChargesFixer()
        result = fixer.transform(sample_df)
        assert result.loc[0, "TotalCharges"] == 29.85
        assert result.loc[1, "TotalCharges"] == 1889.50

    def test_output_dtype_is_numeric(self, sample_df):
        fixer = TotalChargesFixer()
        result = fixer.transform(sample_df)
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])


# Tests each engineered feature independently. Each test verifies both the
# column's existence and a specific computed value, ensuring the formula is
# correct. The charges_per_tenure test uses np.testing.assert_almost_equal
# for floating-point comparison.
class TestFeatureEngineer:
    """Tests for FeatureEngineer transformer."""

    def test_creates_charges_per_tenure(self, sample_df):
        fixer = TotalChargesFixer()
        fixed = fixer.transform(sample_df)
        engineer = FeatureEngineer()
        result = engineer.transform(fixed)
        assert "charges_per_tenure" in result.columns
        # For row 0: 29.85 / (1+1) = 14.925
        np.testing.assert_almost_equal(result.loc[0, "charges_per_tenure"], 14.925)

    def test_creates_service_count(self, sample_df):
        fixer = TotalChargesFixer()
        fixed = fixer.transform(sample_df)
        engineer = FeatureEngineer()
        result = engineer.transform(fixed)
        assert "service_count" in result.columns
        # All values should be non-negative integers
        assert (result["service_count"] >= 0).all()

    def test_creates_is_new_customer(self, sample_df):
        fixer = TotalChargesFixer()
        fixed = fixer.transform(sample_df)
        engineer = FeatureEngineer()
        result = engineer.transform(fixed)
        assert "is_new_customer" in result.columns
        # Row 0: tenure=1, should be 1 (new)
        assert result.loc[0, "is_new_customer"] == 1
        # Row 1: tenure=34, should be 0 (not new)
        assert result.loc[1, "is_new_customer"] == 0

    def test_creates_high_value_flag(self, sample_df):
        fixer = TotalChargesFixer()
        fixed = fixer.transform(sample_df)
        engineer = FeatureEngineer()
        result = engineer.transform(fixed)
        assert "high_value_flag" in result.columns
        # Row 4: MonthlyCharges=70.70, should be 1 (>70)
        assert result.loc[4, "high_value_flag"] == 1
        # Row 0: MonthlyCharges=29.85, should be 0
        assert result.loc[0, "high_value_flag"] == 0

    def test_does_not_drop_original_columns(self, sample_df):
        fixer = TotalChargesFixer()
        fixed = fixer.transform(sample_df)
        engineer = FeatureEngineer()
        result = engineer.transform(fixed)
        assert "tenure" in result.columns
        assert "MonthlyCharges" in result.columns


class TestFullPipeline:
    """Tests for the complete preprocessing pipeline."""

    def test_pipeline_runs_without_error(self, sample_df):
        # Smoke test — verifies the entire pipeline (fix_charges → engineer
        # → preprocessor) runs end-to-end without crashing.
        pipeline = build_full_pipeline()
        X = sample_df.drop(columns=["Churn", "customerID"])
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] > 0

    def test_no_nan_in_output(self, sample_df):
        # Critical contract test. NaN in transformed features would cause
        # model training to fail or produce undefined predictions. The
        # pipeline must guarantee NaN-free output.
        pipeline = build_full_pipeline()
        X = sample_df.drop(columns=["Churn", "customerID"])
        X_transformed = pipeline.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_prepare_data_returns_correct_shapes(self, sample_df):
        # Shape contract: output rows must match input rows, y must align
        # with X.
        X, y, pipeline = prepare_data(sample_df)
        assert X.shape[0] == 5
        assert len(y) == 5
        assert pipeline is not None

    def test_prepare_data_target_is_binary(self, sample_df):
        # Target contract: y must be binary {0, 1}.
        _, y, _ = prepare_data(sample_df)
        assert set(np.unique(y)).issubset({0, 1})

    def test_transform_with_fitted_pipeline(self, sample_df):
        # Idempotency test: transforming the same data with a fitted
        # pipeline must produce identical output. This validates that
        # fit_pipeline=False correctly reuses the pipeline without
        # re-fitting.
        X, y, pipeline = prepare_data(sample_df)
        # Transform again with the same pipeline
        X2, _, _ = prepare_data(sample_df, fit_pipeline=False, pipeline=pipeline)
        np.testing.assert_array_almost_equal(X, X2)
