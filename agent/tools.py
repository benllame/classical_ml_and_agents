"""ReAct Agent Tools — 5 tools backed by real models and data.

Each tool's docstring serves dual purpose: Python documentation and
LLM prompt — the agent reads it to decide when to call the tool and
what arguments to pass.

Tools:
1. get_customer_profile(customer_id) → customer data dict
2. predict_churn_risk(customer_id) → P(churn) + risk segment
3. explain_prediction(customer_id) → SHAP top-3 factors
4. recommend_intervention(customer_id, budget) → best policy + ROI
5. simulate_budget_allocation(budget, top_n) → prioritized customer list
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool
from loguru import logger

from src.config import ID_COL, RAW_CSV

# ── Lazy-loaded shared resources ─────────────────────────────────────────────
# Resources are loaded on first use and cached globally. This avoids loading
# all artifacts at import time, which would slow down partial usage and tests.

_df: pd.DataFrame | None = None
_model = None
_pipeline = None
_explainer = None


def _get_df() -> pd.DataFrame:
    """Lazy-load the dataset."""
    global _df
    if _df is None:
        _df = pd.read_csv(RAW_CSV)
        _df["TotalCharges"] = pd.to_numeric(_df["TotalCharges"], errors="coerce")
        _df["Churn"] = (_df["Churn"] == "Yes").astype(int)
        logger.info(f"Loaded dataset: {len(_df)} rows")
    return _df


def _get_model():
    """Lazy-load the prediction model from MLflow or local joblib fallback."""
    global _model
    if _model is None:
        try:
            import mlflow

            from tracking.mlflow_setup import get_production_model_uri

            uri = get_production_model_uri()
            _model = mlflow.pyfunc.load_model(uri)
            logger.info(f"Loaded model from MLflow: {uri}")
        except Exception as e:
            logger.warning(f"MLflow model not available ({e}), trying local fallback...")
            try:
                import joblib

                from src.config import MODELS_DIR

                model_path = MODELS_DIR / "best_model.joblib"
                _model = joblib.load(model_path)
                logger.info(f"Loaded local model: {model_path}")
            except Exception as e2:
                logger.error(f"No model available: {e2}")
                raise RuntimeError(
                    "No prediction model available. Run src/train.py first."
                ) from e2
    return _model


def _get_pipeline():
    """Lazy-load the preprocessing pipeline."""
    global _pipeline
    if _pipeline is None:
        from src.preprocessing import load_pipeline

        _pipeline = load_pipeline()
    return _pipeline


def _get_explainer():
    """Lazy-load the SHAP explainer."""
    global _explainer
    if _explainer is None:
        from src.explainer import load_explainer

        _explainer = load_explainer()
    return _explainer


def init_tools(df=None, model=None, pipeline=None, explainer=None):
    """Pre-initialize tool resources (for testing or API usage).

    Call this before running tools to avoid lazy-loading during agent
    execution. The FastAPI lifespan uses this to pre-load everything on
    startup and avoid cold-start latency on the first request.
    """
    global _df, _model, _pipeline, _explainer
    if df is not None:
        _df = df
    if model is not None:
        _model = model
    if pipeline is not None:
        _pipeline = pipeline
    if explainer is not None:
        _explainer = explainer


# ── Tool 1: Customer Profile ────────────────────────────────────────────────


@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve the complete profile of a customer by their ID.

    Returns demographics, services subscribed, charges, tenure, and contract info.
    Use this tool when you need to understand WHO a customer is before analyzing their risk.

    Args:
        customer_id: The unique customer identifier (e.g., '7590-VHVEG')
    """
    df = _get_df()
    row = df[df[ID_COL] == customer_id]

    if row.empty:
        return json.dumps({"error": f"Customer '{customer_id}' not found in the dataset."})

    customer = row.iloc[0].to_dict()

    # Returns structured JSON with nested demographics/account/services/charges.
    profile = {
        "customer_id": customer.get("customerID"),
        "demographics": {
            "gender": customer.get("gender"),
            "senior_citizen": bool(customer.get("SeniorCitizen")),
            "partner": customer.get("Partner"),
            "dependents": customer.get("Dependents"),
        },
        "account": {
            "tenure_months": int(customer.get("tenure", 0)),
            "contract": customer.get("Contract"),
            "paperless_billing": customer.get("PaperlessBilling"),
            "payment_method": customer.get("PaymentMethod"),
        },
        "services": {
            "phone_service": customer.get("PhoneService"),
            "multiple_lines": customer.get("MultipleLines"),
            "internet_service": customer.get("InternetService"),
            "online_security": customer.get("OnlineSecurity"),
            "online_backup": customer.get("OnlineBackup"),
            "device_protection": customer.get("DeviceProtection"),
            "tech_support": customer.get("TechSupport"),
            "streaming_tv": customer.get("StreamingTV"),
            "streaming_movies": customer.get("StreamingMovies"),
        },
        "charges": {
            "monthly_charges": float(customer.get("MonthlyCharges", 0)),
            "total_charges": (
                float(customer.get("TotalCharges", 0))
                if pd.notna(customer.get("TotalCharges"))
                else None
            ),
        },
        "actual_churn": bool(customer.get("Churn", 0)),
    }

    return json.dumps(profile, indent=2, default=str)


# ── Tool 2: Churn Risk Prediction ───────────────────────────────────────────


@tool
def predict_churn_risk(customer_id: str) -> str:
    """Predict the churn probability and risk segment for a customer.

    Uses the trained ML model from the MLflow Model Registry.
    Returns P(churn), risk segment (low/medium/high), and the model used.

    Args:
        customer_id: The unique customer identifier (e.g., '7590-VHVEG')
    """
    df = _get_df()
    row = df[df[ID_COL] == customer_id]

    if row.empty:
        return json.dumps({"error": f"Customer '{customer_id}' not found."})

    try:
        model = _get_model()
        pipeline = _get_pipeline()
        from src.preprocessing import prepare_data

        X, _, _ = prepare_data(row, fit_pipeline=False, pipeline=pipeline)

        # Handle both sklearn and MLflow pyfunc models
        if hasattr(model, "predict_proba"):
            churn_prob = float(model.predict_proba(X)[0, 1])
        else:
            # MLflow pyfunc
            pred = model.predict(pd.DataFrame(X))
            churn_prob = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred[0])

    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    from policy.intervention_engine import classify_risk

    risk_segment = classify_risk(churn_prob)

    result = {
        "customer_id": customer_id,
        "churn_probability": round(churn_prob, 4),
        "churn_probability_pct": f"{churn_prob * 100:.1f}%",
        "risk_segment": risk_segment,
        "model_source": "MLflow Registry" if _model is not None else "fallback",
    }

    return json.dumps(result, indent=2)


# ── Tool 3: SHAP Explanation ────────────────────────────────────────────────


@tool
def explain_prediction(customer_id: str) -> str:
    """Explain WHY a customer has their predicted churn risk using SHAP.

    Returns the top-3 factors driving the prediction, with direction
    (increases_risk / decreases_risk) and magnitude.

    Args:
        customer_id: The unique customer identifier (e.g., '7590-VHVEG')
    """
    try:
        from src.explainer import get_shap_explanation

        model = _get_model() if _model is not None else None
        pipeline = _get_pipeline() if _pipeline is not None else None
        explainer = _get_explainer() if _explainer is not None else None

        result = get_shap_explanation(
            customer_id=customer_id,
            df=_get_df(),
            pipeline=pipeline,
            explainer=explainer,
            model=model,
            top_n=3,
        )
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        result = {
            "customer_id": customer_id,
            "error": f"SHAP explanation unavailable: {str(e)}",
            "fallback_note": "Run training pipeline and SHAP explainer first.",
        }

    return json.dumps(result, indent=2, default=str)


# ── Tool 4: Policy Recommendation ───────────────────────────────────────────


@tool
def recommend_intervention(customer_id: str, budget: float = 5000.0) -> str:
    """Recommend the best retention intervention for a customer.

    Evaluates all eligible policies (discount, call, plan upgrade) and picks
    the one with the highest expected ROI. Returns the policy, cost, and expected benefit.

    Args:
        customer_id: The unique customer identifier (e.g., '7590-VHVEG')
        budget: Available monthly budget in dollars (default: $5,000)
    """
    from policy.intervention_engine import get_policy

    model = _get_model() if _model is not None else None
    pipeline = _get_pipeline() if _pipeline is not None else None

    result = get_policy(
        customer_id=customer_id,
        budget=budget,
        df=_get_df(),
        model=model,
        pipeline=pipeline,
    )

    return json.dumps(result, indent=2, default=str)


# ── Tool 5: Budget Simulation ───────────────────────────────────────────────


@tool
def simulate_budget_allocation(budget: float = 5000.0, top_n: int = 10) -> str:
    """Simulate how to allocate a retention budget across the highest-ROI customers.

    Uses a greedy algorithm: ranks all customers by expected ROI,
    then allocates interventions top-down until the budget is exhausted.

    Args:
        budget: Total monthly budget in dollars (default: $5,000)
        top_n: Maximum number of customers to target (default: 10)
    """
    from policy.intervention_engine import simulate_budget_allocation as sim

    model = _get_model() if _model is not None else None
    pipeline = _get_pipeline() if _pipeline is not None else None

    result = sim(
        budget=budget,
        top_n=top_n,
        df=_get_df(),
        model=model,
        pipeline=pipeline,
    )

    # Truncate allocations for readability in agent context.
    if "allocations" in result and len(result["allocations"]) > top_n:
        result["allocations"] = result["allocations"][:top_n]

    return json.dumps(result, indent=2, default=str)


# ── Tool List for Graph ──────────────────────────────────────────────────────

ALL_TOOLS = [
    get_customer_profile,
    predict_churn_risk,
    explain_prediction,
    recommend_intervention,
    simulate_budget_allocation,
]
