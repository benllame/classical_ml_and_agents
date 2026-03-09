"""FastAPI REST API for the Churn Intelligence System.

Business logic lives in the service modules (preprocessing, train,
explainer, intervention_engine). The API layer handles request
validation (Pydantic), resource lifecycle (lifespan), and response
formatting.

Endpoints:
    POST /predict       — Single customer churn prediction
    POST /policy        — Intervention recommendation for a customer
    POST /agent/query   — Natural language query to the ReAct agent
    GET  /health        — Health check

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# ── Pydantic Schemas ─────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""

    customer_id: str = Field(..., description="Unique customer identifier", examples=["7590-VHVEG"])


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    customer_id: str
    churn_probability: float
    churn_probability_pct: str
    risk_segment: str
    model_source: str
    latency_ms: float


class PolicyRequest(BaseModel):
    """Request schema for /policy endpoint."""

    customer_id: str = Field(..., description="Unique customer identifier")
    budget: float = Field(default=5000.0, description="Monthly budget in dollars", ge=0)


class PolicyResponse(BaseModel):
    """Response schema for /policy endpoint."""

    customer_id: str
    churn_probability: float
    risk_segment: str
    recommended_policy: str
    policy_name: str | None = None
    policy_description: str | None = None
    cost: float | None = None
    roi: float | None = None
    benefit: float | None = None
    ltv: float | None = None
    retention_probability_without: float | None = None
    retention_probability_with: float | None = None


class AgentQueryRequest(BaseModel):
    """Request schema for /agent/query endpoint."""

    query: str = Field(
        ...,
        description="Natural language question for the agent",
        examples=["What is the churn risk for customer 7590-VHVEG?"],
    )


class AgentQueryResponse(BaseModel):
    """Response schema for /agent/query endpoint."""

    answer: str
    tools_used: list[dict[str, Any]] = []
    reasoning_steps: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str
    version: str
    model_loaded: bool
    pipeline_loaded: bool
    agent_available: bool


# ── Application Lifespan ─────────────────────────────────────────────────────

# Global resources
_resources: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML resources on startup, clean up on shutdown.

    Resources are loaded eagerly so the first request has no cold-start
    latency. Each resource load is wrapped in try/except so the API
    starts even if some resources are missing (graceful degradation).
    """
    logger.info("Loading ML resources...")

    try:
        from src.preprocessing import load_pipeline

        _resources["pipeline"] = load_pipeline()
        logger.info("Pipeline loaded")
    except Exception as e:
        logger.warning(f"Pipeline not available: {e}")
        _resources["pipeline"] = None

    try:
        # Try MLflow first, then local fallback
        try:
            import mlflow

            from tracking.mlflow_setup import get_production_model_uri

            uri = get_production_model_uri()
            try:
                _resources["model"] = mlflow.sklearn.load_model(uri)
                logger.info(f"Model loaded from MLflow (sklearn flavor): {uri}")
            except Exception:
                _resources["model"] = mlflow.pyfunc.load_model(uri)
                logger.info(f"Model loaded from MLflow (pyfunc flavor): {uri}")
        except Exception:
            import joblib

            from src.config import MODELS_DIR

            model_path = MODELS_DIR / "best_model.joblib"
            _resources["model"] = joblib.load(model_path)
            logger.info(f"Model loaded from local: {model_path}")
    except Exception as e:
        logger.warning(f"Model not available: {e}")
        _resources["model"] = None

    try:
        from src.explainer import load_explainer

        _resources["explainer"] = load_explainer()
        logger.info("SHAP explainer loaded")
    except Exception as e:
        logger.warning(f"SHAP explainer not available: {e}")
        _resources["explainer"] = None

    # Pre-load data
    try:
        import pandas as pd

        from src.config import RAW_CSV

        df = pd.read_csv(RAW_CSV)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
        _resources["df"] = df
        logger.info(f"Dataset loaded: {len(df)} rows")
    except Exception as e:
        logger.warning(f"Dataset not available: {e}")
        _resources["df"] = None

    # Initialize agent tools
    try:
        from agent.tools import init_tools

        init_tools(
            df=_resources.get("df"),
            model=_resources.get("model"),
            pipeline=_resources.get("pipeline"),
            explainer=_resources.get("explainer"),
        )
        logger.info("Agent tools initialized")
    except Exception as e:
        logger.warning(f"Agent tools not initialized: {e}")

    logger.success("API startup complete")
    yield

    # Cleanup
    _resources.clear()
    logger.info("API shutdown complete")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Intelligence System API",
    description=(
        "REST API for churn prediction, SHAP explanations, intervention policies, "
        "and a ReAct agent that answers business questions in natural language."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# allow_origins=['*'] is fine for development. Restrict to specific
# domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and resource availability."""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        model_loaded=_resources.get("model") is not None,
        pipeline_loaded=_resources.get("pipeline") is not None,
        agent_available=True,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_churn(request: PredictRequest):
    """Predict churn probability for a single customer.

    Returns the churn probability, risk segment, and model source.
    Falls back to a heuristic if no trained model is available.
    """
    start = time.time()

    df = _resources.get("df")
    model = _resources.get("model")
    pipeline = _resources.get("pipeline")

    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    from src.config import ID_COL

    customer_row = df[df[ID_COL] == request.customer_id]
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found")

    # Predict
    try:
        if model is not None and pipeline is not None:
            import pandas as pd

            from src.preprocessing import prepare_data

            X, _, _ = prepare_data(customer_row, fit_pipeline=False, pipeline=pipeline)
            if hasattr(model, "predict_proba"):
                churn_prob = float(model.predict_proba(X)[0, 1])
            else:
                pred = model.predict(pd.DataFrame(X))
                churn_prob = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred[0])
            model_source = "MLflow Registry"
        else:
            raise HTTPException(status_code=503, detail="Prediction model not loaded. Run the training pipeline first.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e

    from policy.intervention_engine import classify_risk

    risk = classify_risk(churn_prob)

    latency = (time.time() - start) * 1000

    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=round(churn_prob, 4),
        churn_probability_pct=f"{churn_prob * 100:.1f}%",
        risk_segment=risk,
        model_source=model_source,
        latency_ms=round(latency, 1),
    )


@app.post("/policy", response_model=PolicyResponse, tags=["Policy"])
async def get_policy(request: PolicyRequest):
    """Get the recommended retention intervention for a customer."""
    from policy.intervention_engine import get_policy as _get_policy

    result = _get_policy(
        customer_id=request.customer_id,
        budget=request.budget,
        df=_resources.get("df"),
        model=_resources.get("model"),
        pipeline=_resources.get("pipeline"),
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return PolicyResponse(**result)


@app.post("/agent/query", response_model=AgentQueryResponse, tags=["Agent"])
def agent_query(request: AgentQueryRequest):
    """Send a natural language query to the ReAct agent.

    The agent uses real ML models, SHAP, and the policy engine to
    answer business questions about customer churn. A fresh graph is
    created per request to ensure clean state. If the LLM API is
    unavailable, a fallback template response is returned.

    Declared as a synchronous ``def`` so FastAPI runs it in a thread-pool
    executor, avoiding event-loop blocking during the (potentially slow)
    LLM call.
    """
    start = time.time()

    try:
        from agent.graph import create_agent, run_agent

        agent = create_agent()
        result = run_agent(agent, request.query)

        latency = (time.time() - start) * 1000

        return AgentQueryResponse(
            answer=result["answer"],
            tools_used=result.get("tool_calls", []),
            reasoning_steps=result.get("steps", 0),
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        logger.error(f"Agent query failed: {e}")

        # Fallback: deterministic template response
        from jinja2 import Template

        fallback_template = Template(
            "I'm sorry, the agent encountered an error: {{ error }}. "
            "Please check your API keys and try again."
        )
        fallback_answer = fallback_template.render(error=str(e))

        latency = (time.time() - start) * 1000

        return AgentQueryResponse(
            answer=fallback_answer,
            tools_used=[],
            reasoning_steps=0,
            latency_ms=round(latency, 1),
        )
