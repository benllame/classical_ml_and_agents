"""Project configuration — one place for all constants and paths.

Everything lives here: file paths, feature lists, model names, thresholds.
If you need to change a number or a path, this is the only file you touch.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Data files ────────────────────────────────────────────────────────────────
RAW_CSV = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_CSV = PROCESSED_DIR / "telco_churn_processed.csv"

# ── Model artifacts ───────────────────────────────────────────────────────────
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

# ── MLflow ────────────────────────────────────────────────────────────────────
# Defaults to a local folder for development. In production you'd point this
# to a shared server (Databricks, AWS, etc.).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-benchmark")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-predictor")

# ── LLM / Agent ──────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
# gemini-2.0-flash is fast, handles tool-calling well, and has a generous free tier.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
# temperature=0.0 makes the agent's tool selections reproducible.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Feature Engineering ──────────────────────────────────────────────────────
# These match the column names in the Kaggle Telco dataset exactly.
TARGET_COL = "Churn"
ID_COL = "customerID"

# Numeric features are continuous — we z-score them so all models see
# features on the same scale.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Binary features are Yes/No or Male/Female columns.
# We encode them as 0/1 — simple and lossless.
BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Nominal features have multiple unordered categories (e.g. payment method).
# We one-hot encode them to avoid the model treating them as ordered numbers.
NOMINAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# ── Policy Engine ─────────────────────────────────────────────────────────────
# ~$5k/month is roughly 1% of estimated monthly revenue for this dataset
# (~7,000 customers × ~$70 avg charge). A reasonable starting budget for a
# retention campaign.
DEFAULT_MONTHLY_BUDGET = 5000.0

# How much each intervention improves the probability of keeping a customer.
# These are additive boosts to the base retention probability.
RETENTION_PROBABILITY_BOOST = {
    "discount_10pct": 0.15,
    "personal_call": 0.20,
    "plan_upgrade": 0.25,
}

# Cost per customer for each intervention type.
# Discount cost is calculated dynamically (% of charges), so it's 0 here.
# Personal call ≈ 15 min of a retention agent's time.
# Plan upgrade ≈ cost of giving 2 months of upgraded service.
INTERVENTION_COST = {
    "discount_10pct": 0.0,  # cost calculated as % of charges
    "personal_call": 15.0,
    "plan_upgrade": 30.0,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
# We bucket customers into three risk tiers: low / medium / high.
# Below 30% we're not worried. Above 60% it's urgent. In between, watch them.
# These cutoffs roughly split the telco churn distribution into thirds.
CHURN_RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}

# ── Benchmark ────────────────────────────────────────────────────────────────

# How many random seeds to use in the repeated benchmark.
# 10 runs is enough for a meaningful statistical comparison while keeping
# total runtime reasonable (~10-20 min on CPU).
N_BENCHMARK_RUNS: int = int(os.getenv("N_BENCHMARK_RUNS", "10"))

# ── Information Theory ────────────────────────────────────────────────────────

# Features with MI below this value are basically noise at this sample size.
MI_THRESHOLD = 0.01  # minimum MI (nats) to consider a feature informative

# k-nearest neighbours used by sklearn's mutual_info_classif.
# k=5 is a solid default for ~7,000 samples.
MI_N_NEIGHBORS = 5  # k for k-NN MI estimator (sklearn)

# Bins for discretising continuous variables when computing entropy directly.
# 10 bins works well for sample sizes above ~1,000.
MI_N_BINS = 10  # bins for discretizing continuous vars in entropy computations

# Flag feature pairs where one feature shares more than 50% of its
# target information with another — they're largely redundant.
MI_REDUNDANCY_THRESHOLD = 0.5  # flag feature pairs with > 50% redundancy ratio
