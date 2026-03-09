"""Intervention engine — decides what to do with at-risk customers.

For each customer the model flags as likely to churn, we pick the
best retention action (discount, personal call, or plan upgrade)
based on expected ROI.  When working with a list of customers and
a fixed monthly budget, we greedily assign actions starting from
whoever has the highest expected return.

Main entry points:
- `get_policy(customer_id, budget)` — what to do for one customer
- `simulate_budget_allocation(budget, n)` — ranked list for a campaign
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CHURN_RISK_THRESHOLDS,
    DEFAULT_MONTHLY_BUDGET,
    ID_COL,
    RAW_CSV,
)
from src.preprocessing import prepare_data

# ── Load cost matrix ─────────────────────────────────────────────────────────

COST_MATRIX_PATH = Path(__file__).parent / "cost_matrix.json"


def load_cost_matrix() -> dict:
    """Load the cost-benefit matrix from JSON."""
    with open(COST_MATRIX_PATH) as f:
        return json.load(f)


# ── Core Functions ───────────────────────────────────────────────────────────


def estimate_intervention_cost(
    policy: dict,
    monthly_charges: float,
) -> float:
    """How much does this intervention actually cost us?

    Discounts are percentage-based (a cut off their monthly bill for
    a few months), so the cost scales with what the customer pays.
    Calls and upgrades have a flat cost regardless of the customer's
    plan size.

    Parameters
    ----------
    policy : dict
        Policy config from cost_matrix.json.
    monthly_charges : float
        Customer's monthly charge.

    Returns
    -------
    float
        Total cost of the intervention.
    """
    if policy["cost_type"] == "percentage_of_charges":
        return monthly_charges * policy["cost_value"] * policy["cost_months"]
    else:
        return policy["fixed_cost"]


def estimate_ltv(
    monthly_charges: float,
    expected_remaining_months: float = 24,
) -> float:
    """Rough estimate of how much revenue this customer will generate
    if they stay.

    We keep it simple: monthly bill times expected remaining months.
    Default horizon is 24 months, which is a reasonable middle ground
    for telco customers.  Good enough for prioritization — we're not
    doing precise accounting here.

    Parameters
    ----------
    monthly_charges : float
        The customer's current monthly spend.
    expected_remaining_months : float
        How many more months we expect them to stay.

    Returns
    -------
    float
        Estimated remaining LTV.
    """
    return monthly_charges * expected_remaining_months


def compute_expected_roi(
    churn_probability: float,
    monthly_charges: float,
    policy: dict,
    expected_remaining_months: float = 24,
) -> dict[str, float]:
    """Is this intervention worth it for this customer?

    We compare two scenarios: do nothing vs. act.  If we do nothing,
    we keep the customer with probability (1 - churn_prob).  If we
    intervene, that retention probability goes up by whatever boost
    the policy offers.  The ROI is the extra expected revenue from
    the boost, minus the cost of the action.

    A positive ROI means it's worth doing.  A negative ROI means
    we'd spend more than we'd recover.

    Parameters
    ----------
    churn_probability : float
        Model-predicted P(churn).
    monthly_charges : float
        Customer's monthly charge.
    policy : dict
        Policy config.
    expected_remaining_months : float
        Expected customer lifetime in months.

    Returns
    -------
    dict
        benefit, cost, roi, and retention probabilities.
    """
    ltv = estimate_ltv(monthly_charges, expected_remaining_months)
    cost = estimate_intervention_cost(policy, monthly_charges)

    p_retention_no_action = 1 - churn_probability
    p_retention_with_action = min(
        1.0, p_retention_no_action + policy["retention_probability_boost"]
    )

    ev_no_action = ltv * p_retention_no_action
    ev_with_action = ltv * p_retention_with_action - cost
    roi = ev_with_action - ev_no_action

    return {
        "benefit": round(ev_with_action, 2),
        "cost": round(cost, 2),
        "roi": round(roi, 2),
        "ltv": round(ltv, 2),
        "retention_probability_without": round(p_retention_no_action, 4),
        "retention_probability_with": round(p_retention_with_action, 4),
    }


def classify_risk(churn_probability: float) -> str:
    """Bucket the churn probability into low / medium / high.

    Thresholds are 0.3 and 0.6 — below 30% we're not worried,
    above 60% it's urgent, in between we keep an eye on them.

    Returns
    -------
    str
        'low', 'medium', or 'high'
    """
    if churn_probability <= CHURN_RISK_THRESHOLDS["low"]:
        return "low"
    elif churn_probability <= CHURN_RISK_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"


def find_best_policy(
    churn_probability: float,
    monthly_charges: float,
    tenure: int,
) -> dict[str, Any]:
    """Pick the best action for this customer.

    We loop through all policies, filter out the ones this customer
    isn't eligible for (wrong risk tier, too new, too long-tenured),
    and return whichever has the highest expected ROI.  With only
    a handful of policies this is plenty fast.

    Parameters
    ----------
    churn_probability : float
    monthly_charges : float
    tenure : int

    Returns
    -------
    dict
        Best policy name, ROI details, and the policy config.
    """
    cm = load_cost_matrix()
    risk_segment = classify_risk(churn_probability)

    best_policy = None
    best_roi = float("-inf")
    best_details = None

    for policy_name, policy in cm["policies"].items():
        if risk_segment not in policy["target_segments"]:
            continue
        if tenure < policy["min_tenure"] or tenure > policy["max_tenure"]:
            continue

        roi_details = compute_expected_roi(churn_probability, monthly_charges, policy)

        if roi_details["roi"] > best_roi:
            best_roi = roi_details["roi"]
            best_policy = policy_name
            best_details = roi_details

    if best_policy is None:
        return {
            "recommended_policy": "no_action",
            "reason": f"Customer in '{risk_segment}' risk segment — no eligible intervention.",
            "risk_segment": risk_segment,
            "roi": 0.0,
        }

    return {
        "recommended_policy": best_policy,
        "policy_name": cm["policies"][best_policy]["name"],
        "policy_description": cm["policies"][best_policy]["description"],
        "risk_segment": risk_segment,
        **best_details,
    }


# ── Agent Tool: get_policy ───────────────────────────────────────────────────


def get_policy(
    customer_id: str,
    budget: float = DEFAULT_MONTHLY_BUDGET,
    df: pd.DataFrame | None = None,
    model=None,
    pipeline=None,
) -> dict[str, Any]:
    """Get the recommended intervention policy for a customer.

    Designed to be called by the ReAct agent's recommend_intervention tool.

    Parameters
    ----------
    customer_id : str
        The customer ID.
    budget : float
        Available monthly budget (used for context, not constraint in single-customer mode).
    df : pd.DataFrame | None
        Full dataset. Loaded from RAW_CSV if None.
    model
        Fitted prediction model.
    pipeline
        Fitted preprocessing pipeline.

    Returns
    -------
    dict
        Recommended policy with full ROI breakdown.
    """
    if df is None:
        df = pd.read_csv(RAW_CSV)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    customer_row = df[df[ID_COL] == customer_id]
    if customer_row.empty:
        return {"error": f"Customer {customer_id} not found"}

    row = customer_row.iloc[0]
    monthly_charges = float(row["MonthlyCharges"])
    tenure = int(row["tenure"])

    # Get churn probability from model
    if model is not None and pipeline is not None:
        X, _, _ = prepare_data(customer_row, fit_pipeline=False, pipeline=pipeline)
        if hasattr(model, "predict_proba"):
            churn_prob = float(model.predict_proba(X)[0, 1])
        else:
            pred = model.predict(pd.DataFrame(X))
            churn_prob = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred[0])
    else:
        raise RuntimeError(
            "Prediction model and pipeline are required. Run src/train.py first."
        )

    result = find_best_policy(churn_prob, monthly_charges, tenure)
    result["customer_id"] = customer_id
    result["monthly_charges"] = monthly_charges
    result["tenure"] = tenure
    result["churn_probability"] = round(churn_prob, 4)
    result["available_budget"] = budget

    return result


# ── Agent Tool: simulate_budget_allocation ───────────────────────────────────


def simulate_budget_allocation(
    budget: float = DEFAULT_MONTHLY_BUDGET,
    top_n: int = 20,
    df: pd.DataFrame | None = None,
    model=None,
    pipeline=None,
) -> dict[str, Any]:
    """Given a monthly budget, figure out which customers to target and how.

    We score every customer, rank them by expected ROI, and greedily
    assign actions from the top down until we run out of budget or
    hit the top_n cap.  If a particular action would blow the remaining
    budget we skip that customer and try the next one — hence we look
    at top_n * 3 candidates to have enough headroom.

    Parameters
    ----------
    budget : float
        Total monthly budget.
    top_n : int
        Maximum number of customers to target.
    df : pd.DataFrame | None
        Full dataset.
    model
        Prediction model.
    pipeline
        Preprocessing pipeline.

    Returns
    -------
    dict
        Budget summary plus the list of customer allocations.
    """
    if df is None:
        df = pd.read_csv(RAW_CSV)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Get predictions for all customers
    if model is not None and pipeline is not None:
        X, _, _ = prepare_data(df, fit_pipeline=False, pipeline=pipeline)
        if hasattr(model, "predict_proba"):
            churn_probs = model.predict_proba(X)[:, 1]
        else:
            pred = model.predict(pd.DataFrame(X))
            churn_probs = pred.iloc[:, 0].values if hasattr(pred, "iloc") else np.array(pred).flatten()
    else:
        raise RuntimeError(
            "Prediction model and pipeline are required for budget allocation. Run src/train.py first."
        )

    # Build allocation candidates
    candidates = []
    for idx, row in df.iterrows():
        churn_prob = float(churn_probs[idx] if idx < len(churn_probs) else 0.5)
        monthly_charges = float(row["MonthlyCharges"])
        tenure = int(row["tenure"])

        policy_result = find_best_policy(churn_prob, monthly_charges, tenure)
        if policy_result["recommended_policy"] == "no_action":
            continue

        candidates.append(
            {
                "customer_id": row[ID_COL],
                "churn_probability": round(churn_prob, 4),
                "monthly_charges": monthly_charges,
                "tenure": tenure,
                **policy_result,
            }
        )

    candidates.sort(key=lambda x: x.get("roi", 0), reverse=True)

    allocations = []
    total_cost = 0.0
    total_roi = 0.0

    for candidate in candidates[:top_n * 3]:
        cost = candidate.get("cost", 0)
        if total_cost + cost > budget:
            continue
        if len(allocations) >= top_n:
            break

        total_cost += cost
        total_roi += candidate.get("roi", 0)
        allocations.append(candidate)

    return {
        "budget": budget,
        "total_cost": round(total_cost, 2),
        "total_expected_roi": round(total_roi, 2),
        "customers_targeted": len(allocations),
        "budget_utilization": round(total_cost / budget * 100, 1) if budget > 0 else 0,
        "roi_per_dollar_spent": round(total_roi / total_cost, 2) if total_cost > 0 else 0,
        "allocations": allocations[:top_n],
    }


# ── Simulation: With Model vs Without ────────────────────────────────────────


def run_simulation_comparison(
    budget: float = DEFAULT_MONTHLY_BUDGET,
    df: pd.DataFrame | None = None,
    model=None,
    pipeline=None,
) -> dict[str, Any]:
    """Compare targeted vs random outreach with the same budget.

    Targeted: we pick the highest-ROI customers using the model.
    Random: we call the same number of random customers with no
    selection logic, estimating a 5% retention rate for blind
    outreach — a typical result for untargeted telco campaigns.

    Returns
    -------
    dict
        Side-by-side numbers for both strategies.
    """
    targeted = simulate_budget_allocation(
        budget=budget, top_n=100, df=df, model=model, pipeline=pipeline
    )

    # Random strategy: pick random customers and apply cheapest intervention
    if df is None:
        df = pd.read_csv(RAW_CSV)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    cm = load_cost_matrix()
    cheapest_cost = min(
        p["fixed_cost"]
        for p in cm["policies"].values()
        if p["fixed_cost"] > 0
    )
    if cheapest_cost == 0:
        cheapest_cost = 15.0
    random_n = int(budget / cheapest_cost)

    random_sample = df.sample(n=min(random_n, len(df)), random_state=42)
    # 5% retention rate for random outreach — rough but reasonable baseline.
    random_roi = random_sample["MonthlyCharges"].sum() * 0.05

    return {
        "budget": budget,
        "targeted_strategy": {
            "customers_targeted": targeted["customers_targeted"],
            "total_roi": targeted["total_expected_roi"],
            "cost": targeted["total_cost"],
        },
        "random_strategy": {
            "customers_targeted": random_n,
            "estimated_roi": round(random_roi, 2),
            "cost": round(random_n * cheapest_cost, 2),
        },
        "improvement_pct": round(
            (targeted["total_expected_roi"] - random_roi) / max(random_roi, 1) * 100, 1
        ),
    }


def sensitivity_analysis(
    base_budget: float = DEFAULT_MONTHLY_BUDGET,
    variations: list[float] | None = None,
    df: pd.DataFrame | None = None,
    model=None,
    pipeline=None,
) -> list[dict]:
    """How does ROI change as we increase or decrease the budget?

    Runs the allocation at several budget levels (by default from
    60% to 150% of the base) so you can see where you start getting
    diminishing returns and where cutting the budget hurts the most.

    Parameters
    ----------
    base_budget : float
        Reference budget level.
    variations : list[float]
        Budget multipliers to test.

    Returns
    -------
    list[dict]
        Results for each budget level.
    """
    if variations is None:
        variations = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]

    results = []
    for mult in variations:
        budget = base_budget * mult
        sim = simulate_budget_allocation(
            budget=budget, top_n=100, df=df,
            model=model, pipeline=pipeline,
        )
        results.append(
            {
                "budget_multiplier": mult,
                "budget": budget,
                "customers_targeted": sim["customers_targeted"],
                "total_roi": sim["total_expected_roi"],
                "total_cost": sim["total_cost"],
                "roi_per_dollar": sim["roi_per_dollar_spent"],
            }
        )

    return results
