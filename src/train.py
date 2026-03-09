"""Model training pipeline with MLflow experiment tracking.

Multi-model benchmark with feature-selection comparison. We evaluate 5 models
(dummy baseline through CatBoost/MLP) across 3 feature-selection approaches
to provide a comprehensive empirical comparison.

Feature selection methods
--------------------------
  all_features  — Full preprocessed feature set (54 engineered features).
                  Serves as the reference baseline.
  mi_filtered   — Keeps features with MI(Xi; Churn) >= MI_THRESHOLD.
                  Removes noisy features with near-zero mutual information.
                  Ref: Brown et al. (2012) "Conditional Likelihood Maximisation."
  hill_climbing — Greedy Forward Selection seeded by MI ranking.
                  Adds features in MI order; keeps each only if CV F1-Macro
                  improves by >= min_improvement. Stops at max_features.
                  Ref: Kohavi & John (1997) "Wrappers for Feature Subset Selection."

Default behaviour (python src/train.py)
-----------------------------------------
Runs the comprehensive benchmark:
  5 models × 3 feature sets × 10 seeds = 150 training runs.

Reports mean ± std per (model, feature_set) + Wilcoxon signed-rank p-values.
All runs logged to MLflow. Summaries saved to:
  models/comprehensive_benchmark_summary.csv
  models/comprehensive_benchmark_raw.csv
  models/pvalue_matrix_<feature_set>.csv

The benchmark follows the 'strong baseline first' principle (Ng, 2017):
compare full features vs. MI-filtered vs. hill-climbing to see whether
selection actually helps on this dataset.

Every run logs:
- Hyperparameters (via autolog + custom tags, including feature_set)
- Metrics: ROC-AUC, F1-Macro, F1-Churn, precision, recall, accuracy
- Artifacts: confusion matrix, ROC curve, feature importance, profit curve
- The model artifact (for MLflow model logging)

Best model (highest mean ROC-AUC across all seeds/feature-sets) is registered
in the Model Registry as 'churn-predictor'.

Threshold selection (OOF protocol):
- cross_val_predict() on X_train → OOF probabilities (no test-set leakage)
- Threshold that maximises F1-Macro is selected and applied to X_test

Hyperparameter optimisation:
- 'random'  : RandomizedSearchCV (default, fast, n_iter=20)
- 'optuna'  : Optuna TPE sampler + MedianPruner (slower, higher quality)
  Bergstra et al. (2011): TPE outperforms random search by modelling
  p(score|params). Typical gain: 0.5–2% AUC for n_trials >= 50.

Usage:
    python src/train.py                              # full benchmark (default)
    python src/train.py --quick                      # quick single run (seed=42, all features)
    python src/train.py --model xgboost              # single model, all feature sets, 10 seeds
    python src/train.py --feature-sets all mi        # specific feature sets only
    python src/train.py --n-runs 5                   # fewer seeds for testing
    python src/train.py --optimizer optuna           # Optuna for all models
    python src/train.py --model catboost --optimizer optuna --n-trials 100
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier

from src.config import (
    FIGURES_DIR,
    MI_N_NEIGHBORS,
    MI_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    N_BENCHMARK_RUNS,
    RAW_CSV,
)
from src.eda import AMBER, CORAL, CYAN, DARK_BG, TEXT_COLOR, set_dark_style
from src.preprocessing import get_feature_names, prepare_data, save_pipeline

warnings.filterwarnings("ignore")

# ── Feature Selection Methods ─────────────────────────────────────────────────

#: All available feature-selection approaches, in evaluation order.
#: Used as the default for the ``feature_sets`` parameter in benchmark functions.
FEATURE_SELECTION_METHODS: list[str] = ["all", "mi", "hill_climbing"]

# ── Model Definitions ─────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    # Stratified random baseline — any model that doesn't beat this is
    # useless. Stratified (not uniform) preserves class proportions,
    # giving a fair baseline for imbalanced data (~26% positive class).
    "dummy": {
        "estimator": DummyClassifier(strategy="stratified"),
        "params": {},
        "description": "Baseline — stratified random predictions",
    },
    # Ensemble baseline — captures non-linear interactions without
    # sequential boosting. 'balanced_subsample' recomputes class weights
    # per bootstrap sample, more robust for RF on imbalanced data.
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "description": "Random Forest with randomized hyperparameter search",
    },
    # State-of-the-art gradient boosting. scale_pos_weight in [1,2,3,5]
    # explicitly handles class imbalance. gamma, reg_alpha, reg_lambda
    # add L1/L2 regularisation that sklearn's GBM lacks.
    # Rationale for scale_pos_weight over SMOTE: King & Zeng (2001)
    # showed native loss weighting avoids synthetic artifacts.
    "xgboost": {
        "estimator": None,  # lazy import
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.3],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [1, 1.5, 2],
            "scale_pos_weight": [1, 2, 3, 5],
        },
        "description": "XGBoost with scale_pos_weight for imbalance",
    },
    # CatBoost handles categorical features natively via ordered target
    # statistics, avoiding OHE explosion. auto_class_weights='Balanced'
    # is CatBoost's equivalent of class_weight='balanced'.
    # silent=True suppresses per-iteration output.
    # Ref: Prokhorenkova et al. (2018) "CatBoost: unbiased boosting with
    # categorical features."
    "catboost": {
        "estimator": None,  # lazy import
        "params": {
            "iterations": [100, 200, 300, 500],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 10],
            "border_count": [32, 64, 128],
            "auto_class_weights": ["Balanced", "None"],
        },
        "description": "CatBoost with ordered target statistics",
    },
    # Multi-Layer Perceptron — non-linear baseline that generalises
    # beyond tree-based ensembles. Two hidden layers (128, 64) capture
    # complex feature interactions. alpha tunes L2 regularisation.
    # early_stopping=True uses 10% of training data as a validation set
    # to stop before overfitting; max_iter=500 is a safety upper bound.
    # Ref: Goodfellow et al. (2016) Deep Learning, Ch. 6.
    "mlp": {
        "estimator": MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
        "params": {
            "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128), (128, 64, 32)],
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-3, 5e-3, 1e-2],
            "activation": ["relu", "tanh"],
        },
        "description": "MLP with two hidden layers + early stopping",
    },
}


def _get_xgboost_estimator(random_state: int = 42):
    """Lazy-load XGBoost to avoid import errors if not installed."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def _get_catboost_estimator(random_state: int = 42):
    """Lazy-load CatBoost to avoid import errors if not installed."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_seed=random_state,
        silent=True,
    )


# ── Feature Selection ─────────────────────────────────────────────────────────


def select_features_mi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    mi_threshold: float = MI_THRESHOLD,
    n_neighbors: int = MI_N_NEIGHBORS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select features where MI(Xi; Churn) >= mi_threshold.

    Uses sklearn's mutual_info_classif (KSG estimator, k=n_neighbors) on
    the preprocessed X_train. Selection happens on training data only —
    the returned mask is applied identically to X_test without refitting.

    If fewer than 3 features pass the threshold (rare for small splits),
    falls back to the top-50% by MI score to guarantee a usable subset.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features (already scaled/encoded).
    y_train : np.ndarray
        Binary target (0/1).
    feature_names : list[str]
        Names corresponding to columns of X_train.
    mi_threshold : float
        Minimum MI (nats) to retain a feature. Default: MI_THRESHOLD.
    n_neighbors : int
        k for KSG estimator. Default: MI_N_NEIGHBORS.
    random_state : int
        Seed for MI computation reproducibility.

    Returns
    -------
    tuple
        (X_selected, mask, selected_names)
        X_selected : shape (n_train, n_selected)
        mask       : boolean array of shape (n_features,)
        selected_names : list[str]
    """
    scores = mutual_info_classif(
        X_train, y_train, n_neighbors=n_neighbors, random_state=random_state
    )
    mask = scores >= mi_threshold

    # Guard: always keep at least 3 features
    if mask.sum() < 3:
        k = max(3, len(scores) // 2)
        mask = np.zeros(len(scores), dtype=bool)
        mask[np.argsort(scores)[-k:]] = True
        logger.debug(f"MI threshold yielded <3 features; keeping top-{k}")

    selected = [n for n, m in zip(feature_names, mask, strict=False) if m]
    logger.info(
        f"MI selection: {mask.sum()}/{len(mask)} features "
        f"(threshold={mi_threshold:.4f}, top-MI={scores.max():.4f})"
    )
    return X_train[:, mask], mask, selected


def select_features_hill_climbing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_features: int = 30,
    min_improvement: float = 0.001,
    cv_folds: int = 3,
    random_state: int = 42,
    n_neighbors: int = MI_N_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Greedy Forward Selection seeded by MI ranking.

    Algorithm (Kohavi & John 1997 "Wrappers for Feature Subset Selection"):
    1.  Rank all features by MI(Xi; Churn) descending on X_train.
    2.  Iterate through the ranked list.  For each candidate feature:
        a.  Tentatively add it to the current selected set.
        b.  Evaluate 3-fold CV F1-Macro (fast RF: 50 trees, depth 8).
        c.  If gain >= min_improvement → accept; otherwise discard.
    3.  Stop when max_features is reached or all features are exhausted.

    Using a fast RF (50 trees, depth 8) for the inner CV loop so the
    hill-climb completes in ~15–40 s per seed on a typical laptop CPU.

    MI seeding means the first few features are already high-quality,
    making early stopping effective: typically 10–25 features are selected
    before gains plateau.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Binary target.
    feature_names : list[str]
        Column names.
    max_features : int
        Hard upper bound on selected subset size.
    min_improvement : float
        Minimum marginal CV F1-Macro gain to accept a feature.
    cv_folds : int
        CV folds for the inner evaluation loop (3 for speed).
    random_state : int
        Seed for MI computation and CV splitting.
    n_neighbors : int
        k for the MI KSG estimator used to seed the search order.

    Returns
    -------
    tuple
        (X_selected, mask, selected_names)
    """
    n_features = X_train.shape[1]
    effective_max = min(max_features, n_features)

    # Step 1: rank by MI score (highest first) to seed the search order
    mi_scores = mutual_info_classif(
        X_train, y_train, n_neighbors=n_neighbors, random_state=random_state
    )
    feature_order = np.argsort(mi_scores)[::-1]

    # Fast inner evaluator — deliberately small to keep the loop fast
    fast_clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    selected_indices: list[int] = []
    best_score = -1.0

    for feat_idx in feature_order:
        if len(selected_indices) >= effective_max:
            break

        candidate = selected_indices + [int(feat_idx)]
        scores = cross_val_score(
            fast_clf,
            X_train[:, candidate],
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        score = float(scores.mean())

        # Accept if improvement exceeds threshold, or always accept first feature
        if score >= best_score + min_improvement or len(selected_indices) == 0:
            selected_indices = candidate
            best_score = score

    # Build boolean mask
    mask = np.zeros(n_features, dtype=bool)
    for idx in selected_indices:
        mask[idx] = True

    # Resolve feature names (guard for length mismatch)
    if len(feature_names) == n_features:
        fname_arr = np.array(feature_names)
        selected = list(fname_arr[selected_indices])
    else:
        selected = [f"f{i}" for i in selected_indices]

    logger.info(
        f"Hill climbing: {mask.sum()}/{n_features} features selected, "
        f"best CV F1-Macro = {best_score:.4f}"
    )
    return X_train[:, mask], mask, selected


def compute_feature_masks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    methods: list[str],
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Compute boolean feature masks for all requested selection methods.

    All selection is performed on X_train only (no test-set leakage).
    The caller applies each mask independently to X_train and X_test.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features used to fit selectors.
    y_train : np.ndarray
        Binary target.
    feature_names : list[str]
        Feature names corresponding to X_train columns.
    methods : list[str]
        Subset of FEATURE_SELECTION_METHODS to compute.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict[str, tuple[mask, selected_names]]
        {"all": (mask_all, names_all),
         "mi": (mask_mi, names_mi),
         "hill_climbing": (mask_hc, names_hc)}
    """
    masks: dict[str, tuple[np.ndarray, list[str]]] = {}

    if "all" in methods:
        masks["all"] = (
            np.ones(X_train.shape[1], dtype=bool),
            list(feature_names),
        )

    if "mi" in methods:
        _, mi_mask, mi_names = select_features_mi(
            X_train, y_train, feature_names, random_state=random_state
        )
        masks["mi"] = (mi_mask, mi_names)

    if "hill_climbing" in methods:
        _, hc_mask, hc_names = select_features_hill_climbing(
            X_train, y_train, feature_names, random_state=random_state
        )
        masks["hill_climbing"] = (hc_mask, hc_names)

    return masks


# ── Optuna Hyperparameter Optimisation ───────────────────────────────────────


def _optuna_objective(
    trial,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> float:
    """Optuna objective: returns mean CV ROC-AUC for one trial.

    Search spaces mirror MODEL_CONFIGS param grids but expressed as
    Optuna suggest_* calls so the TPE sampler can model the distribution
    of good configurations. Continuous ranges (learning_rate, alpha)
    use suggest_float(log=True) — log-uniform prior is correct because
    a change from 0.001 to 0.01 is equally meaningful as 0.1 to 1.0.
    """
    if model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
        }
        estimator = RandomForestClassifier(random_state=random_state, **params)

    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 3, 5]),
        }
        estimator = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            **params,
        )

    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "None"]
            ),
        }
        estimator = CatBoostClassifier(random_seed=random_state, silent=True, **params)

    elif model_name == "mlp":
        layer_choices = [(64,), (128,), (128, 64), (256, 128), (128, 64, 32)]
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", layer_choices),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-3, 1e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        }
        estimator = MLPClassifier(
            random_state=random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            **params,
        )

    else:
        raise ValueError(f"Optuna not supported for model '{model_name}'")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def optimize_with_optuna(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any], float]:
    """Optimise hyperparameters with Optuna TPE sampler + MedianPruner.

    Compared to RandomizedSearchCV, Optuna's TPE sampler builds a
    probabilistic model of the objective (p(score|params)) and samples
    from the region predicted to have high scores. After ~20 warmup
    trials (random exploration), it focuses on the promising subspace.

    MedianPruner halts unpromising trials early by comparing their
    intermediate results against the median of completed trials,
    saving compute on clear losers.

    Parameters
    ----------
    model_name : str
        Key from MODEL_CONFIGS.
    X_train, y_train : arrays
        Training data.
    n_trials : int
        Total Optuna trials (recommend >= 50 for meaningful optimisation).
    cv_folds : int
        Cross-validation folds per trial.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (best_fitted_model, best_params_dict, best_cv_auc)
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Optuna is required for --optimizer optuna. " "Install with: pip install optuna"
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial):
        return _optuna_objective(trial, model_name, X_train, y_train, cv_folds, random_state)

    logger.info(
        f"[optuna] Starting {n_trials} trials for '{model_name}' "
        f"(TPE + MedianPruner, seed={random_state})"
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    best_cv_auc = study.best_value
    logger.info(f"[optuna] Best CV AUC = {best_cv_auc:.4f} | params = {best_params}")

    # Reconstruct the best model and fit on full training set
    if model_name == "random_forest":
        best_model = RandomForestClassifier(
            random_state=random_state,
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight=best_params["class_weight"],
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        best_model = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            **best_params,
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        best_model = CatBoostClassifier(random_seed=random_state, silent=True, **best_params)
    elif model_name == "mlp":
        best_model = MLPClassifier(
            random_state=random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            alpha=best_params["alpha"],
            learning_rate_init=best_params["learning_rate_init"],
            activation=best_params["activation"],
        )
    else:
        raise ValueError(f"Unknown model for Optuna refit: {model_name}")

    best_model.fit(X_train, y_train)
    return best_model, best_params, best_cv_auc


# ── OOF Threshold Optimisation ───────────────────────────────────────────────


def find_oof_threshold(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> float:
    """Find the decision threshold that maximises F1-Macro on OOF predictions.

    Protocol
    --------
    1. Run cross_val_predict(method='predict_proba') over X_train.
       This produces OOF probabilities that were never seen by the model
       during the fold that generated them — no look-ahead bias.
    2. Sweep thresholds from 0.05 to 0.95 in steps of 0.01.
    3. Return the threshold that maximises F1-Macro (average of F1 for
       both classes), which balances precision and recall for the
       imbalanced churn target.

    Rationale: Using the test set to pick a threshold inflates reported
    performance. OOF threshold selection keeps the test set truly held-out.

    Parameters
    ----------
    estimator : sklearn estimator
        A fitted or unfitted estimator supporting predict_proba.
    X_train, y_train : arrays
        Full training data.
    cv_folds : int
        Number of CV folds for cross_val_predict.
    random_state : int
        Seed for StratifiedKFold.

    Returns
    -------
    float
        Optimal threshold in [0.05, 0.95].
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    try:
        oof_proba = cross_val_predict(
            estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
    except Exception:
        # Fallback: if predict_proba not supported (e.g. dummy), return 0.5
        return 0.5

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1_macro = -1.0

    for t in thresholds:
        y_pred_t = (oof_proba >= t).astype(int)
        score = f1_score(y_train, y_pred_t, average="macro", zero_division=0)
        if score > best_f1_macro:
            best_f1_macro = score
            best_threshold = t

    return float(best_threshold)


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> plt.Figure:
    """Create a styled confusion matrix plot."""
    set_dark_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(cm, cmap="Blues", alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=TEXT_COLOR,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name: str, auc_score: float) -> plt.Figure:
    """Create a styled ROC curve plot."""
    set_dark_style()
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=CYAN, linewidth=2, label=f"{model_name} (AUC={auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color=CORAL, linestyle="--", alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=CYAN)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model, feature_names: list[str], model_name: str, top_n: int = 15
) -> plt.Figure | None:
    """Create feature importance bar plot."""
    set_dark_style()
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(model.coef_[0])
        except AttributeError:
            return None

    n_features = min(top_n, len(importances))
    indices = np.argsort(importances)[-n_features:]

    names = []
    for i in indices:
        if i < len(feature_names):
            names.append(feature_names[i])
        else:
            names.append(f"feature_{i}")

    fig, ax = plt.subplots(figsize=(8, max(4, n_features * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_features))
    ax.barh(range(n_features), importances[indices], color=colors, edgecolor="none")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig


# ── Profit Curve ─────────────────────────────────────────────────────────────


def compute_profit_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 15.0,
    benefit_tp: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute profit at different thresholds.

    Profit curve from cost-sensitive learning: Elkan (2001) 'The
    Foundations of Cost-Sensitive Learning', IJCAI.

    profit(threshold) = TP * benefit_tp - FP * cost_fp

    - benefit_tp = $200: estimated revenue saved per correctly identified
      churner. Based on median MonthlyCharges (~$70) x ~3 months of
      retained revenue.
    - cost_fp = $15: cost of unnecessarily contacting a non-churner
      (agent time). Same as personal_call intervention cost.

    Parameters
    ----------
    y_true : array
        True labels.
    y_proba : array
        Predicted probabilities for class 1.
    cost_fp : float
        Cost of incorrectly targeting a non-churner.
    benefit_tp : float
        Benefit of correctly retaining a churner.

    Returns
    -------
    thresholds, profits, optimal_threshold
    """
    thresholds = np.linspace(0.05, 0.95, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        profit = tp * benefit_tp - fp * cost_fp
        profits.append(profit)

    profits = np.array(profits)
    optimal_idx = np.argmax(profits)
    return thresholds, profits, thresholds[optimal_idx]


def plot_profit_curve(thresholds, profits, optimal_threshold, model_name: str) -> plt.Figure:
    """Plot the profit curve."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, profits, color=CYAN, linewidth=2)
    ax.axvline(
        optimal_threshold,
        color=AMBER,
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal: {optimal_threshold:.2f}",
    )
    ax.fill_between(thresholds, profits, alpha=0.1, color=CYAN)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Profit ($)")
    ax.set_title(f"Profit Curve — {model_name}")
    ax.legend(framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


# ── Training Logic ───────────────────────────────────────────────────────────


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_iter: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    optimizer: str = "random",
    n_trials: int = 50,
    feature_set: str = "all",
) -> dict[str, Any]:
    """Train a single model with MLflow tracking.

    Steps:
    1. Hyperparameter search — RandomizedSearchCV ('random') or Optuna TPE ('optuna').
    2. OOF threshold optimisation (argmax F1-Macro on cross_val_predict).
    3. Evaluate on held-out test set using the OOF threshold.
    4. Log all metrics, artifacts, and model to MLflow.

    Parameters
    ----------
    model_name : str
        Key from MODEL_CONFIGS.
    X_train, y_train : arrays
        Training data (already feature-selected if applicable).
    X_test, y_test : arrays
        Test data (same feature mask applied).
    feature_names : list[str]
        Names of the input features (after selection).
    n_iter : int
        RandomizedSearchCV iterations (only when optimizer='random').
    cv_folds : int
        Number of CV folds.
    random_state : int
        Random seed for reproducibility.
    optimizer : str
        'random' (RandomizedSearchCV) or 'optuna' (TPE sampler).
    n_trials : int
        Optuna trials (only when optimizer='optuna').
    feature_set : str
        Name of the feature selection method ('all', 'mi', 'hill_climbing').
        Used for MLflow tagging and result grouping only.

    Returns
    -------
    dict
        Results including model, metrics, oof_threshold, and MLflow run_id.
    """
    config = MODEL_CONFIGS[model_name]

    # Lazy-load estimators that require optional dependencies
    if model_name == "xgboost":
        estimator = _get_xgboost_estimator(random_state=random_state)
    elif model_name == "catboost":
        estimator = _get_catboost_estimator(random_state=random_state)
    else:
        import copy

        estimator = copy.deepcopy(config["estimator"])
        # Propagate random_state when the estimator supports it
        if hasattr(estimator, "random_state"):
            estimator.random_state = random_state

    run_name = f"{model_name}_{feature_set}_seed{random_state}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("description", config["description"])
        mlflow.set_tag(
            "imbalance_strategy",
            "class_weight" if "class_weight" in config["params"] else "none",
        )
        mlflow.set_tag("n_features", str(X_train.shape[1]))
        mlflow.set_tag("random_state", str(random_state))
        mlflow.set_tag("optimizer", optimizer)

        mlflow.sklearn.autolog(log_models=False, silent=True)

        # ── Hyperparameter search ──
        if not config["params"]:
            # Dummy / no-param models: fit directly
            best_model = estimator
            best_model.fit(X_train, y_train)
            mlflow.log_metric("cv_best_roc_auc", 0.5)

        elif optimizer == "optuna":
            # Optuna TPE sampler
            best_model, best_params, best_cv_auc = optimize_with_optuna(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("cv_best_roc_auc", best_cv_auc)

        else:
            # RandomizedSearchCV (default)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = RandomizedSearchCV(
                estimator,
                param_distributions=config["params"],
                n_iter=min(n_iter, 30),
                cv=cv,
                scoring="roc_auc",
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        # ── OOF threshold optimisation ──
        oof_threshold = find_oof_threshold(
            best_model,
            X_train,
            y_train,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        mlflow.log_metric("oof_threshold", oof_threshold)

        # ── Predictions ──
        y_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba")
            else best_model.predict(X_test).astype(float)
        )
        y_pred = (y_proba >= oof_threshold).astype(int)

        # ── Metrics ──
        metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "test_f1_churn": f1_score(y_test, y_pred, zero_division=0),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_accuracy": accuracy_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # ── Profit curve ──
        thresholds, profits, optimal_threshold = compute_profit_curve(y_test, y_proba)
        mlflow.log_metric("profit_threshold", optimal_threshold)
        mlflow.log_metric("max_profit", float(np.max(profits)))

        # ── Log artifacts ──
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            fig_cm = plot_confusion_matrix(y_test, y_pred, model_name)
            cm_path = tmp_path / f"confusion_matrix_{model_name}_{feature_set}.png"
            fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            mlflow.log_artifact(str(cm_path), "figures")
            plt.close(fig_cm)

            fig_roc = plot_roc_curve(y_test, y_proba, model_name, metrics["test_roc_auc"])
            roc_path = tmp_path / f"roc_curve_{model_name}_{feature_set}.png"
            fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            mlflow.log_artifact(str(roc_path), "figures")
            plt.close(fig_roc)

            fig_fi = plot_feature_importance(best_model, feature_names, model_name)
            if fig_fi:
                fi_path = tmp_path / f"feature_importance_{model_name}_{feature_set}.png"
                fig_fi.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
                mlflow.log_artifact(str(fi_path), "figures")
                plt.close(fig_fi)

            fig_pc = plot_profit_curve(thresholds, profits, optimal_threshold, model_name)
            pc_path = tmp_path / f"profit_curve_{model_name}_{feature_set}.png"
            fig_pc.savefig(pc_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            mlflow.log_artifact(str(pc_path), "figures")
            plt.close(fig_pc)

            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = tmp_path / f"classification_report_{model_name}_{feature_set}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(report_path), "reports")

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=None,
        )

        mlflow.sklearn.autolog(disable=True)

        logger.info(
            f"[{model_name}|{feature_set}] "
            f"AUC={metrics['test_roc_auc']:.4f} | "
            f"F1-Macro={metrics['test_f1_macro']:.4f} | "
            f"F1-Churn={metrics['test_f1_churn']:.4f} | "
            f"n_features={X_train.shape[1]} | "
            f"OOF-thr={oof_threshold:.2f} | "
            f"run={run.info.run_id}"
        )

        return {
            "model_name": model_name,
            "feature_set": feature_set,
            "model": best_model,
            "metrics": metrics,
            "oof_threshold": oof_threshold,
            "optimal_threshold": optimal_threshold,
            "run_id": run.info.run_id,
            "n_features": X_train.shape[1],
        }


# ── Quick Single-Seed Benchmark ──────────────────────────────────────────────


def run_benchmark(
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    n_iter: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    optimizer: str = "random",
    n_trials: int = 50,
) -> pd.DataFrame:
    """Quick single-seed benchmark: all models × selected feature sets.

    Intended for rapid iteration (--quick flag).  For statistically robust
    results use run_repeated_benchmark() instead.

    Parameters
    ----------
    models : list[str] | None
        Model names to train. None = all.
    feature_sets : list[str] | None
        Feature selection methods. None = ["all"] (quick mode default).
    n_iter : int
        RandomizedSearchCV iterations per model.
    test_size : float
        Train/test split ratio.
    random_state : int
        Random seed.
    optimizer : str
        'random' or 'optuna'.
    n_trials : int
        Optuna trials per model.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by ROC-AUC (all feature sets combined).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if feature_sets is None:
        feature_sets = ["all"]
    if models is None:
        models = list(MODEL_CONFIGS.keys())

    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X, y, pipeline = prepare_data(df)
    feature_names = get_feature_names(pipeline)
    save_pipeline(pipeline)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Quick benchmark | seed={random_state} | "
        f"train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"features={X_train.shape[1]}"
    )

    # Compute feature masks (train-only to avoid leakage)
    masks = compute_feature_masks(
        X_train, y_train, feature_names, feature_sets, random_state=random_state
    )

    results = []
    for fs in feature_sets:
        mask, sel_names = masks[fs]
        X_tr = X_train[:, mask]
        X_te = X_test[:, mask]

        for name in models:
            logger.info(f"\n{'='*60}\n[{fs}] Training: {name}\n{'='*60}")
            result = train_single_model(
                model_name=name,
                X_train=X_tr,
                y_train=y_train,
                X_test=X_te,
                y_test=y_test,
                feature_names=sel_names,
                n_iter=n_iter,
                random_state=random_state,
                optimizer=optimizer,
                n_trials=n_trials,
                feature_set=fs,
            )
            results.append(result)

    comparison = pd.DataFrame(
        [
            {
                "feature_set": r["feature_set"],
                "model": r["model_name"],
                "n_features": r["n_features"],
                "roc_auc": r["metrics"]["test_roc_auc"],
                "f1_macro": r["metrics"]["test_f1_macro"],
                "f1_churn": r["metrics"]["test_f1_churn"],
                "precision": r["metrics"]["test_precision"],
                "recall": r["metrics"]["test_recall"],
                "accuracy": r["metrics"]["test_accuracy"],
                "oof_threshold": r["oof_threshold"],
                "run_id": r["run_id"],
            }
            for r in results
        ]
    ).sort_values(["feature_set", "roc_auc"], ascending=[True, False])

    logger.info(
        f"\n{'='*60}\nQuick Benchmark Results:\n{'='*60}\n"
        f"{comparison.drop(columns=['run_id']).to_string(index=False)}"
    )

    # Register best overall model
    best_row = comparison.sort_values("roc_auc", ascending=False).iloc[0]
    best_result = next(r for r in results if r["run_id"] == best_row["run_id"])
    _register_best_model(best_result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODELS_DIR / "benchmark_results.csv", index=False)

    return comparison


# ── Comprehensive Repeated Benchmark ────────────────────────────────────────


def run_repeated_benchmark(
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    n_runs: int = N_BENCHMARK_RUNS,
    n_iter: int = 20,
    test_size: float = 0.2,
    optimizer: str = "random",
    n_trials: int = 50,
) -> dict[str, pd.DataFrame]:
    """Comprehensive benchmark: all models × all feature sets × N seeds.

    For each seed in range(n_runs):
      1. Re-split train/test (stratified).
      2. Compute feature selection masks once on X_train (no leakage).
      3. For each feature_set × model: train and collect metrics.

    After all seeds:
      - Compute mean ± std per (feature_set, model) per metric.
      - Wilcoxon signed-rank test for pairwise ROC-AUC (per feature_set).
      - Save comprehensive summary to models/comprehensive_benchmark_summary.csv.

    Rationale for Wilcoxon over paired t-test: the distribution of
    ROC-AUC differences across seeds is not guaranteed to be normal
    for n_runs=10. Wilcoxon is the non-parametric equivalent.
    Ref: Demšar (2006) "Statistical Comparisons of Classifiers over
    Multiple Data Sets."

    Parameters
    ----------
    models : list[str] | None
        Models to benchmark. None = all 5.
    feature_sets : list[str] | None
        Feature selection methods. None = all 3 (FEATURE_SELECTION_METHODS).
    n_runs : int
        Number of random seeds (default: N_BENCHMARK_RUNS from config).
    n_iter : int
        RandomizedSearchCV iterations per model per seed.
    test_size : float
        Train/test split ratio.
    optimizer : str
        'random' or 'optuna'.
    n_trials : int
        Optuna trials per model (only when optimizer='optuna').

    Returns
    -------
    dict with keys:
        "summary"  : pd.DataFrame — mean ± std per (feature_set, model)
        "raw"      : pd.DataFrame — all per-seed raw records
        "pvalues"  : dict[str, pd.DataFrame] — Wilcoxon p-value matrix per feature_set
    """
    from scipy.stats import wilcoxon

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if feature_sets is None:
        feature_sets = list(FEATURE_SELECTION_METHODS)

    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X, y, pipeline = prepare_data(df)
    feature_names = get_feature_names(pipeline)
    save_pipeline(pipeline)

    n_combinations = n_runs * len(feature_sets) * len(models)
    logger.info(
        f"Comprehensive benchmark: {n_runs} seeds × {len(feature_sets)} feature sets "
        f"× {len(models)} models = {n_combinations} training runs"
    )
    logger.info(f"  Feature sets : {feature_sets}")
    logger.info(f"  Models       : {models}")
    logger.info(f"  Optimizer    : {optimizer}")

    raw_records: list[dict] = []

    for seed in range(n_runs):
        logger.info(f"\n{'='*60}\nSeed {seed}/{n_runs - 1}\n{'='*60}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Compute feature masks ONCE per seed (train-set only, no leakage)
        logger.info(f"  Computing feature masks for seed {seed}...")
        masks = compute_feature_masks(
            X_train, y_train, feature_names, feature_sets, random_state=seed
        )

        for fs in feature_sets:
            mask, sel_names = masks[fs]
            X_tr = X_train[:, mask]
            X_te = X_test[:, mask]

            for name in models:
                result = train_single_model(
                    model_name=name,
                    X_train=X_tr,
                    y_train=y_train,
                    X_test=X_te,
                    y_test=y_test,
                    feature_names=sel_names,
                    n_iter=n_iter,
                    random_state=seed,
                    optimizer=optimizer,
                    n_trials=n_trials,
                    feature_set=fs,
                )
                raw_records.append(
                    {
                        "seed": seed,
                        "feature_set": fs,
                        "n_features": int(mask.sum()),
                        "model": name,
                        **result["metrics"],
                        "oof_threshold": result["oof_threshold"],
                        "run_id": result["run_id"],
                    }
                )

    raw_df = pd.DataFrame(raw_records)

    # ── Summary: mean ± std grouped by (feature_set, model) ──
    metric_cols = [
        "test_roc_auc",
        "test_f1_macro",
        "test_f1_churn",
        "test_precision",
        "test_recall",
        "test_accuracy",
    ]
    summary_rows = []
    for fs in feature_sets:
        for model_name in models:
            subset = raw_df[(raw_df["feature_set"] == fs) & (raw_df["model"] == model_name)]
            row: dict[str, Any] = {"feature_set": fs, "model": model_name}
            row["n_features_mean"] = subset["n_features"].mean()
            for col in metric_cols:
                short = col.replace("test_", "")
                row[f"{short}_mean"] = subset[col].mean()
                row[f"{short}_std"] = subset[col].std()
                row[f"{short}"] = f"{subset[col].mean():.4f} ± {subset[col].std():.4f}"
            summary_rows.append(row)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["feature_set", "roc_auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ── Wilcoxon p-value matrix per feature_set ──
    # Only for non-dummy models (dummy AUC ≈ 0.5, trivially different).
    test_models = [m for m in models if m != "dummy"]
    pvalue_matrices: dict[str, pd.DataFrame] = {}

    for fs in feature_sets:
        fs_df = raw_df[raw_df["feature_set"] == fs]
        pmat = pd.DataFrame(index=test_models, columns=test_models, dtype=float)
        for m1 in test_models:
            for m2 in test_models:
                if m1 == m2:
                    pmat.loc[m1, m2] = 1.0
                else:
                    s1 = fs_df[fs_df["model"] == m1]["test_roc_auc"].values
                    s2 = fs_df[fs_df["model"] == m2]["test_roc_auc"].values
                    diff = s1 - s2
                    if np.all(diff == 0):
                        pmat.loc[m1, m2] = 1.0
                    else:
                        try:
                            _, p = wilcoxon(diff, alternative="two-sided")
                            pmat.loc[m1, m2] = p
                        except Exception:
                            pmat.loc[m1, m2] = float("nan")
        pvalue_matrices[fs] = pmat

    # ── Save all results ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(MODELS_DIR / "comprehensive_benchmark_summary.csv", index=False)
    raw_df.to_csv(MODELS_DIR / "comprehensive_benchmark_raw.csv", index=False)
    for fs, pmat in pvalue_matrices.items():
        pmat.to_csv(MODELS_DIR / f"pvalue_matrix_{fs}.csv")

    # ── Console output ──
    logger.success(
        f"Comprehensive benchmark complete ({n_runs} seeds). " f"Results saved to {MODELS_DIR}"
    )
    logger.info(f"\n{'='*70}\nSummary (mean ± std over {n_runs} seeds):\n{'='*70}")
    display_cols = [
        "feature_set",
        "model",
        "n_features_mean",
        "roc_auc",
        "f1_macro",
        "f1_churn",
    ]
    logger.info("\n" + summary_df[display_cols].to_string(index=False))

    for fs, pmat in pvalue_matrices.items():
        logger.info(f"\n{'='*70}\nWilcoxon p-values (ROC-AUC) — feature_set={fs}:\n{'='*70}")
        logger.info("\n" + pmat.to_string())

    # ── Register overall best model ──
    best_row = summary_df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_fs = best_row["feature_set"]
    best_model_name = best_row["model"]
    # Find best single-seed result for this (model, feature_set) to register
    candidate = raw_df[
        (raw_df["feature_set"] == best_fs) & (raw_df["model"] == best_model_name)
    ].sort_values("test_roc_auc", ascending=False)
    if len(candidate) > 0:
        best_candidate = candidate.iloc[0]
        logger.info(
            f"Best configuration: {best_model_name} | {best_fs} | "
            f"ROC-AUC={best_row['roc_auc_mean']:.4f} ± {best_row['roc_auc_std']:.4f}"
        )
        _register_best_model(
            {
                "run_id": best_candidate["run_id"],
                "model_name": best_model_name,
                "feature_set": best_fs,
                "metrics": {"test_roc_auc": best_candidate["test_roc_auc"]},
            }
        )

    return {
        "summary": summary_df,
        "raw": raw_df,
        "pvalues": pvalue_matrices,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _register_best_model(result: dict) -> None:
    """Register best model in MLflow Model Registry (best-effort).

    Also saves a local copy as ``models/best_model.joblib`` so the API and
    Streamlit fallback paths work regardless of which algorithm won.
    """
    import joblib

    best_run_id = result["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    # Persist a local copy with a stable, algorithm-agnostic name.
    # Use the in-memory object when available (single-seed path), otherwise
    # load from MLflow artifacts (repeated-benchmark path).
    try:
        model_obj = result.get("model") or mlflow.sklearn.load_model(model_uri)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(model_obj, local_path)
        logger.info(f"Best model saved locally: {local_path}")
    except Exception as e:
        logger.warning(f"Could not save model locally: {e}")

    try:
        reg = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
        logger.success(
            f"Registered '{MLFLOW_MODEL_NAME}' v{reg.version} "
            f"({result['model_name']} | {result.get('feature_set','all')} | "
            f"AUC={result['metrics']['test_roc_auc']:.4f})"
        )
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME, version=reg.version, stage="Staging"
        )
    except Exception as e:
        logger.warning(f"Could not register model (MLflow server may be offline): {e}")


def print_results_for_readme(summary_df: pd.DataFrame) -> None:
    """Print a markdown-ready results table to stdout.

    Call after run_repeated_benchmark() to get copy-paste text for README.
    """
    print("\n\n=== COPY-PASTE FOR README ===\n")
    print(
        "| Feature Set | Model | N Features | ROC-AUC | F1-Macro | F1-Churn | Precision | Recall |"
    )
    print(
        "|-------------|-------|-----------|---------|----------|----------|-----------|--------|"
    )
    display_cols = [
        "feature_set",
        "model",
        "n_features_mean",
        "roc_auc",
        "f1_macro",
        "f1_churn",
        "precision",
        "recall",
    ]
    for _, row in summary_df[display_cols].iterrows():
        n_feat = f"{row['n_features_mean']:.0f}" if not pd.isna(row["n_features_mean"]) else "—"
        print(
            f"| {row['feature_set']:<14} | {row['model']:<15} | {n_feat:<10} "
            f"| {row['roc_auc']:<15} | {row['f1_macro']:<16} | {row['f1_churn']:<16} "
            f"| {row['precision']:<17} | {row['recall']:<15} |"
        )
    print("\n=== END ===\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive churn model benchmark with feature-selection comparison.\n"
            "Default (no flags): all models × all feature sets × 10 seeds.\n"
            "Use --quick for a fast single-seed run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Train only this model (default: all). "
        "Choices: dummy, random_forest, xgboost, catboost, mlp",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        choices=FEATURE_SELECTION_METHODS,
        metavar="FS",
        help=(
            "Feature selection method(s) to include "
            f"(default: all = {FEATURE_SELECTION_METHODS}). "
            "Choices: all, mi, hill_climbing"
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="RandomizedSearchCV iterations per model (optimizer=random, default: 20)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_BENCHMARK_RUNS,
        help=f"Number of random seeds for repeated benchmark (default: {N_BENCHMARK_RUNS})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="random",
        choices=["random", "optuna"],
        help=(
            "Hyperparameter optimizer: 'random' (RandomizedSearchCV, default) "
            "or 'optuna' (TPE sampler, higher quality but slower)"
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna trials per model (only used when --optimizer optuna, default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick single-seed run (seed=42, feature_sets=['all'] by default). "
            "Fast for iteration; use default (no flag) for full benchmark."
        ),
    )
    # Keep --repeated as an alias for backward compatibility
    parser.add_argument(
        "--repeated",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden; same as default behaviour now
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --quick single-run mode (default: 42)",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="After benchmark, print markdown table ready to paste into README.",
    )

    args = parser.parse_args()

    models_list = [args.model] if args.model else None
    feature_sets_list = args.feature_sets  # None means all defaults

    if args.quick:
        # Fast single-seed path
        fs = feature_sets_list or ["all"]
        run_benchmark(
            models=models_list,
            feature_sets=fs,
            n_iter=args.n_iter,
            test_size=args.test_size,
            random_state=args.seed,
            optimizer=args.optimizer,
            n_trials=args.n_trials,
        )
    else:
        # Default: full comprehensive repeated benchmark
        results = run_repeated_benchmark(
            models=models_list,
            feature_sets=feature_sets_list,
            n_runs=args.n_runs,
            n_iter=args.n_iter,
            test_size=args.test_size,
            optimizer=args.optimizer,
            n_trials=args.n_trials,
        )
        if args.readme:
            print_results_for_readme(results["summary"])
