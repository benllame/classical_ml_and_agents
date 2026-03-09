"""
Patch script: injects new sections 15-22 into the notebook
before the final Conclusiones markdown cell.
"""
import json, copy

NB_PATH = "notebooks/01_eda_preprocessing_training.ipynb"

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "",
        "metadata": {},
        "source": source
    }

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {},
        "outputs": [],
        "source": source
    }

# ─────────────────────────────────────────────
# SECTION 15 – Optuna TPE
# ─────────────────────────────────────────────
SEC15_MD = """\
---
## 15. Optimización Bayesiana con Optuna (TPE)

Reemplazamos `RandomizedSearchCV` por **Optuna** con el sampler TPE (Tree-structured
Parzen Estimator), que explora el espacio de hiperparámetros de forma adaptativa.

* 200 trials, `TPESampler(seed=42)`
* CV: `StratifiedKFold(5)`, métrica `roc_auc`
* Espacio de búsqueda igual al de `RandomizedSearchCV` (sección 7)\
"""

SEC15_CODE = """\
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def _objective_xgb(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1200),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        gamma             = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 5.0),
        scale_pos_weight  = trial.suggest_categorical(
            "scale_pos_weight", [1.0, 1.5, 2.0, float(class_ratio)]
        ),
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=_skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
)
study_xgb.optimize(_objective_xgb, n_trials=200, show_progress_bar=True)

print(f"Best ROC-AUC (CV): {study_xgb.best_value:.4f}")
print("Best params:", study_xgb.best_params)\
"""

SEC15_CODE2 = """\
# ── Train best Optuna-XGB on full training set ──────────────────────────────
from sklearn.metrics import roc_auc_score, f1_score

_best_p = study_xgb.best_params.copy()
optuna_xgb = xgb.XGBClassifier(
    **_best_p,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
optuna_xgb.fit(X_train, y_train)

# ── OOF threshold optimisation ───────────────────────────────────────────────
oof_proba_optxgb = np.zeros(len(y_train))
for tr_idx, val_idx in _skf.split(X_train, y_train):
    _m = xgb.XGBClassifier(
        **_best_p,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    _m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
    oof_proba_optxgb[val_idx] = _m.predict_proba(X_train[val_idx])[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [f1_score(y_train, (oof_proba_optxgb >= t).astype(int), average="macro")
              for t in thresholds]
thresh_optxgb = thresholds[np.argmax(f1_scores)]
print(f"Optimal OOF threshold (Optuna-XGB): {thresh_optxgb:.3f}")

# ── Test evaluation ──────────────────────────────────────────────────────────
proba_optxgb   = optuna_xgb.predict_proba(X_test)[:, 1]
pred_optxgb    = (proba_optxgb >= thresh_optxgb).astype(int)

roc_optxgb  = roc_auc_score(y_test, proba_optxgb)
f1m_optxgb  = f1_score(y_test, pred_optxgb, average="macro")
f1c_optxgb  = f1_score(y_test, pred_optxgb, pos_label=1)

print(f"Optuna-XGB  |  ROC-AUC={roc_optxgb:.4f}  F1-Macro={f1m_optxgb:.4f}"
      f"  F1-Churn={f1c_optxgb:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 16 – Seed Averaging
# ─────────────────────────────────────────────
SEC16_MD = """\
---
## 16. Seed Averaging

Entrenamos el modelo Optuna-XGB con 5 semillas distintas y **promediamos las
probabilidades** para reducir varianza de inicialización.\
"""

SEC16_CODE = """\
SEEDS = [42, 123, 456, 789, 2024]
proba_seeds = []

for s in SEEDS:
    _m = xgb.XGBClassifier(
        **{**_best_p,
           "random_state": s,
           "use_label_encoder": False,
           "eval_metric": "logloss",
           "n_jobs": -1},
    )
    _m.fit(X_train, y_train)
    proba_seeds.append(_m.predict_proba(X_test)[:, 1])

proba_seed_avg = np.mean(proba_seeds, axis=0)

# OOF threshold for seed-averaged model (use same CV folds, seed=42 model as proxy)
oof_proba_seedavg = np.zeros(len(y_train))
for tr_idx, val_idx in _skf.split(X_train, y_train):
    _preds = []
    for s in SEEDS:
        _m = xgb.XGBClassifier(
            **{**_best_p,
               "random_state": s,
               "use_label_encoder": False,
               "eval_metric": "logloss",
               "n_jobs": -1},
        )
        _m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
        _preds.append(_m.predict_proba(X_train[val_idx])[:, 1])
    oof_proba_seedavg[val_idx] = np.mean(_preds, axis=0)

f1_seed = [f1_score(y_train, (oof_proba_seedavg >= t).astype(int), average="macro")
           for t in thresholds]
thresh_seed = thresholds[np.argmax(f1_seed)]

pred_seed_avg = (proba_seed_avg >= thresh_seed).astype(int)
roc_seed  = roc_auc_score(y_test, proba_seed_avg)
f1m_seed  = f1_score(y_test, pred_seed_avg, average="macro")
f1c_seed  = f1_score(y_test, pred_seed_avg, pos_label=1)

print(f"Seed Averaging ({len(SEEDS)} seeds)  |  "
      f"ROC-AUC={roc_seed:.4f}  F1-Macro={f1m_seed:.4f}  F1-Churn={f1c_seed:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 17 – CatBoost
# ─────────────────────────────────────────────
SEC17_MD = """\
---
## 17. CatBoost (categorías nativas)

CatBoost maneja features categóricas directamente mediante **ordered target
encoding** interno, sin necesidad de OHE.  
Usamos el dataset **sin transformar** (`df` raw) para respetar las columnas
categóricas originales.\
"""

SEC17_CODE = """\
try:
    from catboost import CatBoostClassifier, Pool
    _has_catboost = True
except ImportError:
    _has_catboost = False
    print("CatBoost no instalado: pip install catboost")

if _has_catboost:
    from src.config import TARGET_COL, ID_COL, NUMERIC_FEATURES, BINARY_FEATURES, NOMINAL_FEATURES
    from sklearn.model_selection import train_test_split as tts

    # ── Prepare raw feature matrix (no OHE) ──────────────────────────────────
    _cat_features = NOMINAL_FEATURES + BINARY_FEATURES   # all non-numeric cols
    _all_features = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES

    df_clean = df.copy()
    # fill any NaN that may exist in TotalCharges
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)

    X_raw = df_clean[_all_features]
    y_raw = df_clean[TARGET_COL].map({"Yes": 1, "No": 0}) if df_clean[TARGET_COL].dtype == object else df_clean[TARGET_COL]

    # ── Same train/test split indices as existing split ───────────────────────
    # Reconstruct by matching index: use the same random_state=42, test_size=0.2
    from sklearn.model_selection import train_test_split as _tts
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = _tts(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    cat_idx = [X_raw_train.columns.tolist().index(c)
               for c in _cat_features if c in X_raw_train.columns]

    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=200,
        cat_features=cat_idx,
    )

    cb_model.fit(
        X_raw_train, y_raw_train,
        eval_set=(X_raw_test, y_raw_test),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    # ── OOF threshold ─────────────────────────────────────────────────────────
    _skf_cb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba_cb = np.zeros(len(y_raw_train))

    for tr_idx, val_idx in _skf_cb.split(X_raw_train, y_raw_train):
        _cb = CatBoostClassifier(
            iterations=cb_model.best_iteration_,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            cat_features=cat_idx,
        )
        _cb.fit(X_raw_train.iloc[tr_idx], y_raw_train.iloc[tr_idx])
        oof_proba_cb[val_idx] = _cb.predict_proba(X_raw_train.iloc[val_idx])[:, 1]

    f1_cb = [f1_score(y_raw_train, (oof_proba_cb >= t).astype(int), average="macro")
             for t in thresholds]
    thresh_cb = thresholds[np.argmax(f1_cb)]

    proba_cb  = cb_model.predict_proba(X_raw_test)[:, 1]
    pred_cb   = (proba_cb >= thresh_cb).astype(int)

    roc_cb  = roc_auc_score(y_raw_test, proba_cb)
    f1m_cb  = f1_score(y_raw_test, pred_cb, average="macro")
    f1c_cb  = f1_score(y_raw_test, pred_cb, pos_label=1)

    print(f"CatBoost  |  ROC-AUC={roc_cb:.4f}  F1-Macro={f1m_cb:.4f}"
          f"  F1-Churn={f1c_cb:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 18 – MLP
# ─────────────────────────────────────────────
SEC18_MD = """\
---
## 18. MLP (Multi-Layer Perceptron)

`MLPClassifier` con `StandardScaler` aplicado sobre las features transformadas.
El pipeline de preprocesamiento ya incluye `StandardScaler` en la salida de
`X_train`/`X_test`, por lo que se usa directamente.\
"""

SEC18_CODE = """\
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
mlp.fit(X_train, y_train)

# ── OOF threshold ─────────────────────────────────────────────────────────────
oof_proba_mlp = np.zeros(len(y_train))
for tr_idx, val_idx in _skf.split(X_train, y_train):
    _m = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    _m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
    oof_proba_mlp[val_idx] = _m.predict_proba(X_train[val_idx])[:, 1]

f1_mlp = [f1_score(y_train, (oof_proba_mlp >= t).astype(int), average="macro")
          for t in thresholds]
thresh_mlp = thresholds[np.argmax(f1_mlp)]

proba_mlp  = mlp.predict_proba(X_test)[:, 1]
pred_mlp   = (proba_mlp >= thresh_mlp).astype(int)

roc_mlp  = roc_auc_score(y_test, proba_mlp)
f1m_mlp  = f1_score(y_test, pred_mlp, average="macro")
f1c_mlp  = f1_score(y_test, pred_mlp, pos_label=1)

print(f"MLP  |  ROC-AUC={roc_mlp:.4f}  F1-Macro={f1m_mlp:.4f}"
      f"  F1-Churn={f1c_mlp:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 19 – Balanced Random Forest
# ─────────────────────────────────────────────
SEC19_MD = """\
---
## 19. Balanced Random Forest

`BalancedRandomForestClassifier` de `imbalanced-learn` aplica submuestreo
bootstrapping balanceado en cada árbol, sin necesidad de sobremuestrear
manualmente.\
"""

SEC19_CODE = """\
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    _has_brf = True
except ImportError:
    _has_brf = False
    print("imbalanced-learn no instalado: pip install imbalanced-learn")

if _has_brf:
    brf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        replacement=False,
        sampling_strategy="auto",
    )
    brf.fit(X_train, y_train)

    # ── OOF threshold ─────────────────────────────────────────────────────────
    oof_proba_brf = np.zeros(len(y_train))
    for tr_idx, val_idx in _skf.split(X_train, y_train):
        _m = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            replacement=False,
            sampling_strategy="auto",
        )
        _m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
        oof_proba_brf[val_idx] = _m.predict_proba(X_train[val_idx])[:, 1]

    f1_brf = [f1_score(y_train, (oof_proba_brf >= t).astype(int), average="macro")
              for t in thresholds]
    thresh_brf = thresholds[np.argmax(f1_brf)]

    proba_brf  = brf.predict_proba(X_test)[:, 1]
    pred_brf   = (proba_brf >= thresh_brf).astype(int)

    roc_brf  = roc_auc_score(y_test, proba_brf)
    f1m_brf  = f1_score(y_test, pred_brf, average="macro")
    f1c_brf  = f1_score(y_test, pred_brf, pos_label=1)

    print(f"Balanced RF  |  ROC-AUC={roc_brf:.4f}  F1-Macro={f1m_brf:.4f}"
          f"  F1-Churn={f1c_brf:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 20 – Error Analysis
# ─────────────────────────────────────────────
SEC20_MD = """\
---
## 20. Análisis de Errores (OOF)

Analizamos los **Falsos Negativos (FN)** y **Falsos Positivos (FP)** del mejor
modelo hasta ahora (Optuna-XGB) usando las predicciones OOF sobre el conjunto
de entrenamiento.\
"""

SEC20_CODE = """\
# ── Build OOF prediction dataframe ───────────────────────────────────────────
oof_pred_optxgb = (oof_proba_optxgb >= thresh_optxgb).astype(int)

err_df = pd.DataFrame(X_train, columns=feature_names)
err_df["y_true"]      = y_train.values
err_df["y_pred"]      = oof_pred_optxgb
err_df["proba_churn"] = oof_proba_optxgb

fn_mask = (err_df["y_true"] == 1) & (err_df["y_pred"] == 0)
fp_mask = (err_df["y_true"] == 0) & (err_df["y_pred"] == 1)
tp_mask = (err_df["y_true"] == 1) & (err_df["y_pred"] == 1)

print(f"Falsos Negativos: {fn_mask.sum()} | "
      f"Falsos Positivos: {fp_mask.sum()} | "
      f"Verdaderos Positivos: {tp_mask.sum()}")

# ── Numeric feature comparison FN vs TP ──────────────────────────────────────
from src.config import NUMERIC_FEATURES as _NUM_FEAT

numeric_cols = [c for c in feature_names if c in _NUM_FEAT]
if numeric_cols:
    set_dark_style()
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        err_df.loc[fn_mask, col].plot.kde(ax=ax, label="FN", color="#FF6B6B")
        err_df.loc[tp_mask, col].plot.kde(ax=ax, label="TP", color="#4ECDC4")
        ax.set_title(col, color="white")
        ax.legend()
    plt.suptitle("FN vs TP: distribución de features numéricas (OOF)", color="white")
    plt.tight_layout()
    plt.show()

# ── Top features by probability gap ──────────────────────────────────────────
print("\\nMedia de probabilidad estimada:")
print(f"  FN : {err_df.loc[fn_mask, 'proba_churn'].mean():.3f}")
print(f"  FP : {err_df.loc[fp_mask, 'proba_churn'].mean():.3f}")
print(f"  TP : {err_df.loc[tp_mask, 'proba_churn'].mean():.3f}")\
"""

# ─────────────────────────────────────────────
# SECTION 21 – Hill Climbing blend
# ─────────────────────────────────────────────
SEC21_MD = """\
---
## 21. Hill Climbing: Optimización de Pesos del Ensemble

Usamos `scipy.optimize.minimize` (Nelder-Mead) para encontrar los pesos óptimos
que maximizan el **F1-Macro sobre las predicciones OOF**.\
"""

SEC21_CODE = """\
from scipy.optimize import minimize

# ── Collect OOF probabilities from available models ───────────────────────────
_oof_models = {
    "Optuna-XGB"  : oof_proba_optxgb,
    "Seed-Avg-XGB": oof_proba_seedavg,
    "MLP"         : oof_proba_mlp,
}
if _has_brf:
    _oof_models["Balanced-RF"] = oof_proba_brf

_oof_matrix  = np.column_stack(list(_oof_models.values()))   # (n_train, n_models)
_model_names = list(_oof_models.keys())
n_models     = len(_model_names)
print(f"Models in blend: {_model_names}")

# ── Test probabilities (same order) ───────────────────────────────────────────
_test_models = {
    "Optuna-XGB"  : proba_optxgb,
    "Seed-Avg-XGB": proba_seed_avg,
    "MLP"         : proba_mlp,
}
if _has_brf:
    _test_models["Balanced-RF"] = proba_brf
_test_matrix = np.column_stack(list(_test_models.values()))

# ── Objective: negative F1-Macro on OOF (Nelder-Mead minimises) ───────────────
def _blend_objective(weights):
    w = np.array(weights)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return 1.0
    w = w / w.sum()
    blended = _oof_matrix @ w
    best_t  = max(thresholds,
                  key=lambda t: f1_score(y_train, (blended >= t).astype(int),
                                         average="macro"))
    return -f1_score(y_train, (blended >= best_t).astype(int), average="macro")

x0      = np.ones(n_models) / n_models
result  = minimize(_blend_objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})

opt_weights = np.clip(result.x, 0, None)
opt_weights /= opt_weights.sum()
print("Pesos óptimos:", dict(zip(_model_names, opt_weights.round(4))))

# ── Apply weights to test set ─────────────────────────────────────────────────
proba_blend    = _test_matrix @ opt_weights
f1_blend_oof   = [f1_score(y_train,
                            (_oof_matrix @ opt_weights >= t).astype(int),
                            average="macro")
                  for t in thresholds]
thresh_blend   = thresholds[np.argmax(f1_blend_oof)]

pred_blend     = (proba_blend >= thresh_blend).astype(int)
roc_blend      = roc_auc_score(y_test, proba_blend)
f1m_blend      = f1_score(y_test, pred_blend, average="macro")
f1c_blend      = f1_score(y_test, pred_blend, pos_label=1)

print(f"\\nHill-Climbing Blend  |  ROC-AUC={roc_blend:.4f}"
      f"  F1-Macro={f1m_blend:.4f}  F1-Churn={f1c_blend:.4f}")\
"""

# ─────────────────────────────────────────────
# SECTION 22 – Per-model comparison table
# ─────────────────────────────────────────────
SEC22_MD = """\
---
## 22. Tabla Comparativa por Modelo

Comparamos todos los modelos entrenados bajo el **mismo protocolo de evaluación**:
predicciones OOF → threshold óptimo (argmax F1-Macro) → métricas en test set.\
"""

SEC22_CODE = """\
# ── Gather all results ────────────────────────────────────────────────────────
_rows = [
    # From previous sections (already computed)
    {"Modelo": "Baseline XGBoost (RandomSearch)",
     "ROC-AUC": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]),
     "F1-Macro": f1_score(y_test,
                          (best_model.predict_proba(X_test)[:, 1] >= optimal_thresh).astype(int),
                          average="macro"),
     "F1-Churn": f1_score(y_test,
                          (best_model.predict_proba(X_test)[:, 1] >= optimal_thresh).astype(int),
                          pos_label=1)},
    {"Modelo": "Optuna-XGB (200 trials TPE)",
     "ROC-AUC": roc_optxgb, "F1-Macro": f1m_optxgb, "F1-Churn": f1c_optxgb},
    {"Modelo": f"Seed Averaging ({len(SEEDS)} seeds)",
     "ROC-AUC": roc_seed,   "F1-Macro": f1m_seed,   "F1-Churn": f1c_seed},
    {"Modelo": "MLP (256-128-64)",
     "ROC-AUC": roc_mlp,    "F1-Macro": f1m_mlp,    "F1-Churn": f1c_mlp},
]
if _has_catboost:
    _rows.append({
        "Modelo": "CatBoost (raw cats)",
        "ROC-AUC": roc_cb, "F1-Macro": f1m_cb, "F1-Churn": f1c_cb,
    })
if _has_brf:
    _rows.append({
        "Modelo": "Balanced Random Forest",
        "ROC-AUC": roc_brf, "F1-Macro": f1m_brf, "F1-Churn": f1c_brf,
    })
_rows.append({
    "Modelo": "Hill-Climbing Blend",
    "ROC-AUC": roc_blend, "F1-Macro": f1m_blend, "F1-Churn": f1c_blend,
})

comp_df = pd.DataFrame(_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
comp_df[["ROC-AUC", "F1-Macro", "F1-Churn"]] = comp_df[
    ["ROC-AUC", "F1-Macro", "F1-Churn"]].round(4)
print(comp_df.to_string(index=False))\
"""

SEC22_CODE2 = """\
# ── Visual comparison ─────────────────────────────────────────────────────────
set_dark_style()
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
metrics   = ["ROC-AUC", "F1-Macro", "F1-Churn"]
colors    = ["#4ECDC4", "#FF6B6B", "#FFE66D"]

for ax, metric, color in zip(axes, metrics, colors):
    bars = ax.barh(comp_df["Modelo"], comp_df[metric], color=color, alpha=0.85)
    ax.set_xlabel(metric, color="white")
    ax.set_title(metric, color="white", fontsize=13)
    ax.tick_params(colors="white")
    for bar, val in zip(bars, comp_df[metric]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=9)
    ax.set_xlim(comp_df[metric].min() * 0.97, comp_df[metric].max() * 1.03)

plt.suptitle("Comparación de Modelos — Protocolo Unificado", color="white", fontsize=15)
plt.tight_layout()
plt.show()\
"""

# ─────────────────────────────────────────────
# PATCH the notebook
# ─────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    md(SEC15_MD), code(SEC15_CODE), code(SEC15_CODE2),
    md(SEC16_MD), code(SEC16_CODE),
    md(SEC17_MD), code(SEC17_CODE),
    md(SEC18_MD), code(SEC18_CODE),
    md(SEC19_MD), code(SEC19_CODE),
    md(SEC20_MD), code(SEC20_CODE),
    md(SEC21_MD), code(SEC21_CODE),
    md(SEC22_MD), code(SEC22_CODE), code(SEC22_CODE2),
]

# Insert before last cell (Conclusiones markdown)
insert_pos = len(nb["cells"]) - 1   # just before the last cell
for i, cell in enumerate(new_cells):
    nb["cells"].insert(insert_pos + i, cell)

# Assign unique IDs
import uuid
for cell in nb["cells"]:
    if not cell.get("id"):
        cell["id"] = str(uuid.uuid4())[:8]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print(f"New cells inserted at positions {insert_pos} to {insert_pos + len(new_cells) - 1}.")
