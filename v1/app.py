import os
import re
import io
import copy
import json
import socket
import shutil
import zipfile
import platform
import tempfile
import datetime
import itertools
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

import xgboost as xgb
import shap

from lime.lime_tabular import LimeTabularExplainer
from openai import OpenAI
from sklearn.linear_model import Ridge
from typing import Optional, Dict, Any
from scipy.optimize import minimize
from scipy import stats as sp_stats
try:
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except Exception:
    CoxPHFitter = None
    _HAS_LIFELINES = False


# -----------------------------
# Helpers
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def detect_feature_columns(
    df: pd.DataFrame,
    outcome_col="y",
    treat_col="treatment",
    event_col=None,
    exclude=None
):
    exclude = set(exclude or [])
    exclude |= {outcome_col, treat_col, "sigpos"}
    if event_col:
        exclude |= {event_col}

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in num_cols if c not in exclude]


def detect_categorical_indices(X: pd.DataFrame, max_levels: int = 10):
    """Column indices LIME should treat as categorical.

    LIME's quartile discretiser otherwise splits a 0/1 indicator into quartile bins and
    reports rules such as "sex <= 0.00", which are meaningless. Columns taking few
    distinct integer-valued levels are declared categorical so the discretiser leaves
    them alone. LIME still marks every column categorical internally once discretisation
    is on, but the discretiser itself skips the columns named here.
    """
    idx = []
    for j, c in enumerate(X.columns):
        v = pd.to_numeric(X[c], errors="coerce").dropna()
        if v.empty:
            continue
        u = np.unique(v.values)
        if len(u) <= int(max_levels) and np.allclose(u, np.round(u)):
            idx.append(j)
    return idx


def data_quality_report(
    df: pd.DataFrame, *, outcome_col="y", treat_col="treatment", event_col=None,
    sigpos_col=None, outcome_type="continuous", loss_name="Original (clinical)",
    exclude=None,
):
    """Describe exactly what the framework will and will not use from an upload.

    The checks exist because each corresponds to a way the pipeline previously failed
    quietly or with an unhelpful message: non-numeric columns were dropped without
    comment, near-unique integer identifiers were silently modelled as covariates, and
    missing values fitted under one objective but crashed the explainer under another.
    """
    rep = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "warnings": [], "notes": []}

    reserved = {c for c in [outcome_col, treat_col, event_col, sigpos_col] if c}
    reserved |= set(exclude or [])
    rep["reserved"] = sorted(reserved)

    non_numeric = [(c, str(df[c].dtype)) for c in df.columns
                   if c not in reserved and not pd.api.types.is_numeric_dtype(df[c])]
    rep["dropped_non_numeric"] = non_numeric

    feats = [c for c in df.columns
             if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
    rep["features"] = feats

    # --- required columns present?
    missing_cols = []
    if outcome_col not in df.columns:
        missing_cols.append(f"outcome '{outcome_col}'")
    if outcome_type == "time-to-event" and event_col and event_col not in df.columns:
        missing_cols.append(f"event '{event_col}'")
    if loss_name in ("A-learning", "W-learning") and treat_col not in df.columns:
        missing_cols.append(f"treatment '{treat_col}' (required by {loss_name})")
    rep["missing_columns"] = missing_cols

    # --- missingness
    na_feat = {c: int(df[c].isna().sum()) for c in feats if df[c].isna().any()}
    rep["na_features"] = dict(sorted(na_feat.items(), key=lambda kv: -kv[1]))
    rep["na_outcome"] = int(df[outcome_col].isna().sum()) if outcome_col in df.columns else 0
    rep["na_treat"] = int(df[treat_col].isna().sum()) if treat_col in df.columns else 0
    rep["na_event"] = (int(df[event_col].isna().sum())
                       if (event_col and event_col in df.columns) else 0)
    if feats:
        rep["complete_cases"] = int(df[feats].notna().all(axis=1).sum())
    else:
        rep["complete_cases"] = 0

    # --- degenerate and identifier-like columns
    const, near_const, id_like, binary, lowcard = [], [], [], [], []
    for c in feats:
        v = df[c].dropna()
        nu = int(v.nunique())
        if nu <= 1:
            const.append(c)
            continue
        top = float(v.value_counts(normalize=True).iloc[0])
        if top > 0.99:
            near_const.append((c, round(top, 4)))
        if nu == 2:
            binary.append(c)
        elif nu <= 10 and np.allclose(np.unique(v.values), np.round(np.unique(v.values))):
            lowcard.append((c, nu))
        integral = np.allclose(v.values, np.round(v.values))
        if integral and len(df) > 20 and nu >= 0.95 * len(v):
            id_like.append((c, nu))
    rep["constant"] = const
    rep["near_constant"] = near_const
    rep["binary"] = binary
    rep["low_cardinality"] = lowcard
    rep["id_like"] = id_like

    # --- scale spread
    if feats:
        sd = df[feats].std(ddof=0)
        sd = sd[sd > 0]
        if len(sd):
            rep["sd_min"], rep["sd_max"] = float(sd.min()), float(sd.max())
            rep["sd_min_col"], rep["sd_max_col"] = str(sd.idxmin()), str(sd.idxmax())
            rep["sd_ratio"] = float(sd.max() / sd.min())

    # --- coding checks
    if treat_col in df.columns:
        u = set(pd.unique(df[treat_col].dropna()))
        rep["treat_levels"] = sorted(str(x) for x in u)
        if u - {0, 1} and loss_name in ("A-learning", "W-learning"):
            rep["warnings"].append(
                f"Treatment column '{treat_col}' must be coded 0/1; found {sorted(u)[:6]}.")
    if outcome_type == "binary" and outcome_col in df.columns:
        u = set(pd.unique(df[outcome_col].dropna()))
        if u - {0, 1}:
            rep["warnings"].append(
                f"Binary outcome '{outcome_col}' must be coded 0/1; found {sorted(u)[:6]}.")
    if outcome_type == "time-to-event" and outcome_col in df.columns:
        if (pd.to_numeric(df[outcome_col], errors="coerce") <= 0).any():
            rep["warnings"].append(
                f"Time column '{outcome_col}' contains non-positive values.")

    # --- assemble warnings
    for c in missing_cols:
        rep["warnings"].append(f"Required column not found: {c}.")
    if len(feats) < 2:
        rep["warnings"].append("Fewer than two numeric feature columns were detected.")
    if non_numeric:
        rep["warnings"].append(
            "Non-numeric columns are not used as features and will be excluded: "
            + ", ".join(f"{c} ({t})" for c, t in non_numeric)
            + ". Encode them numerically before upload if they are needed.")
    if na_feat:
        rep["warnings"].append(
            f"{len(na_feat)} feature column(s) contain missing values "
            f"({rep['complete_cases']} of {rep['n_rows']} rows are complete). Choose a "
            "missing-data policy in the sidebar; LIME cannot run on unimputed data.")
    if rep["na_outcome"]:
        rep["warnings"].append(
            f"{rep['na_outcome']} row(s) have a missing outcome and will be dropped.")
    if id_like:
        rep["warnings"].append(
            "Identifier-like column(s) will otherwise be modelled as covariates and "
            "receive attribution: "
            + ", ".join(f"{c} ({n} distinct)" for c, n in id_like)
            + ". Remove them or add them to the excluded columns.")
    if const:
        rep["warnings"].append("Constant column(s) carry no information: " + ", ".join(const))
    if near_const:
        rep["warnings"].append(
            "Near-constant column(s): "
            + ", ".join(f"{c} ({p:.1%} one level)" for c, p in near_const))
    if rep.get("sd_ratio", 1) > 100:
        rep["notes"].append(
            f"Feature standard deviations span a factor of {rep['sd_ratio']:.0f} "
            f"({rep['sd_min_col']} {rep['sd_min']:.3g} to {rep['sd_max_col']} "
            f"{rep['sd_max']:.3g}). Tree models and SHAP are invariant to this, but "
            "penalised regression is not; consider enabling standardisation.")
    return rep


def format_data_report(rep: Optional[Dict[str, Any]]) -> list:
    if not rep:
        return ["Upload a CSV to see the data check."]
    L = ["=== Data check ===",
         f"Rows: {rep['n_rows']}   Columns: {rep['n_cols']}",
         f"Reserved (outcome / treatment / event / ground-truth): "
         f"{', '.join(rep['reserved']) or 'none'}",
         f"Numeric features used ({len(rep['features'])}): "
         f"{', '.join(rep['features']) or 'none'}"]
    if rep["dropped_non_numeric"]:
        L.append(f"Excluded, not numeric: "
                 + ", ".join(f"{c} ({t})" for c, t in rep["dropped_non_numeric"]))
    L.append("")
    L.append("--- Missing data ---")
    L.append(f"Complete feature rows: {rep['complete_cases']} of {rep['n_rows']}")
    if rep["na_features"]:
        for c, n in list(rep["na_features"].items())[:15]:
            L.append(f"  {c}: {n} missing ({n / max(rep['n_rows'], 1):.1%})")
    else:
        L.append("  No missing values in the feature columns.")
    for k, lab in (("na_outcome", "outcome"), ("na_treat", "treatment"), ("na_event", "event")):
        if rep.get(k):
            L.append(f"  {lab}: {rep[k]} missing (these rows are always dropped)")

    L.append("")
    L.append("--- Column types and scale ---")
    if rep["binary"]:
        L.append(f"Binary (declared categorical to LIME): {', '.join(rep['binary'])}")
    if rep["low_cardinality"]:
        L.append("Low-cardinality integer (declared categorical to LIME): "
                 + ", ".join(f"{c} ({n} levels)" for c, n in rep["low_cardinality"]))
    if "sd_ratio" in rep:
        L.append(f"Feature SD range: {rep['sd_min_col']} {rep['sd_min']:.4g} to "
                 f"{rep['sd_max_col']} {rep['sd_max']:.4g} (ratio {rep['sd_ratio']:.1f})")
    if rep.get("treat_levels"):
        L.append(f"Treatment levels: {', '.join(rep['treat_levels'])}")

    if rep["warnings"]:
        L.append("")
        L.append("--- Warnings ---")
        for w in rep["warnings"]:
            L.append(f"  ! {w}")
    if rep["notes"]:
        L.append("")
        L.append("--- Notes ---")
        for n in rep["notes"]:
            L.append(f"  - {n}")
    if not rep["warnings"]:
        L.append("")
        L.append("No blocking issues detected.")

    L.append("")
    L.append("--- Preprocessing the framework expects ---")
    L.append("  1. One row per patient; no clustering or repeated measures.")
    L.append("  2. Features numeric. Encode categorical variables (one-hot or ordinal)")
    L.append("     before upload; non-numeric columns are excluded.")
    L.append("  3. Treatment coded 0/1, and for an ITR objective the outcome coded so that")
    L.append("     a LARGER value is a BETTER outcome.")
    L.append("  4. Identifiers and administrative columns removed.")
    L.append("  5. Missing data resolved by the sidebar policy. Single imputation is")
    L.append("     offered as a convenience only; principled multiple imputation should be")
    L.append("     done upstream, since it is not accounted for in the reported metrics.")
    L.append("  6. Standardisation is optional. Tree models, SHAP and LIME are effectively")
    L.append("     invariant to feature scale; penalised regression is not.")
    return L


def concordance_index(time, event, score):
    """
    Simple Harrell's C-index.
    Higher score => higher risk => shorter survival expected.
    """
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(int)
    score = np.asarray(score).reshape(-1)

    n = len(time)
    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue

            if event[i] == 1 and time[i] < time[j]:
                permissible += 1
                if score[i] > score[j]:
                    concordant += 1
                elif score[i] == score[j]:
                    ties += 1

            elif event[j] == 1 and time[j] < time[i]:
                permissible += 1
                if score[j] > score[i]:
                    concordant += 1
                elif score[i] == score[j]:
                    ties += 1

    if permissible == 0:
        return np.nan
    return float((concordant + 0.5 * ties) / permissible)

# -----------------------------
# Custom modified-loss estimators
# -----------------------------
class ModifiedLinearRegressor:
    def __init__(self, loss_name="A-learning", reg_lambda=1e-6, maxiter=500):
        self.loss_name = loss_name
        self.reg_lambda = reg_lambda
        self.maxiter = maxiter
        self.intercept_ = None
        self.coef_ = None
        self.n_iter_ = None
        self.success_ = None
        self.message_ = None

    def fit(self, X, y, trt_pm, pi):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        trt_pm = np.asarray(trt_pm, dtype=float).reshape(-1)
        pi = np.asarray(pi, dtype=float).reshape(-1)

        n, p = X.shape

        if self.loss_name == "A-learning":
            m = (trt_pm + 1.0) / 2.0 - pi
            w = np.ones(n)
        elif self.loss_name == "W-learning":
            cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
            cw = np.clip(cw, 1e-6, None)
            m = trt_pm
            w = 1.0 / cw
        else:
            raise ValueError("ModifiedLinearRegressor only supports A-learning or W-learning.")

        def obj(theta):
            b = theta[0]
            beta = theta[1:]
            pred_raw = b + X @ beta
            resid = y - m * pred_raw

            loss = np.mean(w * resid ** 2) + self.reg_lambda * np.sum(beta ** 2)

            g_pred = (-2.0 * w * m * resid) / n
            g_b = np.sum(g_pred)
            g_beta = X.T @ g_pred + 2.0 * self.reg_lambda * beta

            grad = np.concatenate([[g_b], g_beta])
            return loss, grad

        theta0 = np.zeros(p + 1)
        res = minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.maxiter)}
        )

        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        self.n_iter_ = getattr(res, "nit", None)
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


class ModifiedLogisticRegressor:
    def __init__(self, loss_name="A-learning", reg_lambda=1e-6, maxiter=500):
        self.loss_name = loss_name
        self.reg_lambda = reg_lambda
        self.maxiter = maxiter
        self.intercept_ = None
        self.coef_ = None
        self.n_iter_ = None
        self.success_ = None
        self.message_ = None

    def fit(self, X, y, trt_pm, pi):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        trt_pm = np.asarray(trt_pm, dtype=float).reshape(-1)
        pi = np.asarray(pi, dtype=float).reshape(-1)

        n, p = X.shape

        if self.loss_name == "A-learning":
            m = (trt_pm + 1.0) / 2.0 - pi
            w = np.ones(n)
        elif self.loss_name == "W-learning":
            cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
            cw = np.clip(cw, 1e-6, None)
            m = trt_pm
            w = 1.0 / cw
        else:
            raise ValueError("ModifiedLogisticRegressor only supports A-learning or W-learning.")

        def obj(theta):
            b = theta[0]
            beta = theta[1:]
            pred_raw = b + X @ beta
            eta = m * pred_raw
            p_hat = sigmoid(eta)

            eps = 1e-12
            nll = -(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
            loss = np.mean(w * nll) + self.reg_lambda * np.sum(beta ** 2)

            g_pred = (w * m * (p_hat - y)) / n
            g_b = np.sum(g_pred)
            g_beta = X.T @ g_pred + 2.0 * self.reg_lambda * beta

            grad = np.concatenate([[g_b], g_beta])
            return loss, grad

        theta0 = np.zeros(p + 1)
        res = minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.maxiter)}
        )

        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        self.n_iter_ = getattr(res, "nit", None)
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def predict(self, X):
        return self.decision_function(X)

    def predict_proba(self, X):
        # Pseudo-probability from raw score only.
        # For modified-loss models, interpret mainly as score, not calibrated probability.
        s = self.decision_function(X)
        p = sigmoid(s)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.vstack([1 - p, p]).T


class ModifiedCoxRegressor:
    def __init__(self, loss_name="A-learning", reg_lambda=1e-6, maxiter=500):
        self.loss_name = loss_name
        self.reg_lambda = reg_lambda
        self.maxiter = maxiter
        self.intercept_ = None
        self.coef_ = None
        self.n_iter_ = None
        self.success_ = None
        self.message_ = None

    def fit(self, X, time, event, trt_pm, pi):
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float).reshape(-1)
        event = np.asarray(event, dtype=float).reshape(-1)
        trt_pm = np.asarray(trt_pm, dtype=float).reshape(-1)
        pi = np.asarray(pi, dtype=float).reshape(-1)

        n, p = X.shape
        R = make_risk_set_matrix(time)

        if self.loss_name == "A-learning":
            m = (trt_pm + 1.0) / 2.0 - pi
            w = np.ones(n)
        elif self.loss_name == "W-learning":
            cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
            cw = np.clip(cw, 1e-6, None)
            m = trt_pm
            w = 1.0 / cw
        else:
            raise ValueError("ModifiedCoxRegressor only supports A-learning or W-learning.")

        wd = w * event

        def obj(theta):
            b = theta[0]
            beta = theta[1:]
            pred_raw = b + X @ beta
            eta = m * pred_raw

            exp_eta = np.exp(np.clip(eta, -50, 50))
            denom = exp_eta @ R
            denom = np.clip(denom, 1e-12, None)

            loss = -np.sum(wd * (eta - np.log(denom))) / n + self.reg_lambda * np.sum(beta ** 2)

            prob = (exp_eta[:, None] / denom[None, :]) * R
            g_eta = -wd + (prob @ wd)

            g_pred = (m * g_eta) / n
            g_b = np.sum(g_pred)
            g_beta = X.T @ g_pred + 2.0 * self.reg_lambda * beta

            grad = np.concatenate([[g_b], g_beta])
            return loss, grad

        theta0 = np.zeros(p + 1)
        res = minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.maxiter)}
        )

        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        self.n_iter_ = getattr(res, "nit", None)
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def predict_partial_hazard(self, X):
        s = self.predict(X)
        return np.exp(np.clip(s, -50, 50))
# -----------------------------
# Custom objectives
# -----------------------------
def make_aw_continuous_objective(loss_name: str, trt_pm: np.ndarray, pi: np.ndarray):
    if loss_name == "Original (clinical)":
        return None, None

    if loss_name == "A-learning":
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label()
            c = (trt_pm + 1.0) / 2.0 - pi
            grad = -2.0 * c * (y - predt * c)
            hess = 2.0 * (c ** 2)
            return grad, hess
        return obj, None

    if loss_name == "W-learning":
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label()
            cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
            cw = np.clip(cw, 1e-6, None)
            grad = (-2.0 * trt_pm) / cw * (y - predt * trt_pm)
            hess = (2.0 * (trt_pm ** 2)) / cw
            return grad, hess
        return obj, None

    raise ValueError(f"Unknown loss: {loss_name}")


def make_aw_binary_objective(loss_name: str, trt_pm: np.ndarray, pi: np.ndarray):
    if loss_name == "Original (clinical)":
        return None, None  # will use built-in binary:logistic

    if loss_name == "A-learning":
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label()
            c = (trt_pm + 1.0) / 2.0 - pi
            eta = c * predt
            p = sigmoid(eta)
            grad = c * (p - y)
            hess = (c ** 2) * p * (1.0 - p)
            return grad, hess
        return obj, None

    if loss_name == "W-learning":
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label()
            cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
            cw = np.clip(cw, 1e-6, None)
            w = 1.0 / cw
            eta = trt_pm * predt
            p = sigmoid(eta)
            grad = w * trt_pm * (p - y)
            hess = w * (trt_pm ** 2) * p * (1.0 - p)
            return grad, hess
        return obj, None

    raise ValueError(f"Unknown loss: {loss_name}")


def make_risk_set_matrix(time_vec: np.ndarray):
    t = np.asarray(time_vec).reshape(-1)
    return (t[:, None] >= t[None, :]).astype(float)


def make_aw_cox_objective(loss_name: str, trt_pm: np.ndarray, pi: np.ndarray, time_vec: np.ndarray, event_vec: np.ndarray):
    time_vec = np.asarray(time_vec).reshape(-1)
    event_vec = np.asarray(event_vec).reshape(-1).astype(float)
    R = make_risk_set_matrix(time_vec)

    if loss_name == "Original (clinical)":
        return None, None  # will use survival:cox

    if loss_name == "A-learning":
        c = (trt_pm + 1.0) / 2.0 - pi

        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            eta = c * predt
            exp_eta = np.exp(np.clip(eta, -50, 50))
            denom = exp_eta @ R
            denom = np.clip(denom, 1e-12, None)
            prob = (exp_eta[:, None] / denom[None, :]) * R
            g_eta = -event_vec + (prob @ event_vec)
            h_eta = (prob * (1.0 - prob)) @ event_vec
            grad = c * g_eta
            hess = (c ** 2) * np.clip(h_eta, 1e-12, None)
            return grad, hess

        return obj, None

    if loss_name == "W-learning":
        cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
        cw = np.clip(cw, 1e-6, None)
        # Unlike the squared/logistic cases, the Cox partial likelihood couples
        # observations through the risk set. The IPW weight therefore belongs to the
        # EVENT being summed over (index j), not to the subject whose prediction is
        # being differentiated (index i):
        #
        #   dL/df_i = T_i * ( -w_i d_i + sum_j w_j d_j p_ij )
        #
        # Weighting the whole bracket by w_i instead yields a direction that still
        # decreases the loss (the signs coincide) but is not its gradient, and it
        # perturbs the fitted feature ranking. ModifiedCoxRegressor.fit below carries
        # wd through both terms; keep the two implementations consistent.
        wd = (1.0 / cw) * event_vec

        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            eta = trt_pm * predt
            exp_eta = np.exp(np.clip(eta, -50, 50))
            denom = exp_eta @ R
            denom = np.clip(denom, 1e-12, None)
            prob = (exp_eta[:, None] / denom[None, :]) * R
            g_eta = -wd + (prob @ wd)
            h_eta = (prob * (1.0 - prob)) @ wd
            grad = trt_pm * g_eta
            hess = (trt_pm ** 2) * np.clip(h_eta, 1e-12, None)
            return grad, hess

        return obj, None

    raise ValueError(f"Unknown loss: {loss_name}")


# -----------------------------
# Metric calculators (YOUR RULES)
# -----------------------------
def _continuous_modified_loss(y, pred_raw, loss_name, trt_pm, pi):
    y = np.asarray(y).reshape(-1)
    pred_raw = np.asarray(pred_raw).reshape(-1)

    if loss_name == "A-learning":
        c = (trt_pm + 1.0) / 2.0 - pi
        return float(np.mean((y - c * pred_raw) ** 2))
    if loss_name == "W-learning":
        cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
        cw = np.clip(cw, 1e-6, None)
        return float(np.mean(((y - trt_pm * pred_raw) ** 2) / cw))
    raise ValueError("continuous modified loss requires A-learning or W-learning")


def _binary_original_metrics(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).reshape(-1)
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "loss": float(log_loss(y_true, prob))
    }


def _binary_modified_loss(y, pred_raw, loss_name, trt_pm, pi):
    y = np.asarray(y).reshape(-1)
    pred_raw = np.asarray(pred_raw).reshape(-1)
    eps = 1e-12

    if loss_name == "A-learning":
        c = (trt_pm + 1.0) / 2.0 - pi
        p = sigmoid(c * pred_raw)
        nll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
        return float(nll)

    if loss_name == "W-learning":
        cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
        cw = np.clip(cw, 1e-6, None)
        w = 1.0 / cw
        p = sigmoid(trt_pm * pred_raw)
        nll = (w * (-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))).mean()
        return float(nll)

    raise ValueError("binary modified loss requires A-learning or W-learning")


def _cox_modified_loss(loss_name, trt_pm, pi, time, event, pred_raw):
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(float)
    pred_raw = np.asarray(pred_raw).reshape(-1)
    R = make_risk_set_matrix(time)

    if loss_name == "A-learning":
        c = (trt_pm + 1.0) / 2.0 - pi
        eta = c * pred_raw
        exp_eta = np.exp(np.clip(eta, -50, 50))
        denom = exp_eta @ R
        denom = np.clip(denom, 1e-12, None)
        pll = -np.sum(event * (eta - np.log(denom))) / len(pred_raw)
        return float(pll)

    if loss_name == "W-learning":
        cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
        cw = np.clip(cw, 1e-6, None)
        w = 1.0 / cw
        eta = trt_pm * pred_raw
        exp_eta = np.exp(np.clip(eta, -50, 50))
        denom = exp_eta @ R
        denom = np.clip(denom, 1e-12, None)
        pll = (w * (-(event * (eta - np.log(denom))))).sum() / len(pred_raw)
        return float(pll)

    raise ValueError("cox modified loss requires A-learning or W-learning")


def _safe_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def benefit_score_sign(outcome_type: str, loss_name: str) -> float:
    """
    Sign that maps the raw model output f(X) onto a *benefiting score* for which
    LARGER means GREATER expected treatment benefit.

    For continuous and binary outcomes the modified losses are written directly in
    terms of the outcome, so f is already oriented that way -- provided the outcome
    is coded so that a larger Y is a better outcome.

    For time-to-event outcomes the Cox loss makes the linear predictor a LOG-HAZARD.
    A larger f therefore means a *higher* hazard under treatment, i.e. LESS benefit,
    and the raw score must be negated before it is compared against a benefiting
    subgroup label, used for ranking, or interpreted directionally via SHAP/LIME.

    Under the original (outcome-prediction) losses the model output is a predicted
    outcome or risk rather than a benefiting score, so no reorientation applies.
    """
    if outcome_type == "time-to-event" and loss_name in ("A-learning", "W-learning"):
        return -1.0
    return 1.0


def benefit_score_note(outcome_type: str, loss_name: str) -> str:
    """One-line description of what larger values of the reported score mean."""
    if loss_name not in ("A-learning", "W-learning"):
        if outcome_type == "time-to-event":
            return "Model output is a risk score: larger = higher hazard."
        return "Model output is the predicted outcome on its natural scale."
    if outcome_type == "time-to-event":
        return ("Benefiting score = -(Cox linear predictor); larger = greater expected "
                "treatment benefit (longer survival under treatment).")
    return ("Benefiting score: larger = greater expected treatment benefit "
            "(assumes the outcome is coded so that larger Y is better).")


# -----------------------------
# Hyperparameter tuning
# -----------------------------
# Two rules govern this section.
#
# 1. The selection criterion is the SAME loss the model is trained on. Choosing
#    hyperparameters by squared error while training under A-learning would select a
#    model for outcome prediction and then interpret it as an ITR model -- exactly the
#    estimand/objective mismatch this application is meant to make explicit. So the
#    criterion is the modified objective itself, evaluated out of fold.
#
# 2. Nothing that depends on the outcome or the treatment assignment may be fitted
#    outside the fold. In particular the propensity score pi(X) is re-estimated on each
#    fold's training rows; estimating it once on all of the training data would leak the
#    validation rows into the weights that define the validation loss.
#
# A ground-truth benefiting label (sigpos), when supplied, is reported as a diagnostic
# only and never used for selection -- it is unobservable in real studies.

# The C that stands in for "no penalty" in sklearn's parameterisation. Large enough that the
# fit is the unpenalised maximum-likelihood solution to the precision that matters, which is
# the fit whose standard errors and confidence intervals are the conventional ones.
LOGREG_C_NO_PENALTY = 1e12

# --- which hyperparameters exist, per model and objective ------------------
# XGBoost: num_boost_round is deliberately absent -- it is set by early stopping on the
# held-out fold rather than searched over. tree_method is a compute choice, not a
# statistical one, so it is a fixed value only.
XGB_TUNABLE = ["learning_rate", "max_depth", "subsample", "colsample_bytree",
               "min_child_weight", "gamma", "reg_lambda", "reg_alpha"]

# Defaults chosen so that the whole space is searched unless the user narrows it.
XGB_TUNED_BY_DEFAULT = list(XGB_TUNABLE)

_INT_PARAMS = {"max_depth"}
_LOG_PARAMS = {"learning_rate", "min_child_weight", "reg_lambda", "reg_alpha",
               "alpha", "C", "penalizer"}

DEFAULT_SPACE_XGB = """learning_rate: 0.01 .. 0.3
max_depth: 2 .. 8
subsample: 0.6 .. 1.0
colsample_bytree: 0.6 .. 1.0
min_child_weight: 0.5 .. 20
gamma: 0, 0.1, 0.5, 1, 2
reg_lambda: 0.1 .. 50
reg_alpha: 0, 0.01, 0.1, 1"""

DEFAULT_SPACE_PENALTY = {
    "reg_lambda": "reg_lambda: 1e-06, 1e-04, 1e-03, 1e-02, 0.1, 1, 10",
    "alpha": "alpha: 0, 0.01, 0.1, 1, 10, 100",
    "C": "C: 0.01, 0.1, 1, 10, 100",
    "penalizer": "penalizer: 0, 1e-03, 1e-02, 0.1, 1",
}

PARAM_LABELS = {
    "reg_lambda": "reg_lambda (L2 penalty on the modified-loss estimator)",
    "alpha": "alpha (ridge penalty; 0 = ordinary least squares)",
    "C": "C (inverse L2 penalty; smaller = stronger penalty)",
    "penalizer": "penalizer (Cox L2 penalty)",
}


def default_fixed_penalty(model_type: str, loss_name: str) -> float:
    """Fixed penalty that reproduces the behaviour before tuning was added."""
    if loss_name in ("A-learning", "W-learning"):
        return 1e-6
    if model_type == "Logistic Regression":
        return 1.0
    return 0.0

MAX_GRID_CONFIGS = 400


def tunable_parameter_names(model_type: str, loss_name: str) -> list:
    """Hyperparameters that can be tuned for this (model, objective) combination."""
    if model_type == "XGBoost":
        return list(XGB_TUNABLE)
    if loss_name in ("A-learning", "W-learning"):
        # All three modified-loss estimators are penalised linear models.
        return ["reg_lambda"]
    if model_type == "Linear Regression":
        return ["alpha"]
    if model_type == "Logistic Regression":
        return ["C"]
    if model_type == "Random Forest":
        # One complexity knob, to match the single penalty field the other non-tree models
        # use. Larger leaves mean a smoother fit, so it plays the role a penalty plays.
        return ["min_samples_leaf"]
    if model_type == "Cox Regression":
        return ["penalizer"]
    return []


def default_search_space_text(model_type: str, loss_name: str) -> str:
    if model_type == "XGBoost":
        return DEFAULT_SPACE_XGB
    names = tunable_parameter_names(model_type, loss_name)
    return "\n".join(DEFAULT_SPACE_PENALTY[n] for n in names if n in DEFAULT_SPACE_PENALTY)


def parse_search_space(text: str, allowed=None):
    """Parse the editable search-space definition.

    One parameter per line, in either of two forms:

        name: lo .. hi          inclusive range (sampled for random search,
                                discretised for grid search)
        name: v1, v2, v3        explicit list of candidate values

    Text after '#' is ignored. Returns (space, errors) where space maps a parameter
    name to ('range', lo, hi) or ('list', [values]).
    """
    space, errors = {}, []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            errors.append(f"missing ':' in {raw.strip()!r}")
            continue
        name, rhs = line.split(":", 1)
        name, rhs = name.strip(), rhs.strip()
        if not name:
            errors.append(f"missing parameter name in {raw.strip()!r}")
            continue
        if allowed is not None and name not in allowed:
            errors.append(f"{name!r} is not tunable for this model/objective")
            continue
        try:
            if ".." in rhs:
                parts = [p for p in rhs.split("..") if p.strip() != ""]
                if len(parts) != 2:
                    raise ValueError("a range needs exactly two values, 'lo .. hi'")
                lo, hi = float(parts[0]), float(parts[1])
                if not hi > lo:
                    raise ValueError(f"range needs hi > lo (got {lo} .. {hi})")
                space[name] = ("range", lo, hi)
            else:
                vals = [float(v) for v in rhs.split(",") if v.strip() != ""]
                if not vals:
                    raise ValueError("no values given")
                space[name] = ("list", vals)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return space, errors


def _cast(name, v):
    return int(round(float(v))) if name in _INT_PARAMS else float(v)


def _sample_value(name, spec, rng: np.random.Generator):
    if spec[0] == "list":
        return _cast(name, rng.choice(spec[1]))
    lo, hi = spec[1], spec[2]
    if name in _LOG_PARAMS and lo > 0:
        return _cast(name, np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return _cast(name, rng.uniform(lo, hi))


def _grid_values(name, spec, grid_points):
    if spec[0] == "list":
        vals = list(spec[1])
    else:
        lo, hi = spec[1], spec[2]
        k = max(2, int(grid_points))
        if name in _LOG_PARAMS and lo > 0:
            vals = list(np.exp(np.linspace(np.log(lo), np.log(hi), k)))
        else:
            vals = list(np.linspace(lo, hi, k))
    out = [_cast(name, v) for v in vals]
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def build_candidates(mode, space, tuned_names, *, model_type="XGBoost",
                     loss_name="Original (clinical)", n_iter=20, seed=42,
                     grid_points=3, max_configs=MAX_GRID_CONFIGS, incumbent=None):
    """Candidate configurations for the chosen search mode.

    Returns (candidates, description). Parameters that are defined in the space but not
    selected for tuning are left out, so they keep their fixed sidebar values.

    The incumbent (the user's fixed sidebar configuration) is always evaluated as the
    first candidate. Without it, a random search with few draws can return a
    configuration worse than the one the user started from, which is a confusing outcome
    for a feature whose purpose is to improve the model. Including it guarantees that the
    selected configuration is no worse than the default on the cross-validated criterion.
    """
    order = tunable_parameter_names(model_type, loss_name)
    sel = set(tuned_names or [])
    names = [n for n in order if n in sel and n in space]
    if not names:
        return [{}], "no parameters selected for tuning; the fixed values are used as-is"

    inc = None
    if incumbent:
        vals = {n: _cast(n, incumbent[n]) for n in names if incumbent.get(n) is not None}
        if len(vals) == len(names):
            inc = vals

    if str(mode).lower().startswith("grid"):
        import itertools
        grids = [_grid_values(n, space[n], grid_points) for n in names]
        total = int(np.prod([len(g) for g in grids]))
        if total > int(max_configs):
            raise ValueError(
                f"Grid search over {', '.join(names)} would evaluate {total} "
                f"configurations, above the limit of {int(max_configs)}. Reduce the "
                f"number of tuned parameters, shorten the candidate lists, or lower "
                f"'grid points per range'."
            )
        cands = [dict(zip(names, combo)) for combo in itertools.product(*grids)]
        sizes = ", ".join(f"{n}({len(g)})" for n, g in zip(names, grids))
        desc = f"grid over {sizes} = {total} configurations"
        if inc is not None and inc not in cands:
            cands = [inc] + cands
            desc += ", plus the fixed configuration"
        return cands, desc

    rng = np.random.default_rng(int(seed))
    draws = [{n: _sample_value(n, space[n], rng) for n in names}
             for _ in range(int(max(1, n_iter)))]
    if inc is not None:
        draws = [inc] + draws
    # Drop repeated draws, which happen whenever the space is a short explicit list (the
    # usual case for the one-dimensional penalty searches). Order is preserved, so the
    # result is still fully determined by the seed.
    seen, cands = set(), []
    for c in draws:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            cands.append(c)
    dup = len(draws) - len(cands)
    desc = f"random search: {len(cands)} distinct configuration(s) over {len(names)} parameter(s)"
    if inc is not None:
        desc += ", including the fixed configuration"
    if dup:
        desc += f" ({dup} duplicate draw(s) dropped)"
    return cands, desc


def describe_search(mode, space_text, tuned_names, *, model_type, loss_name,
                    n_iter, seed, grid_points, n_splits, incumbent=None):
    """One-line summary of the configured search, for display before running."""
    allowed = tunable_parameter_names(model_type, loss_name)
    space, errors = parse_search_space(space_text, allowed=allowed)
    if errors:
        return "Search space has errors: " + "; ".join(errors[:4])
    try:
        cands, desc = build_candidates(
            mode, space, tuned_names, model_type=model_type, loss_name=loss_name,
            n_iter=n_iter, seed=seed, grid_points=grid_points, incumbent=incumbent,
        )
    except Exception as e:
        return str(e)
    n_fits = len(cands) * int(n_splits)
    return f"{desc}; {len(cands)} x {int(n_splits)} folds = {n_fits} model fits"


def cv_criterion_name(outcome_type: str, loss_name: str, model_type: str) -> str:
    if loss_name in ("A-learning", "W-learning"):
        return f"CV {loss_name} objective (out-of-fold)"
    if outcome_type == "continuous":
        return "CV RMSE (out-of-fold)"
    if outcome_type == "binary":
        return "CV log-loss (out-of-fold)"
    if model_type == "XGBoost":
        return "CV Cox negative partial log-likelihood (out-of-fold)"
    return "CV (1 - Harrell C-index) (out-of-fold)"


def _make_folds(n, n_splits, seed, stratify=None):
    n_splits = int(max(2, min(int(n_splits), n)))
    if stratify is not None:
        s = np.asarray(stratify).reshape(-1)
        _, counts = np.unique(s, return_counts=True)
        if len(counts) > 1 and counts.min() >= n_splits:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            return list(skf.split(np.zeros(n), s))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(kf.split(np.zeros(n)))


def _fold_propensity(X_tr, trt_tr, X_va):
    """Propensity fitted on the fold's training rows only. Falls back to the observed
    randomisation ratio if an arm is missing from the fold."""
    if len(np.unique(trt_tr)) < 2:
        p = float(np.clip(np.mean(trt_tr), 1e-3, 1 - 1e-3))
        return np.full(len(X_tr), p), np.full(len(X_va), p)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, trt_tr)
    return lr.predict_proba(X_tr)[:, 1], lr.predict_proba(X_va)[:, 1]


def _make_modified_metric(outcome_type, loss_name, *, y_va=None, tpm_va=None,
                          pi_va=None, time_va=None, event_va=None, name="cv_objective"):
    """XGBoost custom_metric evaluating the modified objective on the validation fold.

    Only ever called with the validation DMatrix, since that is the sole entry in
    `evals`. With a custom objective the predictions passed here are raw margins,
    which is what the modified losses expect.
    """
    def metric(predt, dmat):
        predt = np.asarray(predt).reshape(-1)
        if outcome_type == "continuous":
            v = _continuous_modified_loss(y_va, predt, loss_name, tpm_va, pi_va)
        elif outcome_type == "binary":
            v = _binary_modified_loss(y_va, predt, loss_name, tpm_va, pi_va)
        else:
            v = _cox_modified_loss(loss_name, tpm_va, pi_va, time_va, event_va, predt)
        return name, float(v)
    return metric


def xgb_cox_label(time_vec, event_vec):
    """Label encoding required by XGBoost's built-in `survival:cox`: the survival time,
    made NEGATIVE for right-censored observations. Passing raw times would treat every
    observation as an observed event."""
    t = np.asarray(time_vec, dtype=float).reshape(-1)
    e = np.asarray(event_vec, dtype=float).reshape(-1)
    return np.where(e > 0, t, -t)


def _cv_score_candidate(
    X, folds, *, outcome_type, model_type, loss_name, cand,
    y=None, time=None, event=None, trt01=None, sigpos=None,
    base_xgb_params=None, max_rounds=400, esr=30, feature_names=None,
):
    """Mean out-of-fold criterion for one candidate configuration.

    Returns (score, sd, mean_best_rounds, mean_sigpos_auc). Lower score is better.
    """
    needs_trt = loss_name in ("A-learning", "W-learning")
    bsign = benefit_score_sign(outcome_type, loss_name)
    scores, rounds, aucs = [], [], []

    for idx_tr, idx_va in folds:
        X_tr, X_va = X[idx_tr], X[idx_va]

        pi_tr = pi_va = tpm_tr = tpm_va = None
        if needs_trt:
            pi_tr, pi_va = _fold_propensity(X_tr, trt01[idx_tr], X_va)
            tpm_tr = np.where(trt01[idx_tr] == 1, 1.0, -1.0)
            tpm_va = np.where(trt01[idx_va] == 1, 1.0, -1.0)

        if model_type == "XGBoost":
            params = dict(base_xgb_params or {})
            params.update({k: v for k, v in cand.items()})
            params.setdefault("tree_method", "hist")
            params["verbosity"] = 0

            if outcome_type == "time-to-event":
                if loss_name == "Original (clinical)":
                    lbl_tr = xgb_cox_label(time[idx_tr], event[idx_tr])
                    lbl_va = xgb_cox_label(time[idx_va], event[idx_va])
                else:
                    lbl_tr, lbl_va = time[idx_tr], time[idx_va]
            else:
                lbl_tr, lbl_va = y[idx_tr], y[idx_va]

            dtr = xgb.DMatrix(X_tr, label=np.asarray(lbl_tr, dtype=float))
            dva = xgb.DMatrix(X_va, label=np.asarray(lbl_va, dtype=float))

            obj_fn = None
            custom_metric = None
            if loss_name == "Original (clinical)":
                if outcome_type == "continuous":
                    params["objective"] = "reg:squarederror"
                    params["eval_metric"] = mname = "rmse"
                elif outcome_type == "binary":
                    params["objective"] = "binary:logistic"
                    params["eval_metric"] = mname = "logloss"
                else:
                    params["objective"] = "survival:cox"
                    params["eval_metric"] = mname = "cox-nloglik"
            else:
                mname = "cv_objective"
                if outcome_type == "continuous":
                    obj_fn, _ = make_aw_continuous_objective(loss_name, tpm_tr, pi_tr)
                elif outcome_type == "binary":
                    obj_fn, _ = make_aw_binary_objective(loss_name, tpm_tr, pi_tr)
                else:
                    obj_fn, _ = make_aw_cox_objective(
                        loss_name, tpm_tr, pi_tr, time[idx_tr], event[idx_tr]
                    )
                custom_metric = _make_modified_metric(
                    outcome_type, loss_name,
                    y_va=(y[idx_va] if y is not None else None),
                    tpm_va=tpm_va, pi_va=pi_va,
                    time_va=(time[idx_va] if time is not None else None),
                    event_va=(event[idx_va] if event is not None else None),
                    name=mname,
                )

            callbacks = []
            if int(esr) > 0:
                callbacks.append(xgb.callback.EarlyStopping(
                    rounds=int(esr), metric_name=mname, data_name="va", maximize=False
                ))

            evals_result = {}
            bst = xgb.train(
                params, dtr,
                num_boost_round=int(max_rounds),
                obj=obj_fn,
                evals=[(dva, "va")],
                custom_metric=custom_metric,
                evals_result=evals_result,
                callbacks=callbacks,
                verbose_eval=False,
            )
            curve = evals_result["va"][mname]
            best_it = int(getattr(bst, "best_iteration", len(curve) - 1))
            best_it = max(0, min(best_it, len(curve) - 1))
            sc = float(curve[best_it])
            rounds.append(best_it + 1)
            pred_va = bst.predict(dva, iteration_range=(0, best_it + 1))

        elif model_type == "Linear Regression":
            if loss_name == "Original (clinical)":
                a = float(cand.get("alpha", 0.0))
                est = Ridge(alpha=a) if a > 0 else LinearRegression()
                est.fit(X_tr, y[idx_tr])
                pred_va = est.predict(X_va)
                sc = float(np.sqrt(mean_squared_error(y[idx_va], pred_va)))
            else:
                est = ModifiedLinearRegressor(
                    loss_name=loss_name, reg_lambda=float(cand["reg_lambda"])
                )
                est.fit(X_tr, y[idx_tr], tpm_tr, pi_tr)
                pred_va = est.predict(X_va)
                sc = _continuous_modified_loss(y[idx_va], pred_va, loss_name, tpm_va, pi_va)

        elif model_type == "Logistic Regression":
            if loss_name == "Original (clinical)":
                est = LogisticRegression(C=float(cand.get("C", 1.0)), max_iter=2000)
                est.fit(X_tr, y[idx_tr].astype(int))
                pred_va = np.clip(est.predict_proba(X_va)[:, 1], 1e-12, 1 - 1e-12)
                sc = float(log_loss(y[idx_va].astype(int), pred_va, labels=[0, 1]))
            else:
                est = ModifiedLogisticRegressor(
                    loss_name=loss_name, reg_lambda=float(cand["reg_lambda"])
                )
                est.fit(X_tr, y[idx_tr].astype(float), tpm_tr, pi_tr)
                pred_va = est.predict(X_va)
                sc = _binary_modified_loss(y[idx_va], pred_va, loss_name, tpm_va, pi_va)

        elif model_type == "Cox Regression":
            if loss_name == "Original (clinical)":
                if not _HAS_LIFELINES:
                    raise ValueError("lifelines is required to tune Cox Regression.")
                cols = list(feature_names)
                df_tr = pd.DataFrame(X_tr, columns=cols)
                df_tr["__t__"] = time[idx_tr].astype(float)
                df_tr["__e__"] = event[idx_tr].astype(int)
                cph = CoxPHFitter(penalizer=float(cand.get("penalizer", 0.0)))
                cph.fit(df_tr, duration_col="__t__", event_col="__e__")
                pred_va = cph.predict_partial_hazard(
                    pd.DataFrame(X_va, columns=cols)
                ).values.reshape(-1)
                c = concordance_index(time[idx_va], event[idx_va], pred_va)
                sc = float(1.0 - c) if np.isfinite(c) else np.nan
            else:
                est = ModifiedCoxRegressor(
                    loss_name=loss_name, reg_lambda=float(cand["reg_lambda"])
                )
                est.fit(X_tr, time[idx_tr], event[idx_tr], tpm_tr, pi_tr)
                pred_va = est.predict(X_va)
                sc = _cox_modified_loss(
                    loss_name, tpm_va, pi_va, time[idx_va], event[idx_va], pred_va
                )
        else:
            raise ValueError(f"Tuning not implemented for model_type={model_type}")

        scores.append(float(sc))
        if sigpos is not None and needs_trt:
            aucs.append(_safe_auc(sigpos[idx_va], bsign * np.asarray(pred_va).reshape(-1)))

    scores = np.asarray(scores, dtype=float)
    good = np.isfinite(scores)
    if not good.any():
        return np.nan, np.nan, None, np.nan
    mean_auc = float(np.nanmean(aucs)) if len(aucs) else np.nan
    mean_rounds = int(round(float(np.mean(rounds)))) if rounds else None
    return float(np.mean(scores[good])), float(np.std(scores[good])), mean_rounds, mean_auc


def tune_hyperparameters(
    X_train, *, outcome_type, model_type, loss_name,
    y_train=None, time_train=None, event_train=None, trt01_train=None,
    sigpos_train=None, mode="Random search (CV)", n_iter=20, n_splits=5,
    seed=42, max_rounds=400, esr=30, base_xgb_params=None,
    feature_names=None, top_k=5,
    space_text=None, tuned_names=None, grid_points=3, incumbent=None,
):
    """Select hyperparameters by K-fold CV on the TRAINING data only.

    The test split is never touched, so the reported test metrics remain an honest
    out-of-sample estimate for the selected configuration.
    """
    X = np.asarray(X_train, dtype=float)
    n = X.shape[0]

    if outcome_type == "binary" and y_train is not None:
        strat = np.asarray(y_train).astype(int)
    elif outcome_type == "time-to-event" and event_train is not None:
        strat = np.asarray(event_train).astype(int)
    else:
        strat = None
    folds = _make_folds(n, n_splits, seed, stratify=strat)

    allowed = tunable_parameter_names(model_type, loss_name)
    if space_text is None:
        space_text = default_search_space_text(model_type, loss_name)
    if tuned_names is None:
        tuned_names = list(allowed)

    space, space_errors = parse_search_space(space_text, allowed=allowed)
    if space_errors:
        raise ValueError("Search space definition is invalid -- "
                         + "; ".join(space_errors[:6]))

    cands, search_desc = build_candidates(
        mode, space, tuned_names, model_type=model_type, loss_name=loss_name,
        n_iter=n_iter, seed=seed, grid_points=grid_points, incumbent=incumbent,
    )
    tuned_used = [n for n in allowed if n in (set(tuned_names or [])) and n in space]

    trace = []
    for cand in cands:
        try:
            sc, sd, rnds, auc = _cv_score_candidate(
                X, folds,
                outcome_type=outcome_type, model_type=model_type, loss_name=loss_name,
                cand=cand, y=y_train, time=time_train, event=event_train,
                trt01=trt01_train, sigpos=sigpos_train,
                base_xgb_params=base_xgb_params, max_rounds=max_rounds, esr=esr,
                feature_names=feature_names,
            )
        except Exception as e:
            trace.append({"params": cand, "score": np.nan, "sd": np.nan,
                          "rounds": None, "sigpos_auc": np.nan, "error": str(e)})
            continue
        trace.append({"params": cand, "score": sc, "sd": sd,
                      "rounds": rnds, "sigpos_auc": auc})

    ok = [t for t in trace if np.isfinite(t.get("score", np.nan))]
    if not ok:
        msgs = {t.get("error") for t in trace if t.get("error")}
        raise ValueError("All candidate configurations failed. "
                         + ("; ".join(sorted(m for m in msgs if m))[:400] or "No score returned."))

    ok.sort(key=lambda t: t["score"])
    best = ok[0]
    return {
        "mode": mode,
        "criterion": cv_criterion_name(outcome_type, loss_name, model_type),
        "n_splits": len(folds),
        "n_configs": len(cands),
        "n_failed": len(trace) - len(ok),
        "seed": int(seed),
        "search": search_desc,
        "tuned": tuned_used,
        "not_tuned": [n for n in allowed if n not in tuned_used],
        "space_text": space_text,
        "best_params": dict(best["params"]),
        "best_score": best["score"],
        "best_sd": best["sd"],
        "best_rounds": best["rounds"],
        "best_sigpos_auc": best["sigpos_auc"],
        "top": [
            {k: t[k] for k in ("params", "score", "sd", "rounds", "sigpos_auc")}
            for t in ok[:int(top_k)]
        ],
    }


def format_tuning_report(info: Optional[Dict[str, Any]]) -> list:
    """Human-readable summary of a tuning run, for the Metrics panel."""
    if not info:
        return []
    if "error" in info and "best_params" not in info:
        return ["=== Hyperparameter tuning ===", f"Tuning failed: {info['error']}",
                "Falling back to the values set in the sidebar."]

    lines = ["=== Hyperparameter tuning ===",
             f"Search: {info['mode']} -- {info.get('search', '')}",
             f"Cross-validation: {info['n_splits']} folds, seed {info['seed']}",
             f"Criterion: {info['criterion']}",
             "Selection used the training data only; the test split was held out.",
             "The propensity score was re-estimated within each fold."]
    if info.get("tuned"):
        lines.append(f"Tuned: {', '.join(info['tuned'])}")
    if info.get("not_tuned"):
        lines.append(f"Held at the fixed sidebar values: {', '.join(info['not_tuned'])}")
    if info.get("n_failed"):
        lines.append(f"{info['n_failed']} configuration(s) failed and were skipped.")

    lines.append("")
    lines.append("Selected configuration:")
    for k in sorted(info["best_params"]):
        v = info["best_params"][k]
        lines.append(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}")
    if info.get("best_rounds"):
        lines.append(f"  num_boost_round (from early stopping): {info['best_rounds']}")
    lines.append(f"  {info['criterion']}: {info['best_score']:.5f} "
                 f"(SD across folds {info['best_sd']:.5f})")
    if info.get("best_sigpos_auc") is not None and np.isfinite(info.get("best_sigpos_auc", np.nan)):
        lines.append(f"  out-of-fold AUC vs ground-truth benefiting label: "
                     f"{info['best_sigpos_auc']:.4f}  (diagnostic only, not used to select)")

    lines.append("")
    lines.append(f"Top {len(info['top'])} configurations by CV criterion:")
    for r, t in enumerate(info["top"], 1):
        ps = ", ".join(
            f"{k}={t['params'][k]:.4g}" if isinstance(t["params"][k], float)
            else f"{k}={t['params'][k]}"
            for k in sorted(t["params"])
        )
        extra = f", rounds={t['rounds']}" if t.get("rounds") else ""
        auc = t.get("sigpos_auc")
        extra += (f", sigpos AUC={auc:.3f}"
                  if auc is not None and np.isfinite(auc) else "")
        lines.append(f"  {r}. score={t['score']:.5f} (SD {t['sd']:.5f}){extra}  [{ps}]")
    return lines


# -----------------------------
# Fit model
# -----------------------------
def fit_model(
    df: pd.DataFrame,
    feature_cols,
    outcome_col="y",
    treat_col="treatment",
    outcome_type="continuous",
    event_col=None,
    model_type="XGBoost",
    loss_name="Original (clinical)",
    test_size=0.25,
    seed=42,
    sigpos_col=None,
    xgb_params=None,
    num_boost_round=400,
    tune_mode="None",
    tune_n_iter=20,
    tune_folds=5,
    tune_seed=42,
    tune_esr=30,
    tune_space_text=None,
    tune_params=None,
    tune_grid_points=3,
    # Fixed values for the non-XGBoost models. Defaults reproduce the previous behaviour
    # exactly: unpenalised OLS, sklearn's default C, no Cox penalty, negligible ridge on
    # the modified-loss estimators.
    lin_alpha=0.0,
    logreg_C=1.0,
    rf_n_estimators=400,
    rf_min_samples_leaf=1,
    cox_penalizer=0.0,
    mod_reg_lambda=1e-6,
    na_policy="Complete cases",
    standardize=False,
):
    # --- missing data, resolved explicitly rather than left to whichever estimator
    # --- happens to fail first. Rows whose outcome, treatment or event is missing can
    # --- never be recovered, so they are always dropped and the count is reported.
    prep = {"na_policy": str(na_policy), "standardize": bool(standardize),
            "n_input": int(len(df)), "dropped_target": 0, "dropped_incomplete": 0,
            "imputed": {}, "scaler": None}

    need_trt = loss_name in ("A-learning", "W-learning")
    target_cols = [outcome_col]
    if outcome_type == "time-to-event" and event_col:
        target_cols.append(event_col)
    if need_trt and treat_col in df.columns:
        target_cols.append(treat_col)
    target_cols = [c for c in target_cols if c in df.columns]

    if target_cols:
        keep = df[target_cols].notna().all(axis=1)
        prep["dropped_target"] = int((~keep).sum())
        if prep["dropped_target"]:
            df = df.loc[keep]

    pol = str(na_policy).lower()
    if pol.startswith("complete"):
        keep = df[feature_cols].notna().all(axis=1)
        prep["dropped_incomplete"] = int((~keep).sum())
        if prep["dropped_incomplete"]:
            df = df.loc[keep]
    elif pol.startswith("keep"):
        if model_type != "XGBoost" or loss_name != "Original (clinical)":
            raise ValueError(
                "'Keep missing values' is only supported for XGBoost under the original "
                "loss, which handles them natively. Every other path (the propensity "
                "model, the parametric estimators, and LIME) requires complete data: "
                "choose complete cases or median imputation."
            )

    if len(df) == 0:
        raise ValueError("No rows remain after applying the missing-data policy.")

    X = df[feature_cols].copy()

    # optional sigpos
    sigpos_exists = False
    sigpos_all = None
    if sigpos_col and sigpos_col.strip():
        sigpos_col = sigpos_col.strip()
        if sigpos_col in df.columns:
            sigpos_all = df[sigpos_col].values
            if set(np.unique(sigpos_all)) - {0, 1}:
                raise ValueError(f"Sigpos column '{sigpos_col}' must be coded as 0/1.")
            sigpos_exists = True

    # --- split
    if test_size == 0:
        # Use all data as training; no test set
        X_train = X.copy()
        X_test = X.iloc[0:0].copy()  # empty DF, keeps columns

        if outcome_type == "time-to-event":
            y_train = df[outcome_col].values.astype(float)
            y_test = np.array([])

            time_train = y_train
            time_test = np.array([])

            event_train = df[event_col].values.astype(int)
            event_test = np.array([])

        else:
            y_train = df[outcome_col].values.astype(float)
            y_test = np.array([])

        if sigpos_exists:
            sigpos_train = df[sigpos_col].values.astype(int)
            sigpos_test = np.array([])
        else:
            sigpos_train = sigpos_test = None
    else:
        if outcome_type == "time-to-event":
            if not event_col or event_col.strip() == "":
                raise ValueError("For time-to-event, please provide an event column name.")
            if outcome_col not in df.columns:
                raise ValueError(f"Time column '{outcome_col}' not found.")
            if event_col not in df.columns:
                raise ValueError(f"Event column '{event_col}' not found.")

            y_time = df[outcome_col].values.astype(float)
            y_event = df[event_col].values.astype(int)

            X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
                X, y_time, y_event, test_size=test_size, random_state=seed
            )
            y_train = time_train
            y_test = time_test

            if sigpos_exists:
                sigpos_train = df.loc[X_train.index, sigpos_col].values.astype(int)
                sigpos_test = df.loc[X_test.index, sigpos_col].values.astype(int)
            else:
                sigpos_train = sigpos_test = None

        else:
            if outcome_col not in df.columns:
                raise ValueError(f"Outcome column '{outcome_col}' not found.")
            y = df[outcome_col].values.astype(float)

            strat = None
            if outcome_type == "binary":
                uniq = set(np.unique(y))
                if uniq - {0, 1}:
                    raise ValueError("Binary outcome must be coded as 0/1 in column y.")
                strat = y.astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=strat
            )

            if sigpos_exists:
                sigpos_train = df.loc[X_train.index, sigpos_col].values.astype(int)
                sigpos_test = df.loc[X_test.index, sigpos_col].values.astype(int)
            else:
                sigpos_train = sigpos_test = None

    # --- safety checks after split
    if X_train.shape[0] == 0:
        raise ValueError("Training set is empty. Reduce test_size or provide more samples.")
    if test_size > 0 and X_test.shape[0] == 0:
        raise ValueError("Test set is empty. Reduce test_size or provide more samples.")

    HAS_TEST = (X_test.shape[0] > 0 and len(y_test) > 0)

    # --- median imputation and standardisation are fitted on the TRAINING split only.
    # --- Fitting either on the full data would let the test rows influence the values
    # --- used to build the model, so the reported test metrics would no longer be
    # --- out-of-sample.
    if pol.startswith("median"):
        med = X_train.median(numeric_only=True)
        med = med.fillna(0.0)
        n_before = int(X_train.isna().sum().sum() + (X_test.isna().sum().sum() if HAS_TEST else 0))
        if n_before:
            prep["imputed"] = {c: float(med[c]) for c in feature_cols
                               if c in med.index and (X_train[c].isna().any()
                                                      or (HAS_TEST and X_test[c].isna().any()))}
            X_train = X_train.fillna(med)
            if HAS_TEST:
                X_test = X_test.fillna(med)
        prep["n_imputed_cells"] = n_before

    # Spread of feature scales on the training split, measured before any standardisation.
    _sd = X_train.std(ddof=0)
    _sd = _sd[_sd > 0]
    prep["sd_ratio"] = float(_sd.max() / _sd.min()) if len(_sd) else 1.0

    if standardize:
        mu = X_train.mean()
        sd = X_train.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        X_train = (X_train - mu) / sd
        if HAS_TEST:
            X_test = (X_test - mu) / sd
        prep["scaler"] = {"mean": {k: float(v) for k, v in mu.items()},
                          "sd": {k: float(v) for k, v in sd.items()}}

    # Orientation of the raw model output as a benefiting score. Every comparison of
    # the model output against a benefiting-subgroup label, and every directional
    # interpretation downstream (SHAP / LIME / LLM), must use this sign.
    BSIGN = benefit_score_sign(outcome_type, loss_name)

    # --- propensity for modified losses
    pi_train = pi_test = None
    trt_train_pm = trt_test_pm = None

    if loss_name in ["A-learning", "W-learning"]:
        if treat_col not in df.columns:
            raise ValueError(f"Loss '{loss_name}' requires column '{treat_col}' in the uploaded data.")

        trt_all = df[treat_col].values
        if set(np.unique(trt_all)) - {0, 1}:
            raise ValueError("Treatment column must be coded as 0/1.")

        trt_train = df.loc[X_train.index, treat_col].values.reshape(-1).astype(int)
        trt_test = df.loc[X_test.index, treat_col].values.reshape(-1).astype(int) if X_test.shape[0] > 0 else np.array([], dtype=int)

        logreg = LogisticRegression(max_iter=2000)
        logreg.fit(X_train.values, trt_train)
        pi_train = logreg.predict_proba(X_train.values)[:, 1]
        pi_test = logreg.predict_proba(X_test.values)[:, 1] if X_test.shape[0] > 0 else np.array([])

        trt_train_pm = np.where(trt_train == 1, 1.0, -1.0)
        trt_test_pm = np.where(trt_test == 1, 1.0, -1.0) if len(trt_test) > 0 else np.array([])
    else:
        trt_train = None

    # --- hyperparameter tuning (training data only; the test split is untouched)
    tuning_info = None
    reg_lambda_mod = float(mod_reg_lambda)   # penalty for the Modified* estimators
    ridge_alpha_lin = float(lin_alpha)       # 0 -> plain LinearRegression, as before
    logreg_C = float(logreg_C)
    cox_penalizer = float(cox_penalizer)

    # sklearn's C is INVERSE regularisation strength, so unlike alpha, penalizer and
    # reg_lambda, zero is not "no penalty" -- it is rejected outright. The sidebar offers one
    # numeric field for all four parametric models, so a user who leaves it at its default of
    # 0 previously got an InvalidParameterError instead of a fit. Zero is read here as the
    # intent it can only have had, no penalty, which for C means effectively unbounded.
    if model_type == "Logistic Regression" and loss_name == "Original (clinical)":
        if logreg_C == 0.0:
            logreg_C = LOGREG_C_NO_PENALTY
        elif logreg_C < 0.0:
            raise ValueError(
                "C for Logistic Regression must be positive. It is inverse regularisation "
                "strength, so a larger value means a weaker penalty; use 0 for no penalty.")

    # Name of the single tunable penalty for the non-tree models, and its fixed value
    # before any tuning. The latter is the incumbent handed to the search.
    _pnames = tunable_parameter_names(model_type, loss_name)
    _pname = _pnames[0] if (_pnames and model_type != "XGBoost") else None

    def _current_penalty():
        return {
            "reg_lambda": reg_lambda_mod,
            "alpha": ridge_alpha_lin,
            "C": logreg_C,
            "penalizer": cox_penalizer,
            "min_samples_leaf": float(rf_min_samples_leaf),
        }.get(_pname)

    _incumbent_penalty = _current_penalty()

    if tune_mode and str(tune_mode) != "None":
        if loss_name in ("A-learning", "W-learning") and trt_train is None:
            raise ValueError("Tuning an ITR objective requires the treatment column.")
        try:
            tuning_info = tune_hyperparameters(
                X_train.values,
                outcome_type=outcome_type,
                model_type=model_type,
                loss_name=loss_name,
                y_train=(y_train if outcome_type != "time-to-event" else None),
                time_train=(time_train if outcome_type == "time-to-event" else None),
                event_train=(event_train if outcome_type == "time-to-event" else None),
                trt01_train=trt_train,
                sigpos_train=sigpos_train,
                mode=tune_mode,
                n_iter=tune_n_iter,
                n_splits=tune_folds,
                seed=tune_seed,
                max_rounds=num_boost_round,
                esr=tune_esr,
                base_xgb_params=(xgb_params if model_type == "XGBoost" else None),
                feature_names=list(X_train.columns),
                space_text=tune_space_text,
                tuned_names=tune_params,
                grid_points=tune_grid_points,
                incumbent=(dict(xgb_params or {}) if model_type == "XGBoost"
                           else ({_pname: _incumbent_penalty} if _pname else None)),
            )
        except Exception as e:
            # Never let a tuning failure block the analysis; fall back to the sidebar
            # values and say so in the Metrics panel.
            tuning_info = {"error": str(e)}
        else:
            bp = tuning_info["best_params"]
            if model_type == "XGBoost":
                xgb_params = dict(xgb_params or {})
                xgb_params.update(bp)
                if tuning_info.get("best_rounds"):
                    num_boost_round = int(tuning_info["best_rounds"])
            elif "reg_lambda" in bp:
                reg_lambda_mod = float(bp["reg_lambda"])
            elif "alpha" in bp:
                ridge_alpha_lin = float(bp["alpha"])
            elif "C" in bp:
                logreg_C = float(bp["C"])
            elif "penalizer" in bp:
                cox_penalizer = float(bp["penalizer"])

    # The resolved penalty actually used, after any tuning override, for reporting.
    PENALTY_USED = _current_penalty()

    # A penalised linear model on unstandardised features shrinks each coefficient in
    # proportion to its variable's scale, so a wide-ranging covariate can crowd out a
    # genuinely stronger one purely because its units are larger. Warn when that
    # combination arises rather than leaving the user to discover it from the ranking.
    if (model_type != "XGBoost" and PENALTY_USED and float(PENALTY_USED) > 0
            and not standardize and prep.get("sd_ratio", 1.0) > 10):
        prep["scale_warning"] = (
            f"{_pname} = {float(PENALTY_USED):.4g} on unstandardised features whose "
            f"standard deviations span a factor of {prep['sd_ratio']:.0f}. The penalty is "
            f"applied on the raw scale, so wide-ranging covariates are penalised less and "
            f"may be ranked too highly. Enable standardisation for a penalised model."
        )

    # --- Linear Regression
    if model_type == "Linear Regression":
        if outcome_type != "continuous":
            raise ValueError("Linear Regression option is only supported for continuous outcome.")

        if loss_name == "Original (clinical)":
            model = Ridge(alpha=ridge_alpha_lin) if ridge_alpha_lin else LinearRegression()
            model.fit(X_train, y_train)

            pred_tr = model.predict(X_train)
            metrics_out = {
                "model": "Linear Regression",
                "outcome_type": "continuous",
                "loss": "Original (clinical)",
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
            }
            if HAS_TEST:
                pred_te = model.predict(X_test)
                metrics_out["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred_te)))
        else:
            model = ModifiedLinearRegressor(loss_name=loss_name, reg_lambda=reg_lambda_mod, maxiter=500)
            model.fit(X_train.values, y_train, trt_train_pm, pi_train)

            pred_tr = model.predict(X_train.values)
            pred_te = model.predict(X_test.values) if HAS_TEST else np.array([])

            metrics_out = {
                "model": "Linear Regression",
                "outcome_type": "continuous",
                "loss": loss_name,
                "train_loss": _continuous_modified_loss(
                    y_train, pred_tr, loss_name, trt_train_pm, pi_train
                ),
            }
            if HAS_TEST:
                metrics_out["test_loss"] = _continuous_modified_loss(
                    y_test, pred_te, loss_name, trt_test_pm, pi_test
                )

            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_tr)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_te)

        extra = {
            "sigpos_exists": sigpos_exists,
            "sigpos_train": sigpos_train,
            "sigpos_test": sigpos_test,
            "pred_raw_train": pred_tr,
            "pred_raw_test": pred_te if HAS_TEST else np.array([]),
            "pi_train": pi_train,
            "pi_test": pi_test if HAS_TEST else np.array([]),
            "trt_train_pm": trt_train_pm,
            "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
            "has_test": HAS_TEST,
            "benefit_sign": BSIGN,
            "benefit_note": benefit_score_note(outcome_type, loss_name),
            "tuning": tuning_info,
            "penalty_used": PENALTY_USED,
            "prep": prep,
        }
        return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- Random Forest
    # Available under the Original objective only. A-/W-learning are optimised through
    # gradients of a modified loss, and a forest is grown by recursive partitioning with no
    # gradient interface to supply them, so there is nothing to hand the objective. Refusing
    # is the honest behaviour: silently fitting a forest to the raw outcome and labelling it
    # an ITR model would reproduce exactly the estimand/objective mismatch this app exists to
    # make visible. SHAP reaches it through TreeExplainer, the same exact path as XGBoost.
    if model_type == "Random Forest":
        if loss_name != "Original (clinical)":
            raise ValueError(
                f"Random Forest is available under the Original (clinical) objective only. "
                f"{loss_name} is optimised through the gradient of a modified loss, which a "
                f"forest does not expose. Use XGBoost for {loss_name}, which accepts a custom "
                f"objective, or one of the parametric models.")
        if outcome_type == "time-to-event":
            raise ValueError(
                "Random Forest is not available for time-to-event outcomes here: a survival "
                "forest needs a different implementation (scikit-survival), which is not a "
                "dependency of this app. Use Cox Regression or XGBoost.")

        rf_kw = dict(n_estimators=int(rf_n_estimators),
                     min_samples_leaf=max(1, int(round(float(rf_min_samples_leaf)))),
                     random_state=int(seed), n_jobs=-1)
        if outcome_type == "continuous":
            model = RandomForestRegressor(**rf_kw)
            model.fit(X_train.values, y_train.astype(float))
            pred_tr = model.predict(X_train.values)
            metrics_out = {
                "model": "Random Forest",
                "outcome_type": "continuous",
                "loss": "Original (clinical)",
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
            }
            pred_te = np.array([])
            if HAS_TEST:
                pred_te = model.predict(X_test.values)
                metrics_out["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred_te)))
        else:
            model = RandomForestClassifier(**rf_kw)
            model.fit(X_train.values, y_train.astype(int))
            prob_tr = np.clip(model.predict_proba(X_train.values)[:, 1], 1e-12, 1 - 1e-12)
            metrics_out = {
                "model": "Random Forest",
                "outcome_type": "binary",
                "loss": "Original (clinical)",
                "train_auc": float(_safe_auc(y_train.astype(int), prob_tr)),
                "train_loss": float(log_loss(y_train.astype(int), prob_tr)),
            }
            pred_tr, pred_te = prob_tr, np.array([])
            if HAS_TEST:
                prob_te = np.clip(model.predict_proba(X_test.values)[:, 1], 1e-12, 1 - 1e-12)
                metrics_out["test_auc"] = float(_safe_auc(y_test.astype(int), prob_te))
                metrics_out["test_loss"] = float(log_loss(y_test.astype(int), prob_te))
                pred_te = prob_te

        if sigpos_exists:
            metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_tr)
            if HAS_TEST:
                metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_te)

        extra = {
            "sigpos_exists": sigpos_exists,
            "sigpos_train": sigpos_train,
            "sigpos_test": sigpos_test,
            "pred_raw_train": pred_tr,
            "pred_raw_test": pred_te if HAS_TEST else np.array([]),
            "pi_train": pi_train,
            "pi_test": pi_test if HAS_TEST else np.array([]),
            "trt_train_pm": trt_train_pm,
            "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
            "has_test": HAS_TEST,
            "benefit_sign": BSIGN,
            "benefit_note": benefit_score_note(outcome_type, loss_name),
            "tuning": tuning_info,
            "penalty_used": PENALTY_USED,
            "prep": prep,
        }
        return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- Logistic Regression
    if model_type == "Logistic Regression":
        if outcome_type != "binary":
            raise ValueError("Logistic Regression option is only supported for binary outcome.")

        if loss_name == "Original (clinical)":
            model = LogisticRegression(C=logreg_C, max_iter=2000)
            model.fit(X_train.values, y_train.astype(int))

            prob_tr = np.clip(model.predict_proba(X_train.values)[:, 1], 1e-12, 1 - 1e-12)
            metrics_out = {
                "model": "Logistic Regression",
                "outcome_type": "binary",
                "loss": "Original (clinical)",
                "train_auc": float(_safe_auc(y_train.astype(int), prob_tr)),
                "train_loss": float(log_loss(y_train.astype(int), prob_tr)),
            }

            pred_te = np.array([])
            if HAS_TEST:
                prob_te = np.clip(model.predict_proba(X_test.values)[:, 1], 1e-12, 1 - 1e-12)
                metrics_out["test_auc"] = float(_safe_auc(y_test.astype(int), prob_te))
                metrics_out["test_loss"] = float(log_loss(y_test.astype(int), prob_te))
                pred_te = prob_te

            pred_tr = prob_tr
        else:
            model = ModifiedLogisticRegressor(loss_name=loss_name, reg_lambda=reg_lambda_mod, maxiter=500)
            model.fit(X_train.values, y_train.astype(float), trt_train_pm, pi_train)

            pred_tr = model.predict(X_train.values)
            pred_te = model.predict(X_test.values) if HAS_TEST else np.array([])

            metrics_out = {
                "model": "Logistic Regression",
                "outcome_type": "binary",
                "loss": loss_name,
                "train_loss": _binary_modified_loss(
                    y_train, pred_tr, loss_name, trt_train_pm, pi_train
                ),
            }

            if HAS_TEST:
                metrics_out["test_loss"] = _binary_modified_loss(
                    y_test, pred_te, loss_name, trt_test_pm, pi_test
                )

            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_tr)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_te)

        extra = {
            "sigpos_exists": sigpos_exists,
            "sigpos_train": sigpos_train,
            "sigpos_test": sigpos_test,
            "pred_raw_train": pred_tr,
            "pred_raw_test": pred_te if HAS_TEST else np.array([]),
            "pi_train": pi_train,
            "pi_test": pi_test if HAS_TEST else np.array([]),
            "trt_train_pm": trt_train_pm,
            "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
            "has_test": HAS_TEST,
            "benefit_sign": BSIGN,
            "benefit_note": benefit_score_note(outcome_type, loss_name),
            "tuning": tuning_info,
            "penalty_used": PENALTY_USED,
            "prep": prep,
        }
        return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- Cox Regression
    if model_type == "Cox Regression":
        if outcome_type != "time-to-event":
            raise ValueError("Cox Regression option is only supported for time-to-event outcome.")

        if loss_name == "Original (clinical)":
            if not _HAS_LIFELINES:
                raise ValueError("lifelines is not installed. Please `pip install lifelines` to use Cox Regression.")

            df_tr = X_train.copy()
            df_tr = df_tr.assign(**{
                outcome_col: time_train.astype(float),
                event_col: event_train.astype(int)
            })

            cox = CoxPHFitter(penalizer=cox_penalizer)
            cox.fit(df_tr, duration_col=outcome_col, event_col=event_col)

            score_tr = cox.predict_partial_hazard(df_tr[X_train.columns]).values.reshape(-1)

            metrics_out = {
                "model": "Cox Regression",
                "outcome_type": "time-to-event",
                "loss": "Original (clinical)",
                "train_c_index": float(concordance_index(time_train, event_train, score_tr)),
            }

            score_te = np.array([])
            if HAS_TEST:
                df_te = X_test.copy()
                df_te = df_te.assign(**{
                    outcome_col: time_test.astype(float),
                    event_col: event_test.astype(int)
                })
                score_te = cox.predict_partial_hazard(df_te[X_train.columns]).values.reshape(-1)
                metrics_out["test_c_index"] = float(concordance_index(time_test, event_test, score_te))

            extra = {
                "time_train": time_train,
                "event_train": event_train,
                "time_test": time_test if HAS_TEST else np.array([]),
                "event_test": event_test if HAS_TEST else np.array([]),
                "pred_raw_train": score_tr,
                "pred_raw_test": score_te if HAS_TEST else np.array([]),
                "sigpos_exists": sigpos_exists,
                "sigpos_train": sigpos_train,
                "sigpos_test": sigpos_test,
                "pi_train": pi_train,
                "pi_test": pi_test if HAS_TEST else np.array([]),
                "trt_train_pm": trt_train_pm,
                "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
                "has_test": HAS_TEST,
                "benefit_sign": BSIGN,
                "benefit_note": benefit_score_note(outcome_type, loss_name),
                "tuning": tuning_info,
                "penalty_used": PENALTY_USED,
                "prep": prep,
            }
            return cox, X_train, X_test, y_train, y_test, metrics_out, None, extra

        else:
            model = ModifiedCoxRegressor(loss_name=loss_name, reg_lambda=reg_lambda_mod, maxiter=500)
            model.fit(X_train.values, time_train, event_train, trt_train_pm, pi_train)

            pred_tr = model.predict(X_train.values)
            pred_te = model.predict(X_test.values) if HAS_TEST else np.array([])

            metrics_out = {
                "model": "Cox Regression",
                "outcome_type": "time-to-event",
                "loss": loss_name,
                "train_loss": _cox_modified_loss(
                    loss_name, trt_train_pm, pi_train, time_train, event_train, pred_tr
                ),
            }

            if HAS_TEST:
                metrics_out["test_loss"] = _cox_modified_loss(
                    loss_name, trt_test_pm, pi_test, time_test, event_test, pred_te
                )

            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_tr)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_te)

            extra = {
                "time_train": time_train,
                "event_train": event_train,
                "time_test": time_test if HAS_TEST else np.array([]),
                "event_test": event_test if HAS_TEST else np.array([]),
                "pred_raw_train": pred_tr,
                "pred_raw_test": pred_te if HAS_TEST else np.array([]),
                "sigpos_exists": sigpos_exists,
                "sigpos_train": sigpos_train,
                "sigpos_test": sigpos_test if HAS_TEST else np.array([]),
                "pi_train": pi_train,
                "pi_test": pi_test if HAS_TEST else np.array([]),
                "trt_train_pm": trt_train_pm,
                "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
                "has_test": HAS_TEST,
                "benefit_sign": BSIGN,
                "benefit_note": benefit_score_note(outcome_type, loss_name),
                "tuning": tuning_info,
                "penalty_used": PENALTY_USED,
                "prep": prep,
            }
            return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- XGBoost
    dtrain = xgb.DMatrix(X_train.values, label=np.zeros(len(X_train)))
    dtest = xgb.DMatrix(X_test.values, label=np.zeros(len(X_test)))

    params = dict(
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        tree_method="hist",
        verbosity=0,
    )

    if xgb_params:
        for k, v in xgb_params.items():
            if v is not None:
                params[k] = v

    obj_fn = None

    if outcome_type == "continuous":
        dtrain.set_label(y_train)
        if HAS_TEST:
            dtest.set_label(y_test)
        if loss_name == "Original (clinical)":
            params["objective"] = "reg:squarederror"
        else:
            obj_fn, _ = make_aw_continuous_objective(loss_name, trt_train_pm, pi_train)

    elif outcome_type == "binary":
        dtrain.set_label(y_train.astype(float))
        if HAS_TEST:
            dtest.set_label(y_test.astype(float))
        if loss_name == "Original (clinical)":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            obj_fn, _ = make_aw_binary_objective(loss_name, trt_train_pm, pi_train)

    elif outcome_type == "time-to-event":
        if loss_name == "Original (clinical)":
            # XGBoost's built-in survival:cox encodes censoring in the SIGN of the label:
            # positive = observed event, negative = right censored. Passing raw times
            # would treat every observation as an observed event.
            dtrain.set_label(xgb_cox_label(time_train, event_train))
            if HAS_TEST:
                dtest.set_label(xgb_cox_label(time_test, event_test))
            params["objective"] = "survival:cox"
            params["eval_metric"] = "cox-nloglik"
        else:
            # The modified objectives read time/event from their closure; the DMatrix
            # label is unused, so the raw times are kept for reference.
            dtrain.set_label(time_train)
            if HAS_TEST:
                dtest.set_label(time_test)
            obj_fn, _ = make_aw_cox_objective(loss_name, trt_train_pm, pi_train, time_train, event_train)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        obj=obj_fn
    )

    pred_raw_train = model.predict(xgb.DMatrix(X_train.values))
    pred_raw_test = model.predict(dtest) if X_test.shape[0] > 0 else np.array([])

    metrics_out = {
        "model": "XGBoost",
        "outcome_type": outcome_type,
        "loss": loss_name,
        "xgb_num_boost_round": int(num_boost_round),
        "xgb_params": {k: params[k] for k in sorted(params.keys())},
    }

    # ---- reporting rules (FIXED: only compute test metrics if HAS_TEST)
    if outcome_type == "continuous":
        if loss_name == "Original (clinical)":
            metrics_out["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, pred_raw_train)))
            if HAS_TEST:
                metrics_out["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred_raw_test)))
        else:
            metrics_out["train_loss"] = _continuous_modified_loss(
                y_train, pred_raw_train, loss_name, trt_train_pm, pi_train
            )
            if HAS_TEST:
                metrics_out["test_loss"] = _continuous_modified_loss(
                    y_test, pred_raw_test, loss_name, trt_test_pm, pi_test
                )
            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_raw_test)

    elif outcome_type == "binary":
        if loss_name == "Original (clinical)":
            prob_tr = np.clip(pred_raw_train, 1e-12, 1 - 1e-12)
            mtr = _binary_original_metrics(y_train.astype(int), prob_tr)
            metrics_out["train_auc"] = mtr["auc"]
            metrics_out["train_loss"] = mtr["loss"]

            if HAS_TEST:
                prob_te = np.clip(pred_raw_test, 1e-12, 1 - 1e-12)
                mte = _binary_original_metrics(y_test.astype(int), prob_te)
                metrics_out["test_auc"] = mte["auc"]
                metrics_out["test_loss"] = mte["loss"]
        else:
            metrics_out["train_loss"] = _binary_modified_loss(
                y_train, pred_raw_train, loss_name, trt_train_pm, pi_train
            )
            if HAS_TEST:
                metrics_out["test_loss"] = _binary_modified_loss(
                    y_test, pred_raw_test, loss_name, trt_test_pm, pi_test
                )
            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_raw_test)

    elif outcome_type == "time-to-event":
        if loss_name == "Original (clinical)":
            metrics_out["train_c_index"] = float(concordance_index(time_train, event_train, pred_raw_train))
            if HAS_TEST:
                metrics_out["test_c_index"] = float(concordance_index(time_test, event_test, pred_raw_test))
        else:
            metrics_out["train_loss"] = _cox_modified_loss(
                loss_name, trt_train_pm, pi_train, time_train, event_train, pred_raw_train
            )
            if HAS_TEST:
                metrics_out["test_loss"] = _cox_modified_loss(
                    loss_name, trt_test_pm, pi_test, time_test, event_test, pred_raw_test
                )
            if sigpos_exists:
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, BSIGN * pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, BSIGN * pred_raw_test)

    extra = {
        "time_train": time_train if outcome_type == "time-to-event" else None,
        "event_train": event_train if outcome_type == "time-to-event" else None,
        "time_test": time_test if (outcome_type == "time-to-event" and HAS_TEST) else np.array([]),
        "event_test": event_test if (outcome_type == "time-to-event" and HAS_TEST) else np.array([]),
        "pi_train": pi_train,
        "pi_test": pi_test if HAS_TEST else np.array([]),
        "trt_train_pm": trt_train_pm,
        "trt_test_pm": trt_test_pm if HAS_TEST else np.array([]),
        "pred_raw_train": pred_raw_train,
        "pred_raw_test": pred_raw_test if HAS_TEST else np.array([]),
        "sigpos_exists": sigpos_exists,
        "sigpos_train": sigpos_train,
        "sigpos_test": sigpos_test if HAS_TEST else np.array([]),
        "sigpos_col": sigpos_col if sigpos_exists else None,
        "has_test": HAS_TEST,
        "benefit_sign": BSIGN,
        "benefit_note": benefit_score_note(outcome_type, loss_name),
        "tuning": tuning_info,
        "penalty_used": PENALTY_USED,
        "prep": prep,
    }

    return model, X_train, X_test, y_train, y_test, metrics_out, dtest, extra

# -----------------------------
# SHAP helpers for linear-style models
# -----------------------------
def _get_model_coef_intercept(model, model_type: str, loss_name: str, feature_names):
    feature_names = list(feature_names)

    # sklearn linear / logistic
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1).astype(float)

        intercept = 0.0
        if hasattr(model, "intercept_"):
            intercept_arr = np.asarray(model.intercept_).reshape(-1)
            if len(intercept_arr) > 0:
                intercept = float(intercept_arr[0])

        if len(coef) != len(feature_names):
            raise ValueError(
                f"Coefficient length ({len(coef)}) does not match number of features ({len(feature_names)})."
            )
        return coef, intercept

    # lifelines CoxPHFitter
    if model_type == "Cox Regression" and loss_name == "Original (clinical)":
        if not hasattr(model, "params_"):
            raise ValueError("CoxPHFitter does not have params_.")
        coef = model.params_.reindex(feature_names).values.astype(float)
        intercept = 0.0  # Cox linear predictor usually has no intercept
        return coef, intercept

    raise ValueError(f"Cannot extract coefficients for model_type={model_type}, loss={loss_name}.")


def build_linear_shap_objects(
    model,
    X_background: pd.DataFrame,
    X_plot: pd.DataFrame,
    model_type: str,
    loss_name: str,
):
    if X_plot is None or len(X_plot) == 0:
        raise ValueError("X_plot is empty; cannot generate SHAP values.")
    if X_background is None or len(X_background) == 0:
        raise ValueError("X_background is empty; cannot generate SHAP values.")

    X_background = X_background.copy()
    X_plot = X_plot.copy()

    feature_names = list(X_plot.columns)
    coef, intercept = _get_model_coef_intercept(model, model_type, loss_name, feature_names)

    bg_mean = X_background[feature_names].mean(axis=0).values.astype(float)
    Xv = X_plot[feature_names].values.astype(float)

    # SHAP for linear predictor:
    # f(x) = intercept + sum_j coef_j * x_j
    # phi_j = coef_j * (x_j - E[x_j])
    shap_values = (Xv - bg_mean[None, :]) * coef[None, :]
    expected_value = float(intercept + np.dot(bg_mean, coef))

    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(expected_value, X_plot.shape[0]),
        data=Xv,
        feature_names=feature_names,
    )

    return expected_value, shap_values, shap_exp


# =============================================================================
# Conventional coefficient summaries, and how they relate to SHAP
# =============================================================================
# Linear, logistic and Cox models already have directly interpretable parameters, so a
# reviewer reasonably asked what this framework does with effect estimates, confidence
# intervals and odds ratios. Three things, kept distinct because they are true to different
# degrees:
#
# 1. Where a conventional summary is valid, it is computed and reported: beta, its standard
#    error, a Wald interval, and the exponentiated form (odds ratio for logistic, hazard
#    ratio for Cox).
#
# 2. Where it is NOT valid, it is refused with the reason rather than printed with a
#    disclaimer. Two cases arise here. A penalised fit -- sklearn's default C=1 is already
#    ridge-penalised -- gives a deliberately biased beta whose model-based variance ignores
#    the penalty. And the A-/W-learning estimators optimise an IPW-weighted modified
#    objective with an estimated propensity, so no closed-form variance is available at all.
#    Both are handled by a nonparametric bootstrap over subjects instead.
#
# 3. SHAP adds no inferential content for these models. It is not an approximation of the
#    coefficient, it is an algebraic re-expression: this file computes
#    phi_ij = beta_j (x_ij - xbar_j) in closed form, so
#    mean|SHAP_j| = |beta_j| E|x_j - xbar_j| identically. The consequence worth teaching is
#    that the two rank features differently whenever covariates differ in spread -- a binary
#    indicator with a large coefficient can sit far down a mean|SHAP| ordering. Neither
#    ordering is wrong; they answer different questions, and reconcile_shap_with_coefficients
#    below verifies the identity numerically at runtime rather than asserting it.

def _wald(beta, se, level=0.95):
    z = float(sp_stats.norm.ppf(0.5 + float(level) / 2.0))
    lo = np.asarray(beta) - z * np.asarray(se)
    hi = np.asarray(beta) + z * np.asarray(se)
    p = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(np.asarray(beta) / np.where(
        np.asarray(se) > 0, np.asarray(se), np.nan))))
    return lo, hi, p


def _ols_inference(X, y, beta, intercept):
    """Textbook OLS standard errors: sigma^2 (X'X)^{-1}, with the intercept partialled in."""
    Xi = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
    resid = np.asarray(y, dtype=float) - (intercept + np.asarray(X, dtype=float) @ beta)
    dof = len(X) - Xi.shape[1]
    if dof <= 0:
        return np.full(len(beta), np.nan)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Xi.T @ Xi)
    except np.linalg.LinAlgError:
        return np.full(len(beta), np.nan)
    return np.sqrt(np.clip(np.diag(cov)[1:], 0.0, None))


def _logit_inference(X, beta, intercept):
    """Observed-information standard errors: (X' W X)^{-1} with W = diag(p(1-p))."""
    Xd = np.asarray(X, dtype=float)
    p = sigmoid(intercept + Xd @ beta)
    w = np.clip(p * (1.0 - p), 1e-12, None)
    Xi = np.hstack([np.ones((len(Xd), 1)), Xd])
    try:
        cov = np.linalg.inv(Xi.T @ (Xi * w[:, None]))
    except np.linalg.LinAlgError:
        return np.full(len(beta), np.nan)
    return np.sqrt(np.clip(np.diag(cov)[1:], 0.0, None))


def coefficient_summary(model, X_train, *, model_type, loss_name, outcome_type,
                        feature_names=None, y_train=None, penalty_used=None,
                        level=0.95, cox_model=None):
    """Conventional parameter summary, or an explicit statement of why there isn't one.

    Returns a dict with `rows` (one per feature) and `inference`, a short description of
    where the standard errors came from, plus `valid` and, when False, `reason`.
    """
    feature_names = list(feature_names if feature_names is not None else X_train.columns)
    coef, intercept = _get_model_coef_intercept(model, model_type, loss_name, feature_names)
    Xd = np.asarray(X_train[feature_names].values, dtype=float)

    exp_label = ("Odds ratio" if (model_type == "Logistic Regression" or outcome_type == "binary")
                 else ("Hazard ratio" if outcome_type == "time-to-event" else ""))
    # `inference` is for the Coefficients tab and may use matrix notation; `inference_prose`
    # is what travels in the LLM payload. A token like "WX" from (X'WX)^{-1} is indistinguishable
    # from an invented gene symbol to the faithfulness grader, so symbols stay out of anything
    # a narrative might echo.
    out = {"model_type": model_type, "loss_name": loss_name, "outcome_type": outcome_type,
           "exp_label": exp_label, "level": float(level), "intercept": float(intercept),
           "n": int(len(Xd)), "valid": False, "reason": "", "inference": "",
           "inference_prose": "", "penalty_used": penalty_used, "rows": []}

    modified = loss_name in ("A-learning", "W-learning")
    pen = float(penalty_used) if penalty_used is not None else None

    se = None
    if modified:
        out["reason"] = (
            f"{loss_name} optimises an inverse-probability-weighted modified objective with "
            "an estimated propensity score, so there is no conventional likelihood and no "
            "closed-form variance. The coefficients below are point estimates only; use the "
            "bootstrap for intervals.")
    elif model_type == "Cox Regression" and cox_model is not None and hasattr(cox_model, "summary"):
        # lifelines already does this properly, including the penalised case's caveat.
        try:
            s = cox_model.summary
            se = s["se(coef)"].reindex(feature_names).values.astype(float)
            out["valid"] = not (pen and pen > 0)
            out["inference"] = ("Cox partial-likelihood standard errors from lifelines"
                               + (" (penalised fit: intervals are not valid)" if (pen and pen > 0)
                                  else ""))
            out["inference_prose"] = ("Cox partial-likelihood standard errors, as reported by "
                                      "lifelines")
            if pen and pen > 0:
                out["reason"] = (f"The Cox fit is penalised (penalizer={pen:g}), so the "
                                 "partial-likelihood interval understates the bias that the "
                                 "penalty introduces. Use the bootstrap instead.")
        except Exception:
            se = None
    elif pen is not None and pen > 0 and not (
            model_type == "Logistic Regression" and pen >= LOGREG_C_NO_PENALTY / 1e3):
        out["reason"] = (
            f"This fit is penalised ({(_penalty_label(model_type, loss_name))}={pen:g}). A "
            "penalised coefficient is deliberately shrunk toward zero, and the model-based "
            "variance treats the penalty as if it were not there, so a Wald interval around "
            "it is not a confidence interval for the parameter. Use the bootstrap, or refit "
            "without a penalty.")
    elif model_type == "Linear Regression" and outcome_type == "continuous":
        if y_train is None:
            out["reason"] = "Standard errors need the training outcome, which was not supplied."
        else:
            se = _ols_inference(Xd, y_train, coef, intercept)
            out["valid"] = True
            out["inference"] = "OLS standard errors, sigma^2 (X'X)^{-1}, t on n-p-1 df"
            out["inference_prose"] = ("ordinary least squares standard errors from the "
                                      "residual variance")
    elif model_type == "Logistic Regression" and outcome_type == "binary":
        se = _logit_inference(Xd, coef, intercept)
        out["valid"] = True
        out["inference"] = ("Maximum-likelihood standard errors from the observed "
                            "information, (X'WX)^{-1} with W = diag(p(1-p))")
        out["inference_prose"] = ("maximum-likelihood standard errors from the observed "
                                  "information matrix")
    else:
        out["reason"] = (f"No conventional parameter summary is defined for {model_type} "
                         f"under {loss_name}.")

    lo = hi = pv = None
    if se is not None and np.isfinite(np.asarray(se, dtype=float)).any():
        lo, hi, pv = _wald(coef, se, level)

    for j, f in enumerate(feature_names):
        row = {"feature": str(f), "coef": float(coef[j])}
        if se is not None:
            row.update({"se": float(se[j]), "ci_lo": float(lo[j]), "ci_hi": float(hi[j]),
                        "p": float(pv[j])})
            if exp_label:
                row.update({"exp_coef": float(np.exp(coef[j])),
                            "exp_lo": float(np.exp(lo[j])), "exp_hi": float(np.exp(hi[j]))})
        elif exp_label:
            row["exp_coef"] = float(np.exp(coef[j]))
        out["rows"].append(row)
    return out


def bootstrap_coefficients(df, feature_cols, *, outcome_col, treat_col, outcome_type,
                           model_type, loss_name, event_col=None, sigpos_col=None,
                           n_boot=200, seed=42, level=0.95, progress=None,
                           na_policy="Drop rows with any missing value",
                           standardize=False, **penalties):
    """Percentile bootstrap intervals for the coefficients, resampling subjects.

    This is the answer for the cases where no conventional interval exists: a penalised fit,
    and the A-/W-learning modified objectives. Each replicate refits the whole pipeline on
    the resampled rows, which means the propensity score is re-estimated inside every
    replicate rather than being held fixed at its full-sample value -- the propensity is
    estimated, so its uncertainty belongs inside the interval.

    Refits on the full resample (test_size=0): the interval is a statement about the
    parameter, so there is nothing to hold out.
    """
    if model_type in ("XGBoost", "Random Forest"):
        raise ValueError(f"Bootstrap intervals are for the parametric models; {model_type} "
                         f"has no coefficients to report.")
    feature_cols = list(feature_cols)
    rng = np.random.default_rng(int(seed))
    n = len(df)
    draws, failures = [], []

    for b in range(int(n_boot)):
        if progress is not None:
            progress(b, int(n_boot))
        idx = rng.integers(0, n, size=n)
        dfb = df.iloc[idx].reset_index(drop=True)
        try:
            mb = fit_model(
                dfb, feature_cols, outcome_col=outcome_col, treat_col=treat_col,
                outcome_type=outcome_type, event_col=event_col, model_type=model_type,
                loss_name=loss_name, test_size=0.0, seed=int(seed) + b,
                sigpos_col=sigpos_col, tune_mode="None", na_policy=na_policy,
                standardize=standardize, **penalties)[0]
            cb, _ = _get_model_coef_intercept(mb, model_type, loss_name, feature_cols)
            if not np.all(np.isfinite(cb)):
                raise ValueError("non-finite coefficients")
            draws.append(np.asarray(cb, dtype=float))
        except Exception as e:
            failures.append(str(e)[:120])

    if len(draws) < 20:
        raise ValueError(
            f"Only {len(draws)} of {n_boot} bootstrap replicates fitted; too few for an "
            f"interval. First failure: {failures[0] if failures else 'unknown'}")

    D = np.vstack(draws)
    a = (1.0 - float(level)) / 2.0
    lo = np.percentile(D, 100 * a, axis=0)
    hi = np.percentile(D, 100 * (1.0 - a), axis=0)
    return {
        "rows": [{"feature": str(f), "boot_se": float(np.std(D[:, j], ddof=1)),
                  "boot_lo": float(lo[j]), "boot_hi": float(hi[j]),
                  "boot_median": float(np.median(D[:, j]))}
                 for j, f in enumerate(feature_cols)],
        "n_boot": int(n_boot), "n_ok": int(len(draws)), "n_failed": len(failures),
        "level": float(level), "seed": int(seed),
        "failure_example": (failures[0] if failures else ""),
        "method": ("percentile bootstrap over subjects; propensity re-estimated per "
                   "replicate" if loss_name in ("A-learning", "W-learning")
                   else "percentile bootstrap over subjects"),
    }


def _penalty_label(model_type, loss_name):
    names = tunable_parameter_names(model_type, loss_name)
    return names[0] if names and model_type != "XGBoost" else "penalty"


def reconcile_shap_with_coefficients(shap_values, X_plot, X_background, coef_rows,
                                    feature_names=None, top_k=3):
    """Check phi_ij = beta_j (x_ij - xbar_j) numerically, and compare the two rankings.

    The identity is what makes SHAP redundant as inference for these models, and the ranking
    comparison is what makes it non-redundant as description. Both are computed from the
    fitted objects rather than stated, so a future change that breaks the identity shows up
    here instead of being quietly wrong.
    """
    feature_names = list(feature_names if feature_names is not None else X_plot.columns)
    beta = np.asarray([r["coef"] for r in coef_rows], dtype=float)
    Xv = np.asarray(X_plot[feature_names].values, dtype=float)
    bg = np.asarray(X_background[feature_names].mean(axis=0).values, dtype=float)
    sv = np.asarray(shap_values, dtype=float)

    implied = (Xv - bg[None, :]) * beta[None, :]
    max_abs_dev = float(np.max(np.abs(sv - implied))) if sv.shape == implied.shape else np.nan

    mean_abs_shap = np.abs(sv).mean(axis=0)
    spread = np.abs(Xv - bg[None, :]).mean(axis=0)
    identity_dev = float(np.max(np.abs(mean_abs_shap - np.abs(beta) * spread)))

    by_shap = [feature_names[j] for j in np.argsort(-mean_abs_shap)]
    by_beta = [feature_names[j] for j in np.argsort(-np.abs(beta))]

    rows = []
    for j, f in enumerate(feature_names):
        rows.append({"feature": str(f), "coef": float(beta[j]),
                     "abs_coef": float(abs(beta[j])),
                     "mean_abs_dev": float(spread[j]),
                     "mean_abs_shap": float(mean_abs_shap[j]),
                     "rank_shap": int(by_shap.index(str(f)) + 1),
                     "rank_beta": int(by_beta.index(str(f)) + 1)})
    rows.sort(key=lambda r: -r["mean_abs_shap"])

    moved = [r for r in rows if r["rank_shap"] != r["rank_beta"]]
    return {
        "identity_holds": bool(np.isfinite(max_abs_dev) and max_abs_dev < 1e-8),
        "max_abs_deviation": max_abs_dev,
        "aggregate_identity_deviation": identity_dev,
        "top_k": int(top_k),
        "top_by_shap": by_shap[:int(top_k)],
        "top_by_abs_coef": by_beta[:int(top_k)],
        "rankings_agree": by_shap[:int(top_k)] == by_beta[:int(top_k)],
        "rows": rows,
        "n_moved": len(moved),
        "largest_move": (max(moved, key=lambda r: abs(r["rank_shap"] - r["rank_beta"]))
                         if moved else None),
    }


# -----------------------------
# SHAP plots
# -----------------------------
def save_shap_plots(
    model,
    X_plot: pd.DataFrame,
    out_dir="outputs",
    prefix="Fig_SHAP",
    dep_main="__AUTO_TOP__",
    dep_interaction="auto",
    patient_idx=0,
    model_type="XGBoost",
    loss_name="Original (clinical)",
    X_background: Optional[pd.DataFrame] = None,
    outcome_type: str = "continuous",
):
    os.makedirs(out_dir, exist_ok=True)

    if X_plot is None or len(X_plot) == 0:
        raise ValueError("X_plot is empty; cannot generate SHAP plots.")

    # ---- choose SHAP backend
    if model_type in ("XGBoost", "Random Forest"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_plot)
        shap_exp = explainer(X_plot)

        # A multiclass tree model returns one set of attributions per class: a list of
        # arrays, or a single (n, p, n_classes) array. XGBoost's binary:logistic returns a
        # plain (n, p) on the log-odds scale, but a RandomForestClassifier returns both
        # classes on the probability scale. Class 1 is the one being modelled, and taking
        # class 0 by accident would flip every attribution's sign.
        cls = None
        if isinstance(shap_values, list):
            cls = min(1, len(shap_values) - 1)
            shap_values = shap_values[cls]
        else:
            sv_arr = np.asarray(shap_values)
            if sv_arr.ndim == 3:
                cls = min(1, sv_arr.shape[2] - 1)
                shap_values = sv_arr[:, :, cls]
        if cls is not None:
            base = float(np.ravel(explainer.expected_value)[cls])
            shap_exp = shap.Explanation(
                values=np.asarray(shap_values),
                base_values=np.repeat(base, len(X_plot)),
                data=np.asarray(X_plot.values, dtype=float),
                feature_names=list(X_plot.columns),
            )
            # Collapse the explainer's own base value to the chosen class as well. The
            # waterfall and decision plots and build_shap_payload all read this attribute
            # directly; left as a two-element array it breaks the plots outright and, worse,
            # would quietly resolve to class 0 in the payload.
            explainer.expected_value = base

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.ravel(expected_value)[cls if cls is not None else 0])
        else:
            expected_value = float(expected_value)
        # Collapse the attribute to the same scalar for every tree model, not just the
        # multiclass case. Downstream code reads it directly, and a length-one array is a
        # standing invitation to the indexing bug that the classifier path already hit.
        explainer.expected_value = expected_value

    else:
        if X_background is None or len(X_background) == 0:
            X_background = X_plot

        expected_value, shap_values, shap_exp = build_linear_shap_objects(
            model=model,
            X_background=X_background,
            X_plot=X_plot,
            model_type=model_type,
            loss_name=loss_name,
        )

        class SimpleExplainer:
            pass

        explainer = SimpleExplainer()
        explainer.expected_value = expected_value

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    # Reorient onto the benefiting-score scale where required (time-to-event + ITR
    # objective), so that "increases the prediction" always means "increases expected
    # treatment benefit". SHAP is additive, so negating the base value and every
    # attribution is exact: -f(x) = -E[f] + sum_j (-phi_j).
    bsign = benefit_score_sign(outcome_type, loss_name)
    if bsign < 0:
        shap_values = -shap_values
        expected_value = -expected_value
        shap_exp = shap.Explanation(
            values=-np.asarray(shap_exp.values),
            base_values=-np.asarray(shap_exp.base_values),
            data=np.asarray(shap_exp.data),
            feature_names=list(X_plot.columns),
        )

        class _OrientedExplainer:
            pass

        explainer = _OrientedExplainer()
        explainer.expected_value = expected_value

    paths_disk = {}

    def _save_current_fig(fn):
        p = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(p, dpi=250, bbox_inches="tight")
        plt.close()
        return p

    scale_txt = " [benefiting score]" if bsign < 0 else ""

    # Beeswarm
    plt.figure(figsize=(8.2, 5.2))
    shap.summary_plot(shap_values, X_plot, show=False)
    plt.title(f"SHAP summary (beeswarm){scale_txt}")
    paths_disk["Beeswarm"] = _save_current_fig(f"{prefix}_A_beeswarm.png")

    # Bar
    plt.figure(figsize=(8.2, 4.8))
    shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False)
    plt.title(f"Mean(|SHAP|) feature importance{scale_txt}")
    paths_disk["Bar"] = _save_current_fig(f"{prefix}_B_bar.png")

    # Dependence
    mean_abs = np.abs(shap_values).mean(axis=0)
    auto_top_feat = X_plot.columns[int(np.argmax(mean_abs))]

    main_feat = auto_top_feat if dep_main == "__AUTO_TOP__" else dep_main
    if main_feat not in X_plot.columns:
        main_feat = auto_top_feat

    if dep_interaction == "none":
        interaction_index = None
    elif dep_interaction == "auto":
        interaction_index = "auto"
    else:
        interaction_index = dep_interaction if dep_interaction in X_plot.columns else "auto"

    safe_feat = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(main_feat))

    plt.figure(figsize=(7.6, 5.2))
    shap.dependence_plot(
        main_feat,
        shap_values,
        X_plot,
        interaction_index=interaction_index,
        show=False
    )
    plt.title(f"SHAP dependence: {main_feat} (color={dep_interaction})")
    paths_disk["Dependence"] = _save_current_fig(f"{prefix}_C_dependence_{safe_feat}.png")

    # Waterfall + Decision
    i = int(patient_idx)
    i = max(0, min(i, len(X_plot) - 1))

    plt.figure(figsize=(7.8, 5.2))
    shap.plots.waterfall(shap_exp[i], show=False, max_display=10)
    plt.title(f"SHAP waterfall: patient {i}{scale_txt}")
    paths_disk["Waterfall"] = _save_current_fig(f"{prefix}_D_waterfall_patient{i}.png")

    plt.figure(figsize=(8.0, 4.8))
    shap.decision_plot(explainer.expected_value, shap_values[i, :], X_plot.iloc[i, :], show=False)
    plt.title(f"SHAP decision plot: patient {i}{scale_txt}")
    paths_disk["Decision"] = _save_current_fig(f"{prefix}_E_decision_patient{i}.png")

    return paths_disk, explainer, shap_values, X_plot


# -----------------------------
# LIME plots
# -----------------------------
def _make_lime_predict_fn(model, model_type, outcome_type, loss_name, feature_names=None):
    """
    Prediction function for LIME, expressed on the same oriented scale used for SHAP:
    under an ITR objective, larger output always means greater expected treatment
    benefit. See benefit_score_sign().
    """
    fn, mode = _make_lime_predict_fn_raw(
        model, model_type, outcome_type, loss_name, feature_names=feature_names
    )
    bsign = benefit_score_sign(outcome_type, loss_name)
    if bsign >= 0:
        return fn, mode

    def oriented(X):
        return bsign * np.asarray(fn(X), dtype=float)

    return oriented, mode


def _make_lime_predict_fn_raw(model, model_type, outcome_type, loss_name, feature_names=None):
    if model_type == "Random Forest":
        if outcome_type == "binary":
            def pred_proba(X):
                p = np.clip(model.predict_proba(np.asarray(X))[:, 1], 1e-12, 1 - 1e-12)
                return np.vstack([1 - p, p]).T
            return pred_proba, "classification"

        def pred_reg(X):
            return model.predict(np.asarray(X))
        return pred_reg, "regression"

    if model_type == "Linear Regression":
        def pred_reg(X):
            return model.predict(X)
        return pred_reg, "regression"

    if model_type == "Logistic Regression":
        if loss_name == "Original (clinical)":
            def pred_proba(X):
                p = model.predict_proba(np.asarray(X))[:, 1]
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return np.vstack([1 - p, p]).T
            return pred_proba, "classification"
        else:
            def pred_reg(X):
                X = np.asarray(X)
                if hasattr(model, "decision_function"):
                    return model.decision_function(X)
                return model.predict(X)
            return pred_reg, "regression"

    if model_type == "Cox Regression":
        if loss_name == "Original (clinical)":
            if feature_names is None:
                raise ValueError("Cox Regression LIME needs feature_names.")
            def pred_risk(X):
                Xdf = pd.DataFrame(np.asarray(X), columns=feature_names)
                s = model.predict_partial_hazard(Xdf).values.reshape(-1)
                return s
            return pred_risk, "regression"
        else:
            def pred_reg(X):
                return model.predict(np.asarray(X))
            return pred_reg, "regression"

    dmat = lambda X: xgb.DMatrix(np.asarray(X))

    if outcome_type == "binary" and loss_name == "Original (clinical)":
        def pred_proba(X):
            p = model.predict(dmat(X))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return np.vstack([1 - p, p]).T
        return pred_proba, "classification"

    def pred_reg(X):
        return model.predict(dmat(X))
    return pred_reg, "regression"


def save_lime_plots(
    model,
    X_train: pd.DataFrame,
    X_plot: pd.DataFrame,
    model_type: str,
    outcome_type: str,
    loss_name: str,
    patient_idx: int,
    out_dir="outputs",
    prefix="Fig_LIME",
    num_features=10,
    global_n=60,
    global_num_samples=500,
    ridge_alpha=0.01,
    lime_seed=42,
):
    os.makedirs(out_dir, exist_ok=True)

    if X_plot is None or len(X_plot) == 0:
        raise ValueError("X_plot is empty; cannot generate LIME plots.")

    predict_fn, mode = _make_lime_predict_fn(
        model, model_type, outcome_type, loss_name, feature_names=list(X_train.columns)
    )

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode=mode,
        discretize_continuous=True,
        categorical_features=detect_categorical_indices(X_train),
        random_state=int(lime_seed),
        verbose=False
    )

    paths = {}

    def _save_fig(fn):
        p = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(p, dpi=250, bbox_inches="tight")
        plt.close()
        return p

    i = int(patient_idx)
    i = max(0, min(i, len(X_plot) - 1))
    x_i = X_plot.iloc[i].values

    if mode == "classification":
        exp = explainer.explain_instance(
            data_row=x_i,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1,
            num_samples=int(global_num_samples),
            model_regressor=Ridge(alpha=float(ridge_alpha)),
        )
        label = exp.available_labels()[0]
        items = exp.as_list(label=label)
        title = f"LIME local explanation (patient {i}, class=1)"
    else:
        exp = explainer.explain_instance(
            data_row=x_i,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=int(global_num_samples),
            model_regressor=Ridge(alpha=float(ridge_alpha)),
        )
        items = exp.as_list()
        title = f"LIME local explanation (patient {i}, score)"

    feats = [t[0] for t in items][::-1]
    wts = np.array([t[1] for t in items])[::-1]

    plt.figure(figsize=(8.2, 5.0))
    plt.barh(feats, wts)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    paths["LIME Local"] = _save_fig(f"{prefix}_A_local_patient{i}.png")

    # Global
    n_use = min(int(global_n), len(X_plot))
    p = X_train.shape[1]
    feature_names = list(X_train.columns)

    weights_mat = []
    for k in range(n_use):
        xk = X_plot.iloc[k].values

        if mode == "classification":
            expk = explainer.explain_instance(
                data_row=xk,
                predict_fn=predict_fn,
                num_features=num_features,
                top_labels=1,
                num_samples=int(global_num_samples),
                model_regressor=Ridge(alpha=float(ridge_alpha)),
            )
            label_k = expk.available_labels()[0]
            pairs = expk.local_exp[label_k]
        else:
            expk = explainer.explain_instance(
                data_row=xk,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=int(global_num_samples),
                model_regressor=Ridge(alpha=float(ridge_alpha)),
            )
            label_k = next(iter(expk.local_exp.keys()))
            pairs = expk.local_exp[label_k]

        wvec = np.zeros(p, dtype=float)
        for feat_idx, wt in pairs:
            if 0 <= int(feat_idx) < p:
                wvec[int(feat_idx)] = float(wt)
        weights_mat.append(wvec)

    lime_weight = pd.DataFrame(weights_mat, columns=feature_names)
    abs_mean = lime_weight.abs().mean(axis=0)
    abs_mean = pd.DataFrame({"feature": abs_mean.index, "abs_mean": abs_mean.values}).sort_values("abs_mean")

    plt.figure(figsize=(10, 11))
    y_ticks = range(len(abs_mean))
    y_labels = abs_mean["feature"].tolist()
    plt.barh(y=y_ticks, width=abs_mean["abs_mean"].values)
    plt.yticks(ticks=list(y_ticks), labels=y_labels, size=12)
    plt.xticks(size=12)
    plt.ylabel("Biomarkers", size=14)
    plt.xlabel("Mean Absolute XGBoost-LIME Weights", size=14)
    plt.title("LIME global importance")
    paths["LIME Global"] = _save_fig(f"{prefix}_B_global.png")

    return paths


def _describe_task(outcome_type: str) -> str:
    if outcome_type == "continuous":
        return "continuous clinical outcome prediction"
    if outcome_type == "binary":
        return "binary clinical outcome prediction (risk/probability)"
    if outcome_type == "time-to-event":
        return "time-to-event clinical outcome prediction (risk score)"
    return "clinical prediction"


def _describe_loss(outcome_type: str, loss_name: str) -> str:
    if loss_name in ["A-learning", "W-learning"]:
        return f"{loss_name} (predictive / ITR-style objective)"
    if outcome_type == "continuous":
        return "Square loss (MSE) / reg:squarederror"
    if outcome_type == "binary":
        return "Logistic loss (NLL) / binary:logistic"
    if outcome_type == "time-to-event":
        return "Cox partial likelihood / survival:cox"
    return "Original (clinical)"


def build_shap_payload(explainer, shap_values, X_plot: pd.DataFrame, patient_idx: int, top_k_local=8, top_k_global=12):
    i = int(patient_idx)
    i = max(0, min(i, len(X_plot) - 1))

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.ravel(expected_value)[0])
    else:
        expected_value = float(expected_value)

    sv = np.asarray(shap_values)
    if sv.ndim == 1:
        sv = sv.reshape(-1, 1)

    patient_features = X_plot.iloc[i, :]
    patient_shap = sv[i, :].reshape(-1)

    top_idx = np.argsort(np.abs(patient_shap))[::-1][:int(top_k_local)]
    local_rows = []
    for j in top_idx:
        local_rows.append({
            "feature": str(X_plot.columns[j]),
            "value": float(patient_features.iloc[j]),
            "shap": float(patient_shap[j])
        })

    pred = expected_value + float(patient_shap.sum())

    mean_abs = np.abs(sv).mean(axis=0)
    g_idx = np.argsort(mean_abs)[::-1][:int(top_k_global)]
    global_rows = []
    for j in g_idx:
        global_rows.append({
            "feature": str(X_plot.columns[j]),
            "mean_abs_shap": float(mean_abs[j])
        })

    return {
        "patient_index": i,
        "expected_value": expected_value,
        "prediction_additive": pred,
        "local_top": local_rows,
        "global_top": global_rows
    }


def build_lime_payload(
    model,
    X_train: pd.DataFrame,
    X_plot: pd.DataFrame,
    model_type: str,
    outcome_type: str,
    loss_name: str,
    patient_idx: int,
    num_features=10,
    global_n=60,
    global_num_samples=500,
    ridge_alpha=0.01,
    lime_seed=42,
):
    predict_fn, mode = _make_lime_predict_fn(
        model, model_type, outcome_type, loss_name, feature_names=list(X_train.columns)
    )

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode=mode,
        discretize_continuous=True,
        categorical_features=detect_categorical_indices(X_train),
        random_state=int(lime_seed),
        verbose=False
    )

    i = int(patient_idx)
    i = max(0, min(i, len(X_plot) - 1))
    x_i = X_plot.iloc[i].values

    if mode == "classification":
        exp = explainer.explain_instance(
            data_row=x_i,
            predict_fn=predict_fn,
            num_features=int(num_features),
            top_labels=1,
            num_samples=int(global_num_samples),
            model_regressor=Ridge(alpha=float(ridge_alpha)),
        )
        label = exp.available_labels()[0]
        items = exp.as_list(label=label)
    else:
        exp = explainer.explain_instance(
            data_row=x_i,
            predict_fn=predict_fn,
            num_features=int(num_features),
            num_samples=int(global_num_samples),
            model_regressor=Ridge(alpha=float(ridge_alpha)),
        )
        items = exp.as_list()

    local_rows = [{"rule": str(a), "weight": float(b)} for (a, b) in items]

    n_use = min(int(global_n), len(X_plot))
    p = X_train.shape[1]
    feature_names = list(X_train.columns)

    weights_mat = []
    for k in range(n_use):
        xk = X_plot.iloc[k].values

        if mode == "classification":
            expk = explainer.explain_instance(
                data_row=xk,
                predict_fn=predict_fn,
                num_features=int(num_features),
                top_labels=1,
                num_samples=int(global_num_samples),
                model_regressor=Ridge(alpha=float(ridge_alpha)),
            )
            label_k = expk.available_labels()[0]
            pairs = expk.local_exp[label_k]
        else:
            expk = explainer.explain_instance(
                data_row=xk,
                predict_fn=predict_fn,
                num_features=int(num_features),
                num_samples=int(global_num_samples),
                model_regressor=Ridge(alpha=float(ridge_alpha)),
            )
            label_k = next(iter(expk.local_exp.keys()))
            pairs = expk.local_exp[label_k]

        wvec = np.zeros(p, dtype=float)
        for feat_idx, wt in pairs:
            if 0 <= int(feat_idx) < p:
                wvec[int(feat_idx)] = float(wt)
        weights_mat.append(wvec)

    lime_weight = pd.DataFrame(weights_mat, columns=feature_names)
    abs_mean = lime_weight.abs().mean(axis=0).sort_values(ascending=False)

    global_rows = [{"feature": str(f), "mean_abs_weight": float(abs_mean.loc[f])} for f in abs_mean.index]

    return {
        "patient_index": i,
        "mode": mode,
        "local_top_rules": local_rows,
        "global_mean_abs": global_rows
    }


# =============================================================================
# LLM explanation
# =============================================================================
# The narrative step can run against either a hosted API or a model deployed on the
# analyst's own machine. Both are reached with the same OpenAI-compatible client, pointed
# at a different base URL: llama.cpp, vLLM, LM Studio and Ollama all serve a /v1 surface.
# Three things differ from the hosted API, and all three are handled here rather than left
# to the user.
#
#   1. API surface. The hosted path used above is /v1/responses, which local servers
#      generally do not implement; they serve /v1/chat/completions. Which one exists is
#      discovered on first use and remembered, so a temperature sweep does not pay for a
#      failed request on every generation.
#   2. Credentials. A local server needs none, so requiring OPENAI_API_KEY
#      unconditionally would make a local model unusable. The key is required only when no
#      endpoint is given, i.e. when the hosted API is the target.
#   3. Token accounting. The two surfaces name the usage fields differently and a local
#      server may omit them altogether, so counts are normalised and may be absent.
#
# Running locally also changes what leaves the machine, which is the substantive reason to
# offer it: see data_egress_note and apply_privacy_filter below.

LLM_MODEL_DEFAULT = "gpt-5.2"

# Addresses that cannot leave the host or the local network.
LLM_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")

# What a server returns for a route it does not serve at all. The text patterns are
# deliberately route-shaped: a parameter complaint such as "temperature is not supported"
# must not be mistaken for a missing endpoint, so it is tested for first.
_SURFACE_MISSING_STATUS = (404, 405, 501)
_SURFACE_MISSING_TEXT = ("not found", "invalid url", "unknown path", "no route",
                         "not implemented", "unknown request url", "unrecognized request")

# (base_url, model) -> {"surface": ..., "temperature": ...}. Capabilities are a property of
# the endpoint, not of the run, so they are cached across calls.
_LLM_CAPS: Dict[tuple, Dict[str, Any]] = {}

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)

# The SDK's own default is a 600 s read timeout with two retries, i.e. up to half an hour of
# silence if an endpoint accepts the connection and then never answers. That is a realistic
# way to mistype a local endpoint -- some other service is listening on the port -- and the
# render function is synchronous, so the whole UI freezes and a stall is indistinguishable
# from a slow generation. A finite timeout and one retry bound the damage; the reachability
# probe below turns the common case into an immediate, specific error instead.
LLM_TIMEOUT_S = 180.0
LLM_MAX_RETRIES = 1
LLM_PROBE_TIMEOUT_S = 3.0


def is_local_endpoint(base_url: Optional[str]) -> bool:
    """True if base_url resolves to this machine or a private network address.

    Used to decide whether a credential is required, and whether to warn that patient
    data is about to cross the network.
    """
    u = (base_url or "").strip()
    if not u:
        return False
    if "://" not in u:
        u = "http://" + u
    try:
        host = (urlparse(u).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return (host in LLM_LOCAL_HOSTS
            or host.endswith(".local")
            or host.startswith("192.168.")
            or host.startswith("10.")
            or bool(re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host)))


def probe_endpoint(base_url: str, timeout: float = LLM_PROBE_TIMEOUT_S) -> str:
    """Check that something at base_url is answering, before a real request is sent.

    Returns "" if the endpoint looks alive, otherwise a sentence naming what went wrong.
    Any HTTP reply at all counts as alive -- 401 and 404 both mean a server is there and
    talking, which is the question being asked. Only silence is a failure, and silence is
    the expensive case: a port with some unrelated service on it accepts the connection and
    never answers, which without this probe costs the SDK's full timeout budget.
    """
    u = base_url.strip().rstrip("/")
    if "://" not in u:
        u = "http://" + u
    try:
        req = urllib.request.Request(u + "/models", method="GET")
        urllib.request.urlopen(req, timeout=float(timeout))
        return ""
    except urllib.error.HTTPError:
        return ""                        # answered, just not with a 200
    except Exception as e:
        kind = type(e).__name__
        if isinstance(e, (TimeoutError, socket.timeout)) or "timed out" in str(e).lower():
            return (f"{base_url} accepted the connection but did not answer within "
                    f"{timeout:g}s. Something is listening on that port, but it does not "
                    f"look like an OpenAI-compatible model server -- check the port.")
        return (f"Nothing is answering at {base_url} ({kind}). Is the model server "
                f"running? For Ollama: 'ollama serve' on port 11434; vLLM defaults to "
                f"8000, llama.cpp to 8080, LM Studio to 1234.")


def make_llm_client(base_url: Optional[str] = None, api_key: Optional[str] = None,
                    probe: bool = True):
    """Build the client for a hosted or a locally deployed model.

    Returns (client, base_url, is_local). The credential is required only for the hosted
    API; a local server ignores it, but the SDK still insists on a non-empty string, so a
    placeholder is supplied rather than making the user invent one.
    """
    url = (base_url if base_url is not None else os.getenv("LLM_BASE_URL", "")).strip()
    key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
    opts = dict(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES)

    if not url:
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set in environment variables. Either set it, or "
                "set the LLM endpoint to a locally deployed OpenAI-compatible server "
                "(for example http://localhost:8000/v1), which needs no key."
            )
        return OpenAI(api_key=key, **opts), "", False

    # Only custom endpoints are probed: the hosted API is not worth an extra round trip,
    # and a typo there produces a fast, clear error of its own.
    if probe:
        problem = probe_endpoint(url)
        if problem:
            raise ValueError(problem)

    return (OpenAI(api_key=(key or "local-no-key"), base_url=url, **opts),
            url, is_local_endpoint(url))


def _usage_dict(usage) -> Dict[str, Any]:
    """Token counts, normalised across the two surfaces.

    Responses reports input_tokens/output_tokens, chat.completions reports
    prompt_tokens/completion_tokens, and a local server may report neither. Missing
    counts are omitted rather than reported as zero, so an absent field is not read as
    a free call.
    """
    if usage is None:
        return {}

    def g(*names):
        for n in names:
            v = usage.get(n) if isinstance(usage, dict) else getattr(usage, n, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
        return None

    out = {"prompt_tokens": g("input_tokens", "prompt_tokens"),
           "completion_tokens": g("output_tokens", "completion_tokens"),
           "total_tokens": g("total_tokens")}
    if out["total_tokens"] is None and None not in (out["prompt_tokens"], out["completion_tokens"]):
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return {k: v for k, v in out.items() if v is not None}


def _strip_reasoning(text: str) -> str:
    """Remove inline chain-of-thought that locally served reasoning models emit.

    Distilled reasoning checkpoints write their scratchpad into the same content field,
    wrapped in <think> tags. Left in place it would be graded as part of the narrative,
    so a feature named while thinking aloud would count as a claim about the explanation.
    """
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    if re.search(r"(?i)<think", out) and not re.search(r"(?i)</think", out):
        # An unterminated trace means the generation hit its token limit mid-thought.
        head, tail = re.split(r"(?i)<think[^>]*>", out, maxsplit=1)
        out = head.strip() or tail
    return out.strip()


def _as_messages(prompt):
    """Accept either a single prompt string or an already-formed message list.

    Multi-turn is needed for the follow-up conversation; both API surfaces take a message
    list in the same shape, so one representation serves both.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


def _call_responses(client, model, prompt, temperature):
    kw = dict(model=model, input=(prompt if isinstance(prompt, str) else _as_messages(prompt)))
    if temperature is not None:
        kw["temperature"] = float(temperature)
    r = client.responses.create(**kw)
    return (getattr(r, "output_text", "") or ""), _usage_dict(getattr(r, "usage", None))


def _call_chat(client, model, prompt, temperature):
    kw = dict(model=model, messages=_as_messages(prompt))
    if temperature is not None:
        kw["temperature"] = float(temperature)
    r = client.chat.completions.create(**kw)
    msg = r.choices[0].message
    text = getattr(msg, "content", None) or ""
    if not text.strip():
        # Some local servers split the answer out into a separate field and leave
        # content empty; an empty narrative would otherwise be graded as a failure.
        text = getattr(msg, "reasoning_content", None) or ""
    return text, _usage_dict(getattr(r, "usage", None))


# Signatures of a corporate web proxy answering instead of the API. An enterprise network is
# the normal setting for this app's audience, and the raw failure is an HTML block page dumped
# into the error box, which reads like a broken key rather than a blocked destination.
_PROXY_STATUS = (403, 407, 451, 511)
_PROXY_MARKERS = ("<!doctype html", "<html", "zscaler", "bluecoat", "forcepoint", "netskope",
                  "proxy", "access denied", "blocked by", "web filter", "url filtering")


def _proxy_block_message(exc):
    """A plain explanation when the network, not the API, refused the call."""
    body = str(exc).lower()
    status = getattr(exc, "status_code", None)
    looks_html = ("<html" in body or "<!doctype html" in body)
    named = next((m for m in ("zscaler", "bluecoat", "forcepoint", "netskope")
                  if m in body), None)
    if not (looks_html or (status in _PROXY_STATUS and any(m in body for m in _PROXY_MARKERS))):
        return None
    who = named.capitalize() if named else "A web proxy or firewall"
    return (
        f"{who} on this network intercepted the request and answered instead of the API"
        + (f" (HTTP {status})" if status else "") + ". This is not a problem with the API key "
        "-- the key was never checked, because the request did not reach the endpoint. "
        "Options: ask IT to permit the endpoint, point 'Endpoint (base URL)' at an "
        "organisation-approved gateway, or run a model locally, which needs no outbound "
        "network access at all.")


def _surface_missing(exc) -> bool:
    """Whether the failure means the route does not exist, rather than the request being bad."""
    if isinstance(exc, AttributeError):
        return True                      # SDK too old to expose client.responses
    m = str(exc).lower()
    if "model" in m and ("not found" in m or "does not exist" in m):
        # Both a missing route and an unknown model name answer 404. This one means the
        # route is there and the name is wrong, so retrying on the other surface would
        # only produce the same complaint from a second request.
        return False
    if getattr(exc, "status_code", None) in _SURFACE_MISSING_STATUS:
        return True
    return any(s in m for s in _SURFACE_MISSING_TEXT)


def _llm_call(client, model, prompt, temperature=None, base_url=""):
    """Single generation, on whichever API surface the endpoint actually serves.

    Returns (text, temperature_applied, note, info).

    Two capabilities are discovered on first use and then remembered for that
    (endpoint, model): which surface exists, and whether the model accepts an explicit
    temperature. Some newer reasoning models reject one. Rather than silently dropping it
    -- which would make a temperature sensitivity analysis meaningless while appearing to
    succeed -- the rejection is detected, the call is retried without it, and the fact is
    reported so the analysis can be labelled accordingly.
    """
    key = (str(base_url or ""), str(model))
    caps = _LLM_CAPS.setdefault(key, {"surface": None, "temperature": None})

    if caps["surface"] is None:
        caps["surface"] = "responses" if hasattr(client, "responses") else "chat"
    if caps["temperature"] is False:
        temperature = None               # known to be rejected; skip the failed request

    if temperature is None:
        note = ("temperature rejected by this model; generations use the server default"
                if caps["temperature"] is False else "temperature not set")
    else:
        note = ""

    fallback = False
    while True:
        fn = _call_responses if caps["surface"] == "responses" else _call_chat
        try:
            text, usage = fn(client, model, prompt, temperature)
        except Exception as e:
            # Parameter complaints are checked before route complaints, because the word
            # "supported" appears in both kinds of message.
            if temperature is not None and "temperature" in str(e).lower():
                caps["temperature"] = False
                note = f"model rejected 'temperature': {str(e)[:160]}"
                temperature = None
                continue
            # Checked before the route test: a proxy block is often a 403 or 404 carrying an
            # HTML page, which would otherwise be read as "this endpoint has no /v1/responses"
            # and burn a second request on the same wall.
            blocked = _proxy_block_message(e)
            if blocked:
                raise ValueError(blocked) from e
            if caps["surface"] == "responses" and _surface_missing(e):
                caps["surface"] = "chat"
                fallback = True
                continue
            raise
        if temperature is not None:
            caps["temperature"] = True
        info = {"surface": caps["surface"], "usage": usage}
        if fallback:
            info["surface_note"] = ("endpoint does not serve /v1/responses; "
                                    "used /v1/chat/completions")
        return _strip_reasoning(text), temperature is not None, note, info


# -----------------------------------------------------------------------------
# What leaves the machine
# -----------------------------------------------------------------------------
# In Global scope the transmitted payload is aggregate -- mean absolute SHAP per feature,
# no records. In Local scope it is not: `local_top` carries this individual's observed
# covariate values at full float precision, so running the narrative step against a hosted
# API transmits patient data. This is a separate matter from not exposing API credentials,
# and the mitigations, strongest first, are: deploy the model locally, use Global scope, or
# reduce the precision of what is sent.

def data_egress_note(scope: str, local_endpoint: bool, base_url: str = "") -> str:
    """One line stating what this configuration transmits and to where."""
    local_scope = str(scope).capitalize() == "Local"
    where = ("a local endpoint (" + base_url + "); nothing leaves this machine"
             if local_endpoint else
             ("the hosted API" if not base_url else "a remote endpoint (" + base_url + ")"))
    what = ("individual covariate values for the selected patient, plus their attributions"
            if local_scope else
            "feature names and mean absolute attributions only, no individual records")
    risk = "" if (local_endpoint or not local_scope) else (
        "  Patient-level data crosses the network in this configuration; deploy the model "
        "locally, or switch to Global scope, if that is not acceptable.")
    return f"Data egress: sending {what} to {where}.{risk}"


def _sigfig(x, sig=3):
    v = float(x)
    if not np.isfinite(v) or v == 0.0:
        return v
    return float(round(v, int(sig) - 1 - int(np.floor(np.log10(abs(v))))))


def apply_privacy_filter(payload, *, scope: str, mode: str = "Send as provided", sig: int = 3):
    """Reduce the precision of individual-level numbers before they are transmitted.

    Only the local-scope fields are touched: the global fields are aggregates over the
    explanation set and carry no individual record, so rounding them would cost accuracy
    for no privacy gain.

    Applied by the caller rather than inside llm_explain, so that the grader in the
    sensitivity analysis scores each narrative against exactly the numbers the model was
    shown. Filtering inside the call would leave the grader holding full-precision truth
    while the narrative quotes rounded values, and every quotation would score as wrong.
    """
    if payload is None or str(mode).lower().startswith("send"):
        return payload
    if str(scope).capitalize() != "Local":
        return payload

    p = copy.deepcopy(payload)
    r = lambda v: _sigfig(v, sig)

    for k in ("expected_value", "prediction_additive"):
        if isinstance(p.get(k), (int, float)):
            p[k] = r(p[k])

    for row in p.get("local_top", []) or []:
        for k in ("value", "shap"):
            if k in row:
                row[k] = r(row[k])

    def round_in_text(s):
        return re.sub(r"-?\d+\.\d+(?:[eE][-+]?\d+)?",
                      lambda m: repr(r(m.group(0))), str(s))

    for row in p.get("local_top_rules", []) or []:
        if "weight" in row:
            row["weight"] = r(row["weight"])
        if "rule" in row:
            row["rule"] = round_in_text(row["rule"])
        if isinstance(row.get("value"), (int, float)):
            row["value"] = r(row["value"])
    return p


def llm_explain(
    *,
    explain_method: str,
    explain_scope: str,
    outcome_type: str,
    loss_name: str,
    model_type: str,
    user_prompt: str,
    shap_payload: Optional[Dict[str, Any]] = None,
    lime_payload: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    model: str = LLM_MODEL_DEFAULT,
    feature_note: str = "",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client=None,
):
    # `client` is accepted so a caller making many generations -- the temperature sweep --
    # can build the connection once instead of once per call.
    if client is None:
        client, base_url, local_endpoint = make_llm_client(base_url, api_key)
    else:
        base_url = base_url or ""
        local_endpoint = is_local_endpoint(base_url)

    context, payload_small, scope = build_explanation_context(
        explain_method=explain_method, explain_scope=explain_scope,
        outcome_type=outcome_type, loss_name=loss_name, model_type=model_type,
        shap_payload=shap_payload, lime_payload=lime_payload, feature_note=feature_note)

    full_prompt = f"""{context}

USER PROMPT (style + extra constraints)
{user_prompt}

EXPLANATION PAYLOAD
{payload_small}
"""

    text, temp_applied, temp_note, info = _llm_call(client, model, full_prompt, temperature,
                                                    base_url=base_url)
    meta = {"model": model, "temperature": temperature,
            "temperature_applied": temp_applied,
            "temperature_note": temp_note,
            "endpoint": base_url or "hosted API (api.openai.com)",
            "local_endpoint": bool(local_endpoint),
            "surface": info.get("surface"),
            "usage": info.get("usage", {}),
            "egress": data_egress_note(scope, local_endpoint, base_url or ""),
            }
    if info.get("surface_note"):
        meta["surface_note"] = info["surface_note"]
    return full_prompt, text, meta


def build_explanation_context(*, explain_method, explain_scope, outcome_type, loss_name,
                              model_type, shap_payload=None, lime_payload=None,
                              feature_note=""):
    """The grounding block and the compacted payload, shared by every generation path.

    Factored out so that a follow-up question is answered against byte-identical grounding
    to the narrative it follows up on. If the conversation were grounded differently from
    the first narrative, a change of answer could not be attributed to the question.
    """
    if explain_method.upper() == "SHAP":
        if shap_payload is None:
            raise ValueError("SHAP payload missing.")
        payload = shap_payload
        method_desc = "SHAP (additive feature attributions)"
    elif explain_method.upper() == "LIME":
        if lime_payload is None:
            raise ValueError("LIME payload missing.")
        payload = lime_payload
        method_desc = "LIME (local surrogate explanations; global via mean absolute weights)"
    else:
        raise ValueError("explain_method must be SHAP or LIME")

    scope = explain_scope.capitalize()
    if scope not in ["Local", "Global"]:
        raise ValueError("explain_scope must be Local or Global")

    context = f"""
TASK CONTEXT
- Model type: {model_type}
- Prediction target: {_describe_task(outcome_type)}
- Training objective / loss: {_describe_loss(outcome_type, loss_name)}

SCORE ORIENTATION (read this before describing any direction)
- {benefit_score_note(outcome_type, loss_name)}{feature_note}
- All attributions in the payload below are already expressed on that scale, so a
  POSITIVE contribution always pushes the score UP and a NEGATIVE contribution pushes
  it DOWN. Describe directions using the meaning stated above; do not re-derive them.

INTERPRETATION GOAL
- If loss is Original clinical: interpret as prognostic biomarkers (association with predicted outcome/risk).
- If loss is A-learning or W-learning: interpret as predictive biomarkers / individualized treatment rules (ITR). Focus on treatment effect heterogeneity.

EXPLANATION METHOD
- Method: {method_desc}
- Scope: {scope}

OUTPUT REQUIREMENTS
- Use the user's prompt style/instructions.
- Be faithful to provided explanation payload ONLY. Do not invent features.
- If scope=Global: summarize top drivers and what they imply clinically.
- If scope=Local: explain this individual's top drivers, directionality, and clinical meaning.
- If the payload contains a `coefficients` block, this is a model whose parameters are
  directly interpretable. Lead with that conventional summary -- the effect estimate, its
  interval, and the odds or hazard ratio where given -- and treat the attributions as a
  description of how much each covariate contributes across this population, not as
  separate evidence. Quote intervals as given; do not compute new ones. Respect
  `coefficient_note`: where it says an interval is not valid, do not supply one.
"""

    def _compact_payload(p):
        extra = ({"coefficients": p["coefficients"],
                  "coefficient_note": p.get("coefficient_note", "")}
                 if p.get("coefficients") else {})
        if explain_method.upper() == "SHAP":
            if scope == "Local":
                return {
                    "patient_index": p["patient_index"],
                    "expected_value": p["expected_value"],
                    "prediction_additive": p["prediction_additive"],
                    "local_top": p["local_top"],
                    **extra,
                }
            else:
                return {"global_top": p["global_top"], **extra}
        else:
            if scope == "Local":
                return {
                    "patient_index": p["patient_index"],
                    "mode": p.get("mode"),
                    "local_top_rules": p["local_top_rules"],
                }
            else:
                return {"global_mean_abs_top15": p["global_mean_abs"][:15]}

    return context, _compact_payload(payload), scope


# =============================================================================
# Stability and faithfulness of the explanation layer
# =============================================================================
# Both the surrogate layer (LIME) and the narrative layer (LLM) are stochastic, so both
# are evaluated with the same three questions and, where possible, the same statistics:
#
#   reproducibility  does a repeated run name the same features, with the same signs?
#   faithfulness     does the output agree with the quantity it claims to describe?
#   sensitivity      how do those answers move as the knobs are turned?
#
# For LIME the knobs are the perturbation seed, the number of perturbations, and the
# kernel width; for the LLM the knob is the sampling temperature.

def _spearman(a, b):
    ra = pd.Series(np.asarray(a, dtype=float)).rank().values
    rb = pd.Series(np.asarray(b, dtype=float)).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _mean_pairwise(vectors, fn):
    vals = [fn(vectors[i], vectors[j])
            for i, j in itertools.combinations(range(len(vectors)), 2)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _top_k_set(values, feature_names, k, use_abs=True):
    v = np.abs(np.asarray(values, dtype=float)) if use_abs else np.asarray(values, dtype=float)
    idx = np.argsort(-v)[:int(k)]
    return frozenset(str(feature_names[i]) for i in idx)


def _jaccard(a, b):
    a, b = set(a), set(b)
    return float(len(a & b) / len(a | b)) if (a or b) else np.nan


def _modal_set(sets):
    if not sets:
        return frozenset(), np.nan
    counts = Counter(sets)
    best, n = counts.most_common(1)[0]
    return best, float(n / len(sets))


# -----------------------------------------------------------------------------
# LIME stability
# -----------------------------------------------------------------------------
def lime_default_kernel_width(n_features: int) -> float:
    """LIME's own default: sqrt(n_features) * 0.75."""
    return float(np.sqrt(int(n_features)) * 0.75)


def _lime_run(X_train, feature_names, mode, predict_fn, x_row, *, kernel_width, seed,
              num_features, num_samples, discretize=True, ridge_alpha=0.01,
              categorical_features=None):
    """One LIME explanation. Returns (weight vector over all features, surrogate R^2)."""
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=list(feature_names),
        mode=mode,
        discretize_continuous=bool(discretize),
        categorical_features=(categorical_features or []),
        kernel_width=float(kernel_width),
        random_state=int(seed),
        verbose=False,
    )
    kwargs = dict(data_row=np.asarray(x_row, dtype=float), predict_fn=predict_fn,
                  num_features=int(num_features), num_samples=int(num_samples),
                  model_regressor=Ridge(alpha=float(ridge_alpha)))
    if mode == "classification":
        kwargs["top_labels"] = 1
    exp = explainer.explain_instance(**kwargs)
    label = (exp.available_labels()[0] if mode == "classification"
             else next(iter(exp.local_exp.keys())))

    p = len(feature_names)
    w = np.zeros(p, dtype=float)
    for fi, wt in exp.local_exp[label]:
        if 0 <= int(fi) < p:
            w[int(fi)] = float(wt)

    score = getattr(exp, "score", np.nan)
    if isinstance(score, dict):
        score = score.get(label, np.nan)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = np.nan
    return w, score


def lime_stability_analysis(
    model, X_train: pd.DataFrame, X_plot: pd.DataFrame, *,
    model_type: str, outcome_type: str, loss_name: str, patient_idx: int = 0,
    sample_sizes=(200, 500, 1000), kernel_multipliers=(0.5, 1.0, 2.0),
    repeats: int = 5, scope: str = "Local", global_n: int = 20,
    num_features: int = 10, top_k: int = 3, shap_values=None, seed0: int = 1000,
):
    """Sweep the LIME knobs the reviewer names and report stability at each setting.

    At every (num_samples, kernel_width) cell, LIME is re-run `repeats` times with
    different perturbation seeds. Reported per cell:

      Top-K agreement    proportion of runs whose top-K set equals the modal top-K set,
                         the same statistic as Eq. (4) applied to LIME instead of the LLM
      Top-K Jaccard      mean pairwise overlap of top-K sets across runs
      rank correlation   mean pairwise Spearman correlation of the full weight vector
      sign consistency   for features in the modal top-K, agreement of the weight sign
      surrogate R^2      mean R^2 of LIME's local ridge fit (its own faithfulness measure)
      SHAP top-K Jaccard overlap with the deterministic SHAP ranking, when supplied
    """
    feature_names = list(X_train.columns)
    p = len(feature_names)
    predict_fn, mode = _make_lime_predict_fn(
        model, model_type, outcome_type, loss_name, feature_names=feature_names
    )
    base_kw = lime_default_kernel_width(p)
    cat_idx = detect_categorical_indices(X_train)

    i = int(max(0, min(int(patient_idx), len(X_plot) - 1)))
    is_global = str(scope).lower().startswith("glob")
    n_use = int(min(int(global_n), len(X_plot))) if is_global else 1

    shap_top = None
    if shap_values is not None:
        sv = np.asarray(shap_values)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        ref = np.abs(sv).mean(axis=0) if is_global else sv[i, :]
        shap_top = _top_k_set(ref, feature_names, top_k)

    rows = []
    for ns in sample_sizes:
        for km in kernel_multipliers:
            kw = float(km) * base_kw
            vecs, scores = [], []
            for r in range(int(repeats)):
                if is_global:
                    acc = []
                    for k in range(n_use):
                        w, sc = _lime_run(
                            X_train.values, feature_names, mode, predict_fn,
                            X_plot.iloc[k].values, kernel_width=kw, seed=seed0 + r,
                            num_features=num_features, num_samples=ns,
                            categorical_features=cat_idx,
                        )
                        acc.append(np.abs(w))
                        scores.append(sc)
                    vecs.append(np.mean(acc, axis=0))
                else:
                    w, sc = _lime_run(
                        X_train.values, feature_names, mode, predict_fn,
                        X_plot.iloc[i].values, kernel_width=kw, seed=seed0 + r,
                        num_features=num_features, num_samples=ns,
                        categorical_features=cat_idx,
                    )
                    vecs.append(w)
                    scores.append(sc)

            tops = [_top_k_set(v, feature_names, top_k) for v in vecs]
            modal, agree = _modal_set(tops)
            jac = _mean_pairwise(tops, _jaccard)
            rho = _mean_pairwise(vecs, _spearman)

            if is_global:
                sign_cons = np.nan   # global vectors are mean |weight|, so signs are moot
            else:
                per = []
                for f in modal:
                    j = feature_names.index(f)
                    sg = [np.sign(v[j]) for v in vecs if v[j] != 0]
                    if sg:
                        _, frac = _modal_set(tuple(sg))
                        per.append(frac)
                sign_cons = float(np.mean(per)) if per else np.nan

            finite = [s for s in scores if np.isfinite(s)]
            rows.append({
                "num_samples": int(ns),
                "kernel_mult": float(km),
                "kernel_width": kw,
                "agreement": agree,
                "jaccard": jac,
                "spearman": rho,
                "sign_consistency": sign_cons,
                "surrogate_r2": float(np.mean(finite)) if finite else np.nan,
                "modal_top": sorted(modal),
                "shap_jaccard": (_jaccard(modal, shap_top) if shap_top is not None else np.nan),
                # Direct analogue of Eq. (4) with the SHAP ranking as the reference set S*:
                # the proportion of LIME runs whose top-K set is exactly S*.
                "agreement_vs_shap": (float(np.mean([t == shap_top for t in tops]))
                                      if shap_top is not None else np.nan),
            })

    return {
        "scope": "Global" if is_global else "Local",
        "patient_index": (None if is_global else i),
        "n_instances": n_use,
        "repeats": int(repeats),
        "top_k": int(top_k),
        "num_features": int(num_features),
        "default_kernel_width": base_kw,
        "mode": mode,
        "discretize_continuous": True,
        "ridge_alpha": 0.01,
        "categorical_features": [feature_names[j] for j in cat_idx],
        "shap_top": (sorted(shap_top) if shap_top is not None else None),
        "rows": rows,
    }


def format_lime_stability(res: Optional[Dict[str, Any]]) -> list:
    if not res:
        return []
    if "error" in res:
        return ["=== LIME stability ===", f"Failed: {res['error']}"]

    L = ["=== LIME stability (reproducibility, faithfulness, sensitivity) ===",
         f"Scope: {res['scope']}"
         + (f" (patient {res['patient_index']})" if res["patient_index"] is not None
            else f" (aggregated over {res['n_instances']} instances)"),
         f"Repeats per setting: {res['repeats']} perturbation seeds",
         f"Reported hyperparameters: num_features={res['num_features']}, "
         f"discretize_continuous={res['discretize_continuous']}, "
         f"surrogate=Ridge(alpha={res['ridge_alpha']}), mode={res['mode']}",
         f"LIME default kernel width for this design: {res['default_kernel_width']:.4f} "
         f"(= sqrt(p) * 0.75); multipliers are relative to it",
         ""]
    if res.get("shap_top"):
        L.append(f"Deterministic SHAP top-{res['top_k']}: {', '.join(res['shap_top'])}")
        L.append("")

    K = res["top_k"]
    hdr = (f"{'n_samp':>7s} {'kw_mult':>8s} {'kern_w':>8s} {'agree_modal':>11s} "
           f"{'agree_SHAP':>10s} {'jaccard':>8s} {'spearman':>9s} {'sign':>6s} "
           f"{'surr_R2':>8s} {'jac_SHAP':>8s}  modal top-K")
    L.append(hdr)
    L.append("-" * len(hdr))

    def f(v, w, d=3):
        return f"{v:>{w}.{d}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"

    for r in res["rows"]:
        L.append(
            f"{r['num_samples']:>7d} {r['kernel_mult']:>8.2f} {r['kernel_width']:>8.3f} "
            f"{f(r['agreement'], 11)} {f(r['agreement_vs_shap'], 10)} "
            f"{f(r['jaccard'], 8)} {f(r['spearman'], 9)} "
            f"{f(r['sign_consistency'], 6, 2)} {f(r['surrogate_r2'], 8)} "
            f"{f(r['shap_jaccard'], 8, 2)}  {', '.join(r['modal_top'])}"
        )

    ag = [r["agreement"] for r in res["rows"] if np.isfinite(r["agreement"])]
    sp = [r["spearman"] for r in res["rows"] if np.isfinite(r["spearman"])]
    L.append("")
    if ag:
        L.append(f"Top-{K} agreement (vs modal) across all settings: "
                 f"min {min(ag):.3f}, median {np.median(ag):.3f}, max {max(ag):.3f}")
    if sp:
        L.append(f"Rank correlation across all settings: "
                 f"min {min(sp):.3f}, median {np.median(sp):.3f}, max {max(sp):.3f}")

    L.append("")
    L.append("Metric definitions and provenance")
    L.append(f"  agree_SHAP   Eq. (4) exactly, with the deterministic SHAP top-{K} as the")
    L.append("               reference set S*: proportion of runs whose top-K set equals S*.")
    L.append(f"  agree_modal  Eq. (4) with S* replaced by the most frequent top-{K} set across")
    L.append("               runs. Measures self-consistency alone, with no reference method.")
    L.append("  sign         Eq. (5) in spirit, applied to LIME weight signs instead of")
    L.append("               narrative wording: agreement of each modal-top-K feature's sign.")
    L.append("  jaccard, spearman, surr_R2, jac_SHAP")
    L.append("               NOT in the manuscript. Added because the manuscript contains no")
    L.append("               LIME stability analysis; exact-set agreement alone is harsh when a")
    L.append("               ranking shifts by one position, and the surrogate R^2 is LIME's own")
    L.append("               internal faithfulness measure.")
    L.append("")
    L.append("Agreement below 1.0 means the top-K set itself changed with the perturbation "
             "seed at that setting.")
    return L


# -----------------------------------------------------------------------------
# Automated grading of LLM narratives
# -----------------------------------------------------------------------------
# The metrics of Section 5 are implemented as explicit, auditable rules so that a
# repeated evaluation is reproducible rather than dependent on a single human rater.
# Every rule is a documented heuristic; the raw extractions are returned alongside the
# scores so they can be checked.

# Direction words are matched as non-overlapping stem regexes rather than as a list of
# literal substrings. Counting literals double-counts any word whose stem is also in the
# list ("decreases" scores for both "decrease" and "decreases", while "increasing" scores
# only once because "increase" is not its prefix), which silently biases the comparison
# towards whichever direction has more overlapping entries and produced spurious ties.
_DIR_UP_RE = re.compile(
    r"\b(?:increas\w*|rais\w*|rise\w*|rising|higher|greater|larger|elevat\w*"
    r"|positive\w*|upward\w*|push\w*\s+(?:it\s+)?up)\b")
_DIR_DOWN_RE = re.compile(
    r"\b(?:decreas\w*|reduc\w*|lower\w*|diminish\w*|smaller|drop\w*|fall\w*"
    r"|negative\w*|downward\w*|push\w*\s+(?:it\s+)?down)\b")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Feature-name-shaped tokens (X4, X10, IL6), masked before numeric extraction so the digits
# inside a name are never read as a numeric claim -- even when that name is absent from the
# supplied feature list.
_FEATURE_LIKE_RE = re.compile(r"\b[A-Za-z]{1,4}\d{1,3}\b")

# Acronyms that look like gene symbols but are vocabulary of the method, not features.
_ACRONYM_ALLOW = {
    "SHAP", "LIME", "LLM", "ITR", "AUC", "ROC", "RMSE", "MSE", "NLL", "CV", "BMI", "API",
    "XGB", "XGBOOST", "GPT", "SD", "CI", "ML", "AI", "OK", "IQR", "ATE", "CATE", "OLS",
}
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\d*\b")

_FRAME_PROGNOSTIC = ("prognostic", "prognosis", "outcome prediction", "outcome-predictive",
                     "outcome predictive", "risk stratification", "risk strat",
                     "predicted outcome", "outcome-associated", "outcome associated")
_FRAME_PREDICTIVE = ("treatment effect heterogeneity", "treatment-effect heterogeneity",
                     "predictive biomarker", "treatment benefit", "individualized treatment",
                     "individualised treatment", "itr", "differential treatment",
                     "benefiting", "treatment effect modif")

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# The last five are added so that a narrative citing a conventional parameter summary has
# those numbers graded too. Multi-word phrases are used deliberately: single words like
# "ratio" or "interval" would fire on incidental prose and start counting numbers that were
# never presented as claims about the payload.
_NUM_CONTEXT = ("shap", "value", "contribution", "contributes", "prediction", "predicted",
                "baseline", "expected", "mean", "average", "importance", "score",
                "coefficient", "odds ratio", "hazard ratio", "confidence interval", "95% ci")


# The nominal level of an interval -- the "95" in "95% confidence interval" -- is not a claim
# about the payload, it names the procedure. Masked before extraction, or every narrative that
# correctly reports an interval is charged with one unsupported number.
_LEVEL_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s*(?=(?:confidence\s+interval|ci\b|interval))",
                       re.IGNORECASE)


def _mask_feature_names(text, feature_names):
    """Replace feature names, and anything shaped like one, with a placeholder so digits
    inside them (X4, X10) are not mistaken for numeric claims."""
    out = _LEVEL_RE.sub(" @LEVEL@ ", text)
    for f in sorted(feature_names, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(str(f))}\b", " @FEAT@ ", out, flags=re.IGNORECASE)
    return _FEATURE_LIKE_RE.sub(" @FEAT@ ", out)


def features_mentioned(text, feature_names):
    """Feature names present in the narrative, ordered by first appearance."""
    hits = []
    low = text.lower()
    for f in feature_names:
        m = re.search(rf"\b{re.escape(str(f).lower())}\b", low)
        if m:
            hits.append((m.start(), str(f)))
    return [f for _, f in sorted(hits)]


def top_k_from_text(text, feature_names, k):
    """The first k distinct feature names mentioned, taken as the narrative's top-k claim.

    A deliberately simple rule: the prompt asks for the leading drivers first, so order of
    first mention is the most defensible automatic proxy for the claimed ranking.
    """
    return frozenset(features_mentioned(text, feature_names)[:int(k)])


def directional_claims(text, feature_names):
    """(feature, +1/-1) for every feature given an unambiguous direction.

    Scoped to the sentence containing the feature's first mention. A fixed character
    window bleeds across sentence boundaries and picks up the direction stated for a
    neighbouring feature, which produces ties and silently drops real claims.
    """
    out = []
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    for f in feature_names:
        pat = re.compile(rf"\b{re.escape(str(f))}\b", re.IGNORECASE)
        for s in sentences:
            if not pat.search(s):
                continue
            up = len(_DIR_UP_RE.findall(s.lower()))
            dn = len(_DIR_DOWN_RE.findall(s.lower()))
            if up > dn:
                out.append((str(f), 1))
            elif dn > up:
                out.append((str(f), -1))
            break
    return out


def objective_framing_ok(text, loss_name):
    """Section 5.1.1's rule: the narrative must use the vocabulary of the estimand its
    objective actually targets, and must not lean more on the opposite vocabulary."""
    low = text.lower()
    prog = sum(low.count(w) for w in _FRAME_PROGNOSTIC)
    pred = sum(low.count(w) for w in _FRAME_PREDICTIVE)
    if loss_name in ("A-learning", "W-learning"):
        return (pred > 0 and pred >= prog), {"predictive_hits": pred, "prognostic_hits": prog}
    return (prog > 0 and prog >= pred), {"predictive_hits": pred, "prognostic_hits": prog}


def numeric_claims(text, feature_names):
    """Numbers stated in a quantitative context, with feature names masked out first."""
    masked = _mask_feature_names(text, feature_names)
    claims = []
    for m in _NUM_RE.finditer(masked):
        seg = masked[max(0, m.start() - 60): m.end() + 60].lower()
        if any(w in seg for w in _NUM_CONTEXT):
            try:
                claims.append(float(m.group()))
            except ValueError:
                pass
    return claims


def out_of_vocabulary_mentions(text, feature_names):
    """Gene-symbol-like tokens that are not features of this model and not method
    vocabulary -- e.g. a narrative that invents 'KRAS'."""
    known = {str(f).upper() for f in feature_names}
    hits = []
    for m in _ACRONYM_RE.finditer(text):
        tok = m.group()
        if tok.upper() in known or tok.upper() in _ACRONYM_ALLOW:
            continue
        hits.append(tok)
    return sorted(set(hits))


def grade_narrative(text, *, feature_names, reference_top, truth_values, loss_name,
                    shap_signs=None, top_k=3, tol=0.05, payload_features=None):
    """Score one narrative against the payload it was given.

    `payload_features` is the subset of features actually shown to the model. Naming a
    feature outside that subset counts as an unsupported claim, per Section 5.1.2.
    """
    named = features_mentioned(text, feature_names)
    s_i = top_k_from_text(text, feature_names, top_k)

    dirs = directional_claims(text, feature_names)
    d_ok = d_tot = 0
    if shap_signs:
        for f, sgn in dirs:
            if f in shap_signs and shap_signs[f] != 0:
                d_tot += 1
                d_ok += int(np.sign(shap_signs[f]) == sgn)

    frame_ok, frame_detail = objective_framing_ok(text, loss_name)

    claims = numeric_claims(text, feature_names)
    truth = np.asarray([t for t in truth_values if np.isfinite(t)], dtype=float)
    matched, unmatched = 0, []
    for c in claims:
        if truth.size and np.min(np.abs(truth - c)) <= tol:
            matched += 1
        else:
            unmatched.append(c)

    # A hallucination is an unsupported substantive claim: a number matching nothing in the
    # payload, a feature named that the payload did not contain, or an invented biomarker.
    not_in_payload = ([f for f in named if f not in set(payload_features)]
                      if payload_features is not None else [])
    oov = out_of_vocabulary_mentions(text, feature_names)
    return {
        "top_set": s_i,
        "top_exact": bool(s_i == reference_top),
        "dir_total": d_tot,
        "dir_correct": d_ok,
        "framing_ok": bool(frame_ok),
        "framing_detail": frame_detail,
        "num_total": len(claims),
        "num_matched": matched,
        "num_unmatched": unmatched,
        "not_in_payload": not_in_payload,
        "out_of_vocabulary": oov,
        "hallucinations": len(unmatched) + len(not_in_payload) + len(oov),
        "features_named": named,
        "n_chars": len(text),
    }


def _embed(texts):
    """Unit-norm embeddings for a set of texts, and the backend that produced them.

    All texts are embedded together so that similarities computed within a group and
    between groups are on the same footing -- TF-IDF in particular fits its vocabulary to
    whatever it is shown, so embedding two groups separately would make their cosines
    incomparable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        E = np.asarray(enc.encode(list(texts), normalize_embeddings=True), dtype=float)
        return E, "sentence-transformers all-MiniLM-L6-v2"
    except ImportError:
        pass
    except Exception:
        pass

    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        M = TfidfVectorizer(stop_words="english").fit_transform(list(texts))
    except ValueError:
        # No usable vocabulary: every text was empty or stop words only. Reachable in
        # practice -- a locally served reasoning model can emit nothing but a <think> trace,
        # which _strip_reasoning correctly reduces to an empty string. Similarity is
        # undefined there, and saying so is better than aborting the whole evaluation.
        return None, "n/a (no usable text to compare)"
    E = np.asarray(M.todense(), dtype=float)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(nrm, 1e-12, None), "TF-IDF cosine (sentence-transformers not installed)"


def _cos(a, b):
    """Cosine of two unit vectors, clamped to [-1, 1].

    Rounding puts the dot product of a vector with itself a few 1e-16 either side of 1.0.
    Unclamped, identical texts can report a similarity above 1, and a within-minus-between
    gap that should be exactly zero can come out faintly negative.
    """
    return float(np.clip(float(np.dot(a, b)), -1.0, 1.0))


def text_similarity(texts):
    """Mean pairwise cosine similarity. Uses sentence-transformers when available (as in
    the manuscript), otherwise TF-IDF, and reports which backend was used."""
    if len(texts) < 2:
        return np.nan, "n/a"
    E, backend = _embed(texts)
    if E is None:
        return np.nan, backend
    sims = [_cos(E[i], E[j]) for i, j in itertools.combinations(range(len(E)), 2)]
    return (float(np.mean(sims)) if sims else np.nan), backend


def within_and_between_similarity(groups):
    """Mean cosine within each group and between different groups, on one embedding.

    The distinction is what separates adaptation from noise. A generator that produces
    different text for different requests shows between-group similarity BELOW its
    within-group similarity; a generator that is merely stochastic shows the two at the
    same level; a deterministic generator shows 1.0 for both.
    """
    flat, index = [], []
    for gi, g in enumerate(groups):
        for t in g:
            flat.append(t)
            index.append(gi)
    if len(flat) < 2:
        return {"within": np.nan, "between": np.nan, "backend": "n/a", "per_group": []}

    E, backend = _embed(flat)
    if E is None:
        return {"within": np.nan, "between": np.nan, "backend": backend,
                "per_group": [np.nan for _ in groups]}
    within, between, per_group = [], [], [[] for _ in groups]
    for i, j in itertools.combinations(range(len(flat)), 2):
        s = _cos(E[i], E[j])
        if index[i] == index[j]:
            within.append(s)
            per_group[index[i]].append(s)
        else:
            between.append(s)
    return {
        "within": float(np.mean(within)) if within else np.nan,
        "between": float(np.mean(between)) if between else np.nan,
        "per_group": [float(np.mean(g)) if g else np.nan for g in per_group],
        "backend": backend,
    }


def payload_reference(payload, *, explain_method, explain_scope, top_k=3, feature_names=None):
    """What a narrative about this payload may legitimately say.

    Returns the reference top-K ranking, the SHAP signs used to score directional claims,
    every number traceable to the payload, and the feature subset that was actually shown.
    Factored out of the sweep so the template baseline, the adaptation experiment and the
    follow-up grader all score against exactly the same reference as the LLM does.
    """
    if payload is None:
        raise ValueError(f"No {explain_method} payload available for the evaluation.")
    scope_local = str(explain_scope).capitalize() == "Local"
    truth, shap_signs, reference_top, payload_features = [], {}, frozenset(), None

    if str(explain_method).upper() == "SHAP":
        if scope_local:
            rows = payload["local_top"]
            reference_top = frozenset(r["feature"] for r in rows[:int(top_k)])
            shap_signs = {r["feature"]: r["shap"] for r in rows}
            payload_features = [r["feature"] for r in rows]
            truth = ([r["shap"] for r in rows] + [r["value"] for r in rows]
                     + [payload["expected_value"], payload["prediction_additive"]])
        else:
            rows = payload["global_top"]
            reference_top = frozenset(r["feature"] for r in rows[:int(top_k)])
            truth = [r["mean_abs_shap"] for r in rows]
            payload_features = [r["feature"] for r in rows]
    else:
        if scope_local:
            rows = payload["local_top_rules"]
            truth = [r["weight"] for r in rows]
            reference_top = top_k_from_text(" ".join(r["rule"] for r in rows),
                                            feature_names or [], top_k)
        else:
            rows = payload["global_mean_abs"]
            reference_top = frozenset(r["feature"] for r in rows[:int(top_k)])
            truth = [r["mean_abs_weight"] for r in rows]
            payload_features = [r["feature"] for r in rows[:15]]

    # A parametric model carries its conventional summary in the payload, so the numbers in
    # it -- the coefficient, its interval, the odds or hazard ratio -- are payload content
    # and a narrative quoting them is being faithful, not inventing.
    for c in (payload.get("coefficients") or []):
        for key in ("coef", "se", "ci_lo", "ci_hi", "exp_coef", "exp_lo", "exp_hi",
                    "boot_lo", "boot_hi", "boot_se"):
            v = c.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                truth.append(float(v))

    return {"truth": truth, "shap_signs": shap_signs, "reference_top": reference_top,
            "payload_features": payload_features, "rows": rows}


def _grade_batch(texts, *, feature_names, ref, loss_name, top_k=3, tol=0.05):
    """Score a set of narratives and return the Section 5 metrics for that set.

    The same block the temperature sweep used, lifted out so that a template's output and
    an LLM's output are reduced to numbers by identical code. Equations (5) and (7) are
    means of PER-NARRATIVE ratios, (1/N) sum_i D_i/F_i and (1/N) sum_i A_i/C_i, so every
    narrative carries equal weight regardless of how many claims it makes. The pooled
    ratios are reported alongside as diagnostics, not as the headline numbers.
    """
    grades = [grade_narrative(
        t, feature_names=list(feature_names or []), reference_top=ref["reference_top"],
        truth_values=ref["truth"], loss_name=loss_name, shap_signs=ref["shap_signs"],
        top_k=top_k, tol=tol, payload_features=ref["payload_features"],
    ) for t in texts]

    sim, backend = text_similarity(texts)
    d_ratios = [g["dir_correct"] / g["dir_total"] for g in grades if g["dir_total"] > 0]
    n_ratios = [g["num_matched"] / g["num_total"] for g in grades if g["num_total"] > 0]
    d_tot = sum(g["dir_total"] for g in grades)
    n_tot = sum(g["num_total"] for g in grades)

    return {
        "n": len(grades),
        "top_agreement": float(np.mean([g["top_exact"] for g in grades])) if grades else np.nan,
        "directional": float(np.mean(d_ratios)) if d_ratios else np.nan,
        "directional_pooled": (float(sum(g["dir_correct"] for g in grades) / d_tot)
                               if d_tot else np.nan),
        "directional_narratives": len(d_ratios),
        "directional_claims": d_tot,
        "similarity": sim,
        "framing": float(np.mean([g["framing_ok"] for g in grades])) if grades else np.nan,
        "numeric_accuracy": float(np.mean(n_ratios)) if n_ratios else np.nan,
        "numeric_pooled": (float(sum(g["num_matched"] for g in grades) / n_tot)
                           if n_tot else np.nan),
        "numeric_narratives": len(n_ratios),
        "numeric_claims": n_tot,
        "hallucinations": float(np.mean([g["hallucinations"] for g in grades])) if grades else np.nan,
        "distinct_top_sets": len({g["top_set"] for g in grades}),
        "mean_chars": float(np.mean([g["n_chars"] for g in grades])) if grades else np.nan,
        "unmatched_examples": sorted({round(u, 4) for g in grades
                                     for u in g["num_unmatched"]})[:8],
        "unsupported_features": sorted({f for g in grades
                                       for f in g["not_in_payload"] + g["out_of_vocabulary"]})[:8],
        "similarity_backend": backend,
    }


# =============================================================================
# Deterministic template baseline
# =============================================================================
# A reviewer asked why an LLM is needed at all: given a structured payload of feature names,
# ranks, signs and values, a fixed template should report them perfectly, for free, and
# identically every time. That is correct, and this is that template, written to the
# manuscript's own reporting rules so it can be graded by the same harness as the LLM.
#
# It wins every Section 5 metric by construction, and the point of building it is to make
# that explicit rather than leave it as an untested objection:
#
#   Eq. (4) top-K agreement     it prints the ranking it was handed
#   Eq. (5) directional         it prints the sign of each attribution
#   Eq. (7) numerical accuracy  every number is copied from the payload
#   Eq. (8) hallucinations      it can only emit payload contents
#   Eq. (6) similarity          deterministic, so exactly 1.0
#   framing accuracy            a lookup on the objective, not an inference
#
# What it cannot do is respond to the free-text prompt, adapt to an audience, or answer a
# question -- which is where the argument for the LLM layer has to be made instead, and what
# adaptation_analysis and llm_followup below measure.

def _estimand_sentence(loss_name: str) -> str:
    """The clause that fixes the estimand vocabulary. Correct by lookup, not by inference."""
    if loss_name in ("A-learning", "W-learning"):
        return ("This pattern is consistent with treatment effect heterogeneity, and suggests "
                "these covariates may act as predictive biomarkers relevant to an "
                "individualized treatment rule.")
    return ("This pattern is consistent with a prognostic association with the predicted "
            "outcome, and should not be read as evidence about how the effect of therapy "
            "varies between patients.")


def _dir_phrase(v: float) -> str:
    """A directional verb the grader can read, matching the sign of the attribution."""
    return "increases" if float(v) >= 0 else "decreases"


def template_explain(*, explain_method, explain_scope, outcome_type, loss_name, model_type,
                     shap_payload=None, lime_payload=None, feature_names=None, top_k=3,
                     decimals=3):
    """Report the payload deterministically. Same inputs as llm_explain, no model involved.

    Returns (text, meta). `user_prompt` is deliberately absent from the signature: the
    template cannot honour free-text instructions, and pretending otherwise by accepting and
    ignoring the argument would obscure precisely the capability being compared.
    """
    scope = str(explain_scope).capitalize()
    if scope not in ("Local", "Global"):
        raise ValueError("explain_scope must be Local or Global")
    method = str(explain_method).upper()
    payload = shap_payload if method == "SHAP" else lime_payload
    if payload is None:
        raise ValueError(f"{explain_method} payload missing.")

    d = int(decimals)
    fnames = list(feature_names or [])
    loss_text = _describe_loss(outcome_type, loss_name)
    # Ordinals as words: a digit near a context word such as "mean" would be extracted as a
    # numerical claim about the payload and correctly scored as unsupported.
    ordinal = ["The largest", "The second largest", "The third largest", "The fourth largest",
               "The fifth largest", "The sixth largest", "The seventh largest",
               "The eighth largest"]
    S = []

    if method == "SHAP" and scope == "Global":
        rows = payload["global_top"][:int(top_k)]
        names = [r["feature"] for r in rows]
        S.append(f"The model was fitted under {loss_text}, and {benefit_score_note(outcome_type, loss_name)}")
        S.append("Across the explanation set, the leading contributors ranked by mean "
                 "absolute SHAP value are " + _join_names(names) + ".")
        for r in rows:
            S.append(f"{r['feature']} has a mean absolute SHAP value of "
                     f"{r['mean_abs_shap']:.{d}f}.")
        S.append(_estimand_sentence(loss_name))

    elif method == "SHAP" and scope == "Local":
        rows = payload["local_top"][:int(top_k)]
        S.append(f"The model was fitted under {loss_text}, and {benefit_score_note(outcome_type, loss_name)}")
        S.append("For this patient the additive prediction is "
                 f"{payload['prediction_additive']:.{d}f}, against a baseline expected value "
                 f"of {payload['expected_value']:.{d}f}.")
        for i, r in enumerate(rows):
            # One sentence per feature: the grader scopes a directional claim to the
            # sentence containing the feature's first mention, so combining features into
            # one sentence would mix their directions.
            S.append(f"{ordinal[min(i, len(ordinal) - 1)]} contribution comes from "
                     f"{r['feature']}, which {_dir_phrase(r['shap'])} the score, with a SHAP "
                     f"contribution of {r['shap']:+.{d}f} at an observed value of "
                     f"{r['value']:.{d}f}.")
        S.append(_estimand_sentence(loss_name))

    elif method == "LIME" and scope == "Global":
        rows = payload["global_mean_abs"][:int(top_k)]
        names = [r["feature"] for r in rows]
        S.append(f"The model was fitted under {loss_text}, and {benefit_score_note(outcome_type, loss_name)}")
        S.append("Aggregating the local surrogate fits, the leading contributors ranked by "
                 "mean absolute LIME weight are " + _join_names(names) + ".")
        for r in rows:
            S.append(f"{r['feature']} has a mean absolute LIME weight of "
                     f"{r['mean_abs_weight']:.{d}f}.")
        S.append(_estimand_sentence(loss_name))

    else:
        rows = payload["local_top_rules"][:int(top_k)]
        S.append(f"The model was fitted under {loss_text}, and {benefit_score_note(outcome_type, loss_name)}")
        S.append("For this patient the local surrogate assigns most of its weight to the "
                 "conditions listed below.")
        for i, r in enumerate(rows):
            named = features_mentioned(str(r["rule"]), fnames)
            who = named[0] if named else "the leading condition"
            # The rule's own threshold is not quoted: thresholds are not part of the truth
            # set the numerical-accuracy metric is scored against, so quoting them would be
            # penalised as unsupported even though they came from the payload. The LLM is
            # not shielded from that, which is noted where the two are compared.
            S.append(f"{ordinal[min(i, len(ordinal) - 1)]} weight falls on {who}, which "
                     f"{_dir_phrase(r['weight'])} the score, with a LIME weight of "
                     f"{r['weight']:+.{d}f}.")
        S.append(_estimand_sentence(loss_name))

    # A parametric model's own parameters lead, because they are the primary summary and the
    # attributions are an algebraic re-expression of them.
    coefs = {c["feature"]: c for c in ((payload.get("coefficients") or []))}
    if coefs:
        named = [r.get("feature") for r in rows if r.get("feature") in coefs]
        if not named:
            named = [features_mentioned(str(r.get("rule", "")), fnames)[:1] for r in rows]
            named = [n[0] for n in named if n and n[0] in coefs]
        for f in named[:int(top_k)]:
            c = coefs[f]
            bits = [f"the fitted coefficient for {f} is {c['coef']:+.{d}f}"]
            if c.get("ci_lo") is not None:
                bits.append(f"with a {int(round(100 * float(payload.get('coefficient_level', 0.95))))}"
                            f"% confidence interval of {c['ci_lo']:+.{d}f} to {c['ci_hi']:+.{d}f}")
            elif c.get("boot_lo") is not None:
                bits.append(f"with a bootstrap interval of {c['boot_lo']:+.{d}f} to "
                            f"{c['boot_hi']:+.{d}f}")
            line = "On the model's own scale, " + ", ".join(bits) + "."
            if c.get("exp_coef") is not None and payload.get("exp_label"):
                lab = str(payload["exp_label"]).lower()
                line += (f" The corresponding {lab} is {c['exp_coef']:.{d}f}"
                         + (f" ({c['exp_lo']:.{d}f} to {c['exp_hi']:.{d}f})."
                            if c.get("exp_lo") is not None else "."))
            S.append(line)
        # Only the validity warning belongs in the narrative. How the standard errors were
        # computed is metadata; it is on the Coefficients tab.
        if payload.get("coefficient_warning"):
            S.append(str(payload["coefficient_warning"]))

    S.append("These attributions describe the fitted model's behaviour and are not by "
             "themselves evidence of a causal effect.")
    text = " ".join(S)
    return text, {"generator": "template", "deterministic": True, "cost_tokens": 0,
                  "method": method, "scope": scope, "top_k": int(top_k)}


def _join_names(names):
    names = [str(n) for n in names]
    if len(names) <= 1:
        return names[0] if names else "none"
    return ", ".join(names[:-1]) + " and " + names[-1]


def llm_stability_analysis(
    *, explain_method, explain_scope, outcome_type, loss_name, model_type,
    user_prompt, shap_payload=None, lime_payload=None,
    temperatures=(0.0, 0.3, 0.7, 1.0), n_reps=10, top_k=3, tol=0.05,
    llm_model=LLM_MODEL_DEFAULT, feature_names=None, progress=None,
    base_url=None, api_key=None, client=None,
):
    """Repeat the identical prompt at each temperature and grade every generation.

    This is the Section 5 evaluation, automated. It is deliberately usable on the
    A-learning global scenario, where Objective Framing Accuracy was the one metric with
    headroom, so a temperature sweep there can actually move.
    """
    scope_local = str(explain_scope).capitalize() == "Local"
    payload = shap_payload if str(explain_method).upper() == "SHAP" else lime_payload
    if payload is None:
        raise ValueError(f"No {explain_method} payload available for the evaluation.")

    # One connection for the whole sweep. A local endpoint is the only configuration in
    # which the temperature knob is reliably honoured, so which endpoint was used is
    # recorded alongside the results rather than assumed.
    if client is None:
        client, base_url, local_endpoint = make_llm_client(base_url, api_key)
    else:
        base_url = base_url or ""
        local_endpoint = is_local_endpoint(base_url)

    ref = payload_reference(payload, explain_method=explain_method,
                            explain_scope=explain_scope, top_k=top_k,
                            feature_names=feature_names)
    truth, shap_signs = ref["truth"], ref["shap_signs"]
    reference_top, payload_features = ref["reference_top"], ref["payload_features"]

    feature_names = list(feature_names or [])
    per_temp, notes, applied_any = [], set(), False
    surfaces, tok_in, tok_out = set(), 0, 0

    for t in temperatures:
        texts = []
        for rep in range(int(n_reps)):
            if progress is not None:
                progress(t, rep)
            _, text, meta = llm_explain(
                explain_method=explain_method, explain_scope=explain_scope,
                outcome_type=outcome_type, loss_name=loss_name, model_type=model_type,
                user_prompt=user_prompt, shap_payload=shap_payload,
                lime_payload=lime_payload, temperature=t, model=llm_model,
                client=client, base_url=base_url,
            )
            applied_any |= bool(meta.get("temperature_applied"))
            if meta.get("temperature_note"):
                notes.add(meta["temperature_note"])
            if meta.get("surface_note"):
                notes.add(meta["surface_note"])
            if meta.get("surface"):
                surfaces.add(meta["surface"])
            u = meta.get("usage") or {}
            tok_in += int(u.get("prompt_tokens") or 0)
            tok_out += int(u.get("completion_tokens") or 0)
            texts.append(text)

        row = _grade_batch(texts, feature_names=feature_names, ref=ref,
                           loss_name=loss_name, top_k=top_k, tol=tol)
        row["temperature"] = float(t)
        per_temp.append(row)

    return {
        "method": explain_method,
        "scope": explain_scope,
        "loss_name": loss_name,
        "model": llm_model,
        "n_reps": int(n_reps),
        "top_k": int(top_k),
        "tolerance": float(tol),
        "reference_top": sorted(reference_top),
        "temperature_applied": applied_any,
        "notes": sorted(notes),
        "rows": per_temp,
        "endpoint": base_url or "hosted API (api.openai.com)",
        "local_endpoint": bool(local_endpoint),
        "surface": "/".join(sorted(surfaces)) if surfaces else "",
        "tokens": ({"prompt": tok_in, "completion": tok_out}
                   if (tok_in or tok_out) else {}),
        "egress": data_egress_note(explain_scope, local_endpoint, base_url or ""),
    }


def llm_model_comparison(*, models, progress=None, base_url=None, api_key=None,
                         client=None, **kw):
    """Run the identical evaluation on two or more models at the same endpoint.

    The comparison this is for is a hosted model against a locally deployed one: whether a
    model an analyst can run themselves reproduces the Section 5 metrics well enough to be
    used in place of a hosted one. The scenario, prompt, payload and grader are held fixed
    so the only thing varying is the generator, and one connection serves every model.
    """
    names = [str(m).strip() for m in models if str(m).strip()]
    if not names:
        raise ValueError("No model names given for the comparison.")

    if client is None:
        client, url, local = make_llm_client(base_url, api_key)
    else:
        url, local = (base_url or ""), is_local_endpoint(base_url)
    out = []
    for name in names:
        def _p(t, rep, _n=name):
            if progress is not None:
                progress(_n, t, rep)
        try:
            res = llm_stability_analysis(llm_model=name, progress=_p, client=client,
                                         base_url=url, **kw)
        except Exception as e:
            # One unavailable model must not discard the results of the others; a local
            # server typically serves a single model, so this is the expected outcome when
            # a name is mistyped or not loaded.
            res = {"model": name, "error": str(e)}
        out.append(res)
    return {"models": out, "endpoint": url or "hosted API (api.openai.com)",
            "local_endpoint": bool(local)}


# =============================================================================
# Is the LLM layer necessary? Two experiments
# =============================================================================
# The first grades the deterministic template against the LLM on the manuscript's own
# metrics. The template is expected to win, and reporting that is the point: it establishes
# that faithful reporting of attributions does not require a language model, which bounds
# what the LLM layer can be claimed to contribute.
#
# The second measures something the metrics above cannot see. Holding the payload fixed and
# varying only the requested audience, a template emits identical text -- it has no capacity
# to adapt -- while an LLM should produce materially different text that remains faithful.
# The control that makes this an argument rather than an anecdote is the comparison of
# between-prompt similarity against within-prompt similarity: if a generator's text differs
# no more across audiences than it does across repeats of the same audience, the difference
# is sampling noise and the adaptation claim fails.

def compare_generators(
    *, explain_method, explain_scope, outcome_type, loss_name, model_type,
    user_prompt, shap_payload=None, lime_payload=None, feature_names=None,
    n_reps=5, top_k=3, tol=0.05, temperature=None, llm_model=LLM_MODEL_DEFAULT,
    base_url=None, api_key=None, client=None, progress=None, feature_note="",
):
    """Grade the template baseline and the LLM on the identical payload and grader.

    The template is generated n_reps times rather than once, so its reproducibility is
    measured on the same footing as the LLM's instead of being asserted.
    """
    method = str(explain_method).upper()
    payload = shap_payload if method == "SHAP" else lime_payload
    ref = payload_reference(payload, explain_method=method, explain_scope=explain_scope,
                           top_k=top_k, feature_names=feature_names)

    t_texts = []
    for _ in range(int(n_reps)):
        txt, _ = template_explain(
            explain_method=method, explain_scope=explain_scope, outcome_type=outcome_type,
            loss_name=loss_name, model_type=model_type, shap_payload=shap_payload,
            lime_payload=lime_payload, feature_names=feature_names, top_k=top_k)
        t_texts.append(txt)
    t_row = _grade_batch(t_texts, feature_names=feature_names, ref=ref,
                         loss_name=loss_name, top_k=top_k, tol=tol)

    if client is None:
        client, base_url, local_endpoint = make_llm_client(base_url, api_key)
    else:
        base_url = base_url or ""
        local_endpoint = is_local_endpoint(base_url)

    l_texts, tok_in, tok_out, notes = [], 0, 0, set()
    for rep in range(int(n_reps)):
        if progress is not None:
            progress("llm", rep)
        _, text, meta = llm_explain(
            explain_method=method, explain_scope=explain_scope, outcome_type=outcome_type,
            loss_name=loss_name, model_type=model_type, user_prompt=user_prompt,
            shap_payload=shap_payload, lime_payload=lime_payload, temperature=temperature,
            model=llm_model, client=client, base_url=base_url, feature_note=feature_note)
        u = meta.get("usage") or {}
        tok_in += int(u.get("prompt_tokens") or 0)
        tok_out += int(u.get("completion_tokens") or 0)
        for k in ("temperature_note", "surface_note"):
            if meta.get(k):
                notes.add(meta[k])
        l_texts.append(text)
    l_row = _grade_batch(l_texts, feature_names=feature_names, ref=ref,
                         loss_name=loss_name, top_k=top_k, tol=tol)

    caveats = []
    if method == "LIME" and str(explain_scope).capitalize() == "Local":
        caveats.append(
            "LIME local scope: the numerical-accuracy truth set contains the surrogate "
            "weights but not the rule thresholds, so a narrative quoting a threshold is "
            "scored as unsupported even though the threshold came from the payload. The "
            "template avoids quoting thresholds and the LLM does not, which biases the "
            "numeric column in the template's favour. Compare on SHAP for the headline "
            "numbers.")
    if not np.isfinite(t_row["directional"]) and not np.isfinite(l_row["directional"]):
        caveats.append(
            "Directional consistency is undefined in Global scope: a mean absolute "
            "attribution carries no sign, so neither generator makes a directional claim. "
            "This is consistent with Table 4 reporting '-' for the global columns.")

    return {
        "method": method, "scope": str(explain_scope).capitalize(), "loss_name": loss_name,
        "n_reps": int(n_reps), "top_k": int(top_k), "tolerance": float(tol),
        "reference_top": sorted(ref["reference_top"]),
        "template": t_row, "llm": l_row,
        "llm_model": llm_model, "temperature": temperature,
        "tokens": {"prompt": tok_in, "completion": tok_out} if (tok_in or tok_out) else {},
        "endpoint": base_url or "hosted API (api.openai.com)",
        "local_endpoint": bool(local_endpoint),
        "template_example": t_texts[0] if t_texts else "",
        "llm_example": l_texts[0] if l_texts else "",
        "notes": sorted(notes), "caveats": caveats,
    }


# Default audiences for the adaptation experiment. Deliberately far apart in register while
# asking for the same content, so that a failure to adapt cannot be blamed on the prompts
# being too similar to distinguish.
ADAPTATION_PROMPTS = [
    ("Molecular tumour board",
     "You are presenting to a molecular tumour board of oncologists and statisticians. "
     "Use precise technical language, name the estimand, and state the attribution "
     "magnitudes. Three to five sentences."),
    ("Plain-language summary",
     "Write for a patient with no statistical training. Use everyday words, avoid all "
     "jargon and symbols, do not state numerical values, and explain only what the model "
     "appears to be responding to. Three to five sentences."),
]


def adaptation_analysis(
    *, explain_method, explain_scope, outcome_type, loss_name, model_type,
    shap_payload=None, lime_payload=None, feature_names=None, prompts=None,
    n_reps=3, top_k=3, tol=0.05, temperature=None, llm_model=LLM_MODEL_DEFAULT,
    base_url=None, api_key=None, client=None, progress=None, feature_note="",
):
    """Vary only the requested audience and measure whether the output adapts.

    Reported per generator: between-prompt similarity, within-prompt similarity, and
    faithfulness for each audience. A template's between/within are both exactly 1.0, so
    its adaptivity is zero by construction; the LLM's claim to adapt requires
    between < within, and its adaptation is only worth anything if faithfulness holds.
    """
    method = str(explain_method).upper()
    payload = shap_payload if method == "SHAP" else lime_payload
    ref = payload_reference(payload, explain_method=method, explain_scope=explain_scope,
                           top_k=top_k, feature_names=feature_names)
    prompts = list(prompts or ADAPTATION_PROMPTS)
    if len(prompts) < 2:
        raise ValueError("The adaptation experiment needs at least two audiences to compare.")

    # The template ignores the request by construction, so its groups are the same text
    # repeated. Generated rather than assumed, so the 1.0 is a measurement.
    t_groups = []
    for _ in prompts:
        g = []
        for _ in range(int(n_reps)):
            txt, _ = template_explain(
                explain_method=method, explain_scope=explain_scope,
                outcome_type=outcome_type, loss_name=loss_name, model_type=model_type,
                shap_payload=shap_payload, lime_payload=lime_payload,
                feature_names=feature_names, top_k=top_k)
            g.append(txt)
        t_groups.append(g)

    if client is None:
        client, base_url, local_endpoint = make_llm_client(base_url, api_key)
    else:
        base_url = base_url or ""
        local_endpoint = is_local_endpoint(base_url)

    l_groups, tok_in, tok_out = [], 0, 0
    for label, ptext in prompts:
        g = []
        for rep in range(int(n_reps)):
            if progress is not None:
                progress(label, rep)
            _, text, meta = llm_explain(
                explain_method=method, explain_scope=explain_scope,
                outcome_type=outcome_type, loss_name=loss_name, model_type=model_type,
                user_prompt=ptext, shap_payload=shap_payload, lime_payload=lime_payload,
                temperature=temperature, model=llm_model, client=client,
                base_url=base_url, feature_note=feature_note)
            u = meta.get("usage") or {}
            tok_in += int(u.get("prompt_tokens") or 0)
            tok_out += int(u.get("completion_tokens") or 0)
            g.append(text)
        l_groups.append(g)

    def _per_audience(groups):
        return [_grade_batch(g, feature_names=feature_names, ref=ref,
                             loss_name=loss_name, top_k=top_k, tol=tol) for g in groups]

    return {
        "method": method, "scope": str(explain_scope).capitalize(), "loss_name": loss_name,
        "audiences": [p[0] for p in prompts], "prompts_used": prompts,
        "n_reps": int(n_reps), "top_k": int(top_k),
        "reference_top": sorted(ref["reference_top"]),
        "template": {"similarity": within_and_between_similarity(t_groups),
                     "per_audience": _per_audience(t_groups),
                     "examples": [g[0] for g in t_groups]},
        "llm": {"similarity": within_and_between_similarity(l_groups),
                "per_audience": _per_audience(l_groups),
                "examples": [g[0] for g in l_groups]},
        "llm_model": llm_model, "temperature": temperature,
        "tokens": {"prompt": tok_in, "completion": tok_out} if (tok_in or tok_out) else {},
        "endpoint": base_url or "hosted API (api.openai.com)",
        "local_endpoint": bool(local_endpoint),
    }


# -----------------------------------------------------------------------------
# Interactive follow-up
# -----------------------------------------------------------------------------
# The capability a template cannot imitate at all: an unanticipated question about the
# explanation, answered in context. A template can only emit the report it was written to
# emit, so there is no baseline to compare against here -- which is the argument.
#
# Interaction widens the surface for unsupported claims, though: a question can invite the
# model past the payload ("is this patient a candidate for immunotherapy?"). So every answer
# is graded by the same faithfulness rules as the narratives, and the grade is shown with
# the answer rather than filed away in an evaluation table.

FOLLOWUP_GUARD = """
You are answering follow-up questions about the explanation above, in conversation.

RULES
- Use ONLY the explanation payload and the task context given above. Do not introduce
  features, biomarkers, genes or numbers that are not in the payload.
- If the question cannot be answered from the payload, say so plainly and say what would be
  needed. A refusal is a correct answer; inventing a plausible one is not.
- Do not give clinical recommendations or infer causal treatment effects.
- Keep answers short, two to four sentences unless asked otherwise.
"""


def llm_followup(
    *, question, history=None, explain_method, explain_scope, outcome_type, loss_name,
    model_type, shap_payload=None, lime_payload=None, feature_names=None, top_k=3, tol=0.05,
    temperature=None, llm_model=LLM_MODEL_DEFAULT, base_url=None, api_key=None, client=None,
    feature_note="", user_prompt="",
):
    """Answer one follow-up question about the current explanation, and grade the answer.

    `history` is the prior [(question, answer), ...] and is replayed so the exchange is a
    conversation rather than a series of unrelated calls. Returns (answer, grade, meta).
    """
    q = str(question or "").strip()
    if not q:
        raise ValueError("No question given.")

    method = str(explain_method).upper()
    context, payload_small, scope = build_explanation_context(
        explain_method=method, explain_scope=explain_scope, outcome_type=outcome_type,
        loss_name=loss_name, model_type=model_type, shap_payload=shap_payload,
        lime_payload=lime_payload, feature_note=feature_note)

    if client is None:
        client, base_url, local_endpoint = make_llm_client(base_url, api_key)
    else:
        base_url = base_url or ""
        local_endpoint = is_local_endpoint(base_url)

    grounding = (f"{context}\n\nEXPLANATION PAYLOAD\n{payload_small}\n"
                 + (f"\nORIGINAL REPORTING INSTRUCTIONS\n{user_prompt}\n" if user_prompt else "")
                 + FOLLOWUP_GUARD)

    messages = [{"role": "user", "content": grounding},
                {"role": "assistant",
                 "content": "Understood. I will answer only from the payload above."}]
    for prev_q, prev_a in (history or []):
        messages.append({"role": "user", "content": str(prev_q)})
        messages.append({"role": "assistant", "content": str(prev_a)})
    messages.append({"role": "user", "content": q})

    text, temp_applied, temp_note, info = _llm_call(client, llm_model, messages, temperature,
                                                    base_url=base_url)

    payload = shap_payload if method == "SHAP" else lime_payload
    ref = payload_reference(payload, explain_method=method, explain_scope=explain_scope,
                           top_k=top_k, feature_names=feature_names)
    grade = grade_narrative(
        text, feature_names=list(feature_names or []), reference_top=ref["reference_top"],
        truth_values=ref["truth"], loss_name=loss_name, shap_signs=ref["shap_signs"],
        top_k=top_k, tol=tol, payload_features=ref["payload_features"])

    meta = {"model": llm_model, "turns": len(history or []) + 1,
            "endpoint": base_url or "hosted API (api.openai.com)",
            "local_endpoint": bool(local_endpoint),
            "surface": info.get("surface"), "usage": info.get("usage", {}),
            "egress": data_egress_note(scope, local_endpoint, base_url or ""),
            "temperature_applied": temp_applied, "temperature_note": temp_note}
    if info.get("surface_note"):
        meta["surface_note"] = info["surface_note"]
    return text, grade, meta


def format_followup_turn(question, answer, grade, meta) -> list:
    """One exchange, with its faithfulness grade attached rather than reported elsewhere."""
    L = [f"YOU: {question}", "", f"MODEL ({meta.get('model', '?')}):", answer, ""]
    flags = []
    if grade["num_unmatched"]:
        flags.append("numbers not traceable to the payload: "
                     + ", ".join(str(round(u, 4)) for u in grade["num_unmatched"][:6]))
    if grade["not_in_payload"]:
        flags.append("features not shown to the model: " + ", ".join(grade["not_in_payload"][:6]))
    if grade["out_of_vocabulary"]:
        flags.append("invented biomarker-like tokens: "
                     + ", ".join(grade["out_of_vocabulary"][:6]))
    L.append("  faithfulness: " + ("clean -- every claim traces to the payload"
                                   if not flags else "; ".join(flags)))
    u = meta.get("usage") or {}
    if u:
        L.append(f"  tokens: {u.get('prompt_tokens', '?')} prompt, "
                 f"{u.get('completion_tokens', '?')} completion")
    L.append("-" * 88)
    return L


def format_llm_stability(res: Optional[Dict[str, Any]]) -> list:
    if not res:
        return []
    if "error" in res:
        return ["=== LLM stability ===", f"Failed: {res['error']}"]

    L = ["=== LLM narrative stability (reproducibility, faithfulness, sensitivity) ===",
         f"Scenario: {res['method']} {res['scope']} explanation under {res['loss_name']}",
         f"Model: {res['model']} | {res['n_reps']} generations per temperature",
         f"Reference top-{res['top_k']} from the payload: {', '.join(res['reference_top'])}",
         f"Numerical tolerance: +/-{res['tolerance']}"]
    if res.get("endpoint"):
        line = f"Endpoint: {res['endpoint']}"
        if res.get("surface"):
            line += f" via {res['surface']}"
        if res.get("local_endpoint"):
            line += " (local deployment)"
        L.append(line)
    if res.get("tokens"):
        L.append(f"Tokens: {res['tokens'].get('prompt', 0)} prompt, "
                 f"{res['tokens'].get('completion', 0)} completion, over the whole sweep")
    if res.get("egress"):
        L.append(res["egress"])
    if not res["temperature_applied"]:
        L.append("WARNING: the temperature setting was NOT applied by the API, so the rows "
                 "below differ only by sampling noise at the model default. Any temperature "
                 "sensitivity claim must be labelled accordingly.")
    for n in res["notes"]:
        L.append(f"  note: {n}")
    if res["rows"]:
        L.append(f"Similarity backend: {res['rows'][0]['similarity_backend']}")
    L.append("")

    hdr = (f"{'temp':>5s} {'top' + str(res['top_k']) + '_agree':>10s} {'framing':>8s} "
           f"{'direction':>10s} {'numeric':>8s} {'halluc':>7s} {'similarity':>11s} "
           f"{'distinct':>9s}")
    L.append(hdr)
    L.append("-" * len(hdr))

    def f(v, w, d=3):
        return f"{v:>{w}.{d}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"

    for r in res["rows"]:
        L.append(f"{r['temperature']:>5.2f} {f(r['top_agreement'], 10)} {f(r['framing'], 8)} "
                 f"{f(r['directional'], 10)} {f(r['numeric_accuracy'], 8)} "
                 f"{f(r['hallucinations'], 7, 2)} {f(r['similarity'], 11)} "
                 f"{r['distinct_top_sets']:>9d}")

    L.append("")
    L.append("Metric definitions and provenance")
    L.append(f"  top{res['top_k']}_agree  Eq. (4): (1/N) sum_i I(S_i = S*), S* from the SHAP output.")
    L.append("  framing     Section 5.1.1 rule: proportion of narratives using the vocabulary")
    L.append("              of the estimand the objective actually targets.")
    L.append("  direction   Eq. (5): (1/N) sum_i D_i/F_i, a mean of per-narrative ratios.")
    L.append("  numeric     Eq. (7): (1/N) sum_i A_i/C_i, a mean of per-narrative ratios.")
    L.append("  halluc      Eq. (8): (1/N) sum_i H_i, mean count per narrative.")
    L.append("  similarity  Eq. (6): mean pairwise cosine, 2/(N(N-1)) sum_{i<j}.")
    L.append("")
    L.append("Deviations from the manuscript, stated explicitly:")
    L.append("  - S_i, the directional claims, and the numerical claims are extracted by the")
    L.append("    documented rules in this file rather than by human reading. S_i is taken as")
    L.append("    the first K distinct feature names mentioned; a numerical claim counts as")
    L.append("    accurate if it falls within tolerance of ANY payload quantity, rather than of")
    L.append("    its intended quantity. Inspect the unmatched values listed below.")
    if res["rows"] and "TF-IDF" in res["rows"][0]["similarity_backend"]:
        L.append("  - Eq. (6) is computed with TF-IDF, not all-MiniLM-L6-v2. Install")
        L.append("    sentence-transformers to reproduce the manuscript's embedding.")
    L.append("")
    pooled = [(r["temperature"], r["directional_pooled"], r["numeric_pooled"])
              for r in res["rows"]]
    L.append("Pooled alternatives (sum_i D_i / sum_i F_i and sum_i A_i / sum_i C_i), which "
             "weight narratives by claim count and are NOT the manuscript's estimator:")
    for t, dp, npd in pooled:
        L.append(f"  temp {t:.2f}: direction {f(dp, 6)}  numeric {f(npd, 6)}")
    L.append("")
    L.append("top-K agreement / framing / direction / numeric are proportions; halluc is the "
             "mean count of unsupported claims per narrative; distinct is the number of "
             "different top-K sets seen across the generations.")
    fr = [r["framing"] for r in res["rows"] if np.isfinite(r["framing"])]
    if fr and min(fr) < 1.0:
        L.append(f"Objective framing accuracy ranges {min(fr):.3f} to {max(fr):.3f} across "
                 f"temperatures -- this is the metric with headroom in this scenario.")
    ex = [r for r in res["rows"] if r["unmatched_examples"]]
    if ex:
        L.append("")
        L.append("Unmatched numeric claims (inspect these; the extraction is heuristic):")
        for r in ex:
            L.append(f"  temp {r['temperature']:.2f}: {r['unmatched_examples']}")
    uf = [r for r in res["rows"] if r.get("unsupported_features")]
    if uf:
        L.append("")
        L.append("Features named that the payload did not contain:")
        for r in uf:
            L.append(f"  temp {r['temperature']:.2f}: {', '.join(r['unsupported_features'])}")
    return L


def format_llm_comparison(res: Optional[Dict[str, Any]]) -> list:
    """Side-by-side Section 5 metrics for two or more generators, then each in full."""
    if not res:
        return []
    if "error" in res and "models" not in res:
        return ["=== LLM model comparison ===", f"Failed: {res['error']}"]

    runs = res.get("models", [])
    ok = [r for r in runs if "error" not in r]
    L = ["=== LLM model comparison (same scenario, same payload, same grader) ===",
         f"Endpoint: {res.get('endpoint', '')}"
         + (" (local deployment)" if res.get("local_endpoint") else ""),
         f"Models requested: {', '.join(r.get('model', '?') for r in runs)}"]
    for r in runs:
        if "error" in r:
            L.append(f"  {r.get('model', '?')}: FAILED -- {r['error']}")
    if not ok:
        return L + ["", "No model produced results."]

    k = ok[0]["top_k"]
    L += ["", "Averaged over the temperatures swept, per model:", ""]
    hdr = (f"{'model':<28s} {'top' + str(k) + '_agree':>10s} {'framing':>8s} "
           f"{'direction':>10s} {'numeric':>8s} {'halluc':>7s} {'similarity':>11s} "
           f"{'temp?':>6s} {'tokens':>8s}")
    L += [hdr, "-" * len(hdr)]

    def m(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(v)) if v else np.nan

    def f(v, w, d=3):
        return f"{v:>{w}.{d}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"

    for r in ok:
        rows = r["rows"]
        tok = (r.get("tokens") or {})
        tot = int(tok.get("prompt", 0)) + int(tok.get("completion", 0))
        L.append(f"{str(r['model'])[:28]:<28s} {f(m(rows, 'top_agreement'), 10)} "
                 f"{f(m(rows, 'framing'), 8)} {f(m(rows, 'directional'), 10)} "
                 f"{f(m(rows, 'numeric_accuracy'), 8)} {f(m(rows, 'hallucinations'), 7, 2)} "
                 f"{f(m(rows, 'similarity'), 11)} "
                 f"{('yes' if r.get('temperature_applied') else 'no'):>6s} "
                 f"{(str(tot) if tot else '-'):>8s}")

    L += ["", "temp? is whether the endpoint honoured an explicit temperature. Where it did "
              "not, that model's rows differ only by sampling noise at the server default, "
              "so its similarity column is not comparable with a model where it did.",
          ""]
    if len(ok) > 1:
        ref = ok[0]
        L.append(f"Read against {ref['model']}: a locally deployed model is usable in place of "
                 "a hosted one for this step only if top-K agreement and framing hold up, "
                 "since those are what the narrative is being trusted for.")
        for r in ok[1:]:
            d_top = m(r["rows"], "top_agreement") - m(ref["rows"], "top_agreement")
            d_fr = m(r["rows"], "framing") - m(ref["rows"], "framing")
            d_h = m(r["rows"], "hallucinations") - m(ref["rows"], "hallucinations")
            L.append(f"  {r['model']}: top-{k} agreement {d_top:+.3f}, framing {d_fr:+.3f}, "
                     f"hallucinations {d_h:+.2f} per narrative")
        L.append("")

    for r in ok:
        L += [f"----- {r['model']} -----"] + format_llm_stability(r) + [""]
    return L


# =============================================================================
# Exporting intermediate results
# =============================================================================
# Everything this app computes was previously rendered to the browser and then discarded;
# only the figures reached disk, and on a deployed instance they land on the server where the
# user cannot get at them. That makes downstream work -- comparing attributions across
# models, re-plotting in R, pooling several runs -- impossible without re-deriving the
# numbers by hand.
#
# Two principles here. Attributions are exported in tidy long form as well as the wide
# matrix, because long form is what a plotting or modelling pipeline actually wants. And
# every bundle carries a run manifest: the configuration, the seeds, the tuning outcome, the
# preprocessing decisions and the package versions. A CSV of SHAP values with no record of
# what produced it is not a reproducible intermediate result, it is a table of numbers.

EXPORT_README = """\
Results exported from the SHAP / LIME / LLM interpretability app.

manifest.json         Everything needed to identify this run: data shape, columns, model,
                      objective, penalties, tuning outcome, preprocessing, seeds, and the
                      versions of every package involved. Read this first.
shap_long.csv         Tidy attributions: one row per (observation, feature), with the
                      observed covariate value and its SHAP value. Use this for plotting.
shap_wide.csv         The same attributions as an observation x feature matrix.
shap_importance.csv   Per-feature mean |SHAP|, mean signed SHAP, and rank.
lime_global.csv       Mean absolute LIME weight per feature, aggregated over instances.
lime_local.csv        The selected patient's LIME rules and weights.
coefficients.csv      For a parametric model: estimate, standard error, interval, p-value,
                      and the odds or hazard ratio. Bootstrap columns appear where a
                      conventional interval is not valid. Absent for tree models.
model_comparison.csv  Attribution rankings across model types, where that was run.
sensitivity_*.csv     Stability, baseline and adaptation results, where those were run.
narratives.csv        LLM prompts, generated text, and the faithfulness grade of each.
figures/              The SHAP and LIME figures as PNG.

Scale warning for cross-model comparison: SHAP values are on the scale of the model's own
output -- log-odds for XGBoost binary:logistic, probability for a random forest classifier,
outcome units for a linear model. Raw magnitudes are therefore NOT comparable between model
types. Compare ranks, or the normalised share column, both of which are provided.
"""


def _package_versions():
    """Versions of everything that can change a number in this export."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name, mod in (("numpy", np), ("pandas", pd), ("scipy", sp_stats),
                      ("xgboost", xgb), ("shap", shap)):
        try:
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unknown"
    for name in ("sklearn", "lime", "shiny", "lifelines", "openai", "matplotlib",
                 "sentence_transformers"):
        try:
            out[name] = str(__import__(name).__version__)
        except Exception:
            out[name] = "not installed"
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except Exception:
        pass
    return out


def export_shap_frames(shap_values, X_plot, feature_names=None):
    """Tidy, wide and per-feature-importance views of the attributions."""
    if shap_values is None or X_plot is None:
        return {}
    feature_names = [str(f) for f in (feature_names if feature_names is not None
                                      else X_plot.columns)]
    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim == 1:
        sv = sv.reshape(-1, 1)
    Xv = np.asarray(X_plot[feature_names].values, dtype=float)
    idx = np.asarray(X_plot.index)

    long_rows = pd.DataFrame({
        "observation": np.repeat(idx, len(feature_names)),
        "feature": np.tile(feature_names, len(idx)),
        "feature_value": Xv.reshape(-1),
        "shap_value": sv.reshape(-1),
    })
    wide = pd.DataFrame(sv, columns=feature_names, index=idx)
    wide.index.name = "observation"

    mean_abs = np.abs(sv).mean(axis=0)
    total = float(mean_abs.sum())
    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": sv.mean(axis=0),
        # Share of total attribution: dimensionless, so it survives comparison between
        # models whose outputs are on different scales.
        "share_of_total": (mean_abs / total if total > 0 else np.full(len(mean_abs), np.nan)),
    }).sort_values("mean_abs_shap", ascending=False)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    return {"shap_long": long_rows, "shap_wide": wide.reset_index(),
            "shap_importance": imp}


def export_lime_frames(lime_payload):
    if not lime_payload:
        return {}
    out = {}
    g = lime_payload.get("global_mean_abs")
    if g:
        gf = pd.DataFrame(g)
        gf.insert(0, "rank", np.arange(1, len(gf) + 1))
        out["lime_global"] = gf
    loc = lime_payload.get("local_top_rules")
    if loc:
        lf = pd.DataFrame(loc)
        lf.insert(0, "patient_index", lime_payload.get("patient_index"))
        out["lime_local"] = lf
    return out


def export_coefficient_frame(summary, boot=None):
    """Estimates with whichever interval is defensible for this fit."""
    if not summary or not summary.get("rows"):
        return {}
    bmap = {r["feature"]: r for r in (boot or {}).get("rows", [])}
    rows = []
    for r in summary["rows"]:
        row = dict(r)
        row["interval_valid"] = bool(summary["valid"])
        row["interval_source"] = (summary.get("inference_prose") or summary.get("inference")
                                  if summary["valid"]
                                  else ((boot or {}).get("method", "none available")))
        if not summary["valid"]:
            row["interval_withheld_because"] = summary.get("reason", "")
        row.update({k: v for k, v in (bmap.get(r["feature"]) or {}).items()
                    if k != "feature"})
        rows.append(row)
    df = pd.DataFrame(rows)
    lead = [c for c in ("feature", "coef", "se", "ci_lo", "ci_hi", "p", "exp_coef",
                        "exp_lo", "exp_hi", "boot_se", "boot_lo", "boot_hi") if c in df]
    return {"coefficients": df[lead + [c for c in df.columns if c not in lead]]}


def export_metrics_frame(metrics, extra=None):
    rows = [{"quantity": k, "value": v} for k, v in (metrics or {}).items()
            if not isinstance(v, (list, dict, np.ndarray))]
    tun = ((extra or {}).get("tuning") or {})
    for k, v in tun.items():
        if not isinstance(v, (list, dict, np.ndarray)):
            rows.append({"quantity": f"tuning.{k}", "value": v})
    return {"metrics": pd.DataFrame(rows)} if rows else {}


def _flatten_rows(rows, prefix=""):
    out = []
    for r in rows or []:
        flat = {}
        for k, v in r.items():
            if isinstance(v, (list, tuple, set, frozenset)):
                flat[k] = "; ".join(str(x) for x in v)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        out.append({**({"analysis": prefix} if prefix else {}), **flat})
    return out


def export_sensitivity_frames(sens):
    """Whichever sensitivity analyses were actually run, one frame each."""
    if not sens:
        return {}
    out = {}
    llm = sens.get("llm")
    if llm and "rows" in llm:
        out["sensitivity_llm_temperature"] = pd.DataFrame(_flatten_rows(llm["rows"]))
    lime = sens.get("lime")
    if lime and isinstance(lime, dict) and lime.get("rows"):
        out["sensitivity_lime"] = pd.DataFrame(_flatten_rows(lime["rows"]))
    cmp_ = sens.get("llm_compare")
    if cmp_ and cmp_.get("models"):
        rows = []
        for m in cmp_["models"]:
            if "rows" in m:
                for r in _flatten_rows(m["rows"]):
                    rows.append({"model": m.get("model"), **r})
        if rows:
            out["sensitivity_llm_models"] = pd.DataFrame(rows)
    base = sens.get("baseline")
    if base and "template" in base:
        rows = []
        for gen in ("template", "llm"):
            rows.append({"generator": gen, **{k: v for k, v in base[gen].items()
                                              if not isinstance(v, (list, dict))}})
        out["sensitivity_template_baseline"] = pd.DataFrame(rows)
    ad = sens.get("adaptation")
    if ad and "audiences" in ad:
        rows = []
        for gen in ("template", "llm"):
            sim = ad[gen]["similarity"]
            for i, aud in enumerate(ad["audiences"]):
                per = ad[gen]["per_audience"][i]
                rows.append({"generator": gen, "audience": aud,
                             "within_similarity": sim.get("within"),
                             "between_similarity": sim.get("between"),
                             **{k: v for k, v in per.items()
                                if not isinstance(v, (list, dict))}})
        out["sensitivity_adaptation"] = pd.DataFrame(rows)
    return out


def export_narrative_frame(narratives):
    """The generated text and its grade, so the evaluation can be audited externally."""
    if not narratives:
        return {}
    rows = []
    for n in narratives:
        g = n.get("grade") or {}
        rows.append({
            "kind": n.get("kind", ""),
            "generator": n.get("generator", ""),
            "question": n.get("question", ""),
            "text": n.get("text", ""),
            "top_set": "; ".join(sorted(g.get("top_set", []) or [])),
            "top_exact": g.get("top_exact"),
            "framing_ok": g.get("framing_ok"),
            "dir_correct": g.get("dir_correct"),
            "dir_total": g.get("dir_total"),
            "num_matched": g.get("num_matched"),
            "num_total": g.get("num_total"),
            "hallucinations": g.get("hallucinations"),
            "numbers_not_in_payload": "; ".join(str(x) for x in (g.get("num_unmatched") or [])),
            "features_not_in_payload": "; ".join(g.get("not_in_payload") or []),
            "invented_tokens": "; ".join(g.get("out_of_vocabulary") or []),
            "prompt": n.get("prompt", ""),
        })
    return {"narratives": pd.DataFrame(rows)}


def run_manifest(*, st, config=None, figures=None):
    """Everything needed to say what produced these numbers."""
    extra = (st or {}).get("extra") or {}
    Xtr, Xte = st.get("X_train"), st.get("X_test")
    prep = extra.get("prep") or {}
    return {
        "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "app_file": os.path.basename(__file__) if "__file__" in globals() else "app.py",
        "model": {
            "model_type": st.get("model_type"),
            "loss_name": st.get("loss_name"),
            "outcome_type": st.get("outcome_type"),
            "penalty_used": extra.get("penalty_used"),
            "benefit_sign": extra.get("benefit_sign"),
            "benefit_note": extra.get("benefit_note"),
        },
        "data": {
            "n_rows_uploaded": int(len(st["df"])) if st.get("df") is not None else None,
            "n_train": int(len(Xtr)) if Xtr is not None else None,
            "n_test": int(len(Xte)) if Xte is not None else None,
            "features": [str(c) for c in (Xtr.columns if Xtr is not None else [])],
            "n_features": int(Xtr.shape[1]) if Xtr is not None else None,
            "explanation_set_rows": (int(len(st["X_plot_used"]))
                                     if st.get("X_plot_used") is not None else None),
        },
        "preprocessing": prep,
        "tuning": extra.get("tuning"),
        "metrics": {k: v for k, v in (st.get("metrics") or {}).items()
                    if not isinstance(v, (list, dict, np.ndarray))},
        "config": config or {},
        "figures": sorted(figures or []),
        "versions": _package_versions(),
        "caveats": [
            "SHAP values are on the scale of the model's own output, which differs between "
            "model types; compare ranks or share_of_total, not raw magnitudes.",
            "The explanation set is the test split truncated to the first 200 rows.",
            ("Coefficient intervals are withheld where the fit is penalised or the objective "
             "is modified; see coefficients.csv."),
        ],
    }


def build_export_bundle(frames, manifest, figure_paths=None, readme=EXPORT_README):
    """filename -> bytes, ready to write individually or zip."""
    bundle = {"README.txt": readme.encode("utf-8"),
              "manifest.json": json.dumps(manifest, indent=2, default=str).encode("utf-8")}
    for name, df in (frames or {}).items():
        if df is None or not len(df):
            continue
        bundle[f"{name}.csv"] = df.to_csv(index=False).encode("utf-8")
    # save_shap_plots returns {label: path}; a plain list is accepted too so that callers
    # do not have to care which shape they are holding.
    paths = figure_paths or []
    if isinstance(paths, dict):
        paths = list(paths.values())
    for p in paths:
        if not p:
            continue
        try:
            with open(p, "rb") as fh:
                bundle[f"figures/{os.path.basename(p)}"] = fh.read()
        except Exception:
            continue
    return bundle


def bundle_to_zip(bundle):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in bundle.items():
            z.writestr(name, data)
    return buf.getvalue()


# =============================================================================
# Attributions across model types
# =============================================================================
# The reviewer's example -- compare feature importance between two model families -- needs
# one methodological guard, stated in the output rather than buried: SHAP values carry the
# units of the model's own output. XGBoost's binary:logistic attributions are log-odds, a
# random forest classifier's are probabilities, a linear model's are outcome units. Raw
# magnitudes are not comparable across families, so agreement is measured on ranks and on
# each feature's share of total attribution, both dimensionless.

def compare_model_attributions(
    df, feature_cols, *, outcome_col, treat_col, outcome_type, loss_name, model_types,
    event_col=None, sigpos_col=None, test_size=0.25, seed=42, top_k=3, n_plot=200,
    na_policy="Drop rows with any missing value", standardize=False, progress=None,
    **penalties
):
    """Fit each model type on the identical split and compare its attribution ranking.

    Models that cannot express this objective are reported as unavailable with the reason,
    not silently dropped -- a random forest has no gradient interface for A-/W-learning, and
    that refusal is itself part of the answer about which models the framework covers.
    """
    feature_cols = list(feature_cols)
    per_model, unavailable = {}, {}
    # save_shap_plots is the only route to the attributions, and it always writes figures.
    # They are a by-product here, so they go to a scratch directory that is deleted rather
    # than left cluttering outputs/ with one set per model.
    scratch = tempfile.mkdtemp(prefix="shap_cmp_")

    for mt in model_types:
        if progress is not None:
            progress(mt)
        try:
            fitted = fit_model(
                df, feature_cols, outcome_col=outcome_col, treat_col=treat_col,
                outcome_type=outcome_type, event_col=event_col, model_type=mt,
                loss_name=loss_name, test_size=test_size, seed=seed,
                sigpos_col=sigpos_col, tune_mode="None", na_policy=na_policy,
                standardize=standardize, **penalties)
            model, X_train, X_test = fitted[0], fitted[1], fitted[2]
            metrics = fitted[5]
            X_plot = (X_test if len(X_test) else X_train).copy().reset_index(drop=True)
            if len(X_plot) > int(n_plot):
                X_plot = X_plot.iloc[:int(n_plot)]
            _, _, sv, X_used = save_shap_plots(
                model=model, X_plot=X_plot, out_dir=scratch,
                prefix=f"cmp_{re.sub(r'[^A-Za-z0-9]+', '', mt)}",
                dep_main="__AUTO_TOP__", dep_interaction="none", patient_idx=0,
                model_type=mt, loss_name=loss_name, X_background=X_train,
                outcome_type=outcome_type)
            imp = export_shap_frames(sv, X_used, feature_names=feature_cols)["shap_importance"]
            per_model[mt] = {"importance": imp, "metrics": metrics,
                             "n_explained": int(len(X_used))}
        except Exception as e:
            unavailable[mt] = str(e)

    shutil.rmtree(scratch, ignore_errors=True)

    if not per_model:
        raise ValueError("No model type could be fitted for this configuration. "
                         + "; ".join(f"{k}: {v[:90]}" for k, v in unavailable.items()))

    names = list(per_model)
    combined = pd.DataFrame({"feature": feature_cols})
    for mt in names:
        imp = per_model[mt]["importance"].set_index("feature")
        combined[f"mean_abs_shap[{mt}]"] = combined["feature"].map(imp["mean_abs_shap"])
        combined[f"share[{mt}]"] = combined["feature"].map(imp["share_of_total"])
        combined[f"rank[{mt}]"] = combined["feature"].map(imp["rank"])
    combined = combined.sort_values(f"rank[{names[0]}]").reset_index(drop=True)

    agreement = []
    for a, b in itertools.combinations(names, 2):
        ra = combined[f"rank[{a}]"].values.astype(float)
        rb = combined[f"rank[{b}]"].values.astype(float)
        ta = frozenset(per_model[a]["importance"]["feature"].head(int(top_k)))
        tb = frozenset(per_model[b]["importance"]["feature"].head(int(top_k)))
        agreement.append({
            "model_a": a, "model_b": b,
            "spearman_rank_corr": _spearman(-ra, -rb),
            f"top{int(top_k)}_jaccard": _jaccard(ta, tb),
            f"top{int(top_k)}_identical": ta == tb,
            f"top{int(top_k)}_a": ", ".join(per_model[a]["importance"]["feature"].head(int(top_k))),
            f"top{int(top_k)}_b": ", ".join(per_model[b]["importance"]["feature"].head(int(top_k))),
        })

    return {
        "outcome_type": outcome_type, "loss_name": loss_name, "top_k": int(top_k),
        "models": names, "unavailable": unavailable,
        "combined": combined, "agreement": agreement,
        "per_model": {mt: {"metrics": per_model[mt]["metrics"],
                           "n_explained": per_model[mt]["n_explained"]}
                      for mt in names},
    }


def format_model_comparison(res):
    """Rankings side by side, with the scale caveat stated before the numbers."""
    if not res:
        return []
    if "error" in res:
        return ["=== Attributions across model types ===", f"Failed: {res['error']}"]

    k = res["top_k"]
    L = ["=== Attributions across model types ===",
         f"Outcome: {res['outcome_type']} | Objective: {res['loss_name']}",
         f"Models compared: {', '.join(res['models'])}"]
    for mt, why in (res.get("unavailable") or {}).items():
        L.append(f"  {mt}: UNAVAILABLE -- {why[:150]}")
    L += ["",
          "Scale caveat, read before comparing magnitudes: a SHAP value is on the scale of",
          "the model's own output -- log-odds for XGBoost binary:logistic, probability for a",
          "random forest classifier, outcome units for a linear model. The mean|SHAP| columns",
          "are therefore NOT comparable between models. The share and rank columns are",
          "dimensionless and are what the agreement statistics below use.",
          ""]

    c = res["combined"]
    names = res["models"]
    hdr = f"{'feature':<10s}" + "".join(f"{('share[' + n[:9] + ']'):>18s}" for n in names) \
                              + "".join(f"{('rank[' + n[:9] + ']'):>17s}" for n in names)
    L += [hdr, "-" * len(hdr)]
    for _, row in c.iterrows():
        line = f"{str(row['feature'])[:10]:<10s}"
        for n in names:
            v = row[f"share[{n}]"]
            line += (f"{v:>18.4f}" if pd.notna(v) else f"{'-':>18s}")
        for n in names:
            v = row[f"rank[{n}]"]
            line += (f"{int(v):>17d}" if pd.notna(v) else f"{'-':>17s}")
        L.append(line)

    L += ["", "Agreement between models:", ""]
    for a in res["agreement"]:
        k_j = f"top{k}_jaccard"
        L += [f"  {a['model_a']} vs {a['model_b']}:",
              f"    Spearman rank correlation of importances: "
              + (f"{a['spearman_rank_corr']:.3f}" if np.isfinite(a['spearman_rank_corr'])
                 else "-"),
              f"    top-{k} Jaccard: {a[k_j]:.3f}"
              + ("  (identical sets)" if a[f'top{k}_identical'] else ""),
              f"    top-{k} {a['model_a']}: {a[f'top{k}_a']}",
              f"    top-{k} {a['model_b']}: {a[f'top{k}_b']}"]
    L += ["",
          "How to read this: high rank agreement means the choice of model family does not",
          "change which covariates the explanation points at, so the substantive conclusion",
          "is robust to that choice. Disagreement is a finding, not a bug -- it means the",
          "families are fitting different structure, and the ranking should not be reported",
          "as though it were a property of the data alone.",
          "",
          "Per-model fit quality (so a ranking from a badly fitting model is not over-read):"]
    for mt, d in res["per_model"].items():
        keep = {kk: vv for kk, vv in (d["metrics"] or {}).items()
                if isinstance(vv, (int, float)) and np.isfinite(vv)}
        L.append(f"  {mt}: " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in keep.items())
                 + f"  (explained rows: {d['n_explained']})")
    return L


def _fmt(v, w, d=3):
    return f"{v:>{w}.{d}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"


def format_coefficients(summary, boot=None, reconcile=None) -> list:
    """The conventional summary, its validity, and its relationship to the attributions."""
    if not summary:
        return ["=== Coefficients ===",
                "Run the analysis with Linear, Logistic or Cox Regression to see a "
                "conventional parameter summary. XGBoost has no coefficients; that is the "
                "case where an attribution method is the only option."]
    if "error" in summary:
        return ["=== Coefficients ===", f"Failed: {summary['error']}"]

    s = summary
    lvl = int(round(100 * s["level"]))
    L = [f"=== Conventional parameter summary: {s['model_type']} under {s['loss_name']} ===",
         f"n = {s['n']} training rows | intercept = {s['intercept']:+.4f}"]
    if s.get("penalty_used") is not None:
        pu = float(s["penalty_used"])
        lab = _penalty_label(s["model_type"], s["loss_name"])
        if lab == "C" and pu >= LOGREG_C_NO_PENALTY / 1e3:
            L.append("Penalty: none. These are the unpenalised maximum-likelihood estimates, "
                     "which is what makes the intervals below the conventional ones.")
        elif pu == 0.0:
            L.append(f"Penalty: none ({lab} = 0).")
        else:
            L.append(f"Penalty in force: {lab} = {pu:g}")
    if s["valid"]:
        L.append(f"Inference: {s['inference']}")
    else:
        L += ["Conventional intervals are NOT reported here, because:",
              "  " + s["reason"]]
        if boot and "rows" in boot:
            L.append(f"  Bootstrap used instead: {boot['method']}, {boot['n_ok']} of "
                     f"{boot['n_boot']} replicates fitted"
                     + (f" ({boot['n_failed']} failed)" if boot.get("n_failed") else ""))
    L.append("")

    has_wald = s["valid"] and any("se" in r for r in s["rows"])
    bmap = {r["feature"]: r for r in (boot or {}).get("rows", [])}
    exp_lab = s.get("exp_label") or ""

    hdr = f"{'feature':<10s}{'coef':>10s}"
    if has_wald:
        hdr += f"{'SE':>8s}{f'{lvl}% CI':>20s}{'p':>10s}"
    if bmap:
        hdr += f"{'boot SE':>9s}{f'boot {lvl}% CI':>22s}"
    if exp_lab:
        hdr += f"{exp_lab:>13s}" + (f"{f'{lvl}% CI':>20s}" if (has_wald or bmap) else "")
    L += [hdr, "-" * len(hdr)]

    def _ci(lo, hi, width, dec=3, signed=True):
        sign = "+" if signed else ""
        return format(f"[{lo:{sign}.{dec}f}, {hi:{sign}.{dec}f}]", f">{width}s")

    for r in s["rows"]:
        line = f"{r['feature'][:10]:<10s}{r['coef']:>+10.4f}"
        if has_wald:
            line += (f"{r['se']:>8.3f}" + _ci(r["ci_lo"], r["ci_hi"], 20)
                     + f"{r['p']:>10.2e}")
        b = bmap.get(r["feature"]) if bmap else None
        if bmap:
            line += ((f"{b['boot_se']:>9.3f}" + _ci(b["boot_lo"], b["boot_hi"], 22))
                     if b else f"{'-':>9s}{'-':>22s}")
        if exp_lab:
            ec = r.get("exp_coef")
            line += (f"{ec:>13.3f}" if ec is not None else f"{'-':>13s}")
            if has_wald and r.get("exp_lo") is not None:
                line += _ci(r["exp_lo"], r["exp_hi"], 20, signed=False)
            elif bmap:
                line += (_ci(float(np.exp(b["boot_lo"])), float(np.exp(b["boot_hi"])), 20,
                             signed=False) if b else f"{'-':>20s}")
        L.append(line)

    if exp_lab:
        L += ["", f"{exp_lab} is exp(coef): the multiplicative change in "
                  + ("the odds of the outcome" if exp_lab.startswith("Odds")
                     else "the hazard")
                  + " per one-unit increase in the covariate, holding the others fixed."]
    if boot and bmap:
        L.append("The bootstrap interval is the percentile interval over subject resamples. "
                 "It is the interval to quote for this fit; where a Wald column is also "
                 "shown, prefer the bootstrap.")

    if reconcile:
        L += ["", "=" * 88,
              "How this relates to the SHAP attributions", "=" * 88]
        if reconcile["identity_holds"]:
            L.append(f"Verified on this fit: phi_ij = coef_j x (x_ij - mean_j) exactly "
                     f"(largest deviation {reconcile['max_abs_deviation']:.2e}). SHAP is not "
                     f"an approximation of the coefficients here, it is an algebraic "
                     f"re-expression of them, so it carries no additional inferential "
                     f"content. Aggregated, mean|SHAP_j| = |coef_j| x mean|x_j - mean_j| "
                     f"(deviation {reconcile['aggregate_identity_deviation']:.2e}).")
        else:
            L.append(f"WARNING: the identity phi_ij = coef_j (x_ij - mean_j) does NOT hold on "
                     f"this fit (largest deviation {reconcile['max_abs_deviation']:.3e}). The "
                     f"attributions and the coefficients disagree, which should not happen "
                     f"for a linear model -- treat both with suspicion until resolved.")
        L += ["",
              "What differs is the ordering, because each coefficient is weighted by how far",
              "its covariate actually moves in this population:", ""]
        h2 = (f"{'feature':<10s}{'coef':>10s}{'|coef|':>9s}{'mean|x-mean|':>14s}"
              f"{'mean|SHAP|':>12s}{'rank_SHAP':>11s}{'rank_|coef|':>12s}")
        L += [h2, "-" * len(h2)]
        for r in reconcile["rows"]:
            L.append(f"{r['feature'][:10]:<10s}{r['coef']:>+10.4f}{r['abs_coef']:>9.4f}"
                     f"{r['mean_abs_dev']:>14.4f}{r['mean_abs_shap']:>12.4f}"
                     f"{r['rank_shap']:>11d}{r['rank_beta']:>12d}")
        k = reconcile["top_k"]
        L += ["",
              f"Top-{k} by mean|SHAP|: {', '.join(reconcile['top_by_shap'])}",
              f"Top-{k} by |coef|   : {', '.join(reconcile['top_by_abs_coef'])}",
              ("The two orderings agree here." if reconcile["rankings_agree"] else
               "The two orderings DISAGREE. Neither is wrong: a coefficient is an effect per "
               "unit of the covariate, while mean|SHAP| is the contribution actually realised "
               "across this population, which is the coefficient scaled by the covariate's "
               "spread. A binary indicator can carry a large coefficient and a small "
               "mean|SHAP| because it barely moves.")]
        lm = reconcile["largest_move"]
        if lm:
            L.append(f"Largest disagreement: {lm['feature']} is rank {lm['rank_beta']} by "
                     f"|coef| and rank {lm['rank_shap']} by mean|SHAP| (mean|x - mean| = "
                     f"{lm['mean_abs_dev']:.4f}).")
        L += ["",
              "Practical consequence for this framework: report the coefficient summary as",
              "the primary result whenever the model has one. The attribution layer earns its",
              "place in two situations -- a model with no coefficients at all (XGBoost), and",
              "an objective modified for treatment-effect estimation, where the coefficient",
              "exists but no conventional interval does and the bootstrap above is required."]
    return L


def format_generator_comparison(res: Optional[Dict[str, Any]]) -> list:
    """The template-baseline table the reviewer asked for, with its conclusion stated."""
    if not res:
        return []
    if "error" in res:
        return ["=== Template baseline vs LLM ===", f"Failed: {res['error']}"]

    t, l = res["template"], res["llm"]
    k = res["top_k"]
    L = ["=== Necessity of the LLM layer: deterministic template vs LLM ===",
         f"Scenario: {res['method']} {res['scope']} explanation under {res['loss_name']}",
         f"Generations per generator: {res['n_reps']} | LLM: {res['llm_model']}"
         + (f" at temperature {res['temperature']}" if res["temperature"] is not None else ""),
         f"Reference top-{k} from the payload: {', '.join(res['reference_top'])}",
         f"Both graded by grade_narrative() against the same payload, tolerance "
         f"+/-{res['tolerance']}.",
         ""]

    hdr = (f"{'generator':<12s} {'top' + str(k) + '_agree':>10s} {'framing':>8s} "
           f"{'direction':>10s} {'numeric':>8s} {'halluc':>7s} {'similarity':>11s} "
           f"{'distinct':>9s} {'chars':>7s} {'tokens':>8s}")
    L += [hdr, "-" * len(hdr)]
    tok = res.get("tokens") or {}
    tok_total = int(tok.get("prompt", 0)) + int(tok.get("completion", 0))
    for name, row, tk in (("template", t, 0), ("LLM", l, tok_total)):
        L.append(f"{name:<12s} {_fmt(row['top_agreement'], 10)} {_fmt(row['framing'], 8)} "
                 f"{_fmt(row['directional'], 10)} {_fmt(row['numeric_accuracy'], 8)} "
                 f"{_fmt(row['hallucinations'], 7, 2)} {_fmt(row['similarity'], 11)} "
                 f"{row['distinct_top_sets']:>9d} {_fmt(row['mean_chars'], 7, 0)} "
                 f"{(str(tk) if tk else '0'):>8s}")

    L += ["", "Where the two differ, and by how much:"]
    for label, key, d in (("top-K agreement", "top_agreement", 3),
                          ("objective framing", "framing", 3),
                          ("directional consistency", "directional", 3),
                          ("numerical accuracy", "numeric_accuracy", 3),
                          ("hallucinations per narrative", "hallucinations", 2)):
        a, b = t.get(key), l.get(key)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            L.append(f"  {label:<30s} not defined in this scenario")
            continue
        diff = b - a
        verdict = ("identical" if abs(diff) < 1e-9
                   else ("LLM worse" if (diff < 0) != (key == "hallucinations") else "LLM better"))
        L.append(f"  {label:<30s} template {a:.3f}   LLM {b:.3f}   ({diff:+.3f}, {verdict})")

    L += ["", "Reading this table"]
    L += ["  The template scores what it scores by construction, not by being good at",
          "  writing: it prints the ranking, signs and numbers it was handed, so Eqs. (4),",
          "  (5), (7) are 1.0 and Eq. (8) is 0 necessarily, Eq. (6) is exactly 1.0 because it",
          "  is deterministic, and framing is a lookup on the objective rather than an",
          "  inference. It cost no tokens and made no network request.",
          "  Consequently these metrics cannot establish that the LLM layer is needed. They",
          "  can only establish that using it is not harmful, which is a weaker claim and",
          "  the one the manuscript is entitled to make from them. Any argument for the LLM",
          "  has to rest on capabilities these metrics do not measure -- responding to a",
          "  free-text request, adapting to an audience, answering a question -- which is",
          "  what the adaptation experiment and the Discussion tab address."]
    for c in res.get("caveats", []):
        L += ["", "  Caveat: " + c]
    for n in res.get("notes", []):
        L.append(f"  note: {n}")

    L += ["", "-" * 88, "Template output (identical every run):", res["template_example"],
          "", "-" * 88, "One LLM generation, for comparison:", res["llm_example"]]
    return L


def format_adaptation(res: Optional[Dict[str, Any]]) -> list:
    """Whether the generator responds to the request, with noise as the control."""
    if not res:
        return []
    if "error" in res:
        return ["=== Audience adaptation ===", f"Failed: {res['error']}"]

    k = res["top_k"]
    L = ["=== Capability the metrics above cannot see: adaptation to the request ===",
         f"Scenario: {res['method']} {res['scope']} explanation under {res['loss_name']}",
         f"Audiences: {' | '.join(res['audiences'])}",
         f"Generations per audience: {res['n_reps']} | LLM: {res['llm_model']}",
         "The payload is identical across audiences; only the requested register changes.",
         ""]

    tsim, lsim = res["template"]["similarity"], res["llm"]["similarity"]
    L.append(f"Similarity backend: {lsim.get('backend', tsim.get('backend', 'n/a'))}")
    L.append("")
    hdr = (f"{'generator':<12s} {'within':>9s} {'between':>9s} {'gap':>9s} "
           f"{'adapts?':>9s}")
    L += [hdr, "-" * len(hdr)]
    for name, s in (("template", tsim), ("LLM", lsim)):
        w, b = s.get("within"), s.get("between")
        gap = (w - b) if (w is not None and b is not None
                          and np.isfinite(w) and np.isfinite(b)) else np.nan
        adapts = ("-" if not np.isfinite(gap) else ("yes" if gap > 0.02 else "no"))
        L.append(f"{name:<12s} {_fmt(w, 9)} {_fmt(b, 9)} {_fmt(gap, 9)} {adapts:>9s}")

    L += ["",
          "  within   mean cosine between generations for the SAME audience -- the noise floor.",
          "  between  mean cosine between generations for DIFFERENT audiences.",
          "  gap      within - between. Positive means the text tracks the request by more",
          "           than it varies by chance, which is what adaptation means. A generator",
          "           that merely samples differently each time has a gap near zero.",
          ""]
    tw, tb = tsim.get("within"), tsim.get("between")
    if tw is not None and np.isfinite(tw) and abs(tw - 1.0) < 1e-9 and abs((tb or 0) - 1.0) < 1e-9:
        L.append("The template's within and between similarities are both exactly 1.000: it "
                 "emits the same text no matter who asks. Its adaptivity is zero, measured, "
                 "not assumed.")
    lw, lb = lsim.get("within"), lsim.get("between")
    if all(v is not None and np.isfinite(v) for v in (lw, lb)):
        if lw - lb > 0.02:
            L.append(f"The LLM's text differs across audiences ({lb:.3f}) by more than it "
                     f"differs across repeats of one audience ({lw:.3f}). The request is "
                     f"being responded to, not merely resampled.")
        else:
            L.append(f"The LLM's between-audience similarity ({lb:.3f}) is not meaningfully "
                     f"below its within-audience similarity ({lw:.3f}). On this evidence it "
                     f"is NOT adapting to the audience, and the adaptation argument fails "
                     f"here -- report this rather than the intended conclusion.")

    L += ["", "Faithfulness must survive the adaptation, or it buys nothing:", ""]
    hdr2 = (f"{'audience':<26s} {'gen':<10s} {'top' + str(k) + '_agree':>10s} "
            f"{'framing':>8s} {'numeric':>8s} {'halluc':>7s} {'chars':>7s}")
    L += [hdr2, "-" * len(hdr2)]
    for gi, aud in enumerate(res["audiences"]):
        for name, side in (("template", res["template"]), ("LLM", res["llm"])):
            row = side["per_audience"][gi]
            L.append(f"{str(aud)[:26]:<26s} {name:<10s} {_fmt(row['top_agreement'], 10)} "
                     f"{_fmt(row['framing'], 8)} {_fmt(row['numeric_accuracy'], 8)} "
                     f"{_fmt(row['hallucinations'], 7, 2)} {_fmt(row['mean_chars'], 7, 0)}")

    L += ["", "Note: the plain-language audience is instructed not to state numerical values,",
          "so a '-' in its numeric column means no numerical claims were made, which is",
          "compliance with the request and not a failure. Judge that audience on top-K",
          "agreement and hallucinations instead.", ""]
    for gi, aud in enumerate(res["audiences"]):
        L += ["-" * 88, f"{aud} -- LLM:", res["llm"]["examples"][gi], ""]
    L += ["-" * 88, "Template, for every audience (unchanged):", res["template"]["examples"][0]]
    return L


# -----------------------------
# Shiny UI
# -----------------------------
app_ui = ui.page_fluid(
    ui.h2("LLM-Assisted Explainable Machine Learning (SHAP & LIME) in Precision Medicine"),
    ui.tags.style("""
      .shap-img {
        max-width: 900px;
        width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
      }
    """),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file", "Upload CSV", accept=[".csv"]),

            ui.input_select(
                "outcome_type",
                "Outcome type",
                choices=["continuous", "binary", "time-to-event"],
                selected="continuous"
            ),

            ui.input_text("outcome", "Outcome column name (use y for all types)", value="y"),

            ui.panel_conditional(
                "input.outcome_type == 'time-to-event'",
                ui.input_text("event_col", "Event column name (1=event, 0=censor)", value="event")
            ),

            ui.input_text("treat", "Treatment column name (optional)", value="treatment"),

            ui.input_select(
                "na_policy",
                "Missing data",
                choices=["Complete cases", "Median imputation (training split)",
                         "Keep (XGBoost original loss only)"],
                selected="Complete cases",
            ),
            ui.input_checkbox(
                "standardize",
                "Standardise features (z-score, fitted on the training split)",
                value=False,
            ),
            ui.help_text(
                "Both are fitted on the training split only, so the test metrics stay "
                "out-of-sample. Tree models, SHAP and LIME are effectively invariant to "
                "feature scale; penalised regression is not, so standardisation matters "
                "most when a strong penalty is selected. See the Data check tab."
            ),

            ui.input_slider(
                "test_size",
                "Test set percentage (0 = no test set)",
                min=0.0, max=0.9, value=0.25, step=0.05
            ),

            ui.input_text(
                "sigpos_col",
                "Ground-truth treatment effect label column (optional, 0/1). Leave blank if not available.",
                value="sigpos"
            ),

            ui.input_select(
                "model_type",
                "Model",
                ["XGBoost", "Random Forest", "Linear Regression", "Logistic Regression",
                 "Cox Regression"]
            ),

            ui.input_select(
                "loss",
                "Loss / Objective",
                ["Original (clinical)", "A-learning", "W-learning"]
            ),

            ui.help_text(
                "A-learning / W-learning produce a benefiting score. For continuous and "
                "binary outcomes these objectives assume the outcome is coded so that a "
                "LARGER value of y is a BETTER outcome; if y is a bad-event indicator, "
                "recode it before fitting or every attribution direction will be "
                "reversed. For time-to-event outcomes the score is reported as "
                "-(Cox linear predictor) so that larger always means greater benefit."
            ),

            ui.hr(),
            ui.h4("Hyperparameters and tuning"),
            ui.input_select(
                "tune_mode",
                "Tuning",
                choices=["None", "Random search (CV)", "Grid search (CV)"],
                selected="None",
            ),
            ui.help_text(
                "With tuning off, the fixed values under Advanced are used as-is. With "
                "tuning on, cross-validation is run on the training split only, so the "
                "reported test metrics stay out-of-sample, and the selection criterion is "
                "the same loss the model is trained on: under A-learning or W-learning, "
                "configurations are ranked by the out-of-fold value of that objective, not "
                "by RMSE or log-loss. The propensity score is re-estimated within each "
                "fold. Time-to-event tuning is the slowest case, because the Cox objective "
                "is quadratic in the fold size."
            ),

            ui.panel_conditional(
                "input.tune_mode != 'None'",
                ui.input_numeric("tune_folds", "CV folds", value=5, min=2, max=10, step=1),
                ui.panel_conditional(
                    "input.tune_mode == 'Random search (CV)'",
                    ui.input_numeric("tune_iters", "Random search draws",
                                     value=20, min=1, step=1),
                ),
                ui.panel_conditional(
                    "input.tune_mode == 'Grid search (CV)'",
                    ui.input_numeric("tune_grid_points",
                                     "Grid points per range", value=3, min=2, max=10, step=1),
                ),
                ui.input_numeric("tune_seed", "Tuning seed", value=42, min=0, step=1),
                ui.input_checkbox_group(
                    "tune_params",
                    "Parameters to tune",
                    choices=XGB_TUNABLE,
                    selected=XGB_TUNED_BY_DEFAULT,
                ),
                ui.help_text("Unchecked parameters keep their fixed value under Advanced."),
                ui.output_text_verbatim("tune_size_note"),
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Advanced: search space and fixed values",

                    ui.panel_conditional(
                        "input.tune_mode != 'None'",
                        ui.h5("Search space"),
                        ui.input_text_area(
                            "tune_space",
                            "One parameter per line",
                            value=DEFAULT_SPACE_XGB,
                            rows=9,
                        ),
                        ui.help_text(
                            "Two forms are accepted. 'name: lo .. hi' is an inclusive "
                            "range, sampled log-uniformly for scale-free parameters under "
                            "random search and discretised for grid search. "
                            "'name: v1, v2, v3' is an explicit list of candidates. "
                            "Text after '#' is ignored. This box resets when you change "
                            "the model or the objective."
                        ),
                        ui.hr(),
                    ),

                    ui.h5("Fixed values"),
                    ui.help_text("Used for any parameter that is not being tuned."),

                    ui.panel_conditional(
                        "input.model_type == 'XGBoost'",
                        ui.input_numeric("xgb_n_estimators",
                                         "n_estimators (num_boost_round; upper limit when tuning)",
                                         value=400, min=10),
                        ui.input_numeric("xgb_learning_rate", "learning_rate (eta)",
                                         value=0.05, min=0.001, max=1, step=0.01),
                        ui.input_numeric("xgb_max_depth", "max_depth",
                                         value=4, min=1, max=20, step=1),
                        ui.input_slider("xgb_subsample", "subsample",
                                        min=0.1, max=1.0, value=0.9, step=0.05),
                        ui.input_slider("xgb_colsample", "colsample_bytree",
                                        min=0.1, max=1.0, value=0.9, step=0.05),
                        ui.input_numeric("xgb_min_child_weight", "min_child_weight",
                                         value=1.0, min=0.0, step=0.5),
                        ui.input_numeric("xgb_gamma", "gamma", value=0.0, min=0.0, step=0.1),
                        ui.input_numeric("xgb_reg_lambda", "reg_lambda (L2)",
                                         value=1.0, min=0.0, step=0.5),
                        ui.input_numeric("xgb_reg_alpha", "reg_alpha (L1)",
                                         value=0.0, min=0.0, step=0.5),
                        ui.input_select("xgb_tree_method", "tree_method",
                                        choices=["hist", "approx", "exact"], selected="hist"),
                        ui.input_numeric("tune_esr",
                                         "Early stopping rounds (0 = off)",
                                         value=30, min=0, step=5),
                    ),

                    ui.panel_conditional(
                        "input.model_type == 'Random Forest'",
                        ui.input_numeric("rf_n_estimators", "n_estimators (trees)",
                                         value=400, min=10, step=50),
                    ),

                    ui.panel_conditional(
                        "input.model_type != 'XGBoost'",
                        ui.input_numeric("param_penalty", "penalty", value=0.0, min=0.0, step=0.1),
                        ui.output_text_verbatim("penalty_hint"),
                    ),
                ),
                id="adv_accordion",
                open=False,
            ),

            ui.hr(),

            ui.input_checkbox_group(
                "shap_plots",
                "SHAP plots to generate",
                choices=["Beeswarm", "Bar", "Dependence", "Waterfall", "Decision"],
                selected=["Beeswarm", "Bar", "Dependence"]
            ),

            ui.hr(),
            ui.h4("SHAP dependence plot options"),
            ui.input_select(
                "dep_main",
                "Dependence plot: main feature (x-axis)",
                choices=["__AUTO_TOP__"],
                selected="__AUTO_TOP__"
            ),

            ui.input_select(
                "dep_interaction",
                "Dependence plot: interaction feature (color)",
                choices=["auto", "none"],
                selected="auto"
            ),

            ui.input_checkbox_group(
                "lime_plots",
                "LIME plots to generate",
                choices=["LIME Global", "LIME Local"],
                selected=["LIME Global", "LIME Local"]
            ),
            ui.input_numeric("boot_reps",
                             "Bootstrap replicates for coefficient intervals (0 = off)",
                             value=200, min=0, max=2000, step=50),
            ui.help_text(
                "Used on the Coefficients tab, and only where a conventional interval is not "
                "valid: a penalised fit, or an A-/W-learning objective. Each replicate refits "
                "on resampled subjects and re-estimates the propensity score."
            ),

            ui.input_numeric("lime_seed", "LIME perturbation seed", value=42, min=0, step=1),
            ui.help_text(
                "LIME resamples on every call, so without a fixed seed its output is not "
                "reproducible run to run. Vary this to see how much the explanation moves; "
                "the Sensitivity panel does so systematically."
            ),

            ui.hr(),
            ui.h4("LLM explanation settings"),

            ui.input_select(
                "llm_method",
                "Explain using",
                choices=["SHAP", "LIME"],
                selected="SHAP"
            ),

            ui.input_select(
                "llm_scope",
                "Scope",
                choices=["Local", "Global"],
                selected="Local"
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Endpoint and privacy",
                    ui.input_text("llm_base_url", "Endpoint (base URL)", value="",
                                  placeholder="blank = hosted API"),
                    ui.help_text(
                        "Leave blank to use the hosted API, which needs OPENAI_API_KEY. "
                        "Point it at a locally deployed OpenAI-compatible server -- "
                        "llama.cpp, vLLM, LM Studio, Ollama -- to run the narrative step "
                        "without a key and without sending anything off this machine, e.g. "
                        "http://localhost:8000/v1. Local servers are called through "
                        "/v1/chat/completions automatically."
                    ),
                    ui.input_text("llm_model", "Model", value=LLM_MODEL_DEFAULT),
                    ui.input_select(
                        "llm_privacy", "Individual values sent in Local scope",
                        choices=["Send as provided", "Round to 3 significant figures"],
                        selected="Send as provided",
                    ),
                    ui.output_text_verbatim("egress_note"),
                ),
                id="llm_endpoint_accordion",
                open=False,
            ),

            ui.input_text_area( "prompt", "LLM prompt template (editable)",
                                value=(
                                    "You are a biomedical data scientist.\n"
                                    "Explain the model prediction and SHAP/LIME results using cautious, scientifically appropriate language.\n"
                                    "Write 3-5 sentences in plain but scientific language.\n"
                                    "Focus on whether the observed attribution patterns are more consistent with an outcome-predictive (prognostic) signal, a treatment-effect-heterogeneity (predictive) signal, or inconclusive evidence.\n"
                                    "If A-learning or W-learning is used, describe the findings as potentially consistent with predictive signal, treatment-effect heterogeneity, or an ITR-related signal, rather than making definitive claims.\n"
                                    "If Original loss is used, describe the findings as potentially consistent with prognostic signal, rather than stating that any biomarker is definitively prognostic.\n"
                                    "Use phrases such as 'suggests', 'is consistent with', 'may reflect', or 'is inconclusive'.\n"
                                    "Do not infer causal treatment effects, do not make clinical recommendations, and do not mention features not supported by the explanation.\n"
                                ),
                                rows=10 ),

            ui.input_numeric("patient_idx", "Patient index for local explanation", value=0, min=0),
            ui.input_action_button("run", "Run analysis", class_="btn-primary"),

            ui.hr(),
            ui.accordion(
                ui.accordion_panel(
                    "Export results",
                    ui.help_text(
                        "Everything the app computed, as CSV, for downstream analysis. The "
                        "bundle also carries a manifest.json recording the configuration, "
                        "seeds, tuning outcome and package versions -- without it a table of "
                        "attributions is not a reproducible intermediate result."
                    ),
                    ui.download_button("dl_all", "All results (.zip)", class_="btn-primary"),
                    ui.download_button("dl_shap", "SHAP attributions (.csv)"),
                    ui.download_button("dl_lime", "LIME weights (.csv)"),
                    ui.download_button("dl_coef", "Coefficients (.csv)"),
                    ui.download_button("dl_manifest", "Run manifest (.json)"),
                    ui.help_text(
                        "SHAP values carry the units of the model's own output, so magnitudes "
                        "are not comparable between model families. Compare the rank or "
                        "share_of_total columns, which are dimensionless."
                    ),
                ),
                id="export_accordion",
                open=False,
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Sensitivity analysis",
                    ui.help_text(
                        "Holds the explanation layer to the same standard as the model: "
                        "how much do LIME and the LLM narrative move when their stochastic "
                        "settings change? Run separately from the main analysis, because "
                        "the LLM sweep makes repeated API calls."
                    ),
                    ui.input_select(
                        "sens_which", "Analysis",
                        choices=["LIME stability", "LLM temperature sweep", "Both",
                                 "Template baseline vs LLM", "Audience adaptation",
                                 "Attributions across models"],
                        selected="LIME stability",
                    ),
                    ui.input_select("sens_scope", "Scope",
                                    choices=["Local", "Global"], selected="Global"),

                    ui.panel_conditional(
                        "input.sens_which == 'Template baseline vs LLM' || "
                        "input.sens_which == 'Audience adaptation'",
                        ui.help_text(
                            "Answers the objection that a deterministic report template "
                            "would do this job. The template is graded by the same harness "
                            "as the LLM. It wins every metric in Section 5 by construction, "
                            "so the case for the LLM has to be made on adaptation and "
                            "interaction instead -- which is what the second option, and "
                            "the Discussion tab, measure."
                        ),
                        ui.input_numeric("sens_baseline_reps", "Generations per condition",
                                         value=5, min=2, max=30, step=1),
                    ),

                    ui.panel_conditional(
                        "input.sens_which == 'Attributions across models'",
                        ui.help_text(
                            "Fits the same data and objective under each model family and "
                            "compares which covariates the attributions point at. Models that "
                            "cannot express the current objective are reported as unavailable "
                            "with the reason rather than skipped."
                        ),
                        ui.input_checkbox_group(
                            "cmp_models", "Model families",
                            choices=["XGBoost", "Random Forest", "Linear Regression",
                                     "Logistic Regression", "Cox Regression"],
                            selected=["XGBoost", "Random Forest"],
                        ),
                    ),

                    ui.panel_conditional(
                        "input.sens_which == 'Audience adaptation'",
                        ui.input_text_area(
                            "sens_aud_a", "Audience A",
                            value=ADAPTATION_PROMPTS[0][1], rows=3),
                        ui.input_text_area(
                            "sens_aud_b", "Audience B",
                            value=ADAPTATION_PROMPTS[1][1], rows=3),
                    ),

                    ui.panel_conditional(
                        "input.sens_which == 'LIME stability' || input.sens_which == 'Both'",
                        ui.h5("LIME"),
                        ui.input_numeric("sens_lime_repeats", "Perturbation seeds per setting",
                                         value=5, min=2, max=30, step=1),
                        ui.input_text("sens_lime_samples", "num_samples values",
                                      value="200, 500, 1000"),
                        ui.input_text("sens_lime_kw",
                                      "kernel width multipliers (x LIME default)",
                                      value="0.5, 1.0, 2.0"),
                        ui.input_numeric("sens_lime_global_n",
                                         "Instances aggregated for global LIME",
                                         value=20, min=2, step=1),
                    ),

                    ui.panel_conditional(
                        "input.sens_which != 'LIME stability' && "
                        "input.sens_which != 'Attributions across models'",
                        ui.h5("LLM"),
                        ui.input_text("sens_llm_model", "Model(s)",
                                      value=LLM_MODEL_DEFAULT),
                        ui.input_numeric("sens_tol", "Numerical tolerance", value=0.05,
                                         min=0.0, step=0.01),
                    ),

                    ui.panel_conditional(
                        "input.sens_which == 'LLM temperature sweep' || input.sens_which == 'Both'",
                        ui.input_numeric("sens_llm_reps", "Generations per temperature",
                                         value=10, min=2, max=50, step=1),
                        ui.input_text("sens_llm_temps", "Temperatures",
                                      value="0, 0.3, 0.7, 1.0"),
                        ui.help_text(
                            "Cost scales with generations x temperatures x models. The "
                            "informative scenario is a Global explanation under A-learning, "
                            "where objective framing accuracy has headroom. Name two or "
                            "more models, comma-separated, to grade them side by side -- a "
                            "hosted model against a local one. The endpoint and key come "
                            "from 'Endpoint and privacy' above."
                        ),
                    ),

                    ui.input_numeric("sens_top_k", "K for top-K agreement", value=3,
                                     min=1, max=10, step=1),
                    ui.input_action_button("run_sens", "Run sensitivity analysis",
                                           class_="btn-secondary"),
                ),
                id="sens_accordion",
                open=False,
            ),
        ),

        ui.navset_tab(
            ui.nav_panel("Data preview", ui.output_table("preview")),
            ui.nav_panel("Data check", ui.output_text_verbatim("data_check")),
            ui.nav_panel("Metrics", ui.output_text_verbatim("metrics")),
            ui.nav_panel("Coefficients", ui.output_text_verbatim("coef_out")),
            ui.nav_panel("SHAP plots", ui.output_ui("shap_gallery")),
            ui.nav_panel("LIME plots", ui.output_ui("lime_gallery")),
            ui.nav_panel("LLM explanation", ui.output_text_verbatim("llm_out")),
            ui.nav_panel(
                "Discussion",
                ui.help_text(
                    "Ask follow-up questions about the explanation currently loaded. This is "
                    "the one thing a report template cannot do at all, so there is no "
                    "baseline to compare against here. Every answer is graded by the same "
                    "faithfulness rules as the narratives, and the grade is shown with it: "
                    "interaction widens the opportunity to stray past the payload, so the "
                    "check travels with the answer."
                ),
                ui.input_text_area("chat_q", "Question", rows=2, width="100%",
                                   placeholder="e.g. Which of these features would you "
                                               "measure first, and why?"),
                ui.div(
                    ui.input_action_button("chat_ask", "Ask", class_="btn-primary"),
                    " ",
                    ui.input_action_button("chat_clear", "Clear conversation",
                                           class_="btn-secondary"),
                ),
                ui.output_text_verbatim("chat_out"),
            ),
            ui.nav_panel("Sensitivity", ui.output_text_verbatim("sens_out")),
        )
    )
)


# -----------------------------
# Shiny Server
# -----------------------------
def server(input, output, session):
    state = reactive.Value(None)
    sens_state = reactive.Value(None)
    chat_state = reactive.Value([])

    @reactive.calc
    def raw_df():
        """The uploaded CSV, read once and shared by the data check and the analysis."""
        fi = input.file()
        if not fi:
            return None
        return pd.read_csv(fi[0]["datapath"])

    @output
    @render.text
    def data_check():
        df = raw_df()
        if df is None:
            return ("Upload a CSV to see the data check.\n\n"
                    "This tab reports which columns will be used as features, which are "
                    "excluded and why, where values are missing, and any coding problems "
                    "-- before anything is fitted.")
        ot = input.outcome_type()
        rep = data_quality_report(
            df,
            outcome_col=input.outcome().strip(),
            treat_col=(input.treat().strip() or "treatment"),
            event_col=(input.event_col().strip() if ot == "time-to-event" else None),
            sigpos_col=(input.sigpos_col().strip() or None),
            outcome_type=ot,
            loss_name=input.loss(),
        )
        return "\n".join(format_data_report(rep))

    @output
    @render.text
    def egress_note():
        """State what the current configuration will transmit, before it is transmitted."""
        url = str(input.llm_base_url()).strip()
        local = is_local_endpoint(url)
        lines = [data_egress_note(input.llm_scope(), local, url)]
        if url and not local:
            lines.append("This endpoint is not a local address, so it is treated as remote.")
        if not url and not os.getenv("OPENAI_API_KEY"):
            lines.append("OPENAI_API_KEY is not set: either set it, or give a local endpoint.")
        if str(input.llm_privacy()).startswith("Round"):
            lines.append("Values are rounded to 3 significant figures before being sent.")
        return "\n".join(lines)

    def _feature_scale_note(st):
        """Warn the LLM when the feature values it is shown are z-scores, not
        measurements, so it does not present them as clinical quantities."""
        if not ((st or {}).get("extra", {}).get("prep") or {}).get("standardize"):
            return ""
        return ("\n- Feature VALUES in the payload are standardised (z-scores relative to "
                "the training split), not original clinical units. Report them as standard "
                "deviations from the mean, never as measurements.")

    def _parse_numlist(txt, cast=float):
        out = []
        for tok in str(txt or "").replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    out.append(cast(float(tok)))
                except ValueError:
                    pass
        return out

    def _loss_choices_for(outcome_type: str):
        if outcome_type == "continuous":
            original_label = "Square loss (MSE)"
        elif outcome_type == "binary":
            original_label = "Logistic loss (NLL)"
        elif outcome_type == "time-to-event":
            original_label = "Cox partial likelihood"
        else:
            original_label = "Original (clinical)"
        return {
            "Original (clinical)": original_label,
            "A-learning": "A-learning",
            "W-learning": "W-learning",
        }

    @reactive.effect
    def _sync_loss_label_with_outcome():
        ot = input.outcome_type()
        current = input.loss()
        ui.update_select(
            "loss",
            choices=_loss_choices_for(ot),
            selected=current,
            session=session
        )

    @reactive.effect
    def _sync_tuning_controls_with_model():
        """Adapt the tuning controls to the selected model and objective.

        The set of tunable hyperparameters, the default search space, and the meaning of
        the single fixed-penalty input all depend on (model, objective), so they are
        refreshed together whenever either changes.
        """
        mt = input.model_type()
        ln = input.loss()
        names = tunable_parameter_names(mt, ln)

        ui.update_checkbox_group(
            "tune_params",
            choices=names,
            selected=names,
            session=session,
        )
        ui.update_text_area(
            "tune_space",
            value=default_search_space_text(mt, ln),
            session=session,
        )
        if names and mt != "XGBoost":
            pname = names[0]
            ui.update_numeric(
                "param_penalty",
                label=PARAM_LABELS.get(pname, pname),
                value=default_fixed_penalty(mt, ln),
                session=session,
            )

    @output
    @render.text
    def penalty_hint():
        mt, ln = input.model_type(), input.loss()
        names = tunable_parameter_names(mt, ln)
        if mt == "XGBoost" or not names:
            return ""
        return (f"This is '{names[0]}' for {mt} under "
                f"{_loss_choices_for(input.outcome_type()).get(ln, ln)}.")

    def _xgb_params_from_sidebar():
        return dict(
            learning_rate=float(input.xgb_learning_rate()),
            max_depth=int(input.xgb_max_depth()),
            subsample=float(input.xgb_subsample()),
            colsample_bytree=float(input.xgb_colsample()),
            min_child_weight=float(input.xgb_min_child_weight()),
            gamma=float(input.xgb_gamma()),
            reg_lambda=float(input.xgb_reg_lambda()),
            reg_alpha=float(input.xgb_reg_alpha()),
            tree_method=str(input.xgb_tree_method()),
            verbosity=0,
        )

    def _incumbent_from_sidebar(model_type, loss_name):
        """The fixed configuration, always evaluated as the search's first candidate."""
        if model_type == "XGBoost":
            return _xgb_params_from_sidebar()
        names = tunable_parameter_names(model_type, loss_name)
        return {names[0]: float(input.param_penalty())} if names else None

    @output
    @render.text
    def tune_size_note():
        if input.tune_mode() == "None":
            return ""
        return describe_search(
            input.tune_mode(),
            input.tune_space(),
            list(input.tune_params()),
            model_type=input.model_type(),
            loss_name=input.loss(),
            n_iter=int(input.tune_iters()),
            seed=int(input.tune_seed()),
            grid_points=int(input.tune_grid_points()),
            n_splits=int(input.tune_folds()),
            incumbent=_incumbent_from_sidebar(input.model_type(), input.loss()),
        )

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        df = raw_df()
        if df is None:
            state.set({"error": "Please upload a CSV first."})
            return

        outcome_type = input.outcome_type()
        outcome_col = input.outcome().strip()

        event_col = None
        if outcome_type == "time-to-event":
            event_col = input.event_col().strip()

        treat_col = input.treat().strip() if input.treat().strip() else "treatment"
        test_size = float(input.test_size())

        sigpos_col = input.sigpos_col().strip()
        if sigpos_col == "":
            sigpos_col = None

        feature_cols = detect_feature_columns(
            df,
            outcome_col=outcome_col,
            treat_col=treat_col,
            event_col=event_col
        )
        if len(feature_cols) < 2:
            state.set({"error": "Not enough numeric feature columns detected."})
            return

        try:
            ui.update_select(
                "dep_main",
                choices=["__AUTO_TOP__"] + feature_cols,
                selected="__AUTO_TOP__",
                session=session
            )
            ui.update_select(
                "dep_interaction",
                choices=["auto", "none"] + feature_cols,
                selected="auto",
                session=session
            )
        except Exception:
            pass

        model_type = input.model_type()
        loss_name = input.loss()

        xgb_params = _xgb_params_from_sidebar()
        num_boost_round = int(input.xgb_n_estimators())

        # Fixed penalty for the non-XGBoost models. One input serves all four cases; its
        # meaning is set by (model, objective) via tunable_parameter_names().
        lin_alpha, logreg_C_val, cox_penalizer, mod_reg_lambda = 0.0, 1.0, 0.0, 1e-6
        rf_min_leaf = 1
        if model_type != "XGBoost":
            pv = float(input.param_penalty())
            if loss_name in ("A-learning", "W-learning"):
                mod_reg_lambda = pv
            elif model_type == "Linear Regression":
                lin_alpha = pv
            elif model_type == "Logistic Regression":
                logreg_C_val = pv
            elif model_type == "Cox Regression":
                cox_penalizer = pv
            elif model_type == "Random Forest":
                # 0 means "no smoothing", which for a leaf-size knob is 1 observation.
                rf_min_leaf = max(1, int(round(pv))) if pv else 1

        if input.tune_mode() != "None":
            ui.notification_show(
                f"Tuning {model_type} under {loss_name} -- "
                + describe_search(
                    input.tune_mode(), input.tune_space(), list(input.tune_params()),
                    model_type=model_type, loss_name=loss_name,
                    n_iter=int(input.tune_iters()), seed=int(input.tune_seed()),
                    grid_points=int(input.tune_grid_points()),
                    n_splits=int(input.tune_folds()),
                    incumbent=_incumbent_from_sidebar(model_type, loss_name),
                )
                + (" Time-to-event is the slowest case."
                   if outcome_type == "time-to-event" else ""),
                duration=None, id="tuning_note", type="message",
            )

        try:
            model, X_train, X_test, y_train, y_test, metrics_out, _, extra = fit_model(
                df,
                feature_cols,
                outcome_col=outcome_col,
                treat_col=treat_col,
                outcome_type=outcome_type,
                event_col=event_col,
                model_type=model_type,
                loss_name=loss_name,
                test_size=test_size,
                seed=42,
                sigpos_col=sigpos_col,
                xgb_params=(xgb_params if model_type == "XGBoost" else None),
                num_boost_round=(num_boost_round if model_type == "XGBoost" else 400),
                tune_mode=input.tune_mode(),
                tune_n_iter=int(input.tune_iters()),
                tune_folds=int(input.tune_folds()),
                tune_seed=int(input.tune_seed()),
                tune_esr=int(input.tune_esr()),
                tune_space_text=input.tune_space(),
                tune_params=list(input.tune_params()),
                tune_grid_points=int(input.tune_grid_points()),
                lin_alpha=lin_alpha,
                logreg_C=logreg_C_val,
                rf_n_estimators=int(input.rf_n_estimators()),
                rf_min_samples_leaf=rf_min_leaf,
                cox_penalizer=cox_penalizer,
                mod_reg_lambda=mod_reg_lambda,
                na_policy=input.na_policy(),
                standardize=bool(input.standardize()),
            )
        except Exception as e:
            state.set({"error": f"Model fitting error: {e}"})
            return
        finally:
            ui.notification_remove("tuning_note")

        artifacts = {
            "df": df,
            "feature_cols": feature_cols,
            "model_type": model_type,
            "loss_name": loss_name,
            "outcome_type": outcome_type,
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "metrics": metrics_out,
            "extra": extra,
            # Enough to refit this exact configuration on a resample, for the bootstrap
            # intervals on the Coefficients tab.
            "fit_args": {
                "outcome_col": outcome_col, "treat_col": treat_col,
                "event_col": event_col, "sigpos_col": sigpos_col,
                "na_policy": input.na_policy(), "standardize": bool(input.standardize()),
                "penalties": {"lin_alpha": lin_alpha, "logreg_C": logreg_C_val,
                              "cox_penalizer": cox_penalizer,
                              "mod_reg_lambda": mod_reg_lambda},
            },
        }

        # Choose plot set:
        # If no test set, use training for plots; else use test
        plot_df = X_test if (X_test is not None and len(X_test) > 0) else X_train
        plot_df = plot_df.copy().reset_index(drop=True)
        if len(plot_df) > 200:
            plot_df = plot_df.iloc[:200, :]

        # --- SHAP for XGBoost + linear/logistic/Cox models
        try:
            paths_disk, explainer, shap_values, X_plot_used = save_shap_plots(
                model=model,
                X_plot=plot_df,
                out_dir="outputs",
                prefix="Fig_SHAP",
                dep_main=input.dep_main(),
                dep_interaction=input.dep_interaction(),
                patient_idx=int(input.patient_idx()),
                model_type=model_type,
                loss_name=loss_name,
                X_background=X_train,
                outcome_type=outcome_type,
            )
            artifacts.update({
                "shap_paths_disk": paths_disk,
                "explainer": explainer,
                "shap_values": shap_values,
                "X_plot_used": X_plot_used,
            })
        except Exception as e:
            artifacts["shap_error"] = str(e)

        # --- LIME
        try:
            lime_paths = save_lime_plots(
                model=model,
                X_train=X_train,
                X_plot=plot_df,
                model_type=model_type,
                outcome_type=outcome_type,
                loss_name=loss_name,
                patient_idx=int(input.patient_idx()),
                out_dir="outputs",
                prefix="Fig_LIME",
                num_features=10,
                global_n=60,
                lime_seed=int(input.lime_seed()),
            )
            artifacts["lime_paths_disk"] = lime_paths
        except Exception as e:
            artifacts["lime_error"] = str(e)

        state.set(artifacts)

    @output
    @render.table
    def preview():
        st = state.get()
        if not st or "df" not in st:
            return pd.DataFrame()
        return st["df"].head(8)

    @output
    @render.text
    def metrics():
        st = state.get()
        if not st:
            return ""
        if "error" in st:
            return st["error"]

        m = st["metrics"]
        ex = st.get("extra", {})
        lines = []
        ot = st["outcome_type"]
        loss_val = st["loss_name"]
        loss_disp = _loss_choices_for(ot).get(loss_val, loss_val)

        lines.append(
            f"Model: {st['model_type']} | Outcome: {ot} | Loss: {loss_disp} | "
            f"Sigpos: {'YES' if ex.get('sigpos_exists') else 'NO'}"
        )

        tune_lines = format_tuning_report(ex.get("tuning"))
        if tune_lines:
            lines.append("")
            lines.extend(tune_lines)

        lines.append("")
        lines.append("=== Hyperparameters used (after tuning, if enabled) ===")
        if st["model_type"] == "XGBoost":
            lines.append(f"num_boost_round: {m.get('xgb_num_boost_round')}")
            xgbp = m.get("xgb_params", {})
            for k in sorted(xgbp.keys()):
                v = xgbp[k]
                lines.append(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
        else:
            names = tunable_parameter_names(st["model_type"], loss_val)
            pname = names[0] if names else "penalty"
            pv = ex.get("penalty_used")
            lines.append(f"{pname}: {pv:.6g}" if isinstance(pv, float)
                         else f"{pname}: {pv}")

        pr = ex.get("prep") or {}
        if pr:
            n_in = int(pr.get("n_input", 0))
            n_used = n_in - int(pr.get("dropped_target", 0)) - int(pr.get("dropped_incomplete", 0))
            lines.append("")
            lines.append("=== Preprocessing ===")
            lines.append(f"Missing-data policy: {pr.get('na_policy')}")
            if pr.get("dropped_target"):
                lines.append(f"Rows dropped for missing outcome/treatment/event: "
                             f"{pr['dropped_target']}")
            if pr.get("dropped_incomplete"):
                lines.append(f"Rows dropped as incomplete: {pr['dropped_incomplete']}")
            if pr.get("n_imputed_cells"):
                lines.append(f"Cells median-imputed: {pr['n_imputed_cells']} in "
                             f"{len(pr.get('imputed', {}))} column(s); medians taken from "
                             f"the training split only")
            lines.append(f"Rows used: {n_used} of {n_in} uploaded")
            lines.append("Standardisation: "
                         + ("z-scored, mean and SD from the training split only"
                            if pr.get("standardize") else "off (raw feature scale)"))
            if pr.get("standardize"):
                lines.append("  NOTE: SHAP and LIME feature values are z-scores, "
                             "not original units.")
            if pr.get("sd_ratio", 1) > 1:
                lines.append(f"Feature SD ratio (training split): {pr['sd_ratio']:.1f}")
            if pr.get("scale_warning"):
                lines.append(f"  ! {pr['scale_warning']}")

        lines.append("")
        lines.append("=== Score orientation ===")
        lines.append(ex.get("benefit_note", ""))
        if ex.get("sigpos_exists"):
            lines.append("*_auc_sigpos below is computed on this oriented score.")

        lines.append("")
        lines.append("=== Performance metrics ===")

        preferred_order = [
            "train_rmse", "test_rmse",
            "train_loss", "test_loss",
            "train_auc", "test_auc",
            "train_auc_sigpos", "test_auc_sigpos",
            "train_c_index", "test_c_index",
        ]
        for k in preferred_order:
            if k in m:
                v = m[k]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v}")

        if st.get("shap_error"):
            lines.append(f"SHAP error: {st['shap_error']}")
        if st.get("lime_error"):
            lines.append(f"LIME error: {st['lime_error']}")

        return "\n".join(lines)

    def _img_payload_from_paths(paths_dict, key):
        p = paths_dict.get(key)
        if not p or not os.path.exists(p):
            return None
        return {"src": p, "alt": key, "class": "shap-img"}

    @output
    @render.ui
    def shap_gallery():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return ui.p("Run analysis to generate SHAP plots.")
        if st.get("shap_error"):
            return ui.p(f"SHAP error: {st['shap_error']}")

        selected = set(input.shap_plots())
        blocks = []

        def add_block(name, output_id):
            blocks.append(ui.h4(name))
            blocks.append(ui.output_image(output_id))
            blocks.append(ui.hr())

        if "Beeswarm" in selected:   add_block("Beeswarm", "img_beeswarm")
        if "Bar" in selected:        add_block("Bar", "img_bar")
        if "Dependence" in selected: add_block("Dependence", "img_dependence")
        if "Waterfall" in selected:  add_block("Waterfall", "img_waterfall")
        if "Decision" in selected:   add_block("Decision", "img_decision")

        return ui.div(*blocks)

    @output
    @render.image
    def img_beeswarm():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["shap_paths_disk"], "Beeswarm")

    @output
    @render.image
    def img_bar():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["shap_paths_disk"], "Bar")

    @output
    @render.image
    def img_dependence():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["shap_paths_disk"], "Dependence")

    @output
    @render.image
    def img_waterfall():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["shap_paths_disk"], "Waterfall")

    @output
    @render.image
    def img_decision():
        st = state.get()
        if not st or "shap_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["shap_paths_disk"], "Decision")

    @output
    @render.ui
    def lime_gallery():
        st = state.get()
        if not st:
            return ui.p("No state available.")
        if st.get("lime_error"):
            return ui.p(f"LIME error: {st['lime_error']}")
        if "lime_paths_disk" not in st:
            return ui.p("Run analysis to generate LIME plots.")

        selected = set(input.lime_plots())
        blocks = []

        def add_block(name, output_id):
            blocks.append(ui.h4(name))
            blocks.append(ui.output_image(output_id))
            blocks.append(ui.hr())

        if "LIME Global" in selected: add_block("LIME Global", "img_lime_global")
        if "LIME Local" in selected:  add_block("LIME Local", "img_lime_local")

        return ui.div(*blocks)

    @output
    @render.image
    def img_lime_global():
        st = state.get()
        if not st or "lime_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["lime_paths_disk"], "LIME Global")

    @output
    @render.image
    def img_lime_local():
        st = state.get()
        if not st or "lime_paths_disk" not in st:
            return None
        return _img_payload_from_paths(st["lime_paths_disk"], "LIME Local")

    def _coef_bundle(st):
        """Coefficient summary, bootstrap where needed, and the SHAP reconciliation.

        Returns (summary, boot, reconcile). Any of them may be None: XGBoost has no
        coefficients, and the bootstrap only runs where a conventional interval is invalid.
        """
        if st["model_type"] in ("XGBoost", "Random Forest"):
            return None, None, None
        cox_model = st["model"] if st["model_type"] == "Cox Regression" else None
        summary = coefficient_summary(
            st["model"], st["X_train"], model_type=st["model_type"],
            loss_name=st["loss_name"], outcome_type=st["outcome_type"],
            feature_names=list(st["X_train"].columns), y_train=st.get("y_train"),
            penalty_used=(st.get("extra") or {}).get("penalty_used"), cox_model=cox_model)

        boot = None
        n_boot = int(input.boot_reps())
        if (not summary["valid"]) and n_boot > 0:
            fa = st.get("fit_args") or {}
            try:
                boot = bootstrap_coefficients(
                    st["df"], st["feature_cols"], outcome_type=st["outcome_type"],
                    model_type=st["model_type"], loss_name=st["loss_name"],
                    n_boot=n_boot, seed=42,
                    outcome_col=fa.get("outcome_col"), treat_col=fa.get("treat_col"),
                    event_col=fa.get("event_col"), sigpos_col=fa.get("sigpos_col"),
                    na_policy=fa.get("na_policy", "Drop rows with any missing value"),
                    standardize=bool(fa.get("standardize", False)),
                    **(fa.get("penalties") or {}))
            except Exception as e:
                boot = {"error": str(e)}

        reconcile = None
        if st.get("shap_values") is not None and st.get("X_plot_used") is not None:
            try:
                reconcile = reconcile_shap_with_coefficients(
                    st["shap_values"], st["X_plot_used"], st["X_train"], summary["rows"],
                    feature_names=list(st["X_train"].columns),
                    top_k=int(input.sens_top_k()))
            except Exception:
                reconcile = None
        return summary, (boot if (boot and "error" not in boot) else None), reconcile

    @output
    @render.text
    @reactive.event(input.run)
    def coef_out():
        st = state.get()
        if not st or "error" in st:
            return st.get("error", "") if st else ""
        if st["model_type"] in ("XGBoost", "Random Forest"):
            return (f"=== Coefficients ===\n"
                    f"{st['model_type']} has no coefficients to report. This is the case the attribution "
                    "layer exists for: with no interpretable parameters, SHAP is the only "
                    "route to a per-feature summary.\n\n"
                    "Switch the model to Linear, Logistic or Cox Regression to see the "
                    "conventional summary -- effect estimates, confidence intervals, odds or "
                    "hazard ratios -- alongside a numerical check that SHAP is an algebraic "
                    "re-expression of those coefficients rather than separate evidence.")
        ui.notification_show("Computing coefficient summary...", duration=None,
                             id="coef_note", type="message")
        try:
            summary, boot, reconcile = _coef_bundle(st)
        except Exception as e:
            return f"=== Coefficients ===\nFailed: {e}"
        finally:
            ui.notification_remove("coef_note")
        return "\n".join(format_coefficients(summary, boot, reconcile))

    def _coefficients_for_payload(st):
        """The conventional summary in payload form, so the narrative layer can cite it."""
        if st["model_type"] in ("XGBoost", "Random Forest"):
            return None
        try:
            summary, boot, _ = _coef_bundle(st)
        except Exception:
            return None
        if not summary:
            return None
        bmap = {r["feature"]: r for r in (boot or {}).get("rows", [])}
        rows = []
        for r in summary["rows"]:
            row = {"feature": r["feature"], "coef": round(float(r["coef"]), 4)}
            if "ci_lo" in r:
                row.update({"se": round(r["se"], 4), "ci_lo": round(r["ci_lo"], 4),
                            "ci_hi": round(r["ci_hi"], 4), "p": float(f"{r['p']:.3g}")})
            b = bmap.get(r["feature"])
            if b:
                row.update({"boot_lo": round(b["boot_lo"], 4),
                            "boot_hi": round(b["boot_hi"], 4),
                            "boot_se": round(b["boot_se"], 4)})
            if r.get("exp_coef") is not None:
                row["exp_coef"] = round(r["exp_coef"], 4)
                if r.get("exp_lo") is not None:
                    row.update({"exp_lo": round(r["exp_lo"], 4),
                                "exp_hi": round(r["exp_hi"], 4)})
            rows.append(row)
        block = {"coefficients": rows,
                 "exp_label": summary.get("exp_label", ""),
                 "coefficient_level": summary.get("level", 0.95)}
        if summary["valid"]:
            block["coefficient_note"] = ("Intervals are "
                                         + (summary.get("inference_prose")
                                            or "conventional standard errors")
                                         + "; quote them as given.")
        else:
            warn = ("No valid conventional interval for this fit: " + summary["reason"]
                    + (" A percentile bootstrap interval is given instead; quote that."
                       if bmap else
                       " No interval is available, so report the point estimate only and "
                       "state that no interval is given."))
            block["coefficient_note"] = warn
            block["coefficient_warning"] = warn
        return block

    def _explain_payloads(st, explain_method, explain_scope):
        """The payload the LLM layer is currently looking at, privacy filter applied.

        Shared by the narrative, the follow-up conversation and the baseline comparisons, so
        all of them are grounded on byte-identical input. Returns (shap, lime, error_text).
        """
        i = int(input.patient_idx())
        X_plot = st["X_test"].copy().reset_index(drop=True)
        if len(X_plot) == 0:
            X_plot = st["X_train"].copy().reset_index(drop=True)
        if len(X_plot) > 200:
            X_plot = X_plot.iloc[:200, :]
        if i < 0 or i >= len(X_plot):
            return None, None, f"Patient index out of range. Choose 0 to {len(X_plot) - 1}."

        shap_payload = lime_payload = None
        if explain_method == "SHAP":
            if "explainer" not in st or "shap_values" not in st:
                return None, None, "No SHAP artifacts available. Please run analysis again."
            shap_payload = build_shap_payload(
                st["explainer"], st["shap_values"], st["X_plot_used"],
                patient_idx=i, top_k_local=8, top_k_global=12)
        else:
            try:
                lime_payload = build_lime_payload(
                    model=st["model"], X_train=st["X_train"], X_plot=X_plot,
                    model_type=st["model_type"], outcome_type=st["outcome_type"],
                    loss_name=st["loss_name"], patient_idx=i, num_features=10,
                    global_n=60, global_num_samples=500, ridge_alpha=0.01,
                    lime_seed=int(input.lime_seed()))
            except Exception as e:
                return None, None, f"LIME payload build error: {e}"

        # A model with interpretable parameters carries them alongside the attributions, so
        # the narrative can lead with the effect estimate and its interval rather than
        # presenting SHAP as though it were the only summary available.
        coef_block = _coefficients_for_payload(st)
        if coef_block:
            for p in (shap_payload, lime_payload):
                if p is not None:
                    p.update(coef_block)

        # Filtered here, on the payload itself, so the prompt displayed is exactly what was
        # transmitted and the graders score against the numbers the model actually saw.
        privacy = input.llm_privacy()
        return (apply_privacy_filter(shap_payload, scope=explain_scope, mode=privacy),
                apply_privacy_filter(lime_payload, scope=explain_scope, mode=privacy), "")

    @output
    @render.text
    @reactive.event(input.run)
    def llm_out():
        st = state.get()
        if not st or "error" in st:
            return st.get("error", "")

        explain_method = input.llm_method()
        explain_scope = input.llm_scope()
        shap_payload, lime_payload, err = _explain_payloads(st, explain_method, explain_scope)
        if err:
            return err

        try:
            full_prompt, text_out, meta = llm_explain(
                explain_method=explain_method,
                explain_scope=explain_scope,
                outcome_type=st["outcome_type"],
                loss_name=st["loss_name"],
                model_type=st["model_type"],
                user_prompt=input.prompt(),
                shap_payload=shap_payload,
                lime_payload=lime_payload,
                feature_note=_feature_scale_note(st),
                base_url=str(input.llm_base_url()).strip(),
                model=str(input.llm_model()).strip() or LLM_MODEL_DEFAULT,
            )
            head = [f"=== Endpoint === {meta['endpoint']} via {meta['surface']}"
                    + (" (local deployment)" if meta["local_endpoint"] else "")]
            if meta.get("surface_note"):
                head.append(f"    note: {meta['surface_note']}")
            head.append(meta["egress"])
            u = meta.get("usage") or {}
            if u:
                head.append(f"Tokens: {u.get('prompt_tokens', '?')} prompt, "
                            f"{u.get('completion_tokens', '?')} completion")
            return ("\n".join(head)
                    + "\n\n=== Prompt sent to LLM ===\n" + full_prompt
                    + f"\n\n=== LLM output ({meta['model']}) ===\n" + text_out)
        except Exception as e:
            return f"LLM call error: {e}"

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def _export_config():
        """The sidebar settings that affect the numbers, for the manifest."""
        def g(name, cast=None):
            try:
                v = getattr(input, name)()
                return cast(v) if cast else v
            except Exception:
                return None
        return {
            "test_size": g("test_size", float), "seed": 42,
            "model_type": g("model_type"), "loss": g("loss"),
            "outcome_type": g("outcome_type"),
            "outcome_col": g("outcome_col"), "treat_col": g("treat_col"),
            "event_col": g("event_col"), "sigpos_col": g("sigpos_col"),
            "na_policy": g("na_policy"), "standardize": g("standardize", bool),
            "tune_mode": g("tune_mode"), "tune_iters": g("tune_iters", int),
            "tune_folds": g("tune_folds", int), "tune_seed": g("tune_seed", int),
            "param_penalty": g("param_penalty", float),
            "rf_n_estimators": g("rf_n_estimators", int),
            "lime_seed": g("lime_seed", int),
            "bootstrap_replicates": g("boot_reps", int),
            "llm": {"endpoint": g("llm_base_url"), "model": g("llm_model"),
                    "scope": g("llm_scope"), "method": g("llm_method"),
                    "privacy": g("llm_privacy")},
        }

    def _collect_frames(st):
        """Every numeric artifact currently available, as named DataFrames."""
        frames = {}
        frames.update(export_shap_frames(st.get("shap_values"), st.get("X_plot_used"),
                                         feature_names=list(st["X_train"].columns)))
        method, scope = input.llm_method(), input.llm_scope()
        try:
            shap_p, lime_p, err = _explain_payloads(st, "LIME", scope)
            if not err:
                frames.update(export_lime_frames(lime_p))
        except Exception:
            pass
        if st["model_type"] not in ("XGBoost", "Random Forest"):
            try:
                summary, boot, _ = _coef_bundle(st)
                frames.update(export_coefficient_frame(summary, boot))
            except Exception:
                pass
        frames.update(export_metrics_frame(st.get("metrics"), st.get("extra")))
        frames.update(export_sensitivity_frames(sens_state.get()))

        cmp_res = (sens_state.get() or {}).get("model_compare")
        if cmp_res and "combined" in cmp_res:
            frames["model_comparison"] = cmp_res["combined"]
            frames["model_comparison_agreement"] = pd.DataFrame(cmp_res["agreement"])

        narratives = []
        for q, a, g, m in (chat_state.get() or []):
            if g is not None:
                narratives.append({"kind": "followup", "generator": m.get("model", ""),
                                   "question": q, "text": a, "grade": g})
        frames.update(export_narrative_frame(narratives))
        return frames

    def _figure_paths(st):
        """Both galleries store {label: path}, so values are collected, not the dicts added."""
        out = []
        for key in ("shap_paths_disk", "lime_paths_disk"):
            d = st.get(key) or {}
            out.extend(list(d.values()) if isinstance(d, dict) else list(d))
        return [p for p in out if p and os.path.exists(p)]

    def _manifest(st):
        figs = [os.path.basename(p) for p in _figure_paths(st)]
        return run_manifest(st=st, config=_export_config(), figures=figs)

    def _stamp():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @render.download(filename=lambda: f"shap_llm_results_{_stamp()}.zip")
    def dl_all():
        st = state.get()
        if not st or "error" in st:
            yield b"Run the analysis first."
            return
        yield bundle_to_zip(build_export_bundle(_collect_frames(st), _manifest(st),
                                               _figure_paths(st)))

    @render.download(filename=lambda: f"shap_long_{_stamp()}.csv")
    def dl_shap():
        st = state.get()
        if not st or "error" in st:
            yield "Run the analysis first."
            return
        f = export_shap_frames(st.get("shap_values"), st.get("X_plot_used"),
                               feature_names=list(st["X_train"].columns))
        yield (f["shap_long"].to_csv(index=False) if f
               else "No SHAP values available for this run.")

    @render.download(filename=lambda: f"lime_weights_{_stamp()}.csv")
    def dl_lime():
        st = state.get()
        if not st or "error" in st:
            yield "Run the analysis first."
            return
        _, lime_p, err = _explain_payloads(st, "LIME", input.llm_scope())
        if err or not lime_p:
            yield f"No LIME payload available: {err or 'not computed'}"
            return
        f = export_lime_frames(lime_p)
        yield (f["lime_global"].to_csv(index=False) if "lime_global" in f
               else "No LIME weights available.")

    @render.download(filename=lambda: f"coefficients_{_stamp()}.csv")
    def dl_coef():
        st = state.get()
        if not st or "error" in st:
            yield "Run the analysis first."
            return
        if st["model_type"] in ("XGBoost", "Random Forest"):
            yield (f"{st['model_type']} has no coefficients. Use a Linear, Logistic or Cox "
                   f"model for a conventional parameter summary.")
            return
        summary, boot, _ = _coef_bundle(st)
        f = export_coefficient_frame(summary, boot)
        yield (f["coefficients"].to_csv(index=False) if f
               else "No coefficient summary available.")

    @render.download(filename=lambda: f"manifest_{_stamp()}.json")
    def dl_manifest():
        st = state.get()
        if not st or "error" in st:
            yield json.dumps({"error": "Run the analysis first."}, indent=2)
            return
        yield json.dumps(_manifest(st), indent=2, default=str)

    # -------------------------------------------------------------------------
    # Interactive follow-up
    # -------------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.chat_clear)
    def _chat_clear():
        chat_state.set([])

    @reactive.effect
    @reactive.event(input.chat_ask)
    def _chat_ask():
        q = str(input.chat_q() or "").strip()
        if not q:
            return
        st = state.get()
        if not st or "error" in st:
            chat_state.set((chat_state.get() or [])
                           + [(q, "Run the main analysis first.", None, {})])
            return

        method, scope = input.llm_method(), input.llm_scope()
        shap_payload, lime_payload, err = _explain_payloads(st, method, scope)
        if err:
            chat_state.set((chat_state.get() or []) + [(q, err, None, {})])
            return

        turns = chat_state.get() or []
        ui.notification_show("Asking the model...", duration=None, id="chat_note",
                             type="message")
        try:
            answer, grade, meta = llm_followup(
                question=q,
                history=[(t[0], t[1]) for t in turns if t[2] is not None],
                explain_method=method, explain_scope=scope,
                outcome_type=st["outcome_type"], loss_name=st["loss_name"],
                model_type=st["model_type"], shap_payload=shap_payload,
                lime_payload=lime_payload,
                feature_names=list(st["X_train"].columns),
                top_k=int(input.sens_top_k()), tol=float(input.sens_tol()),
                llm_model=str(input.llm_model()).strip() or LLM_MODEL_DEFAULT,
                base_url=str(input.llm_base_url()).strip(),
                feature_note=_feature_scale_note(st), user_prompt=input.prompt(),
            )
        except Exception as e:
            answer, grade, meta = f"[failed] {e}", None, {}
        finally:
            ui.notification_remove("chat_note")

        chat_state.set(turns + [(q, answer, grade, meta)])
        ui.update_text_area("chat_q", value="", session=session)

    @output
    @render.text
    def chat_out():
        turns = chat_state.get() or []
        if not turns:
            return ("Run the main analysis, then ask a question above.\n\n"
                    "The conversation is grounded on the same payload as the LLM "
                    "explanation tab -- the same features, values and attributions, "
                    "nothing more. Questions that cannot be answered from it should be "
                    "refused by the model; if instead it answers confidently, the "
                    "faithfulness line under the answer will say so.\n\n"
                    "A deterministic template has no counterpart to this tab: it can only "
                    "emit the report it was written to emit.")
        lines = []
        for q, a, g, m in turns:
            if g is None:
                lines += [f"YOU: {q}", "", a, "-" * 88]
            else:
                lines += format_followup_turn(q, a, g, m)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Sensitivity analysis
    # -------------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.run_sens)
    def _run_sens():
        st = state.get()
        if not st or "error" in st:
            sens_state.set({"error": "Run the main analysis first."})
            return

        which = input.sens_which()
        scope = input.sens_scope()
        top_k = int(input.sens_top_k())
        out = {"which": which, "scope": scope}

        X_plot = st.get("X_plot_used")
        if X_plot is None or len(X_plot) == 0:
            X_plot = st["X_test"] if len(st["X_test"]) else st["X_train"]
            X_plot = X_plot.copy().reset_index(drop=True).iloc[:200]
        i = int(input.patient_idx())

        ui.notification_show("Running sensitivity analysis...", duration=None,
                             id="sens_note", type="message")
        try:
            if which in ("LIME stability", "Both"):
                try:
                    out["lime"] = lime_stability_analysis(
                        st["model"], st["X_train"], X_plot,
                        model_type=st["model_type"], outcome_type=st["outcome_type"],
                        loss_name=st["loss_name"], patient_idx=i,
                        sample_sizes=_parse_numlist(input.sens_lime_samples(), int) or [500],
                        kernel_multipliers=_parse_numlist(input.sens_lime_kw()) or [1.0],
                        repeats=int(input.sens_lime_repeats()),
                        scope=scope, global_n=int(input.sens_lime_global_n()),
                        num_features=10, top_k=top_k,
                        shap_values=st.get("shap_values"),
                    )
                except Exception as e:
                    out["lime"] = {"error": str(e)}

            if which in ("LLM temperature sweep", "Both"):
                try:
                    if "explainer" not in st or "shap_values" not in st:
                        raise ValueError("No SHAP artifacts available; run the analysis again.")
                    payload = build_shap_payload(
                        st["explainer"], st["shap_values"], st["X_plot_used"],
                        patient_idx=i, top_k_local=8, top_k_global=12,
                    )
                    payload = apply_privacy_filter(payload, scope=scope,
                                                   mode=input.llm_privacy())
                    names = [m.strip() for m in
                             str(input.sens_llm_model()).replace(";", ",").split(",")
                             if m.strip()] or [LLM_MODEL_DEFAULT]
                    common = dict(
                        explain_method="SHAP", explain_scope=scope,
                        outcome_type=st["outcome_type"], loss_name=st["loss_name"],
                        model_type=st["model_type"], user_prompt=input.prompt(),
                        shap_payload=payload,
                        temperatures=_parse_numlist(input.sens_llm_temps()) or [0.0],
                        n_reps=int(input.sens_llm_reps()), top_k=top_k,
                        tol=float(input.sens_tol()),
                        feature_names=list(st["X_train"].columns),
                        base_url=str(input.llm_base_url()).strip(),
                    )
                    if len(names) > 1:
                        out["llm_compare"] = llm_model_comparison(models=names, **common)
                    else:
                        out["llm"] = llm_stability_analysis(llm_model=names[0], **common)
                except Exception as e:
                    out["llm"] = {"error": str(e)}

            if which == "Attributions across models":
                try:
                    fa = st.get("fit_args") or {}
                    chosen = list(input.cmp_models())
                    if len(chosen) < 2:
                        raise ValueError("Select at least two model families to compare.")
                    out["model_compare"] = compare_model_attributions(
                        st["df"], st["feature_cols"],
                        outcome_type=st["outcome_type"], loss_name=st["loss_name"],
                        model_types=chosen, top_k=top_k,
                        outcome_col=fa.get("outcome_col"), treat_col=fa.get("treat_col"),
                        event_col=fa.get("event_col"), sigpos_col=fa.get("sigpos_col"),
                        test_size=float(input.test_size()), seed=42,
                        na_policy=fa.get("na_policy", "Drop rows with any missing value"),
                        standardize=bool(fa.get("standardize", False)),
                        rf_n_estimators=int(input.rf_n_estimators()),
                        **(fa.get("penalties") or {}))
                except Exception as e:
                    out["model_compare"] = {"error": str(e)}

            if which in ("Template baseline vs LLM", "Audience adaptation"):
                try:
                    if "explainer" not in st or "shap_values" not in st:
                        raise ValueError("No SHAP artifacts available; run the analysis again.")
                    payload = build_shap_payload(
                        st["explainer"], st["shap_values"], st["X_plot_used"],
                        patient_idx=i, top_k_local=8, top_k_global=12,
                    )
                    payload = apply_privacy_filter(payload, scope=scope,
                                                   mode=input.llm_privacy())
                    model_name = (str(input.sens_llm_model()).replace(";", ",").split(",")[0]
                                  .strip() or LLM_MODEL_DEFAULT)
                    shared = dict(
                        explain_method="SHAP", explain_scope=scope,
                        outcome_type=st["outcome_type"], loss_name=st["loss_name"],
                        model_type=st["model_type"], shap_payload=payload,
                        feature_names=list(st["X_train"].columns),
                        top_k=top_k, tol=float(input.sens_tol()),
                        llm_model=model_name,
                        base_url=str(input.llm_base_url()).strip(),
                        feature_note=_feature_scale_note(st),
                        n_reps=int(input.sens_baseline_reps()),
                    )
                    if which == "Template baseline vs LLM":
                        out["baseline"] = compare_generators(
                            user_prompt=input.prompt(), **shared)
                    else:
                        out["adaptation"] = adaptation_analysis(
                            prompts=[("Audience A", str(input.sens_aud_a()).strip()),
                                     ("Audience B", str(input.sens_aud_b()).strip())],
                            **shared)
                except Exception as e:
                    key = ("baseline" if which == "Template baseline vs LLM" else "adaptation")
                    out[key] = {"error": str(e)}
        finally:
            ui.notification_remove("sens_note")

        sens_state.set(out)

    @output
    @render.text
    @reactive.event(input.run_sens, ignore_none=False)
    def sens_out():
        s = sens_state.get()
        if not s:
            return ("Run the main analysis, then open 'Sensitivity analysis' in the sidebar "
                    "and press 'Run sensitivity analysis'.\n\n"
                    "LIME stability needs no API key. The LLM temperature sweep makes "
                    "generations x temperatures API calls.")
        keys = ("lime", "llm", "llm_compare", "baseline", "adaptation", "model_compare")
        if "error" in s and not any(k in s for k in keys):
            return s["error"]

        lines = []
        if "lime" in s:
            lines += format_lime_stability(s["lime"])
        for key, fmt in (("llm", format_llm_stability),
                         ("llm_compare", format_llm_comparison),
                         ("baseline", format_generator_comparison),
                         ("adaptation", format_adaptation),
                         ("model_compare", format_model_comparison)):
            if key in s:
                if lines:
                    lines += ["", ""]
                lines += fmt(s[key])
        return "\n".join(lines) if lines else "Nothing to report."


app = App(app_ui, server)
