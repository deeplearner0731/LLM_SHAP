import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

import xgboost as xgb
import shap

from lime.lime_tabular import LimeTabularExplainer
from openai import OpenAI
from sklearn.linear_model import Ridge
from typing import Optional, Dict, Any

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
            h_eta = (prob * (1.0 - prob) @ event_vec)
            grad = c * g_eta
            hess = (c ** 2) * np.clip(h_eta, 1e-12, None)
            return grad, hess

        return obj, None

    if loss_name == "W-learning":
        cw = (1.0 - trt_pm) / 2.0 + pi * trt_pm
        cw = np.clip(cw, 1e-6, None)
        w = 1.0 / cw

        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            eta = trt_pm * predt
            exp_eta = np.exp(np.clip(eta, -50, 50))
            denom = exp_eta @ R
            denom = np.clip(denom, 1e-12, None)
            prob = (exp_eta[:, None] / denom[None, :]) * R
            g_eta = -event_vec + (prob @ event_vec)
            h_eta = (prob * (1.0 - prob) @ event_vec)
            grad = w * trt_pm * g_eta
            hess = w * (trt_pm ** 2) * np.clip(h_eta, 1e-12, None)
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
    num_boost_round=400
):
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

    # --- Linear Regression
    if model_type == "Linear Regression":
        if outcome_type != "continuous":
            raise ValueError("Linear Regression option is only supported for continuous outcome.")
        if loss_name != "Original (clinical)":
            raise ValueError("Linear Regression only supports Original (clinical) in this template.")

        model = LinearRegression()
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

        extra = {
            "sigpos_exists": sigpos_exists,
            "sigpos_train": sigpos_train,
            "sigpos_test": sigpos_test,
        }
        return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- Logistic Regression (binary, original-style only)
    if model_type == "Logistic Regression":
        if outcome_type != "binary":
            raise ValueError("Logistic Regression option is only supported for binary outcome.")
        if loss_name != "Original (clinical)":
            raise ValueError("Logistic Regression baseline supports only Original (clinical).")

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train.values, y_train.astype(int))

        prob_tr = np.clip(model.predict_proba(X_train.values)[:, 1], 1e-12, 1 - 1e-12)
        metrics_out = {
            "model": "Logistic Regression",
            "outcome_type": "binary",
            "loss": "Original (clinical)",
            "train_auc": float(_safe_auc(y_train.astype(int), prob_tr)),
            "train_loss": float(log_loss(y_train.astype(int), prob_tr)),
        }

        if HAS_TEST:
            prob_te = np.clip(model.predict_proba(X_test.values)[:, 1], 1e-12, 1 - 1e-12)
            metrics_out["test_auc"] = float(_safe_auc(y_test.astype(int), prob_te))
            metrics_out["test_loss"] = float(log_loss(y_test.astype(int), prob_te))

        extra = {
            "sigpos_exists": sigpos_exists,
            "sigpos_train": sigpos_train,
            "sigpos_test": sigpos_test,
        }
        return model, X_train, X_test, y_train, y_test, metrics_out, None, extra

    # --- Cox Regression baseline (time-to-event, original-style only; requires lifelines)
    if model_type == "Cox Regression":
        if outcome_type != "time-to-event":
            raise ValueError("Cox Regression option is only supported for time-to-event outcome.")
        if loss_name != "Original (clinical)":
            raise ValueError("Cox Regression baseline supports only Original (clinical).")
        if not _HAS_LIFELINES:
            raise ValueError("lifelines is not installed. Please `pip install lifelines` to use Cox Regression.")

        df_tr = X_train.copy()
        df_tr = df_tr.assign(**{
            outcome_col: time_train.astype(float),
            event_col: event_train.astype(int)
        })

        cox = CoxPHFitter()
        cox.fit(df_tr, duration_col=outcome_col, event_col=event_col)

        score_tr = cox.predict_partial_hazard(df_tr[X_train.columns]).values.reshape(-1)

        metrics_out = {
            "model": "Cox Regression",
            "outcome_type": "time-to-event",
            "loss": "Original (clinical)",
            "train_c_index": float(concordance_index(time_train, event_train, score_tr)),
        }

        score_te = None
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
        }
        return cox, X_train, X_test, y_train, y_test, metrics_out, None, extra

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
        dtrain.set_label(time_train)
        if HAS_TEST:
            dtest.set_label(time_test)
        if loss_name == "Original (clinical)":
            params["objective"] = "survival:cox"
            params["eval_metric"] = "cox-nloglik"
        else:
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
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, pred_raw_test)

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
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, pred_raw_test)

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
                metrics_out["train_auc_sigpos"] = _safe_auc(sigpos_train, pred_raw_train)
                if HAS_TEST:
                    metrics_out["test_auc_sigpos"] = _safe_auc(sigpos_test, pred_raw_test)

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
    }

    return model, X_train, X_test, y_train, y_test, metrics_out, dtest, extra


# -----------------------------
# SHAP plots
# -----------------------------
def save_shap_plots(model, X_plot: pd.DataFrame, out_dir="outputs", prefix="Fig_SHAP",
                    dep_main="__AUTO_TOP__", dep_interaction="auto"):
    os.makedirs(out_dir, exist_ok=True)

    if X_plot is None or len(X_plot) == 0:
        raise ValueError("X_plot is empty; cannot generate SHAP plots.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_plot)
    shap_exp = explainer(X_plot)

    paths_disk = {}

    def _save_current_fig(fn):
        p = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(p, dpi=250, bbox_inches="tight")
        plt.close()
        return p

    # Beeswarm
    plt.figure(figsize=(8.2, 5.2))
    shap.summary_plot(shap_values, X_plot, show=False)
    plt.title("SHAP summary (beeswarm)")
    paths_disk["Beeswarm"] = _save_current_fig(f"{prefix}_A_beeswarm.png")

    # Bar
    plt.figure(figsize=(8.2, 4.8))
    shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False)
    plt.title("Mean(|SHAP|) feature importance")
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
    i = 0
    plt.figure(figsize=(7.8, 5.2))
    shap.plots.waterfall(shap_exp[i], show=False, max_display=10)
    plt.title("SHAP waterfall: patient 0")
    paths_disk["Waterfall"] = _save_current_fig(f"{prefix}_D_waterfall_patient0.png")

    plt.figure(figsize=(8.0, 4.8))
    shap.decision_plot(explainer.expected_value, shap_values[i, :], X_plot.iloc[i, :], show=False)
    plt.title("SHAP decision plot: patient 0")
    paths_disk["Decision"] = _save_current_fig(f"{prefix}_E_decision_patient0.png")

    return paths_disk, explainer, shap_values, X_plot


# -----------------------------
# LIME plots
# -----------------------------
def _make_lime_predict_fn(model, model_type, outcome_type, loss_name, feature_names=None):
    if model_type == "Linear Regression":
        def pred_reg(X):
            return model.predict(X)
        return pred_reg, "regression"

    if model_type == "Logistic Regression":
        def pred_proba(X):
            p = model.predict_proba(np.asarray(X))[:, 1]
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return np.vstack([1 - p, p]).T
        return pred_proba, "classification"

    if model_type == "Cox Regression":
        if feature_names is None:
            raise ValueError("Cox Regression LIME needs feature_names.")
        def pred_risk(X):
            Xdf = pd.DataFrame(np.asarray(X), columns=feature_names)
            s = model.predict_partial_hazard(Xdf).values.reshape(-1)
            return s
        return pred_risk, "regression"

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
):
    predict_fn, mode = _make_lime_predict_fn(
        model, model_type, outcome_type, loss_name, feature_names=list(X_train.columns)
    )

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode=mode,
        discretize_continuous=True,
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


# -----------------------------
# LLM explanation
# -----------------------------
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
):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    task_text = _describe_task(outcome_type)
    loss_text = _describe_loss(outcome_type, loss_name)

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
"""

    def _compact_payload(p):
        if explain_method.upper() == "SHAP":
            if scope == "Local":
                return {
                    "patient_index": p["patient_index"],
                    "expected_value": p["expected_value"],
                    "prediction_additive": p["prediction_additive"],
                    "local_top": p["local_top"],
                }
            else:
                return {"global_top": p["global_top"]}
        else:
            if scope == "Local":
                return {
                    "patient_index": p["patient_index"],
                    "mode": p.get("mode"),
                    "local_top_rules": p["local_top_rules"],
                }
            else:
                return {"global_mean_abs_top15": p["global_mean_abs"][:15]}

    payload_small = _compact_payload(payload)

    full_prompt = f"""{context}

USER PROMPT (style + extra constraints)
{user_prompt}

EXPLANATION PAYLOAD
{payload_small}
"""

    resp = client.responses.create(model="gpt-5.2", input=full_prompt)
    return full_prompt, resp.output_text


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
                ["XGBoost", "Linear Regression", "Logistic Regression", "Cox Regression"]
            ),

            ui.input_select(
                "loss",
                "Loss / Objective",
                ["Original (clinical)", "A-learning", "W-learning"]
            ),

            ui.hr(),
            ui.h4("XGBoost hyperparameters (optional)"),
            ui.p("Defaults are the same as the current template."),

            ui.input_numeric("xgb_n_estimators", "n_estimators (num_boost_round)", value=400, min=10),
            ui.input_numeric("xgb_learning_rate", "learning_rate (eta)", value=0.05, min=0.001, max=1, step=0.01),
            ui.input_numeric("xgb_max_depth", "max_depth", value=4, min=1, max=20, step=1),

            ui.input_slider("xgb_subsample", "subsample", min=0.1, max=1.0, value=0.9, step=0.05),
            ui.input_slider("xgb_colsample", "colsample_bytree", min=0.1, max=1.0, value=0.9, step=0.05),

            ui.input_numeric("xgb_min_child_weight", "min_child_weight", value=1.0, min=0.0, step=0.5),
            ui.input_numeric("xgb_gamma", "gamma", value=0.0, min=0.0, step=0.1),

            ui.input_numeric("xgb_reg_lambda", "reg_lambda (L2)", value=1.0, min=0.0, step=0.5),
            ui.input_numeric("xgb_reg_alpha", "reg_alpha (L1)", value=0.0, min=0.0, step=0.5),

            ui.input_select("xgb_tree_method", "tree_method", choices=["hist", "approx", "exact"], selected="hist"),

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

            ui.input_text_area(
                "prompt",
                "LLM prompt template (editable)",
                value=(
                    "You are a biomedical data scientist.\n"
                    "Explain the model prediction and SHAP/LIME results.\n"
                    "Write 3-5 sentences in plain but scientific language.\n"
                    "If A-learning or W-learning is used, interpret results as predictive biomarkers / ITR.\n"
                    "If Original loss is used, interpret results as prognostic biomarkers.\n"
                    "Avoid mentioning features not supported by the explanation.\n"
                ),
                rows=10
            ),

            ui.input_numeric("patient_idx", "Patient index for local explanation", value=0, min=0),
            ui.input_action_button("run", "Run analysis", class_="btn-primary"),
        ),

        ui.navset_tab(
            ui.nav_panel("Data preview", ui.output_table("preview")),
            ui.nav_panel("Metrics", ui.output_text_verbatim("metrics")),
            ui.nav_panel("SHAP plots", ui.output_ui("shap_gallery")),
            ui.nav_panel("LIME plots", ui.output_ui("lime_gallery")),
            ui.nav_panel("LLM explanation", ui.output_text_verbatim("llm_out")),
        )
    )
)


# -----------------------------
# Shiny Server
# -----------------------------
def server(input, output, session):
    state = reactive.Value(None)

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
    @reactive.event(input.run)
    def _run():
        fileinfo = input.file()
        if not fileinfo:
            state.set({"error": "Please upload a CSV first."})
            return

        df = pd.read_csv(fileinfo[0]["datapath"])

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

        xgb_params = dict(
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
        num_boost_round = int(input.xgb_n_estimators())

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
            )
        except Exception as e:
            state.set({"error": f"Model fitting error: {e}"})
            return

        artifacts = {
            "df": df,
            "feature_cols": feature_cols,
            "model_type": model_type,
            "loss_name": loss_name,
            "outcome_type": outcome_type,
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "metrics": metrics_out,
            "extra": extra,
        }

        # Choose plot set:
        # If no test set, use training for plots; else use test
        plot_df = X_test if (X_test is not None and len(X_test) > 0) else X_train
        plot_df = plot_df.copy().reset_index(drop=True)
        if len(plot_df) > 200:
            plot_df = plot_df.iloc[:200, :]

        # --- SHAP for XGBoost
        if model_type == "XGBoost":
            try:
                paths_disk, explainer, shap_values, X_plot_used = save_shap_plots(
                    model,
                    plot_df,
                    out_dir="outputs",
                    prefix="Fig_SHAP",
                    dep_main=input.dep_main(),
                    dep_interaction=input.dep_interaction()
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
                global_n=60
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

        if st["model_type"] == "XGBoost":
            lines.append("")
            lines.append("=== XGBoost hyperparameters used ===")
            lines.append(f"num_boost_round: {m.get('xgb_num_boost_round')}")
            xgbp = m.get("xgb_params", {})
            for k in sorted(xgbp.keys()):
                lines.append(f"{k}: {xgbp[k]}")

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
            return ui.p("Run analysis with XGBoost to generate SHAP plots.")
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

    @output
    @render.text
    @reactive.event(input.run)
    def llm_out():
        st = state.get()
        if not st or "error" in st:
            return st.get("error", "")

        explain_method = input.llm_method()
        explain_scope = input.llm_scope()
        i = int(input.patient_idx())

        shap_payload = None
        lime_payload = None

        # For LLM payloads, use same plot set logic:
        X_plot = st["X_test"].copy().reset_index(drop=True)
        if len(X_plot) == 0:
            X_plot = st["X_train"].copy().reset_index(drop=True)
        if len(X_plot) > 200:
            X_plot = X_plot.iloc[:200, :]

        if explain_method == "SHAP":
            if st["model_type"] != "XGBoost":
                return "SHAP LLM explanation currently requires XGBoost in this template."
            if "explainer" not in st or "shap_values" not in st:
                return "No SHAP artifacts available. Please run analysis again with XGBoost + SHAP."
            if i < 0 or i >= len(X_plot):
                return f"Patient index out of range. Choose 0 to {len(X_plot)-1}."

            shap_payload = build_shap_payload(
                st["explainer"],
                st["shap_values"],
                st["X_plot_used"],
                patient_idx=i,
                top_k_local=8,
                top_k_global=12
            )

        if explain_method == "LIME":
            if i < 0 or i >= len(X_plot):
                return f"Patient index out of range. Choose 0 to {len(X_plot)-1}."
            try:
                lime_payload = build_lime_payload(
                    model=st["model"],
                    X_train=st["X_train"],
                    X_plot=X_plot,
                    model_type=st["model_type"],
                    outcome_type=st["outcome_type"],
                    loss_name=st["loss_name"],
                    patient_idx=i,
                    num_features=10,
                    global_n=60,
                    global_num_samples=500,
                    ridge_alpha=0.01,
                )
            except Exception as e:
                return f"LIME payload build error: {e}"

        try:
            full_prompt, text_out = llm_explain(
                explain_method=explain_method,
                explain_scope=explain_scope,
                outcome_type=st["outcome_type"],
                loss_name=st["loss_name"],
                model_type=st["model_type"],
                user_prompt=input.prompt(),
                shap_payload=shap_payload,
                lime_payload=lime_payload,
            )
            return "=== Prompt sent to LLM ===\n" + full_prompt + "\n\n=== LLM output ===\n" + text_out
        except Exception as e:
            return f"LLM call error: {e}"


app = App(app_ui, server)
