import math

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.adapters import get_adapter
from src.config import cfg


def cv_result(
    model,
    X_train,
    y_train,
    y_strat,
    cv_splitter,
    preprocessor,
    name=None,
    fit_params=None,
    use_early_stopping=False,
    early_stopping_callback=None,
    calculate_importance=False,
    perm_n_repeats=5,
    random_state=cfg.general.gseed,
):
    """
    Universal cv function, but could use LGBM early stopping
    """
    fit_params = fit_params or {}
    model_pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    fitted_models = []
    oof_preds_sum = np.zeros(len(X_train), dtype=float)
    oof_counts = np.zeros(len(X_train), dtype=float)

    tr_scores, val_scores = [], []

    fold_importances = []

    splits = list(cv_splitter.split(X_train, y_strat))

    for train_idx, val_idx in tqdm(
        splits, desc=name or "CV folds", unit="fold", leave=False
    ):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipe = clone(model_pipe)

        fold_fit_params = fit_params.copy()

        if use_early_stopping:
            fold_preprocessor = clone(preprocessor)
            X_tr_trans = fold_preprocessor.fit_transform(X_tr, y_tr)
            X_val_trans = fold_preprocessor.transform(X_val)

            fold_fit_params["model__eval_X"] = X_val_trans
            fold_fit_params["model__eval_y"] = y_val

            if early_stopping_callback is not None:
                fold_fit_params["model__callbacks"] = [early_stopping_callback]

            fold_pipe.fit(X_tr, y_tr, **fold_fit_params)
        else:
            fold_pipe.fit(X_tr, y_tr, **fold_fit_params)

        if calculate_importance:
            res = permutation_importance(
                fold_pipe,
                X_val,
                y_val,
                scoring="neg_root_mean_squared_error",
                n_repeats=perm_n_repeats,
                random_state=random_state,
                n_jobs=-1,
            )
            fold_importances.append(res.importances_mean)

        fold_preds_tr = fold_pipe.predict(X_tr)
        fold_preds_val = fold_pipe.predict(X_val)

        oof_preds_sum[val_idx] += fold_preds_val
        oof_counts[val_idx] += 1

        fitted_models.append(fold_pipe)

        tr_scores.append(root_mean_squared_error(y_tr, fold_preds_tr))
        val_scores.append(root_mean_squared_error(y_val, fold_preds_val))

    oof_preds = oof_preds_sum / oof_counts
    oof_mse = root_mean_squared_error(y_train, oof_preds)

    metrics = pd.DataFrame(
        [
            {
                "model": name,
                "TRAIN_rmsle_MEAN": np.mean(tr_scores),
                "TRAIN_rmsle_STD": np.std(tr_scores),
                "VAL_rmsle_MEAN": np.mean(val_scores),
                "VAL_rmsle_STD": np.std(val_scores),
                "OOF_rmsle": oof_mse,
            }
        ]
    )

    importance_df = None
    if calculate_importance:
        importance_df = pd.DataFrame(fold_importances, columns=X_train.columns).T
        fold_cols = importance_df.columns.tolist()

        importance_df["importance_mean"] = importance_df[fold_cols].mean(axis=1)
        importance_df["importance_std"] = importance_df[fold_cols].std(axis=1)

        importance_df = importance_df.sort_values(by="importance_mean", ascending=False)

        return fitted_models, oof_preds, metrics, val_scores, importance_df
    return fitted_models, oof_preds, metrics, val_scores


def train_cv_ensemble(model, pipeline, X, y, y_strat, cv_splitter):
    fitted_models, oof_preds, metrics, val_scores = cv_result(
        model,
        X,
        y,
        y_strat,
        cv_splitter,
        pipeline,
    )
    return fitted_models, metrics, val_scores


def train_full_refit(model, pipeline, X, y, fit_params=None, auto_cat_features=False):
    fit_params = fit_params or {}
    prep = clone(pipeline)
    X_t = prep.fit_transform(X, y)

    if auto_cat_features:
        cat_cols = X_t.select_dtypes(include="category").columns.tolist()
        fit_params = {**fit_params, "cat_features": cat_cols}

    model_clone = clone(model)
    prefixed_fit_params = {k: v for k, v in fit_params.items()}
    model_clone.fit(X_t, y, **prefixed_fit_params)

    full_pipe = Pipeline([("preprocessor", prep), ("model", model_clone)])
    full_pipe.steps[0] = ("preprocessor", prep)
    full_pipe.steps[1] = ("model", model_clone)
    return [full_pipe]


def train_early_stopping_refit(
    model,
    pipeline,
    X,
    y,
    holdout_frac: float = 0.1,
    early_stopping_rounds: int = 100,
    eval_metric: str = "rmse",
    random_state: int = cfg.general.seed,
):
    adapter = get_adapter(model)

    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=holdout_frac, random_state=random_state
    )

    probe_preprocessor = clone(pipeline)
    X_fit_t = probe_preprocessor.fit_transform(X_fit, y_fit)
    X_es_t = probe_preprocessor.transform(X_es)
    # var A: base iter
    model_a = clone(model)
    model_a = adapter.fit_with_eval(
        model_a, X_fit_t, y_fit, X_es_t, y_es, early_stopping_rounds, eval_metric
    )
    base_iterations = adapter.get_best_iteration(model_a)
    score_base = root_mean_squared_error(
        y_es, adapter.predict_at_best(model_a, X_es_t, base_iterations)
    )
    # var B: extra iter
    scale = 1.0 / (1.0 - holdout_frac)
    scaled_iterations = math.ceil(base_iterations * scale)

    model_b = clone(model).set_params(**{adapter.n_estimators_param: scaled_iterations})
    model_b.fit(X_fit_t, y_fit)
    score_scaled = root_mean_squared_error(y_es, model_b.predict(X_es_t))

    final_iterations = (
        base_iterations if score_base <= score_scaled else scaled_iterations
    )

    final_model = clone(model).set_params(
        **{adapter.n_estimators_param: final_iterations}
    )
    final_pipe = Pipeline([("preprocessor", clone(pipeline)), ("model", final_model)])
    final_pipe.fit(X, y)

    return [final_pipe]


def train_topk_cv_ensemble(
    base_model, param_sets, pipeline, X, y, y_strat, cv_splitter, name=None
):
    all_fitted = []
    summaries = []

    for rank, params in enumerate(param_sets, start=1):
        model = clone(base_model).set_params(**params)
        fitted_models, oof_preds, metrics, val_scores = cv_result(
            model,
            X,
            y,
            y_strat,
            cv_splitter,
            pipeline,
            name=f"{name}_top{rank}" if name else f"top{rank}",
        )
        all_fitted.extend(fitted_models)
        summaries.append(
            metrics.assign(rank=rank, n_folds=len(fitted_models), params=[params])
        )

    return all_fitted, pd.concat(summaries, ignore_index=True)


def get_top_k_params(study_name: str, storage: str, k: int = 3) -> list[dict]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed = sorted(completed, key=lambda t: t.value)
    return [t.params for t in completed[:k]]


def run_final_training(
    strategy,
    model,
    pipeline,
    X,
    y,
    y_strat=None,
    cv_splitter=None,
    strategy_params=None,
):
    strategy_params = strategy_params or {}
    if strategy == "full_refit":
        fit_params = strategy_params.get("fit_params")
        auto_cat_features = strategy_params.get("auto_cat_features", False)
        return train_full_refit(
            model,
            pipeline,
            X,
            y,
            fit_params=fit_params,
            auto_cat_features=auto_cat_features,
        )
    elif strategy == "cv_ensemble":
        fitted_models, _, _ = train_cv_ensemble(
            model, pipeline, X, y, y_strat, cv_splitter
        )
        return fitted_models
    elif strategy == "early_stopping_refit":
        return train_early_stopping_refit(model, pipeline, X, y, **strategy_params)

    elif strategy == "topk_cv_ensemble":
        param_sets = get_top_k_params(
            strategy_params["optuna_study_name"],
            strategy_params["optuna_storage"],
            k=strategy_params.get("top_k", 3),
        )
        fitted_models, summary = train_topk_cv_ensemble(
            model, param_sets, pipeline, X, y, y_strat, cv_splitter
        )
        print(summary[["rank", "VAL_rmsle_MEAN", "params"]])
        return fitted_models  # summary можно вернуть отдельно/залогировать
    raise ValueError(f"Unknown strategy: {strategy}")
