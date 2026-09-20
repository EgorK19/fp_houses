import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from src.config import cfg
from src.processing import build_cat_features_fit_params


def prepare_early_stopping_params(model, X_val, y_val, early_stopping_callback=None):
    model_name = type(model).__name__
    fit_params = {}

    if "LGBM" in model_name:
        fit_params["model__eval_X"] = X_val
        fit_params["model__eval_y"] = y_val

        if early_stopping_callback is not None:
            fit_params["model__callbacks"] = [early_stopping_callback]
        else:
            import lightgbm as lgb

            fit_params["model__callbacks"] = [
                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ]

    elif "XGB" in model_name:
        fit_params["model__eval_set"] = [(X_val, y_val)]

    elif "CatBoost" in model_name:
        fit_params["model__eval_set"] = (X_val, y_val)
        fit_params["model__early_stopping_rounds"] = 100
        fit_params["model__verbose"] = False

    elif "HistGradientBoosting" in model_name:
        fit_params["model__X_val"] = X_val
        fit_params["model__y_val"] = y_val

    elif "GradientBoosting" in model_name:
        pass

    elif "MLP" in model_name:
        fit_params["model__eval_X"] = X_val
        fit_params["model__eval_y"] = y_val

    return fit_params


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
    random_state=cfg.general.seed,
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

            framework_params = prepare_early_stopping_params(
                model=model,
                X_val=X_val_trans,
                y_val=y_val,
                early_stopping_callback=early_stopping_callback,
            )
            fold_fit_params.update(framework_params)

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


def train_cv_ensemble(
    model, pipeline, X, y, y_strat, cv_splitter, use_early_stopping=True
):
    fitted_models, oof_preds, metrics, val_scores = cv_result(
        model,
        X,
        y,
        y_strat,
        cv_splitter,
        pipeline,
        use_early_stopping=use_early_stopping,
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
    return [full_pipe]  # метрик нет - датасет использован целиком, holdout'а нет


def train_early_stopping_holdout(
    model,
    pipeline,
    X,
    y,
    holdout_frac: float = 0.1,
    random_state: int = cfg.general.seed,
    auto_cat_features: bool = False,
):

    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=holdout_frac, random_state=random_state
    )

    prep_for_es = clone(pipeline)
    prep_for_es.fit(X_fit, y_fit)
    X_es_trans = prep_for_es.transform(X_es)

    fit_params = prepare_early_stopping_params(
        model=model, X_val=X_es_trans, y_val=y_es
    )
    if auto_cat_features:
        fit_params.update(
            build_cat_features_fit_params(model, pipeline, prefix="model__")
        )

    model_pipe = Pipeline([("preprocessor", clone(pipeline)), ("model", clone(model))])
    model_pipe.fit(X_fit, y_fit, **fit_params)

    train_pred = model_pipe.predict(X_fit)
    es_pred = model_pipe.predict(X_es)
    metrics = pd.DataFrame(
        [
            {
                "model": None,  # проставит main.py
                "TRAIN_rmsle_MEAN": root_mean_squared_error(y_fit, train_pred),
                "TRAIN_rmsle_STD": np.nan,  # один сплит - std не из чего считать
                "VAL_rmsle_MEAN": root_mean_squared_error(y_es, es_pred),
                "VAL_rmsle_STD": np.nan,
                "OOF_rmsle": root_mean_squared_error(y_es, es_pred),  # = VAL, не OOF
            }
        ]
    )
    return [model_pipe], metrics


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
        final_models = train_full_refit(
            model,
            pipeline,
            X,
            y,
            fit_params=fit_params,
            auto_cat_features=auto_cat_features,
        )
        return final_models, None
    elif strategy == "cv_ensemble":
        fitted_models, metrics, _ = train_cv_ensemble(
            model,
            pipeline,
            X,
            y,
            y_strat,
            cv_splitter,
            use_early_stopping=strategy_params.get("use_early_stopping", True),
        )
        return fitted_models, metrics
    elif strategy == "early_stopping_holdout":
        return train_early_stopping_holdout(model, pipeline, X, y, **strategy_params)

    raise ValueError(f"Unknown strategy: {strategy}")
