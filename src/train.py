import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def cv_result(
    model,
    X_train,
    y_train,
    y_strat,
    cv_splitter,
    preprocessor,
    name=None,
    fit_params=None,
):
    fit_params = fit_params or {}
    model_pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    fitted_models = []
    oof_preds_sum = np.zeros(len(X_train), dtype=float)
    oof_counts = np.zeros(len(X_train), dtype=float)

    tr_scores, val_scores = [], []

    splits = list(cv_splitter.split(X_train, y_strat))

    for train_idx, val_idx in tqdm(
        splits, desc=name or "CV folds", unit="fold", leave=False
    ):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipe = clone(model_pipe)

        fold_pipe.fit(X_tr, y_tr, **fit_params)

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

    return fitted_models, oof_preds, metrics, val_scores


def train_cv_ensemble(model, pipeline, X, y, y_strat, cv_splitter):
    fitted_models, oof_preds, metrics, val_scores = cv_result(
        model, X, y, y_strat, cv_splitter, pipeline
    )
    return fitted_models, metrics, val_scores


def train_full_refit(model, pipeline, X, y):
    full_pipe = clone(Pipeline([("preprocessor", pipeline), ("model", model)]))
    full_pipe.fit(X, y)
    return [full_pipe]  # список из одной модели — единый интерфейс с ensemble-вариантом


def run_final_training(strategy, model, pipeline, X, y, y_strat, cv_splitter):
    if strategy == "cv_ensemble":
        fitted_models, _, _ = train_cv_ensemble(
            model, pipeline, X, y, y_strat, cv_splitter
        )
        return fitted_models
    elif strategy == "full_refit":
        return train_full_refit(model, pipeline, X, y)
    raise ValueError(strategy)
