from abc import ABC, abstractmethod

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# from xgboost import XGBRegressor


class EarlyStoppingAdapter(ABC):
    n_estimators_param: str

    @abstractmethod
    def fit_with_eval(self, model, X_fit, y_fit, X_es, y_es, rounds, eval_metric):
        """Обучает model с eval_set, возвращает обученную модель."""

    @abstractmethod
    def get_best_iteration(self, fitted_model) -> int: ...

    @abstractmethod
    def predict_at_best(self, fitted_model, X, best_iteration):
        """Предсказание с учётом лучшей итерации (для сравнения кандидатов на holdout)."""


class LGBMAdapter(EarlyStoppingAdapter):
    n_estimators_param = "n_estimators"

    def fit_with_eval(self, model, X_fit, y_fit, X_es, y_es, rounds, eval_metric):
        import lightgbm as lgb

        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=rounds, verbose=False)],
        )
        return model

    def get_best_iteration(self, fitted_model) -> int:
        return fitted_model.best_iteration_

    def predict_at_best(self, fitted_model, X, best_iteration):
        return fitted_model.predict(X, num_iteration=best_iteration)


class CatBoostAdapter(EarlyStoppingAdapter):
    n_estimators_param = "iterations"

    def fit_with_eval(self, model, X_fit, y_fit, X_es, y_es, rounds, eval_metric):
        model.fit(
            X_fit,
            y_fit,
            eval_set=(X_es, y_es),
            early_stopping_rounds=rounds,
            use_best_model=True,
            verbose=False,
        )
        return model

    def get_best_iteration(self, fitted_model) -> int:
        return fitted_model.get_best_iteration()

    def predict_at_best(self, fitted_model, X, best_iteration):
        return fitted_model.predict(X)


ADAPTER_REGISTRY = {
    LGBMRegressor: LGBMAdapter(),
    CatBoostRegressor: CatBoostAdapter(),
}


def get_adapter(model) -> EarlyStoppingAdapter:
    for cls, adapter in ADAPTER_REGISTRY.items():
        if isinstance(model, cls):
            return adapter
    raise NotImplementedError(
        f"Нет early-stopping адаптера для {type(model).__name__}. Добавьте новый адаптер в ADAPTER_REGISTRY."
    )
