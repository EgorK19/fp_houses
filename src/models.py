from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.nn import MLPRegressor

REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "LGBMRegressor": LGBMRegressor,
    "SVR": SVR,
    "CatBoostRegressor": CatBoostRegressor,
    "ElasticNet": ElasticNet,
    "MLPRegressor": MLPRegressor,
}


def build_model(model_cfg):
    cls = REGISTRY[model_cfg["estimator"]]
    return cls(**model_cfg.get("params", {}))
