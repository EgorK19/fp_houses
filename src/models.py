from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "LGBMRegressor": LGBMRegressor,
    "SVR": SVR,
    "CatBoostRegressor": CatBoostRegressor,
    "ElasticNet": ElasticNet,
}


def build_model(model_cfg):
    cls = REGISTRY[model_cfg["estimator"]]
    return cls(**model_cfg.get("params", {}))
