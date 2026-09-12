from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "LGBMRegressor": LGBMRegressor,
    "SVR": SVR,
}


def build_model(model_cfg):
    cls = REGISTRY[model_cfg["estimator"]]
    return cls(**model_cfg.get("params", {}))
