from pathlib import Path

from omegaconf import OmegaConf

raw_data_dir = Path("d:/vs_projects/fp_houses/data/raw")

config_dict = {
    "general": {
        "seed": 101,
    },
    "paths": {
        "raw_dir": str(raw_data_dir),
        "pictures": str(Path("d:/vs_projects/fp_houses/data/pictures")),
        "train": str(raw_data_dir / "train.csv"),
        "test": str(raw_data_dir / "test.csv"),
        "models_dir": str(Path("d:/vs_projects/fp_houses/models")),
        "logs": str(Path("d:/vs_projects/fp_houses/logs")),
        "outputs": str(Path("d:/vs_projects/fp_houses/outputs")),
    },
    "cv": {"n_splits": 10, "n_repeats": 5},
    "mode": "train_and_predict",  # "train_and_predict" | "train_only" | "predict_only"
    "inference": {
        # путь к конкретному артефакту для predict_only режима
        # если None — main.py попробует взять "последний лучший" через pointer-файл
        "artifact_path": None,
    },
    "models": [
        {
            "name": "linear_regression",
            "pipeline": "linear",
            "estimator": "LinearRegression",
            "params": {},
        },
        {
            "name": "ridge_regression",
            "pipeline": "linear",
            "estimator": "Ridge",
            "params": {"alpha": 1},
        },
        {
            "name": "knn_regressor",
            "pipeline": "linear",
            "estimator": "KNeighborsRegressor",
            "params": {},
        },
        {
            "name": "rfr_regressor",
            "pipeline": "tree_oe&ohe",
            "estimator": "RandomForestRegressor",
            "params": {},
        },
        {
            "name": "lgbm_oe&ohe",
            "pipeline": "tree_oe&ohe",
            "estimator": "LGBMRegressor",
            "params": {"verbosity": -1},
        },
        {
            "name": "lgbm_oe&lgbm",
            "pipeline": "tree_oe&lgbm",
            "estimator": "LGBMRegressor",
            "params": {"verbosity": -1},
        },
        {
            "name": "lgbm_encoder",
            "pipeline": "tree_lgbm_only",
            "estimator": "LGBMRegressor",
            "params": {"verbosity": -1},
        },
    ],
    "final": {
        "selection_metric": "OOF_rmsle",  # по какой метрике выбирать лучшую модель
        "strategy": "cv_ensemble",  # "cv_ensemble" | "full_refit"
    },
}


cfg = OmegaConf.create(config_dict)
