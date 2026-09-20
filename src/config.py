from pathlib import Path

from omegaconf import OmegaConf

raw_data_dir = Path("d:/vs_projects/fp_houses/data/raw")
database_path = Path(
    "d:/vs_projects/fp_houses/notebooks/optuna_results/optuna_studies.db"
)
optuna_storage_uri = f"sqlite:///{database_path.as_posix()}"
gseed = 101


config_dict = {
    "general": {"seed": gseed, "use_raw": False},
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
        # если None - main.py попробует взять "последний лучший" через pointer-файл
        "artifact_path": None,
    },
    "models": [
        {
            "name": "baseline",
            "estimator": "LGBMRegressor",
            "pipeline": "tree_oe&lgbm",
            "params": {
                "random_state": gseed,
                "verbosity": -1,
            },
            "run_cv": True,
            "final_strategy": "full_refit",
        },
        {
            "name": "elasticnet",
            "estimator": "ElasticNet",
            "pipeline": "final_tree_for_lin",
            "params": {
                "alpha": 0.0007518534274624432,
                "l1_ratio": 0.9928780888562401,
                "random_state": gseed,
                "max_iter": 20000,
            },
            "run_cv": True,
            "final_strategy": "full_refit",
        },
        {
            "name": "lgbm",
            "estimator": "LGBMRegressor",
            "pipeline": "final_tree",
            "params": {
                "num_leaves": 10,
                "max_depth": 3,
                "learning_rate": 0.021539596638002458,
                "min_child_samples": 10,
                "subsample": 0.6056914016781052,
                "subsample_freq": 1,
                "colsample_bytree": 0.5086201003596847,
                "reg_alpha": 2.3384784804742476e-05,
                "reg_lambda": 0.17188789086338954,
                "n_estimators": 5000,
                "random_state": gseed,
                "verbosity": -1,
            },
            "run_cv": True,
            "final_strategy": "cv_ensemble",
        },
        {
            "name": "catboost",
            "estimator": "CatBoostRegressor",
            "pipeline": "final_tree",
            "params": {
                "depth": 6,
                "learning_rate": 0.014078507251350254,
                "l2_leaf_reg": 1.1965964748905311,
                "random_strength": 5.161452661662928,
                "bagging_temperature": 0.6846172064818904,
                "border_count": 217,
                "iterations": 5000,
                "loss_function": "RMSE",
                "random_state": gseed,
                "thread_count": -1,
                "silent": True,
            },
            "run_cv": False,
            "final_strategy": "early_stopping_holdout",
            "final_strategy_params": {"holdout_frac": 0.1, "auto_cat_features": True},
        },
        {
            "name": "nn",
            "estimator": "MLPRegressor",
            "pipeline": "final_lin",
            "params": {
                "hidden_size": 128,
                "lr": 0.1,
                "epochs": 500,
                "patience": 50,
                "scheduler_patience": 15,
                "batch_size": 32,
                "seed": gseed,
            },
            "run_cv": True,
            "final_strategy": "cv_ensemble",
        },
    ],
    "final": {
        "selection_metric": "OOF_rmsle",  # по какой метрике выбирать лучшую модель
        "default_strategy": "full_refit",  # "cv_ensemble" | "full_refit"
        "submission_files": "all",  # "all" | "best"
    },
}


cfg = OmegaConf.create(config_dict)
