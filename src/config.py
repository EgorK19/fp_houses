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
    "models": [],
    "final": {
        "selection_metric": "OOF_rmsle",  # по какой метрике выбирать лучшую модель
        "default_strategy": "full_refit",  # "cv_ensemble" | "full_refit"
        "submission_files": "all",  # "all" | "best"
    },
}


cfg = OmegaConf.create(config_dict)
