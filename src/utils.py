import hashlib
import json
import os
import random
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import cfg
from src.processing import fix_data_bugs, remove_train_outliers


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        pass


def load_data(cfg, use_raw=cfg.general.use_raw):
    if use_raw:
        df_train = pd.read_csv(Path(cfg.paths.train))
        df_test = pd.read_csv(Path(cfg.paths.test))

        X_train = df_train.drop(columns=["Id", "SalePrice"])
        y_train_raw = df_train["SalePrice"]

        test_ids = df_test["Id"]
        X_test = df_test.drop(columns=["Id"])

    else:
        df_train = pd.read_csv(Path(cfg.paths.train))
        df_test = pd.read_csv(Path(cfg.paths.test))

        df_all_data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
        df_all_data = df_all_data.drop(columns=["SalePrice"])
        df_all_data = fix_data_bugs(df_all_data)

        X_train_raw, X_test = (
            df_all_data[: df_train.shape[0]],
            df_all_data[df_train.shape[0] :],
        )

        df_raw = remove_train_outliers(
            pd.concat([X_train_raw, df_train["SalePrice"]], axis=1)
        )

        X_train = df_raw.drop(columns=["Id", "SalePrice"])
        y_train_raw = df_raw["SalePrice"]

        test_ids = X_test["Id"]
        X_test = X_test.drop(columns=["Id"])

    return X_train, y_train_raw, X_test, test_ids


def get_git_hash():
    try:
        h = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        )
        return f"{h}{'-dirty' if dirty else ''}"
    except Exception as e:
        return f"Something wrong with get_git_hash: {e}"


def feature_set_hash(columns: list[str]) -> str:
    return hashlib.md5(",".join(sorted(columns)).encode()).hexdigest()[:8]


def build_experiment_record(
    model_name: str,
    metrics_df: pd.DataFrame,
    val_scores: list[float],
    model_params: dict,
    feature_columns: list[str],
    cv_config: dict,
    git_commit: str = "",
) -> dict:
    return {
        "experiment_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,  # сначала коммит, потом запуск
        "model": model_name,
        "model_params": model_params,
        "feature_hash": feature_set_hash(feature_columns),
        "n_features": len(feature_columns),
        "cv_config": cv_config,  # {"n_splits": __ ,"n_repeats": __ ,"random_state": __ }
        "metrics": metrics_df.to_dict(orient="records")[0],
        "val_scores_raw": val_scores,  # для paired-тестов задним числом
    }


def log_fe_experiment(metrics: dict, note: str = ""):
    log_dir = Path(cfg.paths.logs)
    log_file_path = log_dir / "fe_experiments.log"

    log_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "note": note,
    }

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def log_experiment(record: dict, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "experiments.jsonl"
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# usage
# fitted_models, oof_preds, metrics, val_scores = cv_result(...)
# record = build_experiment_record(...)
# log_experiment(record, log_dir=Path(cfg.paths.logs))


# просмотр экспериментов
def load_experiments(log_dir: Path) -> pd.DataFrame:
    path = log_dir / "experiments.jsonl"
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    df = pd.json_normalize(records)
    return df


def save_artifact(
    models,
    cfg,
    experiment_id: str,
    model_name: str,
):
    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = models_dir / f"{model_name}_{experiment_id}.pkl"
    joblib.dump(models, artifact_path)

    return artifact_path


def load_artifact(artifact_path: str):
    return joblib.load(artifact_path)


def update_best_pointer(artifact_path: str, experiment_id: str, cfg):
    pointer_path = Path(cfg.paths.models_dir) / "best_model_pointer.json"
    pointer_path.write_text(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "experiment_id": experiment_id,
            },
            indent=2,
        )
    )


def get_best_artifact_path(cfg):
    pointer_path = Path(cfg.paths.models_dir) / "best_model_pointer.json"
    if not pointer_path.exists():
        raise FileNotFoundError("Нет сохранённых моделей - сначала запустите train.")
    return json.loads(pointer_path.read_text())["artifact_path"]


set_seed(cfg.general.seed)
