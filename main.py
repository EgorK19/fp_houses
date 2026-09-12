from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import cfg
from src.models import build_model
from src.processing import build_pipeline
from src.train import cv_result, run_final_training
from src.utils import (
    build_experiment_record,
    get_best_artifact_path,
    get_git_hash,
    load_artifact,
    load_data,
    log_experiment,
    save_artifact,
    set_seed,
    update_best_pointer,
)


def main():
    set_seed(cfg.general.seed)

    if cfg.mode in ("train_and_predict", "train_only"):
        X_train, y_train_raw, X_test, test_ids = load_data(cfg)
        y_train = np.log1p(y_train_raw)
        y_binned = pd.qcut(y_train, q=10, labels=False)

        rskf = RepeatedStratifiedKFold(
            n_splits=cfg.cv.n_splits,
            n_repeats=cfg.cv.n_repeats,
            random_state=cfg.general.seed,
        )

    results = []
    for model_cfg in cfg.models:
        model = build_model(model_cfg)
        pipeline = build_pipeline(model_cfg["pipeline"])
        _, _, metrics, val_scores = cv_result(
            model, X_train, y_train, y_binned, rskf, pipeline, model_cfg["name"]
        )
        results.append(metrics)

        record = build_experiment_record(
            model_cfg["name"],
            metrics,
            val_scores,
            model.get_params(),
            X_train.columns.tolist(),
            {"n_splits": cfg.cv.n_splits, "n_repeats": cfg.cv.n_repeats},
            git_commit=get_git_hash(),
        )

        comparison_table = pd.concat(results, ignore_index=True)
        print("=" * 64)
        print(comparison_table.sort_values(cfg.final.selection_metric))

        comparison_table.to_csv(
            Path(cfg.paths.outputs) / "comparison_table.csv", index=False
        )

        best_name = comparison_table.sort_values(cfg.final.selection_metric).iloc[0][
            "model"
        ]
        best_cfg = next(m for m in cfg.models if m["name"] == best_name)

        model = build_model(best_cfg)
        pipeline = build_pipeline(best_cfg["pipeline"])

        final_models = run_final_training(
            cfg.final.strategy, model, pipeline, X_train, y_train, y_binned, rskf
        )

        artifact_path = save_artifact(
            final_models,
            cfg,
            record["experiment_id"],
            best_name,
        )
        record["artifact_path"] = str(artifact_path)
        log_experiment(record, log_dir=Path(cfg.paths.logs))
        update_best_pointer(artifact_path, record["experiment_id"], cfg)

        if cfg.mode == "train_only":
            print(f"Training done. Artifact saved to {artifact_path}")
            return

    if cfg.mode in ("train_and_predict", "predict_only"):
        artifact_path = cfg.inference.artifact_path or get_best_artifact_path(cfg)
        final_models = load_artifact(artifact_path)

        _, _, X_test, test_ids = load_data(
            cfg
        )  # если predict_only - данные всё равно нужны заново
        preds_log = np.mean([m.predict(X_test) for m in final_models], axis=0)
        preds = np.expm1(preds_log)

        submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
        submission.to_csv(Path(cfg.paths.outputs) / "submission.csv", index=False)
        print(f"Submission saved using artifact: {artifact_path}")


if __name__ == "__main__":
    main()
