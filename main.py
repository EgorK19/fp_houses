import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import cfg
from src.models import build_model
from src.processing import build_cat_features_fit_params, build_pipeline
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
    trained_artifacts = {}
    experiment_ids = {}

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

            fit_params = build_cat_features_fit_params(
                model, pipeline, prefix="model__"
            )

            _, _, metrics, val_scores = cv_result(
                model,
                X_train,
                y_train,
                y_binned,
                rskf,
                pipeline,
                model_cfg["name"],
                fit_params=fit_params,
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
            experiment_ids[model_cfg["name"]] = record["experiment_id"]
            log_experiment(record, log_dir=Path(cfg.paths.logs))

        comparison_table = pd.concat(results, ignore_index=True)
        comparison_table = comparison_table.sort_values(cfg.final.selection_metric)
        print(comparison_table)
        comparison_table.to_csv(
            Path(cfg.paths.outputs) / "comparison_table.csv", index=False
        )

        best_name = comparison_table.iloc[0]["model"]

        if cfg.final.submission_files == "best":
            names_to_finalize = [best_name]
        elif cfg.final.submission_files == "all":
            names_to_finalize = comparison_table["model"].tolist()
        else:
            raise ValueError(f"Unknown submission_files: {cfg.final.submission_files}")

        for name in names_to_finalize:
            model_cfg = next(m for m in cfg.models if m["name"] == name)
            model = build_model(model_cfg)
            pipeline = build_pipeline(model_cfg["pipeline"])

            strategy = model_cfg.get("final_strategy", cfg.final.default_strategy)
            strategy_params = model_cfg.get("final_strategy_params", {})

            final_models = run_final_training(
                strategy,
                model,
                pipeline,
                X_train,
                y_train,
                y_binned,
                rskf,
                strategy_params=strategy_params,
            )

            artifact_path = save_artifact(
                final_models,
                cfg,
                experiment_ids[name],
                name,
            )
            trained_artifacts[name] = str(artifact_path)

        manifest_path = Path(cfg.paths.models_dir) / "artifacts_manifest.json"
        manifest_path.write_text(
            json.dumps(trained_artifacts, indent=2, ensure_ascii=False)
        )

        update_best_pointer(
            trained_artifacts[best_name], experiment_ids[best_name], cfg
        )

        if cfg.mode == "train_only":
            print(f"Training done. Artifacts: {trained_artifacts}")
            return

    if cfg.mode in ("train_and_predict", "predict_only"):
        _, _, X_test, test_ids = load_data(cfg)

        if cfg.final.submission_files == "best":
            artifact_path = cfg.inference.artifact_path or get_best_artifact_path(cfg)
            artifacts_to_predict = {"best": artifact_path}
        elif cfg.final.submission_files == "all":
            if trained_artifacts:
                artifacts_to_predict = trained_artifacts
            else:
                manifest_path = Path(cfg.paths.models_dir) / "artifacts_manifest.json"
                artifacts_to_predict = json.loads(manifest_path.read_text())
        else:
            raise ValueError(f"Unknown submission_files: {cfg.final.submission_files}")

        for name, artifact_path in artifacts_to_predict.items():
            final_models = load_artifact(artifact_path)
            preds_log = np.mean([m.predict(X_test) for m in final_models], axis=0)
            preds = np.expm1(preds_log)
            submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
            out_path = Path(cfg.paths.outputs) / f"submission_{name}.csv"
            submission.to_csv(out_path, index=False)
            print(f"Submission saved: {out_path}")


if __name__ == "__main__":
    main()
