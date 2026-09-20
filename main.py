import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
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
        oof_store = {}
        for model_cfg in cfg.models:
            if not model_cfg.get("run_cv", True):
                continue
            model = build_model(model_cfg)
            pipeline = build_pipeline(model_cfg["pipeline"])
            fit_params = build_cat_features_fit_params(
                model, pipeline, prefix="model__"
            )
            _, oof_preds, metrics, val_scores = cv_result(
                model,
                X_train,
                y_train,
                y_binned,
                rskf,
                pipeline,
                model_cfg["name"],
                fit_params=fit_params,
                use_early_stopping=model_cfg.get("use_early_stopping", True),
            )
            oof_store[model_cfg["name"]] = oof_preds
            results.append(metrics)
            record = build_experiment_record(
                model_name=model_cfg["name"],
                metrics_df=metrics,
                val_scores=val_scores,
                model_params=OmegaConf.to_container(
                    model_cfg.get("params", {}), resolve=True
                ),
                feature_columns=list(X_train.columns),
                cv_config={
                    "n_splits": cfg.cv.n_splits,
                    "n_repeats": cfg.cv.n_repeats,
                    "random_state": cfg.general.seed,
                },
                git_commit=get_git_hash(),
            )
            experiment_ids[model_cfg["name"]] = record["experiment_id"]
            log_experiment(record, log_dir=Path(cfg.paths.logs))

        comparison_table = (
            pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        )
        if not comparison_table.empty:
            comparison_table["protocol"] = f"rskf({cfg.cv.n_splits}x{cfg.cv.n_repeats})"
            comparison_table = comparison_table.sort_values(cfg.final.selection_metric)
            print(comparison_table)
            comparison_table.to_csv(
                Path(cfg.paths.outputs) / "comparison_table.csv", index=False
            )
            best_name = comparison_table.iloc[0]["model"]
        else:
            best_name = None

        if oof_store:
            oof_df = pd.DataFrame(oof_store, index=X_train.index)
            oof_df.insert(0, "y_true_log", y_train)
            oof_path = Path(cfg.paths.outputs) / "oof_predictions.csv"
            oof_df.to_csv(oof_path)
            print(
                f"OOF-предсказания ({list(oof_store.keys())}) сохранены в {oof_path} "
                f"- для блендинга; только с CV"
            )

        if cfg.final.submission_files == "best":
            names_to_finalize = [best_name] if best_name else []
        elif cfg.final.submission_files == "all":
            names_to_finalize = [m["name"] for m in cfg.models]
        else:
            raise ValueError(f"Unknown submission_files: {cfg.final.submission_files}")

        skipped_cv_rows = []

        for name in names_to_finalize:
            model_cfg = next(m for m in cfg.models if m["name"] == name)
            model = build_model(model_cfg)
            pipeline = build_pipeline(model_cfg["pipeline"])

            strategy = model_cfg.get("final_strategy", cfg.final.default_strategy)
            strategy_params = model_cfg.get("final_strategy_params", {})

            final_models, final_metrics = run_final_training(
                strategy,
                model,
                pipeline,
                X_train,
                y_train,
                y_binned,
                rskf,
                strategy_params=strategy_params,
            )

            if not model_cfg.get("run_cv", True) and final_metrics is not None:
                final_metrics = final_metrics.copy()
                final_metrics["model"] = name
                final_metrics["protocol"] = (
                    f"{strategy}(holdout_frac={strategy_params.get('holdout_frac')})"
                )
                skipped_cv_rows.append(final_metrics)

                record = build_experiment_record(
                    model_name=name,
                    metrics_df=final_metrics.drop(columns=["protocol"]),
                    val_scores=[float(final_metrics["VAL_rmsle_MEAN"].iloc[0])],
                    model_params=OmegaConf.to_container(
                        model_cfg.get("params", {}), resolve=True
                    ),
                    feature_columns=list(X_train.columns),
                    cv_config={"protocol": final_metrics["protocol"].iloc[0]},
                    git_commit=get_git_hash(),
                )
                experiment_ids[name] = record["experiment_id"]
                log_experiment(record, log_dir=Path(cfg.paths.logs))
            elif name not in experiment_ids:
                experiment_ids[name] = str(uuid.uuid4())[:8]

            artifact_path = save_artifact(
                final_models,
                cfg,
                experiment_ids[name],
                name,
            )
            trained_artifacts[name] = str(artifact_path)

        if skipped_cv_rows:
            extended_table = pd.concat(
                [comparison_table, *skipped_cv_rows], ignore_index=True
            )
            extended_table = extended_table.sort_values(cfg.final.selection_metric)
            print(extended_table)
            extended_table.to_csv(
                Path(cfg.paths.outputs) / "comparison_table_extended.csv", index=False
            )

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
