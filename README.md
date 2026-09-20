# House Prices: Advanced Regression Techniques

Полный цикл решения регрессионной задачи от EDA до воспроизводимого инференса - исследование в ноутбуках, финальный пайплайн вынесен в `src/` и запускается через `main.py` по конфигу.
## О соревновании

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) - датасет Ames Housing : 1460 объектов в train, 1459 в test, 79 признаков дома

## Результаты

| модель | OOF rmsle (внутр. CV) | протокол CV | Public LB |
|---|---|---|---|
| lgbm | 0.11290 | rskf(10×5) | **0.12513** |
| catboost | 0.11003 | *holdout 90/10, не CV* | 0.12542 |
| baseline | 0.11872 | rskf(10×5) | 0.13195 |
| nn | 0.11175 | rskf(10×5) | 0.13302 |
| elasticnet | 0.11206 | rskf(10×5) | 0.13540 |

Лучшие на паблик-лидерборде - `lgbm` и `catboost` (тюненные через Optuna, см. `4.0_final_tree.ipynb` и `notebooks/optuna_results/optuna_studies.db`), почти вровень друг с другом.


Важный методологический момент: ранжирование по внутренней CV (`nn` > `elasticnet` > `lgbm` > `baseline`) почти полностью переворачивается на реальном тесте (`lgbm`/`catboost` - лучшие, `nn`/`elasticnet` - худшие). Это не опечатка - вероятная причина в том, что `nn` и `elasticnet` сильнее переобучаются на структуру train-фолдов, которая не полностью повторяется в test. Поэтому:
- `OOF_rmsle` у `catboost` - это **не** CV-метрика (у него `run_cv: False` в конфиге, слишком долгая полная CV), а скор на одном 90/10 holdout-сплите ранней остановки - сравнивать её напрямую с rskf-метриками остальных моделей некорректно, отсюда пометка протокола.
- Полная сравнительная таблица - `outputs/comparison_table.csv` (только модели с честной `10×5` CV) и `outputs/comparison_table_extended.csv` (+ catboost, с колонкой `protocol`).
- OOF-предсказания моделей с честной CV сохраняются в `outputs/oof_predictions.csv` - заготовка под блендинг/стэкинг поверх уже обученных моделей.

## Структура проекта

  

```
fp_houses/
├── main.py                        # точка входа: leaderboard CV → финальное обучение → predict → submission_*.csv
├── pyproject.toml                 # зависимости проекта (uv)
├── src/
│   ├── config.py                  # cfg (OmegaConf): пути, список моделей и их гиперпараметры, CV, финальные стратегии
│   ├── models.py                  # REGISTRY: имя эстиматора → класс
│   ├── processing.py              # build_pipeline(kind) - именованные sklearn-пайплайны препроцессинга, доп. трансформеры
│   ├── train.py                   # cv_result, prepare_early_stopping_params, стратегии финального обучения
│   ├── nn.py                      # кастомный MLPRegressor (torch) со sklearn-совместимым интерфейсом (fit/predict, early stopping)
│   ├── utils.py                   # загрузка данных, логирование экспериментов, сохранение/загрузка артефактов
│   └── tests/                     # тесты
├── notebooks/
│   ├── 0.0_draft.ipynb
│   ├── 1.0_eda.ipynb                     # разведочный анализ
│   ├── 2.0_baseline.ipynb                # зафиксированный бейзлайн (LGBM без тюнинга) - отправная точка, дальше не трогается
│   ├── 3.0_feature_engeneering.ipynb     # поиск фичей и препроцессинга
│   ├── 4.0_final_tree.ipynb              # тюнинг бустингов (Optuna), финальные деревья
│   ├── 5.0_nn.ipynb                      # эксперименты с нейросетью на torch
│   └── optuna_results/optuna_studies.db  # сырые результаты Optuna-исследований (SQLite)
├── data/
│   ├── raw/                       # train.csv, test.csv, data_description.txt, sample_submission.csv (не в репозитории)
│   ├── processed/, external/, pictures/
├── models/                        # сохранённые артефакты (*.pkl), artifacts_manifest.json, best_model_pointer.json
├── outputs/                       # comparison_table(.csv|_extended.csv), oof_predictions.csv, submission_*.csv
└── logs/                          # experiments.jsonl, fe_experiments.log

```


## Установка и запуск


Требования: Python 3.14, [uv](https://docs.astral.sh/uv/).

1. Клонировать репозиторий и перейти в его директорию.

2. Установить зависимости:

   ```bash

   uv sync

   ```

3. Скачать данные соревнования со [страницы Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) и положить `train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv` в `data/raw/` - они не хранятся в репозитории (см. `.gitignore`).

4. Запустить пайплайн:

   ```bash

   uv run main.py

   ```


Поведение целиком задаётся в `src/config.py`:
- `cfg.mode` - `train_and_predict` / `train_only` / `predict_only`;
- `cfg.models` - список моделей, у каждой свои гиперпараметры, `run_cv` (гонять ли леардборд-CV - для медленных моделей вроде catboost можно выключить) и `final_strategy` (`full_refit` / `cv_ensemble` / `early_stopping_holdout`);
- `cfg.cv`, `cfg.final` - параметры кросс-валидации и выбора/сохранения финальных моделей.

Отдельного CLI/YAML-конфига нет - правки вносятся прямо в `config.py`.

### Ограничения

- Пути в `config.py` захардкожены под Windows (`d:/vs_projects/fp_houses/...`) - при другом расположении репозитория их нужно поправить руками.

- Блендинг/стэкинг нескольких моделей в `main.py` пока не реализован - `oof_predictions.csv` подготовлен как заготовка под него (подробнее см. историю решений).