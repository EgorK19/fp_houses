import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.config import cfg
from src.processing import (
    FinalTreeTransformer,
    MemoryOptimizer,
    new_num_columns,
    new_ohe_columns,
    new_ordinal_categories,
    new_ordinal_columns,
)

final_lin = Pipeline(
    [
        ("processor", FinalTreeTransformer()),
        ("memory_optimizer", MemoryOptimizer()),
        (
            "column_transformer",
            ColumnTransformer(
                transformers=[
                    ("num_features", "passthrough", new_num_columns),
                    (
                        "cat_features_oe",
                        OrdinalEncoder(
                            categories=new_ordinal_categories,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        new_ordinal_columns,
                    ),
                    (
                        "cat_features_ohe",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                        new_ohe_columns,
                    ),
                ],
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        ("scaler", StandardScaler().set_output(transform="pandas")),
    ]
)


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_size=128,
        lr=3e-3,
        epochs=500,
        patience=30,
        scheduler_patience=10,
        batch_size=32,
        device="cpu",
        seed=cfg.general.seed,
        eval_set=None,
    ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.eval_set = eval_set

    def _build_model(self, n_features):
        return nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def fit(self, X, y):
        torch.manual_seed(self.seed)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.model_ = self._build_model(X_arr.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=self.scheduler_patience,
        )

        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_eval = self.eval_set is not None
        if has_eval:
            X_val, y_val = self.eval_set

            X_val_t = torch.as_tensor(
                np.asarray(X_val), dtype=torch.float32, device=self.device
            )
            y_val_t = torch.as_tensor(
                np.asarray(y_val), dtype=torch.float32, device=self.device
            ).view(-1, 1)

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            min_delta = 1e-4

        for epoch in tqdm(range(self.epochs), desc="nn"):
            self.model_.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

            if has_eval:
                self.model_.eval()
                with torch.no_grad():
                    val_preds = self.model_(X_val_t)
                    val_loss = loss_fn(val_preds, y_val_t).item()

                scheduler.step(val_loss)

                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_state = {
                        k: v.clone().cpu() for k, v in self.model_.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"[Early Stopping] Обучение остановлено на эпохе {epoch}")
                    break

        if has_eval and best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.to("cpu")
        return self

    def predict(self, X):
        self.model_.to(self.device)
        self.model_.eval()

        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_arr).to(self.device)

        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().ravel()

        self.model_.to("cpu")
        return preds
