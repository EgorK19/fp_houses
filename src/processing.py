import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Found unknown categories.*"
)


class BaselineTransformer(BaseEstimator, TransformerMixin):
    # признаки, где NaN = категория
    NONE_COLS: ClassVar[list[str]] = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "MasVnrType",
        "GarageQual",
        "GarageFinish",
        "GarageType",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]

    # числовые признаки, где NaN = 0
    ZERO_COLS: ClassVar[list[str]] = [
        "MasVnrArea",
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
    ]

    # признаки, заполняемые модой (считаем на fit)
    MODE_COLS: ClassVar[list[str]] = [
        "Electrical",
        "MSZoning",
        "Exterior1st",
        "Exterior2nd",
        "SaleType",
        "KitchenQual",
    ]

    def fit(self, X_original, y=None):
        X = X_original.copy()

        self.lotfrontage_medians_ = X.groupby(by=["Neighborhood"])[
            "LotFrontage"
        ].median()

        self.lotfrontage_global_median_ = X["LotFrontage"].median()

        self.modes_ = {col: X[col].mode()[0] for col in self.MODE_COLS}

        return self

    def transform(self, X_original):
        X = X_original.copy()

        for col in self.NONE_COLS:
            X[col] = X[col].fillna("None")

        for col in self.ZERO_COLS:
            X[col] = X[col].fillna(0)

        X["LotFrontage"] = X.apply(
            lambda row: (
                self.lotfrontage_medians_.get(
                    row["Neighborhood"], self.lotfrontage_global_median_
                )
                if pd.isna(row["LotFrontage"])
                else row["LotFrontage"]
            ),
            axis=1,
        )

        for col, mode_value in self.modes_.items():
            X[col] = X[col].fillna(mode_value)

        X["Functional"] = X["Functional"].fillna("Typ")

        if "Id" in X.columns:
            X = X.drop(columns=["Id"])

        if "Utilities" in X.columns:
            X = X.drop(columns=["Utilities"])

        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        return X


class MemoryOptimizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        float16_as32=True,
        optimize_categories=True,
        cat_threshold=None,
        opt_log=False,
    ):
        self.float16_as32 = float16_as32
        self.optimize_categories = optimize_categories
        self.cat_threshold = cat_threshold
        self.opt_log = opt_log

    def fit(self, X, y=None):
        self._validate_input(X)

        self.categorical_cols_ = []
        self.numeric_downcast_rules_ = {}
        self.category_levels_ = {}

        for col in X.columns:
            col_type = X[col].dtype

            if self.optimize_categories and (
                col_type == "object"
                or col_type.name == "string"
                or col_type.name == "category"
            ):
                nunique = X[col].nunique()
                if self.cat_threshold is None or nunique <= self.cat_threshold:
                    self.categorical_cols_.append(col)
                    # запоминаем уровни, известные на fit, чтобы ловить unseen-категории на transform
                    self.category_levels_[col] = set(X[col].dropna().unique())
                continue

            if "int" in str(col_type):
                downcasted = pd.to_numeric(X[col], downcast="integer")
                self.numeric_downcast_rules_[col] = downcasted.dtype

            elif "float" in str(col_type):
                downcasted = pd.to_numeric(X[col], downcast="float")
                if self.float16_as32 and downcasted.dtype == np.float16:
                    self.numeric_downcast_rules_[col] = np.dtype(np.float32)
                else:
                    self.numeric_downcast_rules_[col] = downcasted.dtype

        return self

    def transform(self, X):
        self._validate_input(X)
        self._check_is_fitted()

        X_out = X.copy()
        start_mem = X_out.memory_usage(deep=True).sum() / 1024**2

        if self.optimize_categories:
            for col in self.categorical_cols_:
                if col not in X_out.columns:
                    continue

                original_na = X_out[col].isna().sum()

                X_out[col] = X_out[col].astype("object")

                known_levels = self.category_levels_[col]
                unseen_mask = X_out[col].notna() & ~X_out[col].isin(known_levels)
                n_unseen = unseen_mask.sum()

                if n_unseen > 0:
                    X_out.loc[unseen_mask, col] = np.nan
                    warnings.warn(
                        f"Column '{col}': {n_unseen} unseen categories converted to NaN on transform."
                    )

                new_na = X_out[col].isna().sum()
                if new_na > original_na and n_unseen == 0:
                    pass

        for col, target_dtype in self.numeric_downcast_rules_.items():
            if col not in X_out.columns:
                continue

            col_min, col_max = X_out[col].min(), X_out[col].max()

            if np.issubdtype(target_dtype, np.integer):
                dtype_info = np.iinfo(target_dtype)
            else:
                dtype_info = np.finfo(target_dtype)

            if col_min < dtype_info.min or col_max > dtype_info.max:
                warnings.warn(
                    f"Column '{col}': values [{col_min}, {col_max}] exceed {target_dtype} range on transform. Falling back to original dtype."
                )
                continue

            X_out[col] = X_out[col].astype(target_dtype)

        end_mem = X_out.memory_usage(deep=True).sum() / 1024**2
        percent_decrease = (
            100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        )
        if self.opt_log:
            print(
                f"Оптимизация памяти завершена: {start_mem:.2f} MB -> {end_mem:.2f} MB (-{percent_decrease:.1f}%)"
            )

        return X_out

    @staticmethod
    def _validate_input(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X должен быть экземпляром pandas.DataFrame")

    def _check_is_fitted(self):
        if not hasattr(self, "categorical_cols_") or not hasattr(
            self, "numeric_downcast_rules_"
        ):
            raise RuntimeError("Трансформер не обучен. Сначала вызовите fit().")


class ToCategory(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns)
        self.categories_ = {
            col: pd.Index(X[col].astype("category").cat.categories)
            for col in self.columns
        }
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = pd.Categorical(X_copy[col], categories=self.categories_[col])
        return X_copy


from sklearn.pipeline import Pipeline

num_columns = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

ordinal_columns = [
    "LotShape",  # Reg > IR1 > IR2 > IR3
    "LandSlope",  # Gtl > Mod > Sev
    "ExterQual",  # Ex > Gd > TA > Fa > Po
    "ExterCond",  # Ex > Gd > TA > Fa > Po
    "BsmtQual",  # Ex > Gd > TA > Fa > Po > None
    "BsmtCond",  # Ex > Gd > TA > Fa > Po > None
    "BsmtExposure",  # Gd > Av > Mn > No > None
    "BsmtFinType1",  # GLQ > ALQ > BLQ > Rec > LwQ > Unf > None
    "BsmtFinType2",  # GLQ > ALQ > BLQ > Rec > LwQ > Unf > None
    "HeatingQC",  # Ex > Gd > TA > Fa > Po
    "KitchenQual",  # Ex > Gd > TA > Fa > Po
    "Functional",  # Typ > Min1 > Min2 > Mod > Maj1 > Maj2 > Sev > Sal
    "FireplaceQu",  # Ex > Gd > TA > Fa > Po > None
    "GarageFinish",  # Fin > RFn > Unf > None
    "GarageQual",  # Ex > Gd > TA > Fa > Po > None
    "GarageCond",  # Ex > Gd > TA > Fa > Po > None
    "PavedDrive",  # Y > P > N
    "PoolQC",  # Ex > Gd > TA > Fa > None
    "CentralAir",  # Y > N
]

ordinal_categories = [
    ["IR3", "IR2", "IR1", "Reg"],  # LotShape
    ["Sev", "Mod", "Gtl"],  # LandSlope
    ["Po", "Fa", "TA", "Gd", "Ex"],  # ExterQual
    ["Po", "Fa", "TA", "Gd", "Ex"],  # ExterCond
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # BsmtQual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # BsmtCond
    ["None", "No", "Mn", "Av", "Gd"],  # BsmtExposure
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType1
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType2
    ["Po", "Fa", "TA", "Gd", "Ex"],  # HeatingQC
    ["Po", "Fa", "TA", "Gd", "Ex"],  # KitchenQual
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],  # Functional
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # FireplaceQu
    ["None", "Unf", "RFn", "Fin"],  # GarageFinish
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # GarageQual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # GarageCond
    ["N", "P", "Y"],  # PavedDrive
    ["None", "Fa", "TA", "Gd", "Ex"],  # PoolQC
    ["N", "Y"],  # CentralAir
]

ohe_columns = [
    "Fence",
    "Electrical",
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]


def build_pipeline(kind: str) -> Pipeline:
    match kind:
        case "linear":
            return Pipeline(
                [
                    ("baseline_transformer", BaselineTransformer()),
                    ("memory_optimizer", MemoryOptimizer()),
                    (
                        "column_transformer",
                        ColumnTransformer(
                            transformers=[
                                ("num_features", StandardScaler(), num_columns),
                                (
                                    "cat_features_oe",
                                    OrdinalEncoder(
                                        categories=ordinal_categories,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                    ordinal_columns,
                                ),
                                (
                                    "cat_features_ohe",
                                    OneHotEncoder(
                                        drop="first",
                                        handle_unknown="infrequent_if_exist",
                                    ),
                                    ohe_columns,
                                ),
                            ]
                        ),
                    ),
                ]
            )
        case "tree_oe&ohe":
            return Pipeline(
                [
                    ("baseline_transformer", BaselineTransformer()),
                    ("memory_optimizer", MemoryOptimizer()),
                    (
                        "column_transformer",
                        ColumnTransformer(
                            transformers=[
                                ("num_features", "passthrough", num_columns),
                                (
                                    "cat_features_oe",
                                    OrdinalEncoder(
                                        categories=ordinal_categories,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                    ordinal_columns,
                                ),
                                (
                                    "cat_features_ohe",
                                    OneHotEncoder(
                                        drop="first",
                                        handle_unknown="infrequent_if_exist",
                                    ),
                                    ohe_columns,
                                ),
                            ]
                        ),
                    ),
                ]
            )
        case "tree_oe&lgbm":
            return Pipeline(
                [
                    ("baseline_transformer", BaselineTransformer()),
                    ("memory_optimizer", MemoryOptimizer()),
                    (
                        "column_transformer",
                        ColumnTransformer(
                            transformers=[
                                ("num_features", "passthrough", num_columns),
                                (
                                    "cat_features_oe",
                                    OrdinalEncoder(
                                        categories=ordinal_categories,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                    ordinal_columns,
                                ),
                                (
                                    "cat_features_ohe",
                                    "passthrough",
                                    ohe_columns,
                                ),
                            ],
                            verbose_feature_names_out=False,
                        ).set_output(transform="pandas"),
                    ),
                    ("to_category", ToCategory(ohe_columns)),
                ]
            )
        case "tree_lgbm_only":
            return Pipeline(
                [
                    ("baseline_transformer", BaselineTransformer()),
                    ("memory_optimizer", MemoryOptimizer()),
                    (
                        "column_transformer",
                        ColumnTransformer(
                            transformers=[
                                ("num_features", "passthrough", num_columns),
                                (
                                    "cat_features_oe",
                                    "passthrough",
                                    ordinal_columns,
                                ),
                                (
                                    "cat_features_ohe",
                                    "passthrough",
                                    ohe_columns,
                                ),
                            ],
                            verbose_feature_names_out=False,
                        ).set_output(transform="pandas"),
                    ),
                    ("to_category", ToCategory(ordinal_columns + ohe_columns)),
                ]
            )
    raise ValueError(f"Unknown pipeline kind: {kind}")
