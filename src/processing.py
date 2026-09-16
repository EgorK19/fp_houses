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
import warnings

warnings.filterwarnings("ignore", message="Found unknown categories in columns")
warnings.filterwarnings("ignore", message=".*unseen categories converted to Unseen.*")


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
            dtype = X[col].dtype

            is_categorical_like = (
                dtype == object
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(
                    dtype
                )  # покрывает object, "string", "str" и любые будущие строковые backend'ы
            )

            if self.optimize_categories and is_categorical_like:
                nunique = X[col].nunique()
                if self.cat_threshold is None or nunique <= self.cat_threshold:
                    self.categorical_cols_.append(col)
                    self.category_levels_[col] = set(X[col].dropna().unique())
                continue

            if "int" in str(dtype):
                downcasted = pd.to_numeric(X[col], downcast="integer")
                self.numeric_downcast_rules_[col] = downcasted.dtype

            elif "float" in str(dtype):
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
                    X_out.loc[unseen_mask, col] = "Unseen"
                    warnings.warn(
                        f"Column '{col}': {n_unseen} unseen categories converted to Unseen on transform."
                    )

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
    def __init__(self, columns, unseen_placeholder="Unseen"):
        self.columns = columns
        self.unseen_placeholder = unseen_placeholder

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns)
        self.categories_ = {}

        for col in self.columns:
            cats = pd.Index(X[col].astype("category").cat.categories)
            if self.unseen_placeholder not in cats:
                cats = cats.insert(len(cats), self.unseen_placeholder)
            self.categories_[col] = cats

        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = pd.Categorical(X_copy[col], categories=self.categories_[col])
        return X_copy


def fix_data_bugs(df):
    df = df.copy()

    idx_2126 = 2126
    detchd_mask = df["GarageType"] == "Detchd"

    for col in ["GarageQual", "GarageCond"]:
        df.loc[idx_2126, col] = df.loc[detchd_mask, col].mode()[0]

    df.loc[idx_2126, "GarageFinish"] = df.loc[detchd_mask, "GarageFinish"].mode()[0]
    df.loc[idx_2126, "GarageYrBlt"] = df.loc[idx_2126, "YearBuilt"]

    idx_2576 = 2576
    df.loc[idx_2576, "GarageType"] = "None"
    for col in ["GarageQual", "GarageFinish", "GarageCond"]:
        df.loc[idx_2576, col] = "None"
    df.loc[idx_2576, ["GarageYrBlt", "GarageCars", "GarageArea"]] = 0

    idx_exposure_missing = [948, 1487, 2348]
    idx_cond_missing = [2040, 2185, 2524]
    idx_qual_missing = [2217, 2218]
    idx_fintype2_missing = [332]

    exposure_mode = df.loc[df["TotalBsmtSF"] > 0, "BsmtExposure"].mode()[0]
    df.loc[idx_exposure_missing, "BsmtExposure"] = exposure_mode

    cond_mode = df.loc[df["TotalBsmtSF"] > 0, "BsmtCond"].mode()[0]
    df.loc[idx_cond_missing, "BsmtCond"] = cond_mode

    qual_mode = df.loc[df["TotalBsmtSF"] > 0, "BsmtQual"].mode()[0]
    df.loc[idx_qual_missing, "BsmtQual"] = qual_mode

    fintype2_mode = df.loc[df["BsmtFinSF2"] > 0, "BsmtFinType2"].mode()[0]
    df.loc[idx_fintype2_missing, "BsmtFinType2"] = fintype2_mode

    df.loc[2592, "GarageYrBlt"] = 2007

    return df


def remove_train_outliers(df_train):
    outlier_mask = (df_train["GrLivArea"] > 4000) & (df_train["SalePrice"] < 300000)
    return df_train[~outlier_mask].reset_index(drop=True)


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
