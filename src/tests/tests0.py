from typing import ClassVar

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.processing import BaselineTransformer, MemoryOptimizer, ToCategory


class AdvancedTransformer(BaseEstimator, TransformerMixin):
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


advanced = Pipeline(
    [
        ("baseline_transformer", BaselineTransformer()),  ### advanced
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
