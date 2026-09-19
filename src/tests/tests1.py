from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.processing import MemoryOptimizer, ToCategory


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

        self.lotfrontage_medians2_ = X.groupby(by=["Neighborhood", "LotConfig"])[
            "LotFrontage"
        ].median()

        self.lotfrontage_medians1_ = X.groupby(by=["Neighborhood"])[
            "LotFrontage"
        ].median()

        self.lotfrontage_global_median_ = X["LotFrontage"].median()

        self.modes_ = {col: X[col].mode()[0] for col in self.MODE_COLS}

        return self

    def transform(self, X_original):
        X = X_original.copy()

        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        for col in self.NONE_COLS:
            X[col] = X[col].fillna("None")

        for col in self.ZERO_COLS:
            X[col] = X[col].fillna(0)

        keys2 = list(zip(X["Neighborhood"], X["LotConfig"]))
        filled_2 = pd.Series(keys2, index=X.index).map(self.lotfrontage_medians2_)
        X["LotFrontage"] = X["LotFrontage"].fillna(filled_2)

        filled_1 = X["Neighborhood"].map(self.lotfrontage_medians1_)
        X["LotFrontage"] = X["LotFrontage"].fillna(filled_1)

        X["LotFrontage"] = X["LotFrontage"].fillna(self.lotfrontage_global_median_)

        for col, mode_value in self.modes_.items():
            X[col] = X[col].fillna(mode_value)

        X["Functional"] = X["Functional"].fillna("Typ")

        # ! new features
        # фичи возраста
        X["MoSold_sin"] = np.sin(2 * np.pi * X["MoSold"] / 12)
        X["MoSold_cos"] = np.cos(2 * np.pi * X["MoSold"] / 12)
        # фичи дома
        # X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        # X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        # X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        # X["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)
        # X["IsNewHouse"] = (X["YrSold"] == X["YearBuilt"]).astype(int)
        # X.loc[X["GarageType"] == "None", "GarageAge"] = 0
        # X["AreaPerCar"] = np.where(
        #     X["GarageCars"] > 0, X["GarageArea"] / X["GarageCars"], 0
        # )
        # X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        # X["TotalPorchSF"] = (
        #     X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        # )
        # X["TotalBathrooms"] = (
        #     X["FullBath"]
        #     + 0.5 * X["HalfBath"]
        #     + X["BsmtFullBath"]
        #     + 0.5 * X["BsmtHalfBath"]
        # )

        # X["AreaPerRoom"] = X["GrLivArea"] / X["TotRmsAbvGrd"].replace(0, np.nan)
        # X["LivingAreaRatio"] = X["GrLivArea"] / X["LotArea"].replace(0, np.nan)

        # X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        # X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        # X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        # X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        # X["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)

        # ! end new features

        # drop
        X = X.drop(
            columns=["Id"]
            + [
                "EnclosedPorch",
                "Exterior2nd",
                "GarageCond",
                "BedroomAbvGr",
                "Electrical",
                "Foundation",
                "BsmtFinType2",
                "LandSlope",
                "RoofStyle",
                "Alley",
                "BsmtHalfBath",
                "GarageQual",
                "HouseStyle",
                "LowQualFinSF",
                "MiscVal",
                "LandContour",
                "Street",
                "Utilities",
                "Condition2",
                "PoolQC",
                "PoolArea",
                "RoofMatl",
            ],
            errors="ignore",
        )

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
    # "LowQualFinSF", # 0 importance
    "GrLivArea",
    "BsmtFullBath",
    # "BsmtHalfBath", # 0 importance
    "FullBath",
    "HalfBath",
    # "BedroomAbvGr", # 0 importance
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    # "EnclosedPorch", # 0 importance
    "3SsnPorch",
    "ScreenPorch",
    # "PoolArea", # 0 importance
    # "MiscVal", # 0 importance
    "MoSold",
    "YrSold",
    # new features
    "MoSold_sin",
    "MoSold_cos",
    # "AreaPerCar",
    # "HouseAge",
    # "RemodAge",
    # "GarageAge",
    # "IsRemodeled",
    # "IsNewHouse",
    # "TotalSF",
    # "TotalBathrooms",
    # "TotalPorchSF",
    # "AreaPerRoom",
    # "LivingAreaRatio",
    # "HasPool",
    # "HasGarage",
    # "HasBsmt",
    # "HasFireplace",
    # "Has2ndFloor",
]

ordinal_columns = [
    "LotShape",  # Reg > IR1 > IR2 > IR3
    # "LandSlope",  # Gtl > Mod > Sev # 0 importance
    "ExterQual",  # Ex > Gd > TA > Fa > Po
    "ExterCond",  # Ex > Gd > TA > Fa > Po
    "BsmtQual",  # Ex > Gd > TA > Fa > Po > None
    "BsmtCond",  # Ex > Gd > TA > Fa > Po > None
    "BsmtExposure",  # Gd > Av > Mn > No > None
    "BsmtFinType1",  # GLQ > ALQ > BLQ > Rec > LwQ > Unf > None
    # "BsmtFinType2",  # GLQ > ALQ > BLQ > Rec > LwQ > Unf > None # 0 importance
    "HeatingQC",  # Ex > Gd > TA > Fa > Po
    "KitchenQual",  # Ex > Gd > TA > Fa > Po
    "Functional",  # Typ > Min1 > Min2 > Mod > Maj1 > Maj2 > Sev > Sal
    "FireplaceQu",  # Ex > Gd > TA > Fa > Po > None
    "GarageFinish",  # Fin > RFn > Unf > None
    # "GarageQual",  # Ex > Gd > TA > Fa > Po > None # 0 importance
    # "GarageCond",  # Ex > Gd > TA > Fa > Po > None # 0 importance
    "PavedDrive",  # Y > P > N
    # "PoolQC",  # Ex > Gd > TA > Fa > None # 0 importance
    "CentralAir",  # Y > N
]

ordinal_categories = [
    ["IR3", "IR2", "IR1", "Reg"],  # LotShape
    # ["Sev", "Mod", "Gtl"],  # LandSlope # 0 importance
    ["Po", "Fa", "TA", "Gd", "Ex"],  # ExterQual
    ["Po", "Fa", "TA", "Gd", "Ex"],  # ExterCond
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # BsmtQual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # BsmtCond
    ["None", "No", "Mn", "Av", "Gd"],  # BsmtExposure
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType1
    # ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType2 # 0 importance
    ["Po", "Fa", "TA", "Gd", "Ex"],  # HeatingQC
    ["Po", "Fa", "TA", "Gd", "Ex"],  # KitchenQual
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],  # Functional
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # FireplaceQu
    ["None", "Unf", "RFn", "Fin"],  # GarageFinish
    # ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # GarageQual # 0 importance
    # ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # GarageCond # 0 importance
    ["N", "P", "Y"],  # PavedDrive
    # ["None", "Fa", "TA", "Gd", "Ex"],  # PoolQC
    ["N", "Y"],  # CentralAir
]

ohe_columns = [
    "Fence",
    # "Electrical", # 0 importance
    "MSSubClass",
    "MSZoning",
    # "Street", # 0 importance
    # "Alley", # 0 importance
    # "LandContour", # 0 importance
    "LotConfig",
    "Neighborhood",
    "Condition1",
    # "Condition2", # 0 importance
    "BldgType",
    # "HouseStyle", # 0 importance
    # "RoofStyle", # 0 importance
    # "RoofMatl", # 0 importance
    "Exterior1st",
    # "Exterior2nd", # 0 importance
    "MasVnrType",
    # "Foundation", # 0 importance
    "Heating",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    # new cat
    # "OverallQual",
    # "OverallCond",
    # "MoSold",
    # "YrSold",
]

advanced = Pipeline(
    [
        ("baseline_transformer", AdvancedTransformer()),  ### advanced
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
