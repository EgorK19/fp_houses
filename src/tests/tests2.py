from typing import ClassVar

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.processing import MemoryOptimizer


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

    CONTINUOUS_CANDIDATES: ClassVar[list[str]] = [
        "LotFrontage",
        "LotArea",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
    ]

    def fit(self, X_original, y=None):
        X = X_original.copy()

        self.lotfrontage_medians_ = X.groupby(by=["Neighborhood"])[
            "LotFrontage"
        ].median()

        self.lotfrontage_global_median_ = X["LotFrontage"].median()

        self.modes_ = {col: X[col].mode()[0] for col in self.MODE_COLS}
        from scipy.stats import skew

        skewed = X[self.CONTINUOUS_CANDIDATES].apply(lambda x: skew(x.dropna()))
        self.skewed_cols_ = skewed[abs(skewed) > 0.85].index.tolist()

        return self

    def transform(self, X_original):
        X = X_original.copy()

        for col in self.NONE_COLS:
            X[col] = X[col].fillna("None")

        for col in self.ZERO_COLS:
            X[col] = X[col].fillna(0)

        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.lotfrontage_medians_)
        )

        X["LotFrontage"] = X["LotFrontage"].fillna(self.lotfrontage_global_median_)

        for col, mode_value in self.modes_.items():
            X[col] = X[col].fillna(mode_value)

        X["Functional"] = X["Functional"].fillna("Typ")

        if "Id" in X.columns:
            X = X.drop(columns=["Id"])

        if "Utilities" in X.columns:
            X = X.drop(columns=["Utilities"])

        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        # for col in ["GrLivArea", "LotArea", "1stFlrSF"]:
        #     X[col] = np.log1p(X[col])
        import numpy as np

        for col in self.skewed_cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col])

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

advanced = Pipeline(
    [
        ("baseline_transformer", AdvancedTransformer()),
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


def build_test_pipeline():
    return advanced
