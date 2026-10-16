import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_pdays_flag(X):
    X = X.copy()
    X["pdays_never"] = (X["pdays"] == 999).astype(int)
    X["pdays"] = X["pdays"].replace(999, np.nan)   # ТАҢДА: NaN + median толтыру, не басқа жол?
    return X


def unknown_to_nan(X):
    X = X.copy()
    cat = X.select_dtypes(include="object").columns
    X[cat] = X[cat].replace("unknown", np.nan)
    return X


def build_preprocessor(unknown="category"):
    """unknown='category' -> 'unknown' бөлек санат; 'nan' -> NaN + most_frequent."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    if unknown == "category":
        cat = OneHotEncoder(handle_unknown="ignore")
        steps = [FunctionTransformer(add_pdays_flag)]
    else:
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore"))])
        steps = [FunctionTransformer(add_pdays_flag),
                 FunctionTransformer(unknown_to_nan)]
    ct = ColumnTransformer([
        ("num", num, make_column_selector(dtype_include=np.number)),
        ("cat", cat, make_column_selector(dtype_include=object)),
    ])
    return Pipeline(steps + [("ct", ct)])