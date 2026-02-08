from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"

FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "OverallQual", "YearBuilt"]
TARGET = "SalePrice"

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

# log transform (helps stability)
y_log = np.log1p(y)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES)
    ]
)

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("✅ New model saved:", MODEL_PATH)
