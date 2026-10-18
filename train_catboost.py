"""
Step 5c: CatBoost on the Qwen embeddings + catalog features -> catboost.csv
Also saves validation predictions to catboost_oof_validation.csv.

Usage: python train_catboost.py
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from common import RANDOM_STATE, calculate_smape, load_tabular

OUTPUT_CSV = "catboost.csv"
OOF_CSV = "catboost_oof_validation.csv"
CAT_FEATURES = ["units"]

# --- Data ---
df = load_tabular("train")
X = df.drop(["price", "sample_id"], axis=1)
y_log = np.log1p(df["price"])
X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)
print(f"Training set size: {X_train.shape[0]} | Validation set size: {X_val.shape[0]}")

# --- Train ---
model = CatBoostRegressor(
    loss_function="MAE",
    iterations=6000,
    learning_rate=0.05,
    depth=5,
    subsample=0.8,
    colsample_bylevel=0.8,
    eval_metric="SMAPE",
    random_seed=RANDOM_STATE,
)
model.fit(
    X_train, y_train_log,
    cat_features=CAT_FEATURES,
    eval_set=[(X_val, y_val_log)],
    early_stopping_rounds=100,
    verbose=100,
)

# --- Validate on the original price scale ---
y_pred = np.expm1(model.predict(X_val))
y_true = np.expm1(y_val_log)
print(f"Validation SMAPE: {calculate_smape(y_true, y_pred):.4f}%")
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

pd.DataFrame({
    "sample_id": df.loc[X_val.index, "sample_id"].values,
    "True_Price": y_true.values,
    "OOF_Prediction": y_pred,
}).to_csv(OOF_CSV, index=False)
print(f"Saved validation predictions to {OOF_CSV}")

# --- Predict test ---
df_test = load_tabular("test")
test_pred = np.expm1(model.predict(df_test.drop(["sample_id"], axis=1)))
pd.DataFrame({"sample_id": df_test["sample_id"], "price": test_pred}).to_csv(OUTPUT_CSV, index=False)
print(f"Saved test predictions to {OUTPUT_CSV}")
