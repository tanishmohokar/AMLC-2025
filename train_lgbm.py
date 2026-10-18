"""
Step 5b: LightGBM on the Qwen embeddings + catalog features -> lgbm.csv

Usage: python train_lgbm.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from common import RANDOM_STATE, calculate_smape, load_tabular

OUTPUT_CSV = "lgbm.csv"


def lgbm_smape(y_true, y_pred):
    return "smape", calculate_smape(y_true, y_pred), False


# --- Data ---
df = load_tabular("train")
X = df.drop(["price", "sample_id"], axis=1)
y_log = np.log1p(df["price"])
X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)
print(f"Training set size: {X_train.shape[0]} | Validation set size: {X_val.shape[0]}")

# --- Train ---
model = lgb.LGBMRegressor(
    objective="mae",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)
model.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    eval_metric=lgbm_smape,
    categorical_feature=["units"],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)

# --- Validate on the original price scale ---
y_pred = np.expm1(model.predict(X_val))
y_true = np.expm1(y_val_log)
print(f"Validation SMAPE: {calculate_smape(y_true, y_pred):.4f}%")
print(f"Validation MSE: {mean_squared_error(y_true, y_pred):.4f}")

# --- Predict test ---
df_test = load_tabular("test")
# Use the training categories so unseen/missing units map consistently
df_test["units"] = pd.Categorical(df_test["units"], categories=X["units"].cat.categories)
test_pred = np.expm1(model.predict(df_test.drop(["sample_id"], axis=1)))
pd.DataFrame({"sample_id": df_test["sample_id"], "price": test_pred}).to_csv(OUTPUT_CSV, index=False)
print(f"Saved test predictions to {OUTPUT_CSV}")
