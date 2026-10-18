"""
Step 6: weighted average of the GBM predictions -> test_out.csv (final submission)

Usage: python ensemble.py
"""
import pandas as pd

OUTPUT_CSV = "test_out.csv"

# Blend weights; XGBoost is currently given no weight
WEIGHTS = {
    "lgbm.csv": 0.2656,
    "catboost.csv": 0.7344,
    "xgb.csv": 0.0,
}

merged = None
for path in WEIGHTS:
    preds = pd.read_csv(path).rename(columns={"price": path})
    # Join on sample_id so a different row order between files can't misalign predictions
    merged = preds if merged is None else merged.merge(preds, on="sample_id", how="inner")

merged["price"] = sum(weight * merged[path] for path, weight in WEIGHTS.items())
merged[["sample_id", "price"]].to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(merged)} blended predictions to {OUTPUT_CSV}")
print(merged[["sample_id", "price"]].head())
