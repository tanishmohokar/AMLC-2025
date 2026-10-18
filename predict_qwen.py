"""
Optional: standalone price prediction with the fine-tuned Qwen model.

Loads CHECKPOINT_DIR, predicts every row of test.csv with the regression head
and writes qwen_submission.csv (sample_id, price). This is the Qwen model on
its own; the final ensemble submission comes from ensemble.py.

Usage: python predict_qwen.py
"""
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import TEST_CSV, forward_pooled, load_finetuned_model, load_raw, make_collate_fn, make_dataset

SUBMISSION_CSV = "qwen_submission.csv"
BATCH_SIZE = 64  # Adjust based on your GPU memory

if __name__ == "__main__":
    model, processor, device, compute_dtype = load_finetuned_model()

    test_df = load_raw(TEST_CSV)
    print(f"Loaded {len(test_df)} records from {TEST_CSV}")
    loader = DataLoader(make_dataset(test_df, processor, with_price=False), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=make_collate_fn(processor.tokenizer.pad_token_id))

    all_sample_ids, all_preds_log = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting Prices"):
            with autocast(device, dtype=compute_dtype, enabled=(device == "cuda")):
                preds = model.regression_head(forward_pooled(model, batch, device)).squeeze(-1)
            all_sample_ids.extend(batch["sample_id"])
            all_preds_log.extend(preds.float().cpu().tolist())

    # The head is trained on log1p(price), so convert back to the price scale
    submission_df = pd.DataFrame({"sample_id": all_sample_ids, "price": np.expm1(all_preds_log)})
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Predictions saved to {SUBMISSION_CSV}")
    print(submission_df.head())
