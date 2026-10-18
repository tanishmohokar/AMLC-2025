"""
Step 3: extract Qwen embeddings for the GBM models.

Runs the fine-tuned model (CHECKPOINT_DIR) over train.csv and test.csv and
saves the last-token pooled hidden state (3584 dims for Qwen2.5-VL-7B), i.e.
the same vector the regression head sees during fine-tuning.

Outputs:
  train_embeddings.csv  -> sample_id, emb_0 .. emb_3583, price
  test_embeddings.csv   -> sample_id, emb_0 .. emb_3583

Usage: python extract_embeddings.py
"""
import os

import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    TEST_CSV, TEST_EMBEDDINGS_CSV, TRAIN_CSV, TRAIN_EMBEDDINGS_CSV,
    forward_pooled, load_finetuned_model, load_raw, make_collate_fn, make_dataset,
)

BATCH_SIZE = 16
FLUSH_EVERY = 50  # batches buffered in memory before appending to the CSV


def extract(df, out_path, model, processor, device, compute_dtype):
    if os.path.exists(out_path):
        os.remove(out_path)

    loader = DataLoader(make_dataset(df, processor, with_price=False), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=make_collate_fn(processor.tokenizer.pad_token_id))
    prices = df.set_index("sample_id")["price"] if "price" in df.columns else None

    buffer, header_written = [], False

    def flush():
        nonlocal buffer, header_written
        if not buffer:
            return
        chunk = pd.concat(buffer, ignore_index=True)
        if prices is not None:
            chunk["price"] = chunk["sample_id"].map(prices).values
        chunk.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written, buffer = True, []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Embedding -> {out_path}")):
            with autocast(device, dtype=compute_dtype, enabled=(device == "cuda")):
                pooled = forward_pooled(model, batch, device)
            emb = pooled.float().cpu().numpy()
            chunk = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
            chunk.insert(0, "sample_id", batch["sample_id"])
            buffer.append(chunk)
            if (i + 1) % FLUSH_EVERY == 0:
                flush()
    flush()
    print(f"Saved embeddings for {len(df)} samples to {out_path}")


if __name__ == "__main__":
    model, processor, device, compute_dtype = load_finetuned_model()
    extract(load_raw(TRAIN_CSV), TRAIN_EMBEDDINGS_CSV, model, processor, device, compute_dtype)
    extract(load_raw(TEST_CSV), TEST_EMBEDDINGS_CSV, model, processor, device, compute_dtype)
