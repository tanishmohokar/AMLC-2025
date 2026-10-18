"""
Step 2: fine-tune Qwen2.5-VL-7B with LoRA and a regression head on log1p(price).

Saves the LoRA adapters, regression_head.pth and the processor after every
epoch to ./price_predictor_epoch_{n}. The shipped price_predictor_epoch_*
folders were produced by this configuration.

Usage: python finetune_qwen.py
"""
import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import (
    MODEL_ID, RANDOM_STATE, TRAIN_CSV, build_regression_head, calculate_smape,
    forward_pooled, get_compute_dtype, load_raw, make_collate_fn, make_dataset,
)

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# ---------------------------
#  Data
# ---------------------------
df = load_raw(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

# ---------------------------
#  Processor + Model
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = get_compute_dtype()
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True
)

# Freeze the base model, then add the trainable regression head and LoRA adapters
for param in model.parameters():
    param.requires_grad = False
model.regression_head = build_regression_head(model.config.hidden_size, device)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    use_dora=False,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ---------------------------
#  Dataset + DataLoader
# ---------------------------
collate_fn = make_collate_fn(processor.tokenizer.pad_token_id)
train_loader = DataLoader(make_dataset(train_df, processor, with_price=True),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(make_dataset(val_df, processor, with_price=True),
                        batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn)

# ---------------------------
#  Optimizer + Loss + GradScaler
# ---------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.L1Loss()
scaler = GradScaler(enabled=(device == "cuda"))

# ---------------------------
#  Training loop
# ---------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for i, batch in enumerate(loop):
        prices = batch["price"].to(device)  # log-scale
        with autocast(device, dtype=compute_dtype, enabled=(device == "cuda")):
            preds = model.regression_head(forward_pooled(model, batch, device)).squeeze(-1)
            loss = loss_fn(preds, prices) / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        loop.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)

    # --- Validation SMAPE on the original price scale ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            with autocast(device, dtype=compute_dtype, enabled=(device == "cuda")):
                preds = model.regression_head(forward_pooled(model, batch, device)).squeeze(-1)
            all_preds.extend(preds.float().cpu().tolist())
            all_targets.extend(batch["price"].tolist())

    smape_val = calculate_smape(torch.expm1(torch.tensor(all_targets)).numpy(),
                                torch.expm1(torch.tensor(all_preds)).numpy())
    print(f"\nEpoch {epoch + 1} | Avg Train Loss: {total_loss / len(train_loader):.4f} | Val SMAPE: {smape_val:.4f}%\n")

    # --- Save checkpoint ---
    output_dir = f"./price_predictor_epoch_{epoch + 1}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    torch.save(model.regression_head.state_dict(), os.path.join(output_dir, "regression_head.pth"))
    processor.save_pretrained(output_dir)
    print(f"Model for epoch {epoch + 1} saved to {output_dir}")
