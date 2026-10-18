"""
Shared paths, data helpers and Qwen model utilities used by every pipeline stage.
"""
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------------------------
#  Paths and constants
# ---------------------------
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
IMAGE_CACHE_DIR = "image_cache"

TRAIN_EMBEDDINGS_CSV = "train_embeddings.csv"
TEST_EMBEDDINGS_CSV = "test_embeddings.csv"
TRAIN_FEATURES_CSV = "features.csv"
TEST_FEATURES_CSV = "features_test.csv"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT_DIR = "./price_predictor_epoch_3"
HIDDEN_DIM = 512  # Intermediate dimension of the regression head
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42


# ---------------------------
#  Metrics
# ---------------------------
def calculate_smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    raw_values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return np.mean(raw_values) * 100


# ---------------------------
#  Raw data
# ---------------------------
def load_raw(path):
    """Loads a raw challenge CSV and makes sure catalog content is always a string."""
    df = pd.read_csv(path)
    df["catalog_content"] = df["catalog_content"].fillna("")
    return df


def image_cache_path(sample_id, cache_dir=IMAGE_CACHE_DIR):
    """Cache filename for a sample, using a sanitized version of sample_id."""
    filename = "".join(c for c in str(sample_id) if c.isalnum()) + ".jpg"
    return os.path.join(cache_dir, filename)


def load_image(url, sample_id, cache_dir=IMAGE_CACHE_DIR):
    """Loads an image from the local cache, downloading it first if needed.
    Returns a black placeholder if the image can't be fetched."""
    cache_path = image_cache_path(sample_id, cache_dir)
    try:
        if not os.path.exists(cache_path):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_path)
        else:
            img = Image.open(cache_path).convert("RGB")
        return img.resize(IMAGE_SIZE)
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Warning: Could not load image for sample {sample_id}. Error: {e}. Using placeholder.")
        return Image.new("RGB", IMAGE_SIZE, (0, 0, 0))


# ---------------------------
#  Tabular data for the GBM models
# ---------------------------
def load_tabular(split, with_features=True):
    """Loads Qwen embeddings for 'train' or 'test', optionally merged with the
    parsed catalog features (value, units)."""
    emb_path = TRAIN_EMBEDDINGS_CSV if split == "train" else TEST_EMBEDDINGS_CSV
    df = pd.read_csv(emb_path)
    if with_features:
        feat_path = TRAIN_FEATURES_CSV if split == "train" else TEST_FEATURES_CSV
        df = pd.merge(df, pd.read_csv(feat_path), on="sample_id", how="left")
        df["units"] = df["units"].astype("category")
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


# ---------------------------
#  Qwen model utilities (torch / transformers imported lazily so the
#  GBM scripts don't need them)
# ---------------------------
def build_prompt(processor, catalog_content):
    """Chat-format prompt that maps the image to `<img>` tokens."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Catalog: {catalog_content}\nInstruction: Encode product for price prediction."},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_compute_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def make_dataset(df, processor, with_price, cache_dir=IMAGE_CACHE_DIR):
    """Dataset yielding processor inputs plus `sample_id` and, if requested, log1p(price)."""
    import torch
    from torch.utils.data import Dataset

    class QwenPriceDataset(Dataset):
        def __init__(self):
            self.df = df.reset_index(drop=True)
            os.makedirs(cache_dir, exist_ok=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = load_image(row["image_link"], row["sample_id"], cache_dir)
            text_prompt = build_prompt(processor, row["catalog_content"])
            inputs = processor(text=text_prompt, images=image, return_tensors="pt")
            # The processor adds a batch dimension, which we remove for the dataset item
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["sample_id"] = row["sample_id"]
            if with_price:
                inputs["price"] = torch.log1p(torch.tensor(row["price"], dtype=torch.float32))
            return inputs

    return QwenPriceDataset()


def make_collate_fn(pad_token_id):
    """Pads input_ids/attention_mask to the longest sequence in the batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        out = {
            "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),
            "sample_id": [b["sample_id"] for b in batch],
        }
        if "price" in batch[0]:
            out["price"] = torch.stack([b["price"] for b in batch])
        return out

    return collate_fn


def build_regression_head(hidden_size, device):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(hidden_size, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, 1),
    ).to(device)


def forward_pooled(model, batch, device):
    """Runs the model and returns the hidden state of the last non-padding token."""
    import torch

    attention_mask = batch["attention_mask"].to(device)
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=attention_mask,
        pixel_values=batch["pixel_values"].to(device),
        image_grid_thw=batch["image_grid_thw"].to(device),
        output_hidden_states=True,
    )
    last_hidden_state = outputs.hidden_states[-1]
    last_token_indices = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=device), last_token_indices]


def load_finetuned_model(checkpoint_dir=CHECKPOINT_DIR):
    """Loads the base Qwen model, the LoRA adapters and the regression head.
    Returns (model, processor, device, compute_dtype)."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = get_compute_dtype()
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True
    )
    base_model.regression_head = build_regression_head(base_model.config.hidden_size, device)

    print(f"Loading LoRA adapters from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    head_path = os.path.join(checkpoint_dir, "regression_head.pth")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"regression_head.pth not found at {head_path}")
    model.regression_head.load_state_dict(torch.load(head_path, map_location=device))

    model.eval()
    return model, processor, device, compute_dtype
