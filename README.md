# Multi-Modal Price Prediction: Fine-Tuned Qwen2.5-VL + GBM Ensemble

**Amazon ML Challenge 2025 · October 2025**

`Vision-Language Models` `PEFT / LoRA` `Hugging Face` `XGBoost` `LightGBM` `CatBoost`

## Results

| | |
|---|---|
| **Final rank** | **All India Rank 14** out of 7,000+ teams |
| **Leaderboard score** | **~41 SMAPE** (lower is better) |
| **Test set** | 75K+ products |
| **Task** | Predict a product's price from its catalog text and product image |

## Highlights
- **Multimodal fine-tuning:** fine-tuned the Qwen2.5-VL-7B vision-language model with LoRA (PEFT) and a custom regression head. It learns joint visual–textual embeddings from product images and catalog text.
- **Stable training on long-tailed prices:** the model regresses `log1p(price)` with MAE/L1-aligned objectives, which matches the SMAPE metric. This keeps training stable on heavily right-skewed price data.
- **Ensemble for generalization:** LightGBM, XGBoost and CatBoost are trained on the fine-tuned Qwen embeddings plus parsed catalog features, then blended with a weighted average.

## Overview

This pipeline predicts product prices from catalog text and product images. It works in four stages:

1. **Fine-tune** Qwen2.5-VL-7B-Instruct with LoRA and a small regression head on `log1p(price)`, using the catalog text and the product image together.
2. **Extract** the fine-tuned model's pooled embedding (3584 dimensions) for every product. Also parse the `Value` / `Unit` fields from the catalog text.
3. **Train** XGBoost, LightGBM and CatBoost on those embeddings and features.
4. **Blend** the three models' predictions with a weighted average to make the final submission.

The metric throughout is SMAPE, computed on the original price scale.

## Repository layout

| File | Stage | Produces |
|---|---|---|
| `common.py` | shared | Paths, SMAPE, image loading/caching, prompt, dataset/collate, model loading |
| `download_images.py` | 1 (optional) | `image_cache/` |
| `finetune_qwen.py` | 2 (GPU) | `price_predictor_epoch_{1,2,3}/` |
| `extract_embeddings.py` | 3 (GPU) | `train_embeddings.csv`, `test_embeddings.csv` |
| `extract_features.py` | 4 | `features.csv`, `features_test.csv` |
| `train_xgb.py` | 5 | `xgb.csv` |
| `train_lgbm.py` | 5 | `lgbm.csv` |
| `train_catboost.py` | 5 | `catboost.csv`, `catboost_oof_validation.csv` |
| `ensemble.py` | 6 | `test_out.csv` (**final submission**) |
| `predict_qwen.py` | optional (GPU) | `qwen_submission.csv` (the Qwen head on its own) |
| `price_predictor_epoch_*/` | — | Shipped LoRA checkpoints; `common.CHECKPOINT_DIR` uses epoch 3 |
| `experiments/` | — | Earlier experiments that the pipeline doesn't use (see its README) |

Input data: `train.csv` (`sample_id, catalog_content, image_link, price`) and `test.csv` (the same without `price`). Both go in this folder.

## Setup

```bash
python -m venv venv
```

Activate it with `source venv/bin/activate` on Linux/macOS, or `venv\Scripts\activate` on Windows. Then:

```bash
pip install -r requirements.txt
```

Steps 2, 3 and the optional `predict_qwen.py` need a CUDA GPU with enough memory for a 7B vision-language model in bf16/fp16. Steps 4–6 run on CPU.

## How to run

Run every script from this folder, in this order:

```bash
python download_images.py
```
```bash
python finetune_qwen.py
```
```bash
python extract_embeddings.py
```
```bash
python extract_features.py
```
```bash
python train_xgb.py
```
```bash
python train_lgbm.py
```
```bash
python train_catboost.py
```
```bash
python ensemble.py
```

- `download_images.py` is optional. It caches every image once, so the GPU steps don't wait on the network.
- You can skip `finetune_qwen.py` and use the shipped `price_predictor_epoch_3/`.
- The three `train_*.py` scripts are independent of each other.

## Method

### Fine-tuning (`finetune_qwen.py`)
- **Input**: a chat-format prompt with the product image (resized to 224×224) and `Catalog: <catalog_content>`. A black placeholder replaces images that fail to download.
- **Target**: `log1p(price)`, which tames the right-skewed price distribution.
- **Model**: the base Qwen2.5-VL-7B weights are frozen. LoRA (r=16, alpha=32, dropout 0.1) is applied to `q_proj`, `k_proj`, `v_proj` and `o_proj`.
- **Regression head**: `Linear(3584→512) → ReLU → Linear(512→1)`. It reads the hidden state of the last non-padding token from the final layer.
- **Training**: L1 loss, AdamW at lr 5e-5, batch size 4, mixed precision, gradient clipping at 1.0, and 3 epochs on an 80/20 split. A checkpoint is saved after every epoch.

### Embeddings and features
- `extract_embeddings.py` loads the epoch-3 checkpoint and saves the same pooled vector the head sees, as columns `emb_0 … emb_3583`.
- `extract_features.py` parses `Value: <number>` into `value`, and `Unit: <text>` into a normalised categorical `units` (for example `oz`/`Ounces` → `ounce` and `ct` → `count`).

### Gradient-boosted models
All three train on `log1p(price)` with an 80/20 split (`random_state=42`) and MAE-style objectives, then predict with `expm1`.

| Model | Inputs | Key settings |
|---|---|---|
| XGBoost | embeddings | `reg:absoluteerror`, 2000 trees, depth 5, lr 0.05 |
| LightGBM | embeddings + `value`, `units` | `mae`, 1000 trees, depth 5, lr 0.05, early stopping 100 |
| CatBoost | embeddings + `value`, `units` | `MAE`, up to 6000 iterations, depth 5, lr 0.05, early stopping 100 |

### Ensemble (`ensemble.py`)
Predictions are joined on `sample_id` and blended:

```
price = 0.2656 · LightGBM + 0.7344 · CatBoost + 0.0 · XGBoost
```

You can change the weights in `WEIGHTS` at the top of `ensemble.py`.

## Configuration
All paths and shared constants live at the top of `common.py`:
- `TRAIN_CSV` and `TEST_CSV`;
- `CHECKPOINT_DIR`, `IMAGE_CACHE_DIR` and `MODEL_ID`;
- the names of the embedding and feature CSVs.

Batch sizes are set at the top of each GPU script.

## Tech stack
- **Modelling:** PyTorch, Hugging Face Transformers, PEFT (LoRA), Accelerate
- **Base model:** [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Gradient boosting:** XGBoost, LightGBM, CatBoost
- **Data:** pandas, NumPy, scikit-learn, Pillow

## Notes
- `features.csv` / `features_test.csv` were not in the original repo. `extract_features.py` rebuilds them from what the models use (`value` plus the categorical `units`). The original files may have had extra columns.
- The original inference notebook wrote log-scale prices. `predict_qwen.py` applies `expm1` to convert them back.
