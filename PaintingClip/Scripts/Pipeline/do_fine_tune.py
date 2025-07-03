#!/usr/bin/env python

import os
import random
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ---------------------------
# Configurable defaults
# ---------------------------
EXCEL_PATH       = "fine_tune_dataset.xlsx"
IMAGE_DIR        = os.path.join("..", "Dataset", "Images")
OUTPUT_MODEL     = "clip_finetuned_lora"
OUTPUT_PLOT      = "val_nce_over_epochs.png"
TRAIN_PLOT       = "train_loss_over_epochs.png"
BEST_MODEL_DIR   = "clip_finetuned_lora_best"

CLIP_MODEL_ID    = "openai/clip-vit-base-patch32"

# LoRA defaults (increased capacity)
DEFAULT_LORA_R       = 16
DEFAULT_LORA_ALPHA   = 32
DEFAULT_LORA_DROPOUT = 0.05

# Training defaults
DEFAULT_EPOCHS    = 15
DEFAULT_BATCH     = 16
DEFAULT_LR        = 2e-4
DEFAULT_TEST_SZ   = 0.1
DEFAULT_SEED      = 42
DEFAULT_WORKERS   = 4
DEFAULT_PIN_MEM   = True


class PaintingDataset(Dataset):
    """Dataset wrapping (image, text) pairs for CLIP fine-tuning."""
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["File Name"])
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, ValueError):
            return None
        text = row["Textual Label"]
        return {"image": image, "text": text}


class Collator:
    """Picklable collator that applies a CLIPProcessor to a batch."""
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        images = [item["image"] for item in batch]
        texts  = [item["text"]  for item in batch]
        return self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on paintings with LoRA adapters"
    )
    parser.add_argument("--data",        "-d", default=EXCEL_PATH)
    parser.add_argument("--images",      "-i", default=IMAGE_DIR)
    parser.add_argument("--output-model","-o", default=OUTPUT_MODEL)
    parser.add_argument("--output-plot", "-p", default=OUTPUT_PLOT,
                        help="PNG file for validation NCE plot")
    parser.add_argument("--train-plot",  "-t", default=TRAIN_PLOT)
    parser.add_argument("--best-model-dir", default=BEST_MODEL_DIR)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--test-size",   type=float, default=DEFAULT_TEST_SZ)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--pin-memory",  action="store_true", default=DEFAULT_PIN_MEM)
    parser.add_argument("--lora-r",      type=int,   default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha",  type=int,   default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout",type=float, default=DEFAULT_LORA_DROPOUT)
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load & filter dataset
    df    = pd.read_excel(args.data)
    total = len(df)
    exists = df["File Name"].apply(
        lambda fn: os.path.exists(os.path.join(args.images, fn))
    )
    missing = (~exists).sum()
    df = df[exists].reset_index(drop=True)

    bad = []
    for fn in df["File Name"]:
        path = os.path.join(args.images, fn)
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
        except (UnidentifiedImageError, ValueError):
            bad.append(fn)
            continue
        if rgb.size[0] <= 1 or rgb.size[1] <= 1:
            bad.append(fn)
    if bad:
        df = df[~df["File Name"].isin(bad)].reset_index(drop=True)

    print(f"Dropped {missing}/{total} missing, {len(bad)} malformed files.")

    # Split
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # CLIP & processor
    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=False)
    base_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)

    # LoRA adapters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["visual_projection", "text_projection"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    print("Applied LoRA adapters. Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # DataLoaders
    collator     = Collator(processor)
    train_loader = DataLoader(
        PaintingDataset(train_df, args.images),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collator
    )
    val_loader   = DataLoader(
        PaintingDataset(val_df, args.images),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collator
    )

    # Optimizer
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training & validation
    best_val_nce = float("inf")
    train_losses = []
    val_nces     = []
    tau = 1.0 / model.logit_scale.exp().item()

    for epoch in range(1, args.epochs + 1):
        # training
        model.train()
        run_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, return_loss=True).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        avg_train = run_loss / len(train_loader)
        train_losses.append(avg_train)

        # validation (InfoNCE)
        model.eval()
        run_nce = 0.0
        count   = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                img_feats = F.normalize(
                    model.get_image_features(pixel_values=batch["pixel_values"]), dim=-1
                )
                txt_feats = F.normalize(
                    model.get_text_features(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    ), dim=-1
                )
                # logits
                logits_i2t = (img_feats @ txt_feats.t()) / tau
                logits_t2i = logits_i2t.t()
                labels = torch.arange(logits_i2t.size(0), device=device)
                loss_i2t = F.cross_entropy(logits_i2t, labels)
                loss_t2i = F.cross_entropy(logits_t2i, labels)
                batch_nce = (loss_i2t + loss_t2i) / 2
                run_nce += batch_nce.item()
                count   += 1
        avg_nce = run_nce / count
        val_nces.append(avg_nce)

        print(f"Epoch {epoch}/{args.epochs}"
              f" - Train Loss: {avg_train:.4f}"
              f" - Val InfoNCE: {avg_nce:.4f}")
        if avg_nce < best_val_nce:
            best_val_nce = avg_nce
            model.save_pretrained(args.best_model_dir)
            print("→ New best model saved")

    # Save plots
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.grid(True)
    plt.savefig(args.train_plot)

    plt.figure()
    plt.plot(range(1, args.epochs + 1), val_nces, marker="o")
    plt.title("Validation InfoNCE Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg InfoNCE Loss")
    plt.grid(True)
    plt.savefig(args.output_plot)

    # Final save
    model.save_pretrained(args.output_model)
    print(f"Saved final model to '{args.output_model}'")


if __name__ == "__main__":
    main()
