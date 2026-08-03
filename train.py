"""
train.py
========
Main training script for the ClickbaitDetector model.

================================================================
HOW TO RETRAIN ON YOUR DATASET:
================================================================
1. Set your dataset path in dataset_loader.py (DATASET_PATH variable)
2. Adjust TRAINING CONFIG below if needed
3. Run:
       python train.py

Model checkpoints are saved to: checkpoints/
Best model is saved to:         checkpoints/best_model.pt
================================================================

Training pipeline:
    1. Load & tokenize data (dataset_loader.py)
    2. Initialize DistilBERT + CNN model (model.py)
    3. Train with AdamW optimizer + CrossEntropyLoss
    4. Validate each epoch, save best checkpoint
    5. Early stopping to prevent overfitting
    6. Final evaluation on held-out test set
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm

from model import ClickbaitDetector
from dataset_loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================
# TRAINING CONFIGURATION — ADJUST AS NEEDED
# ================================================================
CONFIG = {
    "pretrained_model":   "distilbert-base-uncased",
    "freeze_bert":        False,     # True = faster but less accurate
    "batch_size":         32,
    "num_epochs":         10,
    "learning_rate":      2e-5,      # Good default for fine-tuning BERT
    "weight_decay":       0.01,      # L2 regularization
    "warmup_steps":       100,
    "early_stop_patience": 3,        # Stop if val loss doesn't improve for N epochs
    "checkpoint_dir":     "checkpoints",
    "best_model_name":    "best_model.pt",
    "dropout":            0.3,
    "grad_clip":          1.0,       # Gradient clipping to prevent exploding gradients
    "seed":               42,
}
# ================================================================


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow — GPU recommended)")
    return device


def train_one_epoch(
    model: ClickbaitDetector,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: float,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        (avg_loss, accuracy, f1_score)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Loss computation
        loss = criterion(outputs.logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping — prevents exploding gradients with transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(
    model: ClickbaitDetector,
    loader,
    criterion,
    device: torch.device,
    split_name: str = "Val",
) -> tuple:
    """
    Evaluate model on a data split without gradient computation.

    Returns:
        (avg_loss, accuracy, f1_score)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Evaluating ({split_name})", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return avg_loss, acc, f1


def train():
    """Main training entrypoint."""
    set_seed(CONFIG["seed"])
    device = get_device()

    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ================================================================
    # STEP 1: Load Data
    # Dataset path is set in dataset_loader.py → DATASET_PATH variable
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading datasets")
    logger.info("=" * 60)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=CONFIG["batch_size"]
    )

    # Save tokenizer alongside checkpoint for consistent inference later
    tokenizer.save_pretrained(os.path.join(CONFIG["checkpoint_dir"], "tokenizer"))
    logger.info("Tokenizer saved to checkpoints/tokenizer/")

    # ================================================================
    # STEP 2: Initialize Model
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing ClickbaitDetector model")
    logger.info("=" * 60)
    model = ClickbaitDetector(
        pretrained_model=CONFIG["pretrained_model"],
        freeze_bert=CONFIG["freeze_bert"],
        cnn_dropout=CONFIG["dropout"],
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ================================================================
    # STEP 3: Optimizer, Loss, Scheduler
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Setting up optimizer")
    logger.info("=" * 60)

    # AdamW: Adam with decoupled weight decay (best for transformers)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Cosine annealing LR — gradually reduces learning rate during training
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"],
        eta_min=1e-6
    )

    # CrossEntropyLoss for binary classification
    criterion = nn.CrossEntropyLoss()

    # ================================================================
    # STEP 4: Training Loop with Early Stopping
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Training")
    logger.info("=" * 60)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_history = []
    best_epoch = 0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Epoch {epoch}/{CONFIG['num_epochs']}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"{'=' * 40}")

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, CONFIG["grad_clip"]
        )

        # Validate
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, split_name="Val"
        )

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log metrics
        logger.info(f"Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        logger.info(f"Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        logger.info(f"Epoch time: {epoch_time:.1f}s")

        # Record history
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss":   val_loss,   "val_acc":   val_acc,   "val_f1":   val_f1,
        })

        # Checkpoint: save if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save model state dict + config
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": CONFIG,
            }
            best_ckpt_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["best_model_name"])
            torch.save(checkpoint, best_ckpt_path)
            logger.info(f"✓ New best model saved (val_loss={val_loss:.4f})")

        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{CONFIG['early_stop_patience']} epochs")

            # Early stopping
            if epochs_no_improve >= CONFIG["early_stop_patience"]:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}!")
                logger.info(f"Best epoch was: {best_epoch}")
                break

    # ================================================================
    # STEP 5: Final Test Evaluation
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Final evaluation on test set")
    logger.info("=" * 60)

    # Load best checkpoint
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info(f"Loaded best model from epoch {best_ckpt['epoch']}")

    # Full evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            outputs        = model(input_ids, attention_mask)
            preds          = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=["Non-Clickbait", "Clickbait"]
    )
    logger.info("\nTest Set Classification Report:\n" + report)

    # Save training history + results
    results = {
        "training_history": training_history,
        "test_classification_report": report,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    results_path = os.path.join(CONFIG["checkpoint_dir"], "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nTraining results saved to: {results_path}")

    # ================================================================
    # NEXT STEP: Export the model for the Chrome extension
    #            Run: python export_model.py
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("Next step: run  python export_model.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
