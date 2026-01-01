import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.cnn import CarBikeCNN
from src.data.dataset import get_dataloader
from src.data.transforms import train_transform, val_test_transform
from src.config.config import *
from src.utils.logger import setup_logger

logger = setup_logger()


def train():
    logger.info("Training started")

    train_loader = get_dataloader(
        "data/processed/train",
        train_transform,
        BATCH_SIZE,
        shuffle=True
    )

    val_loader = get_dataloader(
        "data/processed/val",
        val_test_transform,
        BATCH_SIZE,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = CarBikeCNN().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    patience = 5
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        # Progressive fine-tuning
        lr = 1e-3 if epoch < 8 else 1e-4
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | LR={lr}")

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        logger.info(
            f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("Best model saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered")
            break

    logger.info("Training completed")


if __name__ == "__main__":
    train()
