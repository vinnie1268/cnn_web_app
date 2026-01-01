import torch
from sklearn.metrics import accuracy_score
from src.models.cnn import CarBikeCNN
from src.data.dataset import get_dataloader
from src.data.transforms import val_test_transform
from src.config.config import *

def evaluate():
    loader = get_dataloader(
        "data/processed/test",
        val_test_transform,
        BATCH_SIZE,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CarBikeCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).cpu().tolist()

            logits = model(images)          # shape: (B, 2)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()  # (B,)

            preds.extend(batch_preds)
            labels_all.extend(labels)

    print("Total labels:", len(labels_all))
    print("Total preds:", len(preds))
    print("Labels sample:", labels_all[:10])
    print("Preds sample:", preds[:10])

    print("Accuracy:", accuracy_score(labels_all, preds))


if __name__ == "__main__":
    evaluate()
