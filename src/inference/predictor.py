import os
import io
import torch
from PIL import Image
from torchvision import transforms

#  USE QUANTIZED MODEL CLASS
from src.models.cnn_quantized import CarBikeCNNQuantized

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_quantized.pth")

print("Loading model from:", MODEL_PATH)

# -------------------------------------------------
# Preprocessing (MUST match val/test transform)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -------------------------------------------------
# Load model (DYNAMIC quantization only)
# -------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # 1️⃣ Create model
    model = CarBikeCNNQuantized()

    # 2️⃣ Apply dynamic quantization (Linear only)
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # 3️⃣ Load weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)

    print("Quantized model loaded successfully")
    return model

# -------------------------------------------------
# Prediction
# -------------------------------------------------
CLASS_NAMES = {
    0: "Bike 🏍️",   #  MUST match training class_to_idx
    1: "Car 🚗"
}

def predict_image_from_bytes(image_bytes: bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(logits, dim=1).item()

    return CLASS_NAMES[pred]
