import torch
from src.models.cnn import CarBikeCNN

FP32_MODEL = "models/best_model.pth"
INT8_MODEL = "models/best_model_quantized.pth"

# Load FP32 model
model = CarBikeCNN()
model.load_state_dict(torch.load(FP32_MODEL, map_location="cpu"))
model.eval()

# SAFE quantization (Linear layers only)
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(model.state_dict(), INT8_MODEL)
print("Dynamic quantized model saved successfully")
