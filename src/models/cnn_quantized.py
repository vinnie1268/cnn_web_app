import torch
import torch.nn as nn

# Import the original FP32 model
from src.models.cnn import CarBikeCNN


class CarBikeCNNQuantized(CarBikeCNN):
    """
    Wrapper class for dynamic quantization.

    IMPORTANT:
    - Inherits exact architecture from CarBikeCNN
    - No QuantStub / DeQuantStub
    - No static quantization
    - Quantization is applied at runtime (Linear layers only)
    """

    def __init__(self):
        super().__init__()

    def quantize(self):
        """
        Apply dynamic quantization to Linear layers only.
        Call this AFTER loading weights.
        """
        return torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=torch.qint8
        )
