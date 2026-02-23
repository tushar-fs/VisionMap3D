"""
engine.py - Feature extraction using a pre-trained ResNet18 model.

This is the core ML component. It takes a raw image and converts it into
a 512-dimensional vector (embedding) that captures the visual/spatial
features of the image. These embeddings are what make similarity search possible.

"""

import io
import time
import logging
from typing import List

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger("visionmap.engine")


class FeatureExtractor:
    """
    Wraps a pre-trained ResNet18 and uses it as a feature extractor
    (not a classifier). The trick is replacing the final fully-connected
    layer with nn.Identity() so the forward pass outputs the raw 512-d
    embedding from the avgpool layer instead of 1000 ImageNet class logits.
    """

    def __init__(self):
        # Pick the best available hardware accelerator.
        # PyTorch defaults to CPU if you don't explicitly set this, which
        # makes inference ~10-50x slower on machines that have a GPU.
        # On my Mac, MPS (Metal Performance Shaders) gives a nice speedup.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"FeatureExtractor using device: {self.device}")

        # Load ResNet18 with pre-trained ImageNet weights, then chop off the
        # classification head. The `fc` layer normally maps 512 features -> 1000
        # classes. Replacing it with Identity() means we get the raw 512-d vector.
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Identity()

        # IMPORTANT: .to(device) moves model weights to GPU/MPS memory,
        # and .eval() switches BatchNorm + Dropout to inference mode.
        # Forgetting eval() is a sneaky bug — BatchNorm will use per-batch
        # statistics instead of the learned running mean/variance, which
        # makes embeddings non-deterministic between calls.
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing. The model was trained with these
        # exact transforms, so we MUST use the same values or our embeddings
        # will be garbage. This is sometimes called "preprocessing parity."
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info("FeatureExtractor initialized (ResNet18, 512-d output)")

    def extract(self, image_bytes: bytes) -> List[float]:
        """
        Take raw image bytes and return a 512-d feature vector.

        Uses torch.no_grad() to disable gradient tracking during inference.
        Without it, PyTorch builds a computational graph for backpropagation
        which roughly doubles memory usage — a real problem when you're
        handling concurrent requests on a GPU with limited VRAM.
        """
        start = time.perf_counter()

        # Decode bytes -> PIL -> RGB (handles grayscale/RGBA edge cases)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess and add batch dimension: [C, H, W] -> [1, C, H, W]
        tensor = self.preprocess(image).unsqueeze(0)

        # Move input to the same device as the model — forgetting this
        # is the #1 cause of "expected cuda tensor, got cpu tensor" errors
        tensor = tensor.to(self.device)

        # Run the forward pass with gradients disabled
        with torch.no_grad():
            embedding = self.model(tensor)

        # Squeeze batch dim, move back to CPU, convert to plain Python list
        result = embedding.squeeze(0).cpu().tolist()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Inference: {elapsed_ms:.1f}ms | device={self.device} | dims={len(result)}")

        return result
