"""
Shared inference core for PhytoVision AI.

Both streamlit_app.py (UI) and app.py (FastAPI service) import from here so
the model-loading, Grad-CAM, and pre/post-processing logic lives in exactly
one place instead of being duplicated (and drifting) across two files.
"""
import io
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
import timm

from advisory import get_treatment_plan

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
CONFIDENCE_THRESHOLD = 65.0
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_MB = 10
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mobilenetv4_plant_disease.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class ImageValidationError(ValueError):
    """Raised when an uploaded file isn't a usable image."""


def load_and_validate_image(file_bytes: bytes, max_mb: int = MAX_UPLOAD_MB) -> Image.Image:
    """Decode raw bytes into a clean, upright RGB PIL image.

    Handles the two most common real-world gotchas that the original code
    didn't: phone photos with EXIF rotation tags, and corrupt/non-image
    uploads (previously these threw an unhandled PIL exception that crashed
    the whole page instead of showing a friendly error).
    """
    if len(file_bytes) > max_mb * 1024 * 1024:
        raise ImageValidationError(f"File exceeds the {max_mb}MB limit.")
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # force decode now so truncated files fail here, not later
    except Exception as exc:
        raise ImageValidationError("That file isn't a readable image.") from exc

    img = ImageOps.exif_transpose(img)  # fix sideways/upside-down phone photos
    return img.convert("RGB")


class GradCAM:
    """Minimal Grad-CAM implementation hooked onto a single conv layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        output[0, class_idx].backward(retain_graph=True)

        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, F.softmax(output, dim=1).detach().cpu().numpy()[0]


def build_model() -> torch.nn.Module:
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def build_grad_cam(model: torch.nn.Module) -> GradCAM:
    target_layer = model.conv_head if hasattr(model, "conv_head") else list(model.children())[-2]
    return GradCAM(model, target_layer)


def make_heatmap_overlay(pil_image: Image.Image, cam: np.ndarray) -> bytes:
    """Blend a Grad-CAM activation map over the (resized) source image, PNG bytes out."""
    base = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * base + 0.4 * heatmap)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode heatmap overlay.")
    return buffer.tobytes()


@dataclass
class DiagnosisResult:
    label: str
    display_title: str
    confidence: float
    is_uncertain: bool
    probabilities: dict
    heatmap_png: bytes
    treatment: dict = field(default_factory=dict)


def diagnose_image(pil_image: Image.Image, model: torch.nn.Module, grad_cam: GradCAM) -> DiagnosisResult:
    """Run the full pipeline: preprocess -> forward+Grad-CAM -> heatmap -> advisory."""
    input_tensor = _TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.enable_grad():
        cam, pred_idx, probs = grad_cam.generate(input_tensor)

    label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    heatmap_png = make_heatmap_overlay(pil_image, cam)

    if is_uncertain:
        treatment = {
            "chemical": "Diagnosis uncertain. Do not apply chemical treatments without secondary lab verification.",
            "organic": "Inspect foliar tissue under natural daylight; check for early spore formations.",
            "prevention": "Retake the leaf photograph under even lighting against a neutral background.",
        }
        display_title = "Inconclusive / Low Confidence"
    else:
        treatment = get_treatment_plan(label)
        display_title = label.replace("_", " ").title()

    probabilities = {CLASS_NAMES[i]: round(float(probs[i] * 100), 2) for i in range(len(CLASS_NAMES))}

    return DiagnosisResult(
        label=label,
        display_title=display_title,
        confidence=confidence,
        is_uncertain=is_uncertain,
        probabilities=probabilities,
        heatmap_png=heatmap_png,
        treatment=treatment,
    )
