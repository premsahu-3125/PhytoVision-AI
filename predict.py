import argparse
import os

import numpy as np
from PIL import Image

from model_utils import build_model, build_grad_cam, diagnose_image


def predict(image_path: str):
    model = build_model()
    grad_cam = build_grad_cam(model)

    if not os.path.exists(image_path):
        print(f"⚠️ Image '{image_path}' not found. Generating a synthetic test specimen...")
        image = Image.fromarray(np.random.randint(40, 180, (224, 224, 3), dtype=np.uint8))
    else:
        image = Image.open(image_path).convert("RGB")

    result = diagnose_image(image, model, grad_cam)

    print("\n==========================================")
    print("🌿 PHYTOMOBILE INFERENCE DIAGNOSIS")
    print("==========================================")
    print(f"Specimen Path    : {image_path}")
    print(f"Predicted Disease: {result.display_title}")
    print(f"Model Confidence : {result.confidence:.2f}%")
    print(f"Softmax Breakdown: {result.probabilities}")
    print("==========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhytoVision AI Specimen Inference")
    parser.add_argument("--image", type=str, default="sample_leaf.jpg", help="Path to input leaf image")
    args = parser.parse_args()
    predict(args.image)
