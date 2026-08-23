import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_utils import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    build_model,
    build_grad_cam,
    diagnose_image,
    load_and_validate_image,
    ImageValidationError,
)

app = FastAPI(title="PhytoVision AI - Plant Pathology API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = build_model()
grad_cam = build_grad_cam(model)


@app.get("/")
def root():
    return {
        "status": "online",
        "model": "MobileNetV4",
        "features": ["Classification", "Grad-CAM Localization", "Agronomic Advisory", "Confidence Guardrail"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "device": str(DEVICE),
    }


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    image_bytes = await file.read()
    try:
        pil_image = load_and_validate_image(image_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = diagnose_image(pil_image, model, grad_cam)

    return {
        "disease": result.display_title,
        "raw_label": result.label,
        "confidence": round(result.confidence, 2),
        "is_uncertain": result.is_uncertain,
        "probabilities": result.probabilities,
        "heatmap": base64.b64encode(result.heatmap_png).decode("utf-8"),
        "treatment": result.treatment,
    }
