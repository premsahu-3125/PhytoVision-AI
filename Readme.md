# 🌿 PhytoVision AI: Bean Leaf Disease Diagnostic & Explainable Advisory Platform

[![CI Pipeline](https://github.com/PrachiAg-02/plant_disease_disease/actions/workflows/ci.yml/badge.svg)](https://github.com/PrachiAg-02/plant_disease_disease/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-MobileNetV4-EE4C2C.svg)
![Backend](https://img.shields.io/badge/FastAPI-optional-009688.svg)
![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)

An explainable-AI (XAI) computer vision tool for **bean leaf** disease diagnosis: classification,
Grad-CAM lesion localization, and a structured agronomic treatment plan, with a downloadable PDF report.

Scope note: the shipped model recognizes exactly three classes — `angular_leaf_spot`, `bean_rust`,
and `healthy` — from bean leaf photos. It is not a general-purpose plant identifier.

---

## 🏗️ System Architecture

```text
Image upload (file / camera / example gallery)
     │
     ▼
Validation (format check, <10MB, EXIF-orientation fix, corrupt-file guard)
     │
     ▼
Preprocessing (Resize 224x224, ImageNet Normalization)
     │
     ▼
MobileNetV4 (PyTorch) forward + backward pass for Grad-CAM
     │
     ▼
Confidence threshold (Threshold: 65.0%)
     ├── Low confidence (<65%)  ──► Inconclusive warning & retake guidance
     └── Accepted result (≥65%) ──► Disease classification + Grad-CAM heatmap + Agronomic Advisory + PDF
```

## 📁 Project layout

| File | Purpose |
|---|---|
| `model_utils.py` | Shared core: model loading, Grad-CAM, image validation, inference pipeline. Used by both `streamlit_app.py` and `app.py` so the logic lives in one place. |
| `streamlit_app.py` | Main user-facing app (upload / camera / example photos, results, history, disease guide). |
| `app.py` | Optional FastAPI JSON API exposing the same pipeline (`POST /diagnose`). |
| `advisory.py` | Treatment/severity/symptom reference data per disease class. |
| `pdf_report.py` | Generates the downloadable diagnostic PDF. |
| `predict.py` | CLI inference for a single image file. |
| `models/` | Trained weights (`.pth`) and ONNX export. |

## 🚀 Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ☁️ Deploying on Streamlit Community Cloud

Use the trimmed, CPU-only dependency file instead of the full one — it installs
much faster and uses far less memory on the free tier:

1. In your app's **Settings → Advanced settings**, set "Requirements file" to `requirements-streamlit.txt`.
2. Main file path: `streamlit_app.py`.

`requirements-streamlit.txt` installs CPU-only PyTorch wheels and skips packages the
Streamlit app never imports (FastAPI, ONNX runtime, scikit-learn, matplotlib, pytest).

## 🧪 Tests

```bash
pytest -v
```
