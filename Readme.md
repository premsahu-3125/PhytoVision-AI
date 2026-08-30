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

## 🚀 Try It Live

### 🎉 One-Click Deploy to Streamlit Cloud (3 minutes)

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/new?repo=https://github.com/premsahu-3125/PhytoVision-AI)

**OR follow the quick guide:**
- 📖 [Complete Streamlit Deploy Guide](./STREAMLIT_DEPLOY.md) - Full instructions
- ⚡ [5-Minute Quick Deploy](./QUICK_STREAMLIT_DEPLOY.md) - TL;DR version

**After deploying**, add your live URL to GitHub's About section:
```bash
gh repo edit premsahu-3125/PhytoVision-AI --homepage "https://your-streamlit-url"
```

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

App runs on: `http://localhost:8501`

---

## ☁️ Deploying on Streamlit Community Cloud

**Streamlit Community Cloud is FREE and perfect for hosting this AI app!**

### Quick Deploy (5 minutes)
See [QUICK_STREAMLIT_DEPLOY.md](./QUICK_STREAMLIT_DEPLOY.md) for step-by-step instructions.

### Full Guide (with details)
See [STREAMLIT_DEPLOY.md](./STREAMLIT_DEPLOY.md) for complete deployment guide.

### Manual Steps
1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select this GitHub repo
4. In Advanced settings, set Requirements file to: `requirements-streamlit.txt`
5. Click Deploy

**Why `requirements-streamlit.txt`?**
- Uses CPU-only PyTorch (much faster, fits free tier)
- Skips unnecessary packages
- ~80% smaller, deploys in 2-3 minutes instead of 10+

**Result:** You get a live URL like `https://phytovision-ai-[random].streamlit.app`

---

## 🧪 Tests

```bash
pytest -v
```
