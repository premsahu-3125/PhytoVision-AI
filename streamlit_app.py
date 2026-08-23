import io
from datetime import datetime

import streamlit as st
import plotly.express as px
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_MB,
    build_model,
    build_grad_cam,
    diagnose_image,
    load_and_validate_image,
    ImageValidationError,
)
from pdf_report import generate_pdf_report

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PhytoVision AI | Bean Leaf Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "PhytoVision AI — an explainable-AI diagnostic tool for bean leaf diseases. "
                  "Built with MobileNetV4 + Grad-CAM. For informational use only."
    },
)

EXAMPLE_IMAGES = {
    "Angular Leaf Spot (example)": "assets/examples/angular_leaf_spot.jpg",
    "Bean Rust (example)": "assets/examples/bean_rust.jpg",
    "Healthy Leaf (example)": "assets/examples/healthy.jpg",
}

# ----------------- STYLE -----------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

    .pv-hero {
        background: linear-gradient(120deg, #065F46 0%, #059669 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .pv-hero h1 {margin: 0; font-size: 1.7rem;}
    .pv-hero p {margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem;}

    .pv-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .pv-card h4 {margin-top: 0; margin-bottom: 0.4rem;}
    .pv-badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pv-badge-healthy {background:#D1FAE5; color:#065F46;}
    .pv-badge-disease {background:#FEE2E2; color:#991B1B;}
    .pv-badge-warn {background:#FEF3C7; color:#92400E;}
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADER -----------------
@st.cache_resource(show_spinner="Loading diagnostic model (first run only)…")
def get_model_and_gradcam():
    model = build_model()
    grad_cam = build_grad_cam(model)
    return model, grad_cam

model, grad_cam = get_model_and_gradcam()

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "active_image_bytes" not in st.session_state:
    st.session_state.active_image_bytes = None
if "active_image_name" not in st.session_state:
    st.session_state.active_image_name = None
if "result" not in st.session_state:
    st.session_state.result = None


def _run_diagnosis(image_bytes: bytes):
    """Validate + run the model, store result & history in session_state."""
    try:
        pil_image = load_and_validate_image(image_bytes, max_mb=MAX_UPLOAD_MB)
    except ImageValidationError as exc:
        st.session_state.result = None
        st.error(f"⚠️ {exc}")
        return

    with st.spinner("🔬 Running inference & computing Grad-CAM localization…"):
        try:
            result = diagnose_image(pil_image, model, grad_cam)
        except Exception:
            st.error("⚠️ Something went wrong analyzing that image. Please try a different photo.")
            return

    st.session_state.result = result
    st.session_state.result_thumb = pil_image.copy()
    st.session_state.result_thumb.thumbnail((160, 160))
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": result.display_title,
        "confidence": result.confidence,
        "thumb": st.session_state.result_thumb,
    })
    st.session_state.history = st.session_state.history[:10]


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🌿 PhytoVision AI")
    st.caption("Explainable bean-leaf disease diagnostics")
    st.divider()

    st.markdown("##### 📋 What this model detects")
    st.markdown(
        "- **Angular Leaf Spot**\n"
        "- **Bean Rust**\n"
        "- **Healthy foliage**\n"
    )
    st.caption("Trained only on bean-leaf photos — results on other crops won't be meaningful.")
    st.divider()

    st.markdown("##### 📸 Best photo practices")
    st.markdown(
        "- Fill ~70% of the frame with a single leaf\n"
        "- Use even, diffuse daylight — avoid glare/deep shadow\n"
        "- Focus on the affected area, not the whole plant\n"
        f"- JPG/PNG, up to {MAX_UPLOAD_MB}MB"
    )
    st.divider()

    st.warning(
        "**Agronomic disclaimer:** predictions are for informational guidance only. "
        "Consult a local extension specialist before large-scale chemical application."
    )

# ----------------- HERO -----------------
st.markdown("""
<div class="pv-hero">
    <h1>🌿 PhytoVision AI</h1>
    <p>Upload a bean leaf photo for instant disease classification, Grad-CAM lesion localization, and a tiered treatment plan.</p>
</div>
""", unsafe_allow_html=True)

tab_diagnose, tab_guide, tab_history = st.tabs(["🔬 Diagnose", "📚 Disease Guide", "🕘 History"])

# ===================== DIAGNOSE TAB =====================
with tab_diagnose:
    input_mode = st.radio(
        "Choose an image source",
        ["Upload a photo", "Use my camera", "Try an example"],
        horizontal=True,
        label_visibility="collapsed",
    )

    new_bytes, new_name = None, None

    if input_mode == "Upload a photo":
        uploaded_file = st.file_uploader("Upload Leaf Specimen", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            new_bytes = uploaded_file.getvalue()
            new_name = uploaded_file.name

    elif input_mode == "Use my camera":
        camera_file = st.camera_input("Take a photo of a leaf")
        if camera_file is not None:
            new_bytes = camera_file.getvalue()
            new_name = "camera_capture.jpg"

    else:
        cols = st.columns(3)
        for col, (label, path) in zip(cols, EXAMPLE_IMAGES.items()):
            with col:
                st.image(path, use_container_width=True, caption=label)
                if st.button(f"Use this", key=f"ex_{label}", use_container_width=True):
                    with open(path, "rb") as f:
                        new_bytes = f.read()
                    new_name = path

    # A newly provided image invalidates any previous result until re-diagnosed
    if new_bytes is not None and new_bytes != st.session_state.active_image_bytes:
        st.session_state.active_image_bytes = new_bytes
        st.session_state.active_image_name = new_name
        st.session_state.result = None

    col_left, col_right = st.columns([1, 1.25], gap="large")

    with col_left:
        if st.session_state.active_image_bytes:
            try:
                preview = Image.open(io.BytesIO(st.session_state.active_image_bytes))
                st.image(preview, caption="Selected specimen", use_container_width=True)
                if st.button("🚀 Diagnose & Generate Heatmap", use_container_width=True, type="primary"):
                    _run_diagnosis(st.session_state.active_image_bytes)
            except Exception:
                st.error("⚠️ Couldn't preview this file — it may not be a valid image.")
        else:
            st.info("👆 Upload a photo, use your camera, or try an example to begin.")

    with col_right:
        result = st.session_state.result
        if result is None:
            st.empty()
        else:
            if result.is_uncertain:
                st.warning(
                    f"⚠️ **Low-confidence diagnostic flag (<{CONFIDENCE_THRESHOLD:.0f}%)** — "
                    "the specimen prediction is inconclusive. Try retaking the photo under even lighting."
                )
                badge_class = "pv-badge-warn"
            elif result.label == "healthy":
                badge_class = "pv-badge-healthy"
            else:
                badge_class = "pv-badge-disease"

            st.markdown(
                f"### {result.display_title} "
                f"<span class='pv-badge {badge_class}'>{result.treatment.get('severity', '')}</span>",
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Model Confidence", f"{result.confidence:.1f}%")
            m2.progress(min(int(result.confidence), 100), text="Confidence")

            with st.expander("🎯 Grad-CAM Lesion Localization Overlay", expanded=True):
                st.image(result.heatmap_png, use_container_width=True)
                st.caption(
                    "Warmer colors (red/yellow) show the image regions that most influenced the model's "
                    "prediction — useful for sanity-checking that it's actually looking at the leaf lesion."
                )

            with st.expander("📊 Prediction probability breakdown"):
                df_probs = [
                    {"Class": k.replace("_", " ").title(), "Probability (%)": v}
                    for k, v in sorted(result.probabilities.items(), key=lambda kv: -kv[1])
                ]
                fig = px.bar(df_probs, x="Probability (%)", y="Class", orientation="h", text="Probability (%)")
                fig.update_traces(marker_color="#059669")
                fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if not result.is_uncertain:
                st.markdown(f"**Symptoms observed for this class:** {result.treatment.get('symptoms', 'N/A')}")

            st.markdown("#### 📋 Remediation Protocol")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='pv-card'><h4>🧪 Chemical</h4>{result.treatment.get('chemical','N/A')}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='pv-card'><h4>🌱 Organic</h4>{result.treatment.get('organic','N/A')}</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='pv-card'><h4>🛡️ Prevention</h4>{result.treatment.get('prevention','N/A')}</div>", unsafe_allow_html=True)

            st.write("")
            img_byte_arr = io.BytesIO()
            preview_rgb = Image.open(io.BytesIO(st.session_state.active_image_bytes)).convert("RGB")
            preview_rgb.save(img_byte_arr, format="JPEG")
            pdf_buffer = generate_pdf_report(
                result.display_title, result.confidence, result.treatment,
                img_byte_arr.getvalue(), result.heatmap_png,
            )
            st.download_button(
                "📥 Download Diagnostic PDF Report",
                data=pdf_buffer,
                file_name=f"Pathology_Report_{result.label}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ===================== GUIDE TAB =====================
with tab_guide:
    st.markdown("### About this model")
    st.markdown(
        "PhytoVision AI classifies **bean leaf photos** into three categories using a "
        "MobileNetV4 convolutional network, and highlights the pixels driving that decision "
        "with **Grad-CAM** (Gradient-weighted Class Activation Mapping). It is scoped narrowly "
        "on purpose — it has not been trained on other crops, so predictions on non-bean leaves "
        "won't be reliable."
    )

    st.markdown("### Disease reference")
    for key, info in {
        "angular_leaf_spot": "Angular Leaf Spot",
        "bean_rust": "Bean Rust",
        "healthy": "Healthy",
    }.items():
        from advisory import get_treatment_plan
        plan = get_treatment_plan(key)
        with st.expander(f"**{info}** — severity: {plan.get('severity','')}"):
            st.markdown(f"**Symptoms:** {plan.get('symptoms','N/A')}")
            st.markdown(f"**Chemical treatment:** {plan.get('chemical','N/A')}")
            st.markdown(f"**Organic treatment:** {plan.get('organic','N/A')}")
            st.markdown(f"**Prevention:** {plan.get('prevention','N/A')}")

    st.markdown("### How to read the confidence score")
    st.markdown(
        f"- **≥ {CONFIDENCE_THRESHOLD:.0f}%** — treated as a confident prediction and shown with a full treatment plan.\n"
        f"- **< {CONFIDENCE_THRESHOLD:.0f}%** — flagged as inconclusive; you'll be asked to retake the photo rather than "
        "act on a shaky prediction."
    )

    st.info(
        "This tool provides informational guidance only and is **not a substitute for a certified "
        "agronomist or plant pathologist**, especially before large-scale chemical treatment."
    )

# ===================== HISTORY TAB =====================
with tab_history:
    if not st.session_state.history:
        st.info("No diagnoses yet this session. Results you generate in the Diagnose tab will appear here.")
    else:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
        for entry in st.session_state.history:
            c1, c2 = st.columns([1, 5])
            with c1:
                st.image(entry["thumb"])
            with c2:
                st.markdown(f"**{entry['label']}** — {entry['confidence']:.1f}% confidence")
                st.caption(f"Diagnosed at {entry['time']} (this session only, not saved after you close the tab)")
            st.divider()
