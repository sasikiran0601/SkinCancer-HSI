"""
Hyperspectral Skin Cancer Diagnostic AI — Streamlit Cloud App
=============================================================
100% Free Cloud Deployment on Streamlit Community Cloud (Snowflake).
Processes 116-band HSI cubes and produces interactive lesion heatmaps,
clinical risk triage, and diagnostic breakdowns via ONNX Runtime.
"""
import io
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import streamlit as st

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Skin Cancer HSI Diagnostic AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_VERSION = "v1.0.0-champion"
# Look for model in local folder or model/outputs
POSSIBLE_MODEL_PATHS = [
    Path("champion_model.onnx"),
    Path("deployment/streamlit/champion_model.onnx"),
    Path("model/outputs/champion_model.onnx"),
]

CLASS_NAMES = ["BE", "BM", "ME", "MM"]
CLASS_LABELS = {
    0: "Benign Epidermal (BE)",
    1: "Benign Melanocytic (BM)",
    2: "Malignant Epidermal (ME)",
    3: "Malignant Melanoma (MM)",
}
CLASS_COLORS_HEX = {
    "BE": "#1E90FF",  # Dodger Blue
    "BM": "#2E8B57",  # Sea Green
    "ME": "#FF8C00",  # Dark Orange
    "MM": "#DC143C",  # Crimson Red
}
CLASS_COLORS_RGB = {
    0: (0.12, 0.56, 1.0),
    1: (0.13, 0.70, 0.30),
    2: (0.95, 0.60, 0.10),
    3: (0.88, 0.15, 0.15),
}

# ==============================================================================
# CACHED MODEL LOADER
# ==============================================================================
@st.cache_resource(show_spinner="Loading ONNX Inference Engine...")
def load_onnx_model():
    model_file = None
    for p in POSSIBLE_MODEL_PATHS:
        if p.exists():
            model_file = p
            break

    if model_file is None:
        st.error("Error: 'champion_model.onnx' not found. Please ensure the ONNX model is in the repository.")
        st.stop()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_file),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


session, input_name = load_onnx_model()


# ==============================================================================
# INFERENCE LOGIC
# ==============================================================================
def process_hsi_cube(cube: np.ndarray, batch_size: int = 256):
    """Runs vectorized ONNX inference on an HSI cube (H, W, 116)."""
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, 116), received shape {cube.shape}")

    if cube.shape[0] == 116:
        cube = np.transpose(cube, (1, 2, 0))
    elif cube.shape[2] != 116:
        raise ValueError(f"Expected 116 spectral bands, received {cube.shape[2]}")

    cube = cube.astype(np.float32)
    H, W, B = cube.shape
    total_pixels = H * W
    pixels = cube.reshape(-1, B)

    t0 = time.perf_counter()
    all_probs = []
    for i in range(0, total_pixels, batch_size):
        batch = pixels[i : i + batch_size]
        logits = session.run(None, {input_name: batch})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.append(probs)

    infer_time_ms = (time.perf_counter() - t0) * 1000
    all_probs = np.vstack(all_probs)
    preds = np.argmax(all_probs, axis=1).reshape(H, W)
    return preds, all_probs, infer_time_ms


def generate_heatmap_image(pred_map: np.ndarray) -> np.ndarray:
    """Generates an RGB color-coded spatial lesion segmentation image."""
    H, W = pred_map.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)
    for c_idx, color in CLASS_COLORS_RGB.items():
        rgb_img[pred_map == c_idx] = color
    return (rgb_img * 255).astype(np.uint8)


def generate_bar_chart(percentages: dict):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
    names = [CLASS_LABELS[i] for i in range(4)]
    values = [percentages[CLASS_NAMES[i]] for i in range(4)]
    colors = [CLASS_COLORS_RGB[i] for i in range(4)]

    bars = ax.barh(names, values, color=colors, height=0.55)
    ax.set_xlim(0, max(max(values) * 1.25, 10))
    ax.set_xlabel("Pixel Percentage (%)", fontsize=9, fontweight="bold")
    ax.set_title("Lesion Tissue Composition", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=72)
    st.title("Diagnostic AI System")
    st.caption(f"Model Release: **{MODEL_VERSION}**")

    st.markdown("---")
    st.subheader("🏷️ Class Legend")
    st.markdown(f"- 🟦 <span style='color:{CLASS_COLORS_HEX['BE']};font-weight:bold;'>BE</span>: Benign Epidermal", unsafe_allow_html=True)
    st.markdown(f"- 🟩 <span style='color:{CLASS_COLORS_HEX['BM']};font-weight:bold;'>BM</span>: Benign Melanocytic", unsafe_allow_html=True)
    st.markdown(f"- 🟧 <span style='color:{CLASS_COLORS_HEX['ME']};font-weight:bold;'>ME</span>: Malignant Epidermal", unsafe_allow_html=True)
    st.markdown(f"- 🟥 <span style='color:{CLASS_COLORS_HEX['MM']};font-weight:bold;'>MM</span>: Malignant Melanoma", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Engine Specs")
    st.markdown("- **Architecture:** Self-Attention CNN")
    st.markdown("- **Runtime:** ONNX Runtime CPU")
    st.markdown("- **Spectral Bands:** 116 calibrated channels")
    st.markdown("- **Model Size:** 1.47 MB")


# ==============================================================================
# MAIN PAGE
# ==============================================================================
st.title("🔬 Hyperspectral Skin Cancer Classification Cloud")
st.markdown(
    "Upload a calibrated **Hyperspectral Imaging (HSI) cube** (`.npy` array with 116 spectral bands) "
    "to perform real-time pixel-wise tissue segmentation and clinical triage."
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose an HSI Cube file (.npy format)",
    type=["npy"],
    help="Upload an array of shape (50, 50, 116) or (116, 50, 50)",
)

if uploaded_file is not None:
    try:
        with st.spinner("Analyzing spectral signatures..."):
            cube = np.load(uploaded_file)
            pred_map, all_probs, infer_time_ms = process_hsi_cube(cube)

        total_pixels = pred_map.size
        counts = {CLASS_NAMES[i]: int(np.sum(pred_map == i)) for i in range(4)}
        percents = {CLASS_NAMES[i]: float(counts[CLASS_NAMES[i]] / total_pixels * 100) for i in range(4)}

        dominant = max(counts.keys(), key=lambda k: counts[k])
        dominant_label = CLASS_LABELS[CLASS_NAMES.index(dominant)]

        benign_pct = percents["BE"] + percents["BM"]
        malignant_pct = percents["ME"] + percents["MM"]

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if malignant_pct > benign_pct:
                st.metric("Clinical Risk Triage", "⚠️ MALIGNANT", delta=f"{malignant_pct:.1f}% risk", delta_color="inverse")
            else:
                st.metric("Clinical Risk Triage", "✅ BENIGN", delta=f"{benign_pct:.1f}% safe", delta_color="normal")

        with col2:
            st.metric("Dominant Diagnosis", dominant, f"{percents[dominant]:.1f}% of lesion")

        with col3:
            st.metric("Total Analyzed Pixels", f"{total_pixels:,}", f"Shape {cube.shape[0]}x{cube.shape[1]}")

        with col4:
            st.metric("Inference Latency", f"{infer_time_ms:.1f} ms", "ONNX CPU Engine")

        st.markdown("---")

        # Visualizations
        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            st.subheader("Spatial Lesion Segmentation Map")
            heatmap_img = generate_heatmap_image(pred_map)
            st.image(
                heatmap_img,
                caption="Color-coded 50x50 spatial distribution (Blue: BE | Green: BM | Orange: ME | Red: MM)",
                use_container_width=True,
            )

        with vcol2:
            st.subheader("Tissue Composition Breakdown")
            chart_fig = generate_bar_chart(percents)
            st.pyplot(chart_fig)

        st.markdown("---")

        # Detailed Data Breakdown
        st.subheader("📊 Detailed Quantitative Report")
        table_data = []
        for i in range(4):
            c_name = CLASS_NAMES[i]
            table_data.append({
                "Class Code": c_name,
                "Lesion Diagnosis": CLASS_LABELS[i],
                "Pixel Count": f"{counts[c_name]:,}",
                "Lesion Area (%)": f"{percents[c_name]:.2f}%",
                "Mean Softmax Probability": f"{np.mean(all_probs[:, i]):.4f}",
            })
        st.dataframe(table_data, use_container_width=True)

        # Download Report JSON
        report_dict = {
            "model_version": MODEL_VERSION,
            "filename": uploaded_file.name,
            "inference_time_ms": round(infer_time_ms, 2),
            "dominant_diagnosis": dominant_label,
            "dominant_percentage": percents[dominant],
            "benign_percentage": round(benign_pct, 2),
            "malignant_percentage": round(malignant_pct, 2),
            "class_counts": counts,
            "class_percentages": percents,
            "mean_probabilities": {
                CLASS_NAMES[i]: round(float(np.mean(all_probs[:, i])), 4)
                for i in range(4)
            },
        }

        st.download_button(
            label="📥 Download Structured Diagnostic Report (JSON)",
            data=json.dumps(report_dict, indent=2),
            file_name=f"diagnostic_report_{Path(uploaded_file.name).stem}.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"Error processing HSI file: {e}")
else:
    st.info("👆 Upload an HSI cube (`.npy`) above to begin diagnosis.")
