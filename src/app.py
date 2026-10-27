import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io

from inference_api import LungNodulePredictor

# ==========================================
# ⚙️ System configuration and page initialization
# ==========================================
st.set_page_config(
    page_title="AstraZeneca - Pulmonary Nodule Decision Support and Risk Stratification",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_ai_engine():
    try:
        predictor = LungNodulePredictor(checkpoint_dir="models/checkpoints_attention/")
        return predictor
    except Exception as e:
        return None

predictor = load_ai_engine()

# Initialize session state for diagnosis history
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# ==========================================
# 🎨 Sidebar design (control panel and guidance)
# ==========================================
st.sidebar.title("🫁 Clinical Decision Support System")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Algorithm**: 3D CBAM Attention ResNet\n\n"
    "**Optimal Operating Point (Youden Index)**: 0.45\n\n"
    "**System Sensitivity**: 78% | **System Specificity**: 82%"
)
st.sidebar.markdown(
    "### 💊 Targeted Therapy Guidance\n"
    "- **High risk (>70%)**: Consider genetic testing and targeted therapy such as osimertinib.\n"
    "- **Intermediate risk (45%-70%)**: Consider needle biopsy or follow-up HRCT within 3 months.\n"
    "- **Low risk (<45%)**: Continue routine annual low-dose CT (LDCT) follow-up."
)

# ==========================================
# 🖥️ Main interface (batch upload and results)
# ==========================================
st.title("Pulmonary Nodule Decision Support and Risk Stratification")
st.markdown("Upload nodule ROI `.npy` files extracted by the standardized pipeline. Multiple files are supported. The system will use a 3D convolutional model to analyze spatial features, stratify risk, and generate a report.")

if predictor is None:
    st.error("❌ Failed to load the model. Check that `models/checkpoints_attention/` contains a valid checkpoint.")
    st.stop()

# 🚀 Improvement 1: support batch uploads with accept_multiple_files=True
uploaded_files = st.file_uploader("📂 Upload nodule ROI data (.npy)", type=["npy"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"✅ Received {len(uploaded_files)} nodule file(s). Starting inference...")

    # Start a fresh batch while retaining the session-wide history.
    current_batch_results = []

    # Create a two-column layout.
    col1, col2 = st.columns([1, 1.5])

    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            # Run inference.
            result = predictor.predict_single(temp_path)
            prob_percent = result['malignancy_probability'] * 100

            # 🚀 Use concise clinical recommendations.
            if prob_percent < 45.0:
                risk_level = "Low risk"
                action_text = "Routine annual LDCT follow-up"
            elif 45.0 <= prob_percent <= 70.0:
                risk_level = "Intermediate risk"
                action_text = "Consider needle biopsy or follow-up within 3 months"
            else:
                risk_level = "High risk"
                action_text = "Consider multidisciplinary team review and targeted therapy assessment"

            # Record the result.
            record = {
                "Nodule ID": file.name,
                "Predicted Malignancy Probability": f"{prob_percent:.2f}%",
                "Clinical Risk Level": risk_level,
                "Clinical Recommendation": action_text
            }
            current_batch_results.append(record)
            st.session_state.diagnosis_history.append(record)

            # Preview the final uploaded sample in the left column.
            with col1:
                if file == uploaded_files[-1]:
                    st.subheader(f"👁️ Sample slice preview: {file.name}")
                    cube = np.load(temp_path)
                    if cube.shape == (32, 32, 32):
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(cube[16, :, :], cmap='bone')
                        ax.axis('off')
                        st.pyplot(fig)

            # Remove the temporary file.
            os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    # 🚀 Improvement 2: display a live summary table.
    with col2:
        st.subheader("📊 Batch Diagnostic Summary")
        df_results = pd.DataFrame(current_batch_results)
        # Display results in an interactive dataframe.
        st.dataframe(df_results, use_container_width=True)

        # 🚀 Improvement 3: export the diagnostic report as a CSV.
        csv_buffer = io.BytesIO()
        df_results.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="💾 Export Diagnostic Report (CSV)",
            data=csv_data,
            file_name="pulmonary_nodule_diagnostic_report.csv",
            mime="text/csv",
            type="primary"
        )

st.markdown("---")
st.caption("⚖️ Disclaimer: This interactive prototype is for academic demonstration and research validation only. Its estimated probabilities are based on the historical LIDC-IDRI dataset distribution and must not be used as a definitive diagnosis in clinical practice.")
