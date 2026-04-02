import streamlit as st
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import base64
import tempfile
import os

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Brain Health Analyzer", layout="wide")

# -------- BACKGROUND + STYLE --------
def set_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            color: white;
        }}

        html, body, [class*="css"] {{
            color: white !important;
        }}

        h1, h2, h3 {{
            color: white !important;
        }}

        label {{
            color: white !important;
        }}

        section[data-testid="stFileUploader"] {{
            background-color: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
        }}

        section[data-testid="stFileUploader"] button {{
            color: black !important;
            font-weight: bold;
        }}

        div.stDownloadButton > button {{
            color: black !important;
            font-weight: bold;
        }}

        </style>
        """, unsafe_allow_html=True)

set_bg("brain.jpg")

st.markdown("<h1>🧠 Brain Health Analyzer</h1>", unsafe_allow_html=True)

# -------- MODEL --------
class BrainAgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.pool = nn.AdaptiveAvgPool3d((11, 13, 4))

        self.fc = nn.Sequential(
            nn.Linear(128 * 11 * 13 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = BrainAgeModel()
model.load_state_dict(torch.load("brain_age_model.pth", map_location="cpu"))
model.eval()

# -------- PREPROCESS --------
def load_mri(file):
    return nib.load(file).get_fdata()

def normalize(img):
    return (img - np.mean(img)) / np.std(img)

def resize(img):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    img = F.interpolate(img, size=(91, 109, 91), mode='trilinear')
    return img.squeeze().numpy()

def preprocess(file):
    img = load_mri(file)
    img = normalize(img)
    img = resize(img)
    return img

# -------- GRAD CAM --------
def generate_gradcam(model, img_tensor):
    img_tensor.requires_grad = True
    output = model(img_tensor)
    output.backward()

    gradients = img_tensor.grad.data.numpy()[0, 0]
    heatmap = np.abs(gradients)
    heatmap = heatmap / np.max(heatmap)
    return heatmap

# -------- UI --------
uploaded_file = st.file_uploader("Upload MRI (.nii / .nii.gz)", type=["nii", "gz"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        raw_img = load_mri(temp_path)
        img = preprocess(temp_path)
    except:
        st.error("Invalid MRI file")
        st.stop()

    st.subheader("🧠 MRI Viewer")
    slice_idx = st.slider("Slice", 0, raw_img.shape[2]-1, raw_img.shape[2]//2)
    st.image(raw_img[:, :, slice_idx], clamp=True)

    st.subheader("🧠 Multi-Slice View")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    slices = np.linspace(0, raw_img.shape[2]-1, 10, dtype=int)

    for i, ax in enumerate(axes.flat):
        ax.imshow(raw_img[:, :, slices[i]], cmap='gray')
        ax.axis('off')

    st.pyplot(fig)

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred_age = model(img).item()

    actual_age = st.number_input("Enter Actual Age", 1, 100, 40)
    bag = pred_age - actual_age

    st.metric("Predicted Brain Age", round(pred_age, 2))
    st.metric("Brain Age Gap", round(bag, 2))

    st.subheader("🧾 Brain Health Diagnosis")

    if bag > 10:
        diagnosis = "Possible Neurodegenerative Disorder (Alzheimer’s)"
        st.error(f"🔴 {diagnosis}")
        cause = "Neuron damage, aging, poor lifestyle"
        precautions = "- Exercise\n- Healthy diet\n- Mental activities\n- Consult doctor"
        diseases = "Alzheimer’s Disease, Dementia"

    elif bag > 5:
        diagnosis = "Mild Cognitive Decline"
        st.warning(f"🟠 {diagnosis}")
        cause = "Stress, sleep issues"
        precautions = "- Sleep well\n- Reduce stress\n- Brain exercises"
        diseases = "Mild Cognitive Impairment (MCI), Early Dementia"

    elif bag < -5:
        diagnosis = "Healthy Brain"
        st.success(f"🟢 {diagnosis}")
        cause = "Strong cognitive health"
        precautions = "- Maintain lifestyle"
        diseases = "No disease (Healthy brain)"

    else:
        diagnosis = "Normal Brain Aging"
        st.success(f"🟢 {diagnosis}")
        cause = "Natural aging"
        precautions = "- Balanced lifestyle"
        diseases = "No major neurological disease"

    st.write("Cause:", cause)
    st.write("Precautions:")
    st.write(precautions)

    st.subheader("📌 Final Report")

    st.write(f"""
    - Predicted Brain Age: {round(pred_age,2)}
    - Actual Age: {actual_age}
    - Brain Age Gap: {round(bag,2)}

    👉 Positive BAG → Faster aging  
    👉 Negative BAG → Younger brain  

    🧠 Associated Brain Diseases:
    {diseases}
    """)

    report = f"""
Brain Health Report

Predicted Brain Age: {round(pred_age,2)}
Actual Age: {actual_age}
Brain Age Gap: {round(bag,2)}

Diagnosis: {diagnosis}

Cause:
{cause}

Precautions:
{precautions}

Associated Brain Diseases:
{diseases}
"""

    # -------- NEW: RISK FACTORS --------
    st.subheader("⚠️ Risk Factors")

    if bag > 5:
        st.warning("""
        - High stress levels  
        - Genetic predisposition  
        - Poor sleep cycle  
        - Unhealthy diet  
        - Lack of mental activity  
        - Early neurodegenerative changes  
        """)

    elif bag < -5:
        st.success("""
        - Strong genetics  
        - Active brain usage  
        - Good lifestyle  
        - Balanced nutrition  
        - Regular mental exercises  
        """)

    else:
        st.info("""
        - Normal aging factors  
        - Daily routine impact  
        - Mild lifestyle variations  
        """)

    # -------- EXISTING EXTRA FEATURES (UNCHANGED) --------
    st.subheader("🧠 Brain Age Gap Interpretation")

    if bag > 5:
        st.warning("""
        ⚠️ Brain age is higher than actual age.

        Possible Reasons:
        - Chronic Stress
        - Poor Sleep Cycle
        - Genetic Factors
        - Early Neurodegenerative Changes
        - Unhealthy Lifestyle

        👉 Indicates faster brain aging and potential risk.
        """)

    elif bag < -5:
        st.success("""
        ✅ Brain age is lower than actual age.

        Possible Reasons:
        - Healthy Lifestyle
        - Good Cognitive Activity
        - Strong Genetics
        - Proper Sleep & Nutrition

        👉 Indicates strong and healthy brain condition.
        """)

    else:
        st.info("""
        ℹ️ Brain age is close to actual age.

        👉 Indicates normal brain aging.
        """)

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="brain_health_report.txt",
        mime="text/plain"
    )


    
    
