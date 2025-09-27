import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# ------------------- Model Definition -------------------
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 128)
        self.base = base

    def forward_once(self, x):
        return self.base(x.repeat(1,3,1,1))

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return e1, e2

# ------------------- Load Model -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\gupta\OneDrive\Desktop\misba_model\siamese_signature.pth"

model = SiameseNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------- Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Signature Verification", layout="wide", initial_sidebar_state="expanded")

# --- Light theme & nicer styles ---
st.markdown("""
<style>
    /* Page background and general text */
    html, body, .stApp {
        background: #ffffff;
        color: #0f1720;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    /* Card style for sections */
    .card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(15,23,32,0.06);
        border: 1px solid rgba(15,23,32,0.06);
    }
    h1.app-title { color: #0b61a4; margin: 0; }
    p.app-sub { color: #334155; margin-top:6px; margin-bottom:0; }
    .stProgress > div > div { background: linear-gradient(90deg, #0b61a4, #3ea0ff) !important; }
    .verify-btn { background-color: #0b61a4; color: white !important; }
    .stButton>button { border-radius:8px; }
    .stImage figcaption { color: #0f1720; }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("<div style='display:flex; align-items:center; justify-content:space-between; width:100%'>"
                        "<div style=\"display:flex; flex-direction:column;\">"
                        "<h1 class='app-title'>Signature Verification</h1>"
                        "<p class='app-sub'>Secure, model-driven verification of handwritten signatures</p>"
                        "</div>"
                        "<div style=\"text-align:right; color:#64748b; font-size:13px;\">Model: Siamese ResNet18<br>Version: 1.0</div>"
                        "</div>", unsafe_allow_html=True)


# Sidebar controls and help
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", min_value=0.01, max_value=5.0, value=0.5, step=0.01, help="Lower distance means more similar signatures.")
    st.write("---")
    st.markdown("**Instructions**")
    st.markdown("1. Upload two signature images (PNG/JPG).\n2. Click the 'Verify' button.\n3. Review the result, distance and confidence.")
    st.write("---")
    st.markdown("**Model info**")
    st.write("Siamese ResNet18 — embeddings (128-d)")
    st.caption("Adjust threshold to tune false accept / reject tradeoff.")
    

# Upload area
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("Upload Signature 1", type=["png","jpg","jpeg"], key="file1")
with col2:
    uploaded_file2 = st.file_uploader("Upload Signature 2", type=["png","jpg","jpeg"], key="file2")
st.markdown("</div>", unsafe_allow_html=True)

# Verify button for explicit control
verify_btn = st.button("🔎 Verify")

result_placeholder = st.container()

def compute_and_show(img1, img2, threshold_val):
    img1_t = transform(img1).unsqueeze(0).to(DEVICE)
    img2_t = transform(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        e1, e2 = model(img1_t, img2_t)
        dist = F.pairwise_distance(e1, e2).item()

    # Confidence mapping: smaller distance -> higher confidence (0-100)
    conf = max(0.0, min(100.0, (1.0 - (dist / max(threshold_val * 2.0, 1e-6))) * 100.0))
    is_genuine = dist < threshold_val

    # Show uploaded images and results
    with result_placeholder:
        st.write("---")
        st.markdown("### 🔹 Uploaded Signatures")
        c1, c2 = st.columns(2)
        c1.image(img1, caption="Signature 1", width=320)
        c2.image(img2, caption="Signature 2", width=320)

        st.write("---")
        st.markdown("### 🔹 Verification Result")
        col_r1, col_r2 = st.columns([1,2])
        # Left: badge
        with col_r1:
            if is_genuine:
                st.markdown("<div style='padding:10px; border-radius:8px; background:#ecfdf5; color:#065f46; font-weight:600; text-align:center;'>Genuine ✅</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding:10px; border-radius:8px; background:#fff1f2; color:#9f1239; font-weight:600; text-align:center;'>Forgery ❌</div>", unsafe_allow_html=True)
            st.metric("Distance", f"{dist:.4f}")
            st.metric("Confidence", f"{conf:.1f}%")
        # Right: details
        with col_r2:
            st.markdown(f"<div style='padding:12px; border-radius:8px; background:#f8fafc; box-shadow:0 2px 8px rgba(15,23,32,0.03)'>"
                        f"<p style='margin:0; font-size:16px; color:#334155;'>Threshold: <b>{threshold_val:.2f}</b></p>"
                        f"<p style='margin:6px 0 0 0; color:#475569;'>Interpretation: <small>{'Signatures are similar' if is_genuine else 'Signatures are dissimilar'}</small></p>"
                        f"</div>", unsafe_allow_html=True)

        # A progress-like visualization where lower distance => higher gauge
        gauge = max(0.0, min(1.0, 1.0 - (dist / max(threshold_val*2, 1e-6))))
        st.write("\n")
        st.progress(gauge)

# Only run verification after user clicks Verify
if uploaded_file1 and uploaded_file2 and verify_btn:
    img1 = Image.open(uploaded_file1).convert("L")
    img2 = Image.open(uploaded_file2).convert("L")
    compute_and_show(img1, img2, threshold)
