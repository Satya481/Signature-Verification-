import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ------------------- Model Definition -------------------
import torch.nn as nn
from torchvision import models

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 128)
        self.base = base

    def forward_once(self, x):
        return self.base(x.repeat(1,3,1,1))  # convert gray->3 channels

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

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------------- Streamlit UI -------------------
st.title("Signature Verification System")

uploaded_file1 = st.file_uploader("Upload Signature 1", type=["png","jpg","jpeg"])
uploaded_file2 = st.file_uploader("Upload Signature 2", type=["png","jpg","jpeg"])

if uploaded_file1 and uploaded_file2:
    img1 = Image.open(uploaded_file1).convert("L")
    img2 = Image.open(uploaded_file2).convert("L")

    img1_t = transform(img1).unsqueeze(0).to(DEVICE)
    img2_t = transform(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        e1, e2 = model(img1_t, img2_t)
        dist = F.pairwise_distance(e1, e2).item()

    # You can adjust threshold based on your previous test set
    threshold = 0.5
    result = "Genuine" if dist < threshold else "Forgery"

    st.image([img1, img2], caption=["Signature 1","Signature 2"], width=250)
    st.write(f"Distance: {dist:.4f}")
    st.write(f"Result: **{result}**")
