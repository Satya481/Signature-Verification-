# Signature Verification 🔏

**A modern, easy-to-use web app for verifying the authenticity of handwritten signatures using deep learning.**

<p align="center">
  <img src="https://img.shields.io/github/license/Satya481/Signature-Verification-?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/built%20with-PyTorch-blue?style=flat-square">
  <img src="https://img.shields.io/badge/web-Streamlit-%23ff4b4b?style=flat-square">
</p>

---

## ✨ Features

- **Instant Signature Verification:** Upload two signature images and determine if they belong to the same person.
- **Deep Learning Model:** Utilizes a Siamese ResNet18 neural network for robust feature extraction and comparison.
- **Intuitive UI:** Smooth, modern Streamlit interface with side-by-side signature display, results, confidence gauge, and adjustable threshold.
- **Configurable:** Easily swap out the model, tune thresholds, or run locally/offline.

---

## 🚀 Quick Start

**1. Clone the repository**
```sh
git clone https://github.com/Satya481/Signature-Verification-.git
cd Signature-Verification-
```

**2. Set up your Python environment**
```sh
python -m venv .venv
# Activate your virtual environment:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**3. Install dependencies**
```sh
pip install -r requirements.txt
```

**4. Get the model file**
- Place your trained `siamese_signature.pth` (PyTorch state_dict) in the same folder as `app.py`
- **Or** set its path via environment variable:
  ```sh
  export MODEL_PATH=/path/to/siamese_signature.pth
  ```

**5. Run the app**
```sh
streamlit run app.py
```

**6. Open the link shown in your terminal** (usually [http://localhost:8501](http://localhost:8501))

---

## 🖼️ How It Works

- The app uses a **Siamese Neural Network** based on ResNet18 to generate 128-dimensional embeddings of each uploaded signature image.
- The distance between these embeddings indicates similarity:
  - **Smaller distance** = more similar = more likely to be genuine.
- You control the "Decision Threshold" in the sidebar for fine-tuning.
- After uploading two signatures and clicking **"Verify"**, you'll see:
  - Badge: "Genuine" ✔️ or "Forgery" ❌
  - Distance score
  - Confidence estimate
  - A progress gauge and interpretation

---

## 🧠 Model Details

- **Architecture:** Siamese ResNet18, final layer replaced with `nn.Linear(...,128)`
- **Input:** 224x224 grayscale signature images (PNG/JPG)
- **Output:** Pairwise distance between embeddings
- **Decision:** If distance < threshold, classified as "Genuine", else "Forgery"

**Example Model Definition:**
```python
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 128)
        self.base = base

    def forward_once(self, x):
        return self.base(x.repeat(1,3,1,1))  # Ensure 3 channels

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return e1, e2
```

---

## 🏗️ Training Pipeline (Optional)

- See `signature_verification.ipynb` for full training and evaluation pipeline.
- Dataset structure:
  ```
  signatures/
    ├── writer_001/
    │     ├── original_1.png
    │     ├── forgeries_1.png
    │     └── ...
    ├── writer_002/
    └── ...
  ```
- Model is trained using contrastive loss, evaluated on ROC AUC and genuine/forged accuracy.

---

## 🛠️ File Structure

| File/Folder              | Purpose                                     |
|--------------------------|---------------------------------------------|
| `app.py`                 | Main Streamlit application                  |
| `requirements.txt`       | Python dependencies                         |
| `siamese_signature.pth`  | Pretrained model weights (not included)     |
| `signature_verification.ipynb` | Model training & evaluation notebook  |
| `signatures/`            | (Optional) Signature image dataset          |
| `.streamlit/`            | Streamlit configuration files               |
| `.ipynb_checkpoints/`    | Jupyter notebook checkpoints (ignore)       |

---

## 📝 Example Usage

1. **Upload two signature images in the app.**
2. Adjust the threshold slider as needed.
3. Click "Verify" to see:
   - **Result:** Genuine ✔️ or Forgery ❌
   - **Distance:** Similarity score
   - **Confidence:** Model certainty

---

## 💡 Tips

- Use high-quality, properly scanned images for best results.
- Fine-tune the threshold for your particular dataset or use-case.
- To train your own model, use the provided notebook and follow the code comments.

---

## 📜 License

MIT License © 2025 Satya481

---

## 🤝 Contributions

Pull requests, issues, and suggestions are welcome!

---

## 🙏 Acknowledgements

- Inspired by the deep metric learning community
- Built with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)