# Signature Verification

Simple Streamlit app that verifies pairs of handwritten signatures using a Siamese ResNet18 model that outputs 128‑D embeddings.

## Quick start

1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Place your trained model `siamese_signature.pth` next to `app.py`, or set the `MODEL_PATH` environment variable to point to it.

3. Run the app:

```powershell
streamlit run app.py
```

4. Open the Streamlit URL shown in the terminal (usually http://localhost:8501).

## Notes
- The app expects a PyTorch state_dict compatible with a ResNet18 whose `fc` has been replaced with `nn.Linear(...,128)`.
- For development you can set `MODEL_PATH` to a different location:

```powershell
$env:MODEL_PATH = 'C:\path\to\siamese_signature.pth'
streamlit run app.py
```

## License
MIT
