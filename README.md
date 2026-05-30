# IMDB RNN Demo

Deployed app : https://rnn-imdb.streamlit.app/

Minimal demo showing training, conversion, and serving an IMDB sentiment RNN.

**Files**
- [rnn_pet.ipynb](rnn_pet.ipynb): Notebook that trains the RNN and saves model artifacts.
- [app.py](app.py): Streamlit front-end that prefers ONNX (`onnxruntime`) and can fallback to Keras formats locally.
- [requirements.txt](requirements.txt): Deployment dependencies (TensorFlow-free).

Quickstart
1. (Optional) Train and save the model from the notebook:
   - Open and run all cells in [rnn_pet.ipynb](rnn_pet.ipynb).

2. Convert to ONNX locally (recommended for deployment):
```bash
pip install tensorflow==2.20.0 tf2onnx
streamlit run app.py
```

In the app sidebar, open **Local conversion: Keras -> ONNX** and export:
- Input: `imdb_rnn_model.keras`
- Output: `imdb_rnn_model.onnx`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app (from this folder):
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

Notes
- The app prefers ONNX files: `imdb_rnn_model.onnx` or `imdb_rnn_models.onnx`.
- Keras files (`.keras`, `.h5`, SavedModel, pickle) are local fallback options when TensorFlow is installed.
- By default advanced debug output is hidden. Enable it in the Streamlit sidebar by toggling "Show advanced debug".
- Short single-word inputs (e.g. “awesome”) may be ambiguous; use longer phrases for best results.

Troubleshooting
- If the app reports no model found, ensure `imdb_rnn_model.onnx` is in the same folder as `app.py`.
- If you see preprocessing mismatches, enable advanced debug to inspect token mapping and run the sanity checks.

## Deployment (Streamlit Cloud)

Use ONNX for deployment because Streamlit Cloud Python runtime may not have a compatible TensorFlow wheel.

1. Convert locally (where TensorFlow + tf2onnx are installed):
```bash
python -m venv .venv_onnx
.venv_onnx\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow==2.20.0 tf2onnx
streamlit run app.py
```

Then use the sidebar panel **Local conversion: Keras -> ONNX**.

2. Commit and push:
```bash
git add .
git commit -m "Deploy-ready ONNX Streamlit app"
git push -u origin main
```

3. Streamlit Cloud settings:
- Repository: `SaiSrikar0/rnn`
- Branch: `main`
- Main file path: `app.py`

4. Optional in-app conversion:
- The sidebar in `app.py` includes "Local conversion: Keras -> ONNX".
- This works only on local environments with TensorFlow and tf2onnx installed.
