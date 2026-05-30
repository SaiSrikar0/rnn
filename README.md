# IMDB RNN Demo

Minimal demo showing training, saving, and serving a Keras RNN model for IMDB sentiment.

**Files**
- [rnn_pet.ipynb](rnn_pet.ipynb): Notebook that trains the RNN, saves models (`saved_models/rnn_imdb`, `rnn_imdb.h5`) and a pickle `pets_cnn.pkl`.
- [app.py](app.py): Streamlit front-end that loads the saved model and provides a text-based prediction UI.
- [requirements.txt](requirements.txt): Python dependencies.
- `imdb_rnn_model.keras` or `imdb_rnn_models.keras`: example Keras model file name the app will try to load.

Quickstart
1. (Optional) Train and save the model from the notebook:
   - Open and run all cells in [rnn_pet.ipynb](rnn_pet.ipynb). The notebook saves the model to `saved_models/rnn_imdb` and `rnn_imdb.h5` and writes `pets_cnn.pkl`.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app (from this folder):
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

Notes
- The app prefers a `.keras` file named `imdb_rnn_models.keras` or `imdb_rnn_model.keras`. If present, it will load that first. Otherwise it falls back to `pets_cnn.pkl`, `rnn_imdb.h5`, or `saved_models/rnn_imdb`.
- By default advanced debug output is hidden. Enable it in the Streamlit sidebar by toggling "Show advanced debug".
- Short single-word inputs (e.g. “awesome”) may be ambiguous; use longer phrases for best results.

Troubleshooting
- If the app reports no model found, ensure one of the model files listed above exists in the same folder as `app.py`.
- If you see preprocessing mismatches, enable advanced debug to inspect token mapping and run the sanity checks.

If you want, I can add a Dockerfile or export the model to ONNX next.
