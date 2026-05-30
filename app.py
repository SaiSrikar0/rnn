"""Streamlit front-end for IMDB sentiment inference.

Deployment path (Streamlit Cloud): ONNX + onnxruntime.
Local fallback: Keras/TensorFlow loaders are used only if TensorFlow is installed.
"""

import json
import importlib
import os
import pickle
import re
import urllib.request

import numpy as np
import streamlit as st

try:
	ort = importlib.import_module("onnxruntime")
	_HAS_ONNX = True
except Exception:
	ort = None
	_HAS_ONNX = False

_WORD_INDEX_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"


def _get_keras_helpers():
	"""Import Keras lazily so app still runs in ONNX-only environments."""
	try:
		keras_models = importlib.import_module("tensorflow.keras.models")
		keras_load_model = getattr(keras_models, "load_model")
		keras_model_from_json = getattr(keras_models, "model_from_json")
		return keras_load_model, keras_model_from_json
	except Exception:
		return None, None


def simple_text_to_word_sequence(text):
	text = text.lower()
	return re.findall(r"[a-z0-9']+", text)


def simple_pad_sequences(sequences, maxlen, value=0):
	arr = np.full((len(sequences), maxlen), value, dtype=np.int64)
	for i, seq in enumerate(sequences):
		trunc = seq[-maxlen:]
		arr[i, -len(trunc):] = np.asarray(trunc, dtype=np.int64)
	return arr


@st.cache_data
def get_word_index():
	"""Get IMDB word index without requiring TensorFlow runtime imports."""
	local_path = "imdb_word_index.json"
	if os.path.exists(local_path):
		with open(local_path, "r", encoding="utf-8") as f:
			return json.load(f)

	with urllib.request.urlopen(_WORD_INDEX_URL) as resp:
		data = json.loads(resp.read().decode("utf-8"))
	with open(local_path, "w", encoding="utf-8") as f:
		json.dump(data, f)
	return data


def load_pickled_model(pkl_path):
	load_model, model_from_json = _get_keras_helpers()
	if model_from_json is None:
		raise RuntimeError("TensorFlow is required to load pickled Keras model format.")

	with open(pkl_path, "rb") as f:
		data = pickle.load(f)
	model = model_from_json(data["model_config"])
	model.set_weights(data["model_weights"])
	try:
		model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	except Exception:
		pass
	return model


def export_keras_to_onnx(input_path, output_path, opset=13):
	"""Convert a local Keras model file to ONNX.

	This is intended for local development environments where TensorFlow and
	tf2onnx are installed.
	"""
	tf_module = importlib.import_module("tensorflow")
	tf2onnx = importlib.import_module("tf2onnx")

	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input model not found: {input_path}")

	keras_model = tf_module.keras.models.load_model(input_path)
	tf2onnx.convert.from_keras(keras_model, opset=opset, output_path=output_path)
	return output_path


@st.cache_resource
def load_model_any():
	# Prefer ONNX for deployment (no TensorFlow required at runtime).
	for onnx_name in ("imdb_rnn_model.onnx", "imdb_rnn_models.onnx"):
		if _HAS_ONNX and os.path.exists(onnx_name):
			sess = ort.InferenceSession(onnx_name, providers=["CPUExecutionProvider"])
			return sess, f"onnx ({onnx_name})"

	load_model, _ = _get_keras_helpers()
	if load_model is None:
		return None, None

	# Keras fallbacks for local usage.
	for keras_name in ("imdb_rnn_models.keras", "imdb_rnn_model.keras"):
		if os.path.exists(keras_name):
			m = load_model(keras_name)
			return m, f"keras ({keras_name})"

	if os.path.exists("pets_cnn.pkl"):
		m = load_pickled_model("pets_cnn.pkl")
		return m, "pickled"

	if os.path.exists("rnn_imdb.h5"):
		m = load_model("rnn_imdb.h5")
		return m, "h5"

	if os.path.isdir("saved_models/rnn_imdb"):
		m = load_model("saved_models/rnn_imdb")
		return m, "savedmodel"

	return None, None


def encode_review(text, word_index, maxlen=500, num_words=10000):
	tokens = simple_text_to_word_sequence(text)
	seq = []
	for w in tokens:
		idx = word_index.get(w)
		if idx is None:
			seq.append(2)  # OOV
			continue
		idx = idx + 3  # shift indices as in Keras IMDB sequences
		if idx >= num_words:
			seq.append(2)
		else:
			seq.append(idx)
	return simple_pad_sequences([seq], maxlen=maxlen)


def predict_probability(model, is_onnx, seq):
	if is_onnx:
		input_info = model.get_inputs()[0]
		input_name = input_info.name
		input_type = (input_info.type or "").lower()

		if "float" in input_type:
			out = model.run(None, {input_name: seq.astype("float32")})
		elif "int64" in input_type:
			out = model.run(None, {input_name: seq.astype("int64")})
		elif "int32" in input_type:
			out = model.run(None, {input_name: seq.astype("int32")})
		else:
			# Fallback for uncommon ONNX input dtypes.
			try:
				out = model.run(None, {input_name: seq.astype("int64")})
			except Exception:
				out = model.run(None, {input_name: seq.astype("float32")})
		return float(np.asarray(out[0]).reshape(-1)[0])

	pred = model.predict(seq)
	return float(np.asarray(pred).reshape(-1)[0])


def main():
	st.title("Model Demo - IMDB RNN")

	with st.sidebar.expander("Local conversion: Keras -> ONNX"):
		st.caption("Optional utility. Requires local TensorFlow + tf2onnx.")
		export_in = st.text_input("Keras input file", value="imdb_rnn_model.keras")
		export_out = st.text_input("ONNX output file", value="imdb_rnn_model.onnx")
		opset = st.number_input("ONNX opset", min_value=11, max_value=19, value=13, step=1)
		if st.button("Export ONNX"):
			try:
				with st.spinner("Converting model to ONNX..."):
					saved = export_keras_to_onnx(export_in, export_out, int(opset))
				st.success(f"ONNX exported successfully: {saved}")
			except Exception as e:
				st.error(
					"Export failed. Ensure TensorFlow and tf2onnx are installed locally. "
					f"Details: {e}"
				)

	model, src = load_model_any()
	if model is None:
		st.error(
			"No compatible model found. Add `imdb_rnn_model.onnx` (recommended), or install TensorFlow "
			"and provide a Keras model file."
		)
		return

	st.write("Loaded model from", src)
	debug_enabled = st.sidebar.checkbox("Show advanced debug", value=False)

	is_onnx = _HAS_ONNX and isinstance(model, ort.InferenceSession)
	is_text = True if is_onnx else any("embedding" in layer.__class__.__name__.lower() for layer in model.layers)

	if not is_text:
		st.info("The loaded model does not look like a text model. Only text UI is implemented.")
		return

	st.subheader("Enter a movie review (plain text)")
	review = st.text_area("Review", value="This movie was fantastic! I loved it.")
	maxlen = 500
	num_words = 10000

	if st.button("Predict"):
		word_index = get_word_index()
		seq = encode_review(review, word_index, maxlen=maxlen, num_words=num_words)
		prob = predict_probability(model, is_onnx, seq)
		label = "Positive" if prob >= 0.5 else "Negative"
		st.write(f"Prediction: **{label}** ({prob*100:.2f}%)")

		if debug_enabled:
			tokens = simple_text_to_word_sequence(review)
			mapped = []
			for w in tokens:
				idx = word_index.get(w)
				if idx is None:
					mapped.append((w, None, 2))
				else:
					mapped.append((w, idx, idx + 3))
			st.write("Token mapping (word, imdb_index, used_index):")
			st.write(mapped)
			st.write("Non-zero encoded sequence indices (first 50):", seq[0][seq[0] > 0][:50])

	if debug_enabled:
		with st.expander("Debug / Sanity checks"):
			st.write("Run checks using local model on manual examples.")
			if st.button("Run sanity checks"):
				examples = [
					"This movie was excellent and emotional",
					"Terrible script and bad acting",
					"A bit slow but overall very good",
				]
				word_index = get_word_index()
				for text in examples:
					seq = encode_review(text, word_index, maxlen=maxlen, num_words=num_words)
					prob = predict_probability(model, is_onnx, seq)
					st.write(f"Text: {text}")
					st.write(f"Model prediction: {prob*100:.2f}%")
					st.write("---")


if __name__ == "__main__":
	main()

