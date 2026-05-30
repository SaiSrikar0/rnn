"""
Streamlit front-end for the trained Keras model.

Usage:
  streamlit run app.py

This app attempts to load `pets_cnn.pkl` (pickled model config+weights),
then `rnn_imdb.h5`, then `saved_models/rnn_imdb`.
If the model contains an Embedding layer it exposes a text input
preprocessor for IMDB-style reviews and shows a positive/negative prediction.
"""

import os
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.datasets import imdb


def load_pickled_model(pkl_path):
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	model = model_from_json(data['model_config'])
	model.set_weights(data['model_weights'])
	try:
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	except Exception:
		pass
	return model


@st.cache_resource
def load_model_any():
	# Prefer .keras files (accept singular/plural filename variants)
	for keras_name in ('imdb_rnn_models.keras', 'imdb_rnn_model.keras'):
		if os.path.exists(keras_name):
			try:
				m = load_model(keras_name)
				return m, f'keras ({keras_name})'
			except Exception as e:
				st.warning(f'Failed loading {keras_name}: {e}')

	# Try pickled model next
	if os.path.exists('pets_cnn.pkl'):
		try:
			m = load_pickled_model('pets_cnn.pkl')
			return m, 'pickled'
		except Exception as e:
			st.warning(f'Failed loading pets_cnn.pkl: {e}')

	# Try HDF5
	if os.path.exists('rnn_imdb.h5'):
		try:
			m = load_model('rnn_imdb.h5')
			return m, 'h5'
		except Exception as e:
			st.warning(f'Failed loading rnn_imdb.h5: {e}')

	# Try SavedModel dir
	if os.path.isdir('saved_models/rnn_imdb'):
		try:
			m = load_model('saved_models/rnn_imdb')
			return m, 'savedmodel'
		except Exception as e:
			st.warning(f'Failed loading saved_models/rnn_imdb: {e}')

	return None, None


def encode_review(text, word_index, maxlen=500, num_words=10000):
	tokens = text_to_word_sequence(text)
	seq = []
	for w in tokens:
		idx = word_index.get(w)
		if idx is None:
			seq.append(2)  # OOV
			continue
		idx = idx + 3  # shift indices as in Keras' imdb sequences
		if idx >= num_words:
			seq.append(2)
		else:
			seq.append(idx)
	return pad_sequences([seq], maxlen=maxlen)


def main():
	st.title('Model Demo — IMDB RNN')

	model, src = load_model_any()
	if model is None:
		st.error('No model found. Place `pets_cnn.pkl`, `rnn_imdb.h5`, or `saved_models/rnn_imdb` in this folder.')
		return

	st.write('Loaded model from', src)
	debug_enabled = st.sidebar.checkbox('Show advanced debug', value=False)

	# detect if it's a text model by checking for an Embedding layer
	layer_names = [layer.__class__.__name__.lower() for layer in model.layers]
	is_text = any('embedding' in name for name in layer_names)

	if is_text:
		st.subheader('Enter a movie review (plain text)')
		review = st.text_area('Review', value='This movie was fantastic! I loved it.')
		maxlen = 500
		num_words = 10000

		if st.button('Predict'):
			word_index = imdb.get_word_index()
			seq = encode_review(review, word_index, maxlen=maxlen, num_words=num_words)
			pred = model.predict(seq)
			prob = float(pred[0][0])
			label = 'Positive' if prob >= 0.5 else 'Negative'
			st.write(f'Prediction: **{label}** ({prob*100:.2f}%)')

			# show diagnostic info only when advanced debug is enabled
			if debug_enabled:
				tokens = text_to_word_sequence(review)
				mapped = []
				for w in tokens:
					idx = word_index.get(w)
					if idx is None:
						mapped.append((w, None, 2))
					else:
						mapped.append((w, idx, idx+3))
				st.write('Token mapping (word, imdb_index, used_index):')
				st.write(mapped)
				st.write('Non-zero encoded sequence indices (first 50):', seq[0][seq[0]>0][:50])

		if debug_enabled:
			with st.expander('Debug / Sanity checks'):
				st.write('Run quick checks on real IMDB test samples to verify preprocessing and model behaviour.')
				if st.button('Run sanity checks'):
					(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
					# helper to decode sequence to text
					word_index = imdb.get_word_index()
					reverse_index = {v+3: k for k, v in word_index.items()}
					reverse_index[0] = '<PAD>'
					reverse_index[1] = '<START>'
					reverse_index[2] = '<OOV>'

					# show three positive and three negative examples
					pos_idxs = [i for i, y in enumerate(y_test) if y == 1][:3]
					neg_idxs = [i for i, y in enumerate(y_test) if y == 0][:3]
					sample_idxs = pos_idxs + neg_idxs
					for idx in sample_idxs:
						seq = x_test[idx]
						# decode first 100 tokens
						text = ' '.join([reverse_index.get(i, '?') for i in seq[:100]])
						st.write('---')
						st.write('True label:', 'Positive' if y_test[idx] == 1 else 'Negative')
						st.write('Decoded (first 100 tokens):', text)
						# re-encode using our tokenizer and predict
						text_input = encode_review(text, word_index, maxlen=maxlen, num_words=num_words)
						p = model.predict(text_input)
						st.write(f'Model prediction: {float(p[0][0])*100:.2f}%')
	else:
		st.info('The loaded model does not look like a text (Embedding) model. Only text UI is implemented.')


if __name__ == '__main__':
	main()

