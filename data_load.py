# data_load.py
# author: Diego Magdaleno
# Copy of the data_load.py file in the Tensorflow implementation of
# dc_tts.
# Python 3.7
# Tensorflow 2.4.0


from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import gc


def load_vocab():
	char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
	idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
	return char2idx, idx2char


def text_normalize(text):
	text = "".join(char for char in unicodedata.normalize("NFD", text)
		if unicodedata.category(char) != "Mn"
	) # strip accents.
	text = text.lower()
	text = re.sub("[^{}]".format(hp.vocab), " ", text)
	text = re.sub("[ ]+", " ", text)
	return text


# Load data.
# @param: mode, "train" or "synthesize".
def load_data(mode="train"):
	# Load vocabulary.
	char2idx, idx2char = load_vocab()

	if mode in ("train", "eval"):
		# Parse
		fpaths, text_lengths, texts = [], [], []
		transcript = os.path.join(hp.data, "transcript.csv")
		lines = codecs.open(transcript, "r", "utf-8").readlines()
		total_hours = 0
		
		if mode == "train":
			lines = lines[1:]
		else:
			lines = lines[:1]

		for line in lines:
			fname, _, text = line.strip().split("|")

			fpath = os.path.join(hp.data, "wavs", fname + ".wav")
			fpaths.append(fpath)

			text = text_normalize(text) + "E" # E: EOS
			text = [char2idx[char] for char in text]
			text_lengths.append(len(text))
			texts.append(np.array(text, np.int32).tostring())

		return fpaths, text_lengths, texts
	else: # Synthesize on unseen test text.
		# Parse.
		lines = codecs.open(hp.test_data, "r", "utf-8").readlines()[1:]
		sents = [text_normalize(line.split(" ", 1)[-1]).strip() +"E"
			for line in lines
		] # Text normalization, E: EOS
		lengths = [len(sent) for sent in sents]
		maxlen = sorted(lengths, reverse=True)[0]
		texts = np.zeros((len(sents), hp.max_N), np.int32)
		for i, sent in enumerate(sents):
			texts[i, :len(sent)] = [char2idx[char] for char in sent]
		return texts


# Load training data and put them in queues.
def get_batch():
	# Load data.
	fpaths, text_lengths, texts = load_data() # list
	maxlen, minlen = max(text_lengths), min(text_lengths)

	# Calculate total batch count.
	num_batch = len(fpaths) // hp.batch_size

	fpaths = tf.convert_to_tensor(fpaths)
	text_lengths = tf.convert_to_tensor(text_lengths)
	texts = tf.convert_to_tensor(texts)

	# Initialize dataset from generator.
	dataset = tf.data.Dataset.from_generator(
		generator, args=(fpaths, text_lengths, texts),
		output_signature=(
			tf.TensorSpec(shape=(()), dtype=tf.string),
			tf.TensorSpec(shape=((None,)), dtype=tf.int32),
			tf.TensorSpec(shape=((None, hp.n_mels * hp.r)), dtype=tf.float32),
			tf.TensorSpec(shape=((None, hp.n_fft // 2 + 1)), dtype=tf.float32)
		)
	)
	dataset = dataset.shuffle(256)
	dataset = dataset.padded_batch(hp.batch_size, drop_remainder=True)
	#dataset = dataset.cache()
	dataset = dataset.prefetch(64)

	# Return shuffled dataset.
	return dataset


def get_spectrograms(fpath, text_length, text):
	# Extract fpath string from tensor.
	#fpath = fpath.numpy()

	# Use the file path to pull mel and mag data.
	if hp.prepro:
		def _load_spectrograms(fpath):
			fname = os.path.basename(fpath)
			# Convert fname to string using decode() with utf-8 encoding.
			#mel = "mels/{}".format(fname.replace("wav", "npy"))
			#mag = "mags/{}".format(fname.replace("wav", "npy"))
			mel = "mels/{}".format(fname.decode("utf-8").replace("wav", "npy"))
			mag = "mags/{}".format(fname.decode("utf-8").replace("wav", "npy"))
			return fname, np.load(mel), np.load(mag)

		#fname, mel, mag = tf.py_function(
		#	_load_spectrograms, [fpaths[i]], [tf.string, tf.float32, tf.float32]
		#)
		fname, mel, mag = _load_spectrograms(fpath)
	else:
		#fname, mel, mag = tf.py_function(
		#	load_spectrograms, [fpaths[i]], [tf.string, tf.float32, tf.float32]
		#) # (None, n_mels)
		fname, mel, mag = load_spectrograms(fpath)

	# Convert fname, mel, and mag to tensor.
	#fname = tf.convert_to_tensor(fname, dtype=tf.string)
	#mel = tf.convert_to_tensor(mel, dtype=tf.float32)
	#mag = tf.convert_to_tensor(mag, dtype=tf.float32)

	return fname, mel, mag


def generator(fpaths, text_lengths, texts):
	#print(len(texts))
	#for i in range(len(texts)):
	for i in range(len(texts) // 8):
		text = tf.io.decode_raw(texts[i], tf.int32)
		fpath = fpaths[i]
		text_length = text_lengths[i]
		fname, mel, mag = get_spectrograms(fpath, text_length, text)
		yield fname, text, mel, mag