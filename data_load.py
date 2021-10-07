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
	#dataset = dataset.prefetch(64)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

	# Return shuffled dataset.
	return dataset, num_batch


# Load training data and put them in queues.
def get_batch_new():
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
		generator_new, args=(fpaths, text_lengths, texts),
		output_signature=(
			tf.TensorSpec(shape=(()), dtype=tf.string),
			tf.TensorSpec(shape=((None,)), dtype=tf.int32),
			tf.TensorSpec(shape=((None, hp.n_mels * hp.r)), dtype=tf.float32),
			tf.TensorSpec(shape=((None, hp.n_fft // 2 + 1)), dtype=tf.float32),
			#tf.TensorSpec(shape=(()), dtype=tf.int32),
			#tf.TensorSpec(shape=(()), dtype=tf.int32),
			#tf.TensorSpec(shape=(()), dtype=tf.int32)
		)
	)
	dataset = dataset.shuffle(256)
	dataset = dataset.padded_batch(hp.batch_size, drop_remainder=True)
	#dataset = dataset.cache()
	#dataset = dataset.prefetch(64)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

	# Return shuffled dataset.
	return dataset, num_batch


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


def generator_new(fpaths, text_lengths, texts):
	max_text_len = max(text_lengths)
	max_mel_len = 0
	max_mag_len = 0
	for i in range(len(texts)):
		fname, mel, mag = get_spectrograms(
			fpaths[i], text_lengths[i], texts[i]
		)
		if tf.shape(mel)[0] > max_mel_len:
			max_mel_len = tf.shape(mel)[0]
		if tf.shape(mag)[0] > max_mag_len:
			max_mag_len = tf.shape(mag)[0]

	for i in range(len(texts) // 8):
		text = tf.io.decode_raw(texts[i], tf.int32)
		fpath = fpaths[i]
		text_length = text_lengths[i]
		fname, mel, mag = get_spectrograms(fpath, text_length, text)

		# Right zero-pad text.
		text_padded = tf.zeros([max_text_len]).numpy()
		text_padded[:len(text)] = text
		text_padded = tf.convert_to_tensor(text_padded, dtype=tf.int32)

		# Right zero-pad mel and mag.
		num_mels = tf.shape(mel)[1]
		num_mags = tf.shape(mag)[1]
		mel_padded = tf.zeros([max_mel_len, num_mels]).numpy()
		mag_padded = tf.zeros([max_mag_len, num_mags]).numpy()
		mel_padded[:tf.shape(mel)[0], :] = mel
		mag_padded[:tf.shape(mag)[0], :] = mag
		mel_padded = tf.convert_to_tensor(mel_padded, dtype=tf.float32)
		mag_padded = tf.convert_to_tensor(mag_padded, dtype=tf.float32)
		'''
		yield fname, text, mel, mag, max_text_len, max_mel_len, \
			max_mag_len
		'''
		yield fname, text_padded, mel_padded, mag_padded


def get_max_lengths(dataset):
	x = list(dataset.as_numpy_iterator())
	'''
	text_lengths = []
	mel_lengths = []
	mag_lengths = []
	for i in range(len(x)):
		text_lengths.append(tf.shape(x[i][1]).numpy()[1])
		mel_lengths.append(tf.shape(x[i][2]).numpy()[1])
		mag_lengths.append(tf.shape(x[i][3]).numpy()[1])

	max_text_len = max(text_lengths)
	max_mel_len = max(mel_lengths)
	max_mag_len = max(mag_lengths)
	'''
	return max([tf.shape(x[i][1]).numpy()[1] for i in range(len(x))]), \
		max([tf.shape(x[i][2]).numpy()[1] for i in range(len(x))]), \
		max([tf.shape(x[i][3]).numpy()[1] for i in range(len(x))]),



class TextMelCollate():
	# Zero-pads model inputs and targets based on number of frames per
	# step.
	def __init__(self, n_frames_per_step, max_input_len, max_mel_len, 
			max_mag_len):
		self.n_frames_per_step = n_frames_per_step
		self.max_input_len = max_input_len
		self.max_mel_len = max_mel_len
		self.max_mag_len = max_mag_len


	# Collate's training batch from normalized text and mel-spectrogram
	# #param: batch, [text_normalized, mel_normalized]
	def __call__(self, batch):
		# Right zero-pad all one-hot text sequences to max input 
		# length.
		max_input_len = self.max_input_len
		batch_size = len(batch[0])

		text_padded = tf.zeros([batch_size, max_input_len]).numpy()
		for i in range(batch_size):
			text = batch[1][i]
			text_padded[i, :len(text)] = text
		text_padded = tf.convert_to_tensor(text_padded, dtype=tf.int32)

		# Right zero-pad mel-spec.
		num_mels = tf.shape(batch[2])[2]
		num_mags = tf.shape(batch[3])[2]
		#max_target_len = self.max_mel_len
		if self.max_mel_len % self.n_frames_per_step != 0:
			self.max_mel_len += self.n_frames_per_step - (self.max_mel_len % self.n_frames_per_step)
			assert self.max_mel_len % self.n_frames_per_step == 0

		# Include mel padded and gate padded.
		mel_padded = tf.zeros([batch_size, self.max_mel_len, num_mels]).numpy()
		mag_padded = tf.zeros([batch_size, self.max_mag_len, num_mags]).numpy()
		for i in range(batch_size):
			mel = batch[2][i]
			mel_padded[i, :tf.shape(mel)[0], :] = mel
			mag = batch[3][i]
			mag_padded[i, :tf.shape(mag)[0], :] = mag
		mel_padded = tf.convert_to_tensor(mel_padded, dtype=tf.float32)
		mag_padded = tf.convert_to_tensor(mag_padded, dtype=tf.float32)

		return text_padded, mel_padded, mag_padded, self.max_input_len, \
			self.max_mel_len, self.max_mag_len


if __name__ == '__main__':
	# Debug TextMelCollate from Tacotron 2 adapted to Tacotron 1.
	x, n_batches = get_batch()

	# Find the longest text, mel, and mag lengths in the dataset.
	max_text_len, max_mel_len, max_mag_len = get_max_lengths(x)
	
	print("Max text len: {}\nMax mel len: {}\nMax mag len: {}".format(
		max_text_len, max_mel_len, max_mag_len
	))

	# Pad the batches of data as they come in.
	text_mel_collate = TextMelCollate(
		1, max_text_len, max_mel_len, max_mag_len
	)

	x = list(x.as_numpy_iterator())
	for i in range(len(x)):
		text_mel_collate(x[i])