# create_dataset.py
# author: Diego Magdaleno
# Create a simple program that allows me to tinker with loading all
# necessary data and creating a dataset.


from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
from data_load import *


def create_dataset():
	#available_devices = tf.config.list_physical_devices()
	#with tf.device(available_devices[-1].name):
	# Load data.
	fpaths, text_lengths, texts = load_data() # list
	maxlen, minlen = max(text_lengths), min(text_lengths)

	# Calculate total batch count.
	num_batch = len(fpaths) // hp.B

	# Create three lists to store mels, mags, fnames.
	mels = []
	mags = []
	fnames = []

	# Iterate through all lists.
	for i in range(len(texts)):
		# Parse all text inputs into a (None, ) tensor of ints.
		text = tf.io.decode_raw(texts[i], tf.int32)

		# Use the file path to pull mel and mag data.
		if hp.prepro:
			def _load_spectrograms(fpath):
				fname = os.path.basename(fpath)
				mel = "mels/{}".format(fname.replace("wav", "npy"))
				mag = "mags/{}".format(fname.replace("wav", "npy"))
				return fname, np.load(mel), np.load(mag)

			#fname, mel, mag = tf.py_function(
			#	_load_spectrograms, [fpaths[i]], [tf.string, tf.float32, tf.float32]
			#)
			fname, mel, mag = _load_spectrograms(fpaths[i])
		else:
			#fname, mel, mag = tf.py_function(
			#	load_spectrograms, [fpaths[i]], [tf.string, tf.float32, tf.float32]
			#) # (None, n_mels)
			fname, mel, mag = load_spectrograms(fpaths[i])

		# Convert fname, mel, and mag to tensor.
		fname = tf.convert_to_tensor(fname, dtype=tf.string)
		mel = tf.convert_to_tensor(mel, dtype=tf.float32)
		mag = tf.convert_to_tensor(mag, dtype=tf.float32)

		# Add shape information.
		fname.set_shape(())
		text.set_shape((None,))
		mel.set_shape((None, hp.n_mels))
		mag.set_shape((None, hp.n_fft // 2 + 1))

		fnames.append(fname)
		texts[i] = text
		mels.append(mel)
		mags.append(mag)

	# Convert to dataset.
	data = {"fnames": fnames, "texts": texts, "mels": mels, "mags": mags}
	dataset = tf.data.Dataset.from_generator(data)

	#return texts, mels, mags, fnames, num_batch
	return dataset, num_batch


create_dataset()