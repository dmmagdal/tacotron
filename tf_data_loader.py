# tf_data_loader.py
# author: Diego Magdaleno
# Adapting data_utils.py from Nvidia Tacotron2 repo to be compatible
# with Tensorflow 2. Reworks a lot of the original module to use the
# tf.data.Dataset module.
# Python 3.7
# Tensorflow 2.4.0



import os
import math
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperparams import Hyperparams as hp
from text import text_to_sequence
from utils import load_wav_to_tensor, load_filepaths_and_text
#from layers import TacotronSTFT
from tqdm import tqdm


def load_data(mode="train"):
	'''
	# Initialize Tacotron STFT function module.
	stft = TacotronSTFT(
		hp.filter_length, hp.hop_length, hp.win_length, hp.n_mel_channels,
		hp.sampling_rate, hp.mel_fmin, hp.mel_fmax
	)
	'''

	# Load files depending on whether the function is in training or
	# validation modes.
	if mode == "train":
		audiopaths_and_text = hp.training_files
	else:
		audiopaths_and_text = hp.validation_files
	audiopaths_texts = load_filepaths_and_text(audiopaths_and_text)

	# Iteratively get the (text, mel) pairs for each file.
	print("Processing audio and text...")
	'''
	files = []
	texts = []
	mels = []
	text_lengths = []
	#for audio_text in audiopaths_texts:
	for i in tqdm(range(len(audiopaths_texts) // 8)):
		audio_text = audiopaths_texts[i]
		audiopath, text = "./" + audio_text[0], audio_text[1]
		files.append(audiopath)
		texts.append(get_text(text, hp.text_cleaners))
		#mels.append(get_mel(audiopath, stft, hp.max_wav_value))
		mels.append(get_mel_librosa(audiopath, hp.max_wav_value))
		text_lengths.append([len(texts[-1])])

	# Verify all lists match in lengths.
	assert len(files) == len(audiopaths_texts), \
		"A file(s) were not successfully loaded."
	assert len(texts) == len(audiopaths_texts), \
		"Text file(s) were not successfully loaded."
	assert len(mels) == len(audiopaths_texts), \
		"Mel file(s) were not successfully loaded."
	assert len(text_lengths) == len(audiopaths_texts), \
		"Text file(s) were not successfully loaded."
	'''

	# Initialize dataset from generator.
	dataset = tf.data.Dataset.from_generator(
		generator, args=(audiopaths_texts,),
		output_signature=(
			tf.TensorSpec(shape=(()), dtype=tf.string),
			tf.TensorSpec(shape=((None,)), dtype=tf.int32),
			#tf.TensorSpec(shape=((None, hp.n_mel_channels)), dtype=tf.float32),
			tf.TensorSpec(shape=((hp.n_mel_channels, None)), dtype=tf.float32),
			tf.TensorSpec(shape=(()), dtype=tf.int32),
			tf.TensorSpec(shape=(()), dtype=tf.int32)
		)
	)

	dataset = dataset.shuffle(256)
	dataset = dataset.padded_batch(hp.batch_size, drop_remainder=False)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

	# Calculate the number of batches given the batch_size in the
	# hyperparameters.
	num_batches = math.ceil(len(audiopaths_texts) / hp.batch_size)
	
	# Return the cached dataset and the number of batches.
	return dataset, num_batches


def generator(fpaths):
	for i in range(len(fpaths)):
		audio_text = fpaths[i]
		audiopath, text = "./" + audio_text[0].decode(), audio_text[1].decode()
		text = get_text(text, hp.text_cleaners)
		mel = get_mel_librosa(audiopath, hp.max_wav_value)
		text_length = len(text)
		mel_length = tf.shape(mel)[1]
		yield audiopath, text, mel, text_length, mel_length


class TextMelCollate():
	# Zero-pads model inputs and targets based on number of frames per
	# step.
	def __init__(self, n_frames_per_step):
		self.n_frames_per_step = n_frames_per_step

	# Collate's training batch from normalized text and mel-spectrogram
	# #param: batch, [text_normalized, mel_normalized]
	def __call__(self, batch):
		# Right zero-pad all one-hot text sequences to max input 
		# length.
		input_lengths = tf.sort(batch[-2], direction="DESCENDING")
		ids_sorted_decreasing = tf.argsort(
			batch[-2], direction="DESCENDING"
		)
		'''
		input_lengths, ids_sorted_decreasing = tf.sort(
			tf.convert_to_tensor([len(x[0]) for x in batch], tf.float32),
			axis=0, direction="DESCENDING"
		)
		'''
		max_input_len = input_lengths[0].numpy()

		text_padded = tf.zeros([len(batch[0]), max_input_len]).numpy()
		#text_padded = tf.zeros([len(batch), max_input_len])
		for i in range(len(ids_sorted_decreasing)):
			text = batch[1][ids_sorted_decreasing[i].numpy()]
			text_padded[i, :len(text)] = text
			#text = batch[ids_sorted_decreasing[i]][0]
			#text_padded[i, :text.size(0)] = text
		text_padded = tf.convert_to_tensor(text_padded, dtype=tf.int32)

		# Right zero-pad mel-spec.
		'''
		num_mels = batch[0][1].size(0)
		max_target_len = max([x[1].size(1) for x in batch])
		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0
		'''
		#num_mels = len(batch[2])
		num_mels = tf.shape(batch[2][0])[0].numpy()
		max_target_len = max([tf.shape(mel)[1] for mel in batch[2]]).numpy()
		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - (max_target_len % self.n_frames_per_step)
			assert max_target_len % self.n_frames_per_step == 0

		# Include mel padded and gate padded.
		mel_padded = tf.zeros([len(batch[0]), num_mels, max_target_len]).numpy()
		gate_padded = tf.zeros([len(batch[0]), max_target_len]).numpy()
		output_lengths = tf.zeros([len(batch[0])]).numpy()
		for i in range(len(ids_sorted_decreasing)):
			mel = batch[2][ids_sorted_decreasing[i]]
			mel_padded[i, :, :tf.shape(mel)[1]] = mel
			#gate_padded[i, tf.shape(mel)[1] - 1:] = 1
			#output_lengths[i] = tf.shape(mel)[1].numpy()
			gate_padded[i, batch[-1][ids_sorted_decreasing[i]] - 1:] = 1
			output_lengths[i] = batch[-1][ids_sorted_decreasing[i]]
		mel_padded = tf.convert_to_tensor(mel_padded, dtype=tf.float32)
		gate_padded = tf.convert_to_tensor(gate_padded, dtype=tf.int32)
		output_lengths = tf.convert_to_tensor(output_lengths, dtype=tf.int32)
		'''
		mel_padded = tf.zeros([len(batch), num_mels, max_target_len])
		gate_padded = tf.zeros([len(batch), max_target_len])
		output_lengths = tf.zeros([len(batch)])
		for i in range(len(ids_sorted_decreasing)):
			mel = batch[ids_sorted_decreasing[i]][1]
			mel_padded[i, :, :mel.size(1)] = mel
			gate_padded[i, mel.size(1) - 1:] = 1
			output_lengths[i] = mel.size(1)
		'''

		return text_padded, input_lengths, mel_padded, gate_padded, \
			output_lengths


if __name__ == '__main__':
	# Use for Debug.
	print("Loading data...")
	dataset, num_batches = load_data()
	print("Data loaded.")

	print("Initializing TextMelCollater...")
	textcollate = TextMelCollate(hp.n_frames_per_step)
	print("TextMelCollater initialized.")

	print("Passing single batch to TextMelCollater...")
	x = list(dataset.as_numpy_iterator())
	y = textcollate(x[0])
	print("Successfully passed batch through TextMelCollater.")
	for i in range(len(y)):
		print(y[i])
		print("-"*72)
	exit(0)