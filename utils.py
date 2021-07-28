# utils.py
# author: Diego Magdaleno


import numpy as np
import librosa
import os
import copy
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from scipy import signal
from hyperparams import Hyperparams as hp
import tensorflow as tf


# Parse the wave file in "fpath" and returns normalized mel spectrogram
# and linear spectrogram.
# @param: fpath, a string. The full path of a sound file.
# @return: mel, a 2D array of shape (T, n_mels) and dtype of float32.
# @return: mag, a 2D array of shape (T, 1 + n_fft / 2) and dtype of 
#	float32.
def get_spectrograms(fpath):
	# Loading sound file.
	y, sr = librosa.load(fpath, sr=hp.sr)

	# Trimming.
	y, _ = librosa.effects.trim(y)

	# Preemphasis.
	y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

	# STFT.
	linear = librosa.stft(y=y, n_fft=hp.n_fft, 
		hop_length=hp.hop_length, win_length=hp.win_length
	)

	# Magnitude spectrogram.
	mag = np.abs(linear) # (1 + n_fft // 2, T)

	# Mel spectrogram.
	mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels) # (n_mels, 1 + n_fft // 2)
	mel = np.dot(mel_basis, mag) # (n_mels, T)

	# To decimal.
	mel = 20 * np.log10(np.maximum(1e-5, mel))
	mag = 20 * np.log10(np.maximum(1e-5, mag))

	# Normalize.
	mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
	mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

	# Transpose.
	mel = mel.T.astype(np.float32) # (T, n_mels)
	mag = mag.T.astype(np.float32) # (T, 1 + n_fft // 2)

	return mel, mag


# Generate wave file from linear magnitude spectrogram.
# @param: mag, a numpy array of (T, 1 + n_fft // 2).
# @return: wav, a 1D numpy array.
def spectrogram2wav(mag):
	# Transpose.
	mag = mag.T

	# De-normalize.
	mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

	# To amplitude.
	mag = np.power(10.0, mag * 0.05)

	# Wav reconstruction.
	wav = griffin_lim(mag)

	# De-preemphasis.
	wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

	# Trim.
	wav, _ = librosa.effects.trim(wav)

	return wav.astype(np.float32)


# Applies Griffin-Lim's raw.
def griffin_lim(spectrogram):
	x_best = copy.deepcopy(spectrogram)
	for i in range(hp.n_iter):
		x_t = invert_spectrogram(x_best)
		est = librosa.stft(x_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
		phase = est / np.maximum(1e-8, np.abs(est))
		x_best = spectrogram * phase
	x_t = invert_spectrogram(x_best)
	y = np.real(x_t)

	return y


# Applies inverst fft.
# @param: spectrogram, [1 + n_fft // 2, t]
def invert_spectrogram(spectrogram):
	return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length,
		window="hann"
	)


# Plots the alignment.
# @param: alignment, a numpy array with shape of (encoder_steps, 
#	decoder_steps).
# @param: gs, (int) global step.
# @param: dir, output path.
def plot_alignment(alignment, gs, dir=hp.logdir):
	if not os.path.exists(dir):
		os.mkdir(dir)

	fig, ax = plt.subplots()
	im = ax.imshow(alignment)

	fig.colorbar(im)
	plt.title("{} Steps".format(gs))
	plt.savefig("{}/alignment_{}.png".format(dir, gs), format="png")
	plt.close(fig)


# Noam scheme from tensor2tensor.
def learning_rate_decay(init_lr, global_step, warmup_steps=4000.0):
	step = tf.to_float(global_step + 1)
	return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, 
		step ** -0.5
	)


# Read the wave file in fpath and extracts spectrograms.
def load_spectrograms(fpath):
	fname = os.path.basename(fpath)
	mel, mag = get_spectrograms(fpath)
	t = mel.shape[0]

	# Marginal padding for reduction shape sync.
	num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
	mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
	mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

	# Reduction.
	#mel = mel[::hp.r, :]
	#return fname, mel, mag
	return fname, mel.reshape((-1, np.n_mels * hp.r)), mag