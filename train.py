# train.py


import os
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, load_vocab
#from modules import *
#from layers import Encoder, Decoder1, Decoder2
from layers import *
from utils import *


class Graph:
	def __init__(self, mode="train"):
		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Set phase.
		training = True if mode == "train" else False

		# Graph.
		# Data Feeding
		# X: Text. (N, Tx)
		# Y: Reduced melspectrogram. (N, Ty//r, n_mels*r)
		# Z: Magnitude. (N, Ty, n_fft//2+1)


class Tacotron(tf.keras.Model):
	def __init__(self):
		super(Tacotron, self).__init__()

		#'''
		#self.input_text = tf.keras.layers.InputLayer(input_shape=(None,))
		#self.input_mel = tf.keras.layers.InputLayer(input_shape=(None, hp.n_mels * hp.r))
		#self.input_mag = tf.keras.layers.InputLayer(input_shape=(None, hp.n_fft // 2 + 1))
		self.embedding = EmbeddingLayer(len(hp.vocab), hp.embed_size)
		self.encoder = Encoder(hp)
		self.decoder1 = Decoder1(hp)
		self.decoder2 = Decoder2(hp)
		#'''

		'''
		self.embedding = EmbeddingLayer(
			len(hp.vocab), hp.embed_size, input_shape=(None,)
		)
		self.encoder = Encoder(hp)
		self.decoder1 = Decoder1(
			hp, input_shape=(None, hp.n_mels * hp.r)
		)
		self.decoder2 = Decoder2(
			hp, input_shape=(None, 1 + hp.n_fft // 2)
		)
		'''


	def call(self, inputs, training=None):
		text, mel, mag = inputs

		#text = self.input_text(text)
		embedding_output = self.embedding(text)
		decoder_inputs = tf.concat(
			(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
		)
		decoder_inputs = decoder_inputs[:, :, -hp.n_mels:]

		#decoder_inputs = self.input_mel(decoder_inputs)
		memory = self.encoder(embedding_output, training=training)
		y_hat, alignments = self.decoder1(
			decoder_inputs, memory, training=training
		)

		z_hat = self.decoder2(y_hat, training=training)

		audio = tf.py_function(spectrogram2wav, [z_hat[0]], tf.float32)

		return y_hat, z_hat, audio


	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			# Feed forward in training mode.
			y_hat, z_hat, audio = self(
				(texts, mels, mags), training=True
			)

			# Loss.
			loss1 = tf.reduce_mean(tf.abs(y_hat - mels))
			loss2 = tf.reduce_mean(tf.abs(z_hat - mags))
			loss = loss1 + loss2

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metrics that tracks the loss).
		self.compiled_metrics.update_state(mels, y_hat)
		self.compiled_metrics.update_state(mags, z_hat)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}


optimizer = tf.keras.optimizers.Adam(lr=hp.lr)
#'''
model = Tacotron()
model.compile(optimizer=optimizer, metrics=["accuracy"])
data = get_batch()
#model.build()
model.fit(data, epochs=5)
model.summary()
#'''

'''
text_in = tf.keras.Input(shape=(None,), dtype=tf.int32)
mel_in = tf.keras.Input(shape=(hp.batch_size, None, hp.n_mels * hp.r), dtype=tf.float32)
mag_in = tf.keras.Input(shape=(hp.batch_size, None, hp.n_fft // 2 + 1), dtype=tf.float32)

embedding = EmbeddingLayer(len(hp.vocab), hp.embed_size)
encoder = Encoder(hp)
decoder1 = Decoder1(hp)
decoder2 = Decoder2(hp)

embedding_output = embedding(text_in)
decoder_inputs = tf.concat(
	(tf.zeros_like(mel_in[:, :1, :]), mel_in[:, :-1, :]), 1
)
decoder_inputs = decoder_inputs[:, :, -hp.n_mels:]

memory = encoder(embedding_output)
y_hat, alignments = decoder1(
	decoder_inputs, memory, 
)

z_hat = decoder2(y_hat)

audio = tf.py_function(spectrogram2wav, [z_hat[0]], tf.float32)

model = tf.keras.Model(
	inputs=[text_in, mel_in, mag_in],
	outputs=[audio, z_hat, y_hat], name="tacotron"
)
model.compile(optimizer=optimizer)
model.summary()
'''