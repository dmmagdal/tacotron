# model.py
# author: Diego Magdaleno
# Convert the Graph from train.py to an OOP implementation from the
# Tensorflow implementation of dc_tts
# Python 3.7
# Tensorflow 2.4.0


import os
import json
import sys
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from layers import *
from utils import *
from hyperparams import Hyperparams as hp
from data_load import get_batch, load_vocab



#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


class Graph:
	def __init__(self, num=1, mode="train"):
		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Set flag.
		training = True if mode == "train" else False

		# Graph.
		# Data feeding:
		# L: Text. (B, N), int32
		# mels: Reduced mel spectrogram. (B, T/r, n_mels), float32
		# mags: Magnitude. (B, T, n_fft // 2 + 1), float32
		if mode == "train":
			self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch()
			self.prev_max_attention = tf.ones(shape=(hp.B,), dtype=tf.int32)
			self.gts = tf.convert_to_tensor(guided_attention())
		else: # synthesize.
			# self.L = tf.placeholder(tf.int32, shape=(None, None))
			# self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
			# self.prev_max_attention = tf.placeholder(tf.int32, shape=(None,))
			self.L = tf.keras.Input(shape=(None, None), dtype=tf.int32)
			self.mels = tf.keras.Input(
				shape=(None, None, hp.n_mels), dtype=tf.float32
			)
			self.prev_max_attention = tf.keras.Input(shape=(None,), dtype=tf.int32)

		if num == 1 or not training:
			# Get S or decoder inputs. (B, T//r, n_mels)
			self.S = tf.concat(
				(tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1
			)

			# Networks.
			self.K, self.V = TextEncoder(hp, training=training)

			self.Q = AudioEncoder(hp, training=training)

			self.R, self.alignments, self.max_attentions = Attention(hp,
				monotonic_attention=(not training), 
				prev_max_attention=self.prev_max_attention
			)

			self.Y_logits, self.Y = AudioDecoder(hp, training=training)

		else:
			self.Z_logits, self.Z = SSRN(hp, training=training)

		if not training:
			self.Z_logits, self.Z = SSRN(hp, training=training)

		self.global_step = tf.Variable(0, trainable=False)

		if training:
			if num == 1: #Text2Mel.
				# Mel L1 loss.
				self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

				# Mel binary divergence loss.
				self.loss_bd1 = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits,
						labels=self.mels
					)
				)

				# Guided attention loss.
				self.A = tf.pad(self.alignments, 
					[(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT",
					constant_values=-1.0
				)[:, :hp.max_N, :hp.max_T]
				self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
				self.loss_att = tf.reduce_sum(
					tf.abs(self.A * self.gts) * self.attention_masks
				)
				self.mask_sum = tf.reduce_sum(self.attention_masks)
				self.loss_att /= self.mask_sum

				# Total loss.
				self.loss = self.loss_mels + self.loss_bd1 + self.loss_att
				
			else: # SSRN.
				# Mag L1 loss.
				self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

				# Mag binary divergence loss.
				self.loss_bd2 = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits),
					labels=self.mags
				)

				# Total loss.
				self.loss = self.loss_mags + self.loss_bd2

			# Training scheme.
			self.lr = learning_rate_decay(hp.lr, self.global_step)
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

			# Gradient clipping.
			self.gvs = self.optimizer.compute_gradients(self.loss)
			self.clipped = []
			for grad, var in self.gvs:
				grad = tf.clip_by_value(grad, -1.0, 1.0)
				self.clipped.append((grad, var))
				self.train_op = self.optimizer.apply_gradients(self.clipped,
					global_step=self.global_step
				)

		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)


	def create_model(self):
		# Initialize all model inputs.
		text = tf.keras.Input(shape=(None,), dtype=tf.int32,
			batch_size=hp
		) # (N)
		mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # (T/r, n_mels)
		pma = tf.keras.Input(shape=(), dtype=tf.int32)
		s = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # same shape as mel
		
		# Model layers.
		training = False
		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp, monotonic_attention=(not training), 
			prev_max_attention=pma
		)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)
		
		# Create and compile Text2Mel model.
		k, v = self.textEnc(text, training=False)
		q = self.audioEnc(s, training=False)
		r, alignments, max_attentions = self.attention((q, k, v), training=False)
		y_logits, y = self.audioDec(r, training=False)
		self.text2mel_model = tf.keras.Model(inputs=[text, s], outputs=[y_logits, y], name="text2mel")
		self.text2mel_model.compile(optimizer="adam", loss=self.text2mel_loss)
		self.text2mel_model.summary()

		# Create and compile SSRN model.
		z_logits, z = self.ssrn(mels, training=False)
		self.ssrn_model = tf.keras.Model(inputs=[mels], outputs=[z_logits, z], name="ssrn")
		self.ssrn_model.compile(optimizer="adam", loss=self.ssrn_loss)
		self.ssrn_model.summary()


	def save_model(self, folder_path):
		pass


	def load_model(self, folder_path):
		pass


	def text2mel_loss(self):
		pass


	def ssrn_loss(self):
		pass


	def train_model(self, dataset_path, model=1):
		pass


class TacoTron(Model):
	def __init__(self, input_hp=None):
		if input_hp is None:
			self.hp = hp
		else:
			self.hp = input_hp

		self.embed = EmbeddingLayer(len(self.hp.vocab), self.hp.embed_size)
		self.encoder = Encoder(self.hp)
		self.decoder1 = Decoder1(self.hp)
		self.decoder2 = Decoder2(self.hp)


	def call(self, inputs, training=False):
		# Unpack inputs.
		text, mel = inputs

		# Get encoder/decoder inputs.
		encoder_inputs = self.embed(text) # (N, T_x, E)
		decoder_inputs = tf.concat(
			(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
		) # (N, Ty/r, n_mels*r)
		decoder_inputs = decoder_inputs[:, :, -self.hp.n_mels] # feed last frames only (N, Ty/r, n_mels)

		# Pass through networks.
		memory = self.encoder(encoder_inputs, training=training) # (N, T_x, E)
		y_hat, alignments = self.decoder1(
			decoder_inputs, memory, training=training
		) # (N, T_y//r, n_mels*r)
		z_hat = self.decoder2(y_hat, training=training) # (N, T_y//r, (1+n_fft//2)*r)

		return y_hat, alignments, z_hat


	@tf.function
	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			# Feed forward in training mode.
			y_pred, alignments, z_pred = self((texts, mels), training=True)

			# Total loss.
			loss1 = tf.reduce_mean(tf.abs(y_pred - mels))
			loss2 = tf.reduce_mean(tf.abs(z_pred - mags))
			loss = loss1 + loss2

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metrics that tracks the loss).
		self.compiled_metrics.update_state(mels, y_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}


class Tacotron(Model):
	def __init__(self, input_hp=None):
		super(Tacotron, self).__init__()

		# Load hyperparameters.
		if input_hp:
			self.hp = input_hp
		else:
			self.hp = hp

		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		self.embedding = EmbeddingLayer(
			len(self.hp.vocab), self.hp.embed_size
		)
		self.encoder = Encoder(self.hp)
		self.decoder1 = Decoder1(self.hp)
		self.decoder2 = Decoder2(self.hp)


	def call(self, inputs, training=False):
		text, mel, mag = inputs

		embedding_out = self.embedding(text)
		decoder_in = tf.concat(
			(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
		)
		decoder_in = decoder_in[:, :, -self.hp.n_mels:]

		memory = self.encoder(embedding_out, training=training)
		y = self.decoder1(decoder_in, memory, training=training)
		z = self.decoder2(y, training=training)

		return y, z


	@tf.function
	def train_step(self, data):
		pass


tts = Tacotron()
text_shape = (None, None,)
mel_shape = (None, None, hp.n_mels * hp.r)
mag_shape = (None, None, 1 + hp.n_fft // 2)
tts.build(input_shape=[text_shape, mel_shape, mag_shape])