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
		decoder_in = decoder_in[:, :, -self.hp.n_mels]

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