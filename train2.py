# train2.py


import os
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tqdm import tqdm
from data_load import get_batch, load_vocab
from layers import Prenet, EmbeddingLayer, Conv1DBanks, Conv1DLayer
from layers import GRULayer, HighwayNet
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


class Encoder(tf.keras.Model):
	def __init__(self, hp, **kwargs):
		super(Encoder, self).__init__()
		# Embedding layer
		self.embedding = EmbeddingLayer(
			len(hp.vocab), hp.embed_size, input_shape=(None,)
		)

		# Encoder
		# Encoder prenet.
		self.prenet = Prenet(hp) # (N, T_x, E/2)

		# Encoder CBHG.
		# Conv1D banks
		self.conv1d_banks = Conv1DBanks(hp=hp, K=hp.encoder_num_banks) # (N, T_x, K * E/2)

		# Max pooling
		self.max_pool = layers.MaxPooling1D(
			pool_size=2, strides=1, padding="same"
		) # (N, T_x, K * E/2)

		# Conv1D projections
		self.conv1d_proj1 = Conv1DLayer(
			filters=hp.embed_size // 2, size=3
		)
		self.batch_norm1 = BatchNorm(activation="relu")
		self.conv1d_proj2 = Conv1DLayer(
			filters=hp.embed_size // 2, size=3
		)
		self.batch_norm2 = BatchNorm()

		# Highway Nets.
		self.highwaynet = []
		for i in range(hp.num_highwaynet_blocks):
			self.highwaynet.append(
				HighwayNet(num_units=hp.embed_size // 2)
			) # (N, T_x, E/2)

		# Bidirectional GRU.
		self.gru = GRULayer(num_units=hp.embed_size // 2, bidirection=True) # (N, T_x, E)


	def call(self, inputs, training=None):
		embedding_output = self.embedding(inputs)

		prenet_out = self.prenet(embedding_output, training=training)
		conv1d_bank_out = self.conv1d_banks(prenet_out, training=training)
		max_pool_out = self.max_pool(conv1d_bank_out)

		# Conv1D projection outputs.
		conv1d_out1 = self.conv1d_proj1(max_pool_out)
		conv1d_bn_out1 = self.batch_norm1(conv1d_out1)
		conv1d_out2 = self.conv1d_proj2(conv1d_bn_out1)
		conv1d_bn_out2 = self.batch_norm2(conv1d_out2)

		# Residual connections.
		residual_out = conv1d_bn_out2 + prenet_out # (N, T_x, E/2)

		# Highwaynet outputs.
		highway_out = residual_out
		for net in self.highwaynet:
			highway_out = net(highway_out)

		# Bidirectional GRU output.
		memory = self.gru(highway_out)
		return memory


class Decoder(tf.keras.Model):
	def __init__(self, hp, max_mel_len, **kwargs):
		super(Decoder, self).__init__()

		# Prenet layer.
		self.prenet = Prenet(hp)

		# Stack of GRU cells to wrap around the Attention layer.
		self.rnn_cell = layers.StackedRNNCells(
			[layers.GRUCell(hp.embed_size), layers.GRUCell(hp.embed_size)]
		)

		# Final output layer.
		self.dense = layers.Dense(hp.n_mels * hp.r)

		# Sampler.
		self.sampler = tfa.seq2seq.sampler.TraningSampler()

		# Attention mechanism (with memory None).
		self.attention_mech = self.build_attention_mechanism(
			hp.embed_size, None, hp.batch_size * [max_mel_len]
		)

		# Wrap attention mechanism with RNN cell.
		self.rnn_cell = self.build_rnn_cell(hp.embed_size)

		# Define decoder with respect to fundamental RNN cell.
		self.decoder = tfa.seq2seq.BasicDecoder(
			self.rnn_cell, sampler=self.sampler, 
			output_layer=self.dense
		)


	def build_attention_mechanism(self, num_units, memory, 
			memory_sequence_length):
		return tfa.seq2seq.BahdanauAttention(
			units=num_units, memory=memory, 
			memory_sequence_length=memory_sequence_length
		)


	def build_rnn_cell(self, num_units):
		return tfa.seq2seq.AttentionWrapper(
			self.rnn_cell, self.attention_mech, 
			attenion_layer_size=num_units
		)


	def call(self, inputs, max_mel_len, training=None):
		prenet_out = self.prenet(inputs, training=training)

		decoder_outputs, _, _ = self.decoder(
			prenet_out, sequence_length=hp.batch_size * [max_mel_len - 1]
		)
		return outputs


class PostNet(tf.keras.Model):
	def __init__(self, hp, **kwargs):
		super(PostNet, self).__init__()

		# Conv1D Bank.
		self.conv1d_banks = Conv1DBanks(
			hp=hp, K=hp.decoder_num_banks
		) # (N, T_y, E * K/2)

		# Max Pooling.
		self.max_pool = layers.MaxPooling1D(
			pool_size=2, strides=1, padding="same"
		) # (N, T_y, E * K/2)

		# Conv1D projections.
		self.conv1d_proj1 = Conv1DLayer(
			filters=hp.embed_size // 2, size=3
		) # (N, T_x, E/2)
		self.batch_norm1 = BatchNorm(activation="relu")
		self.conv1d_proj2 = Conv1DLayer( filters=hp.n_mels, size=3) # (N, T_x, E/2)
		self.batch_norm2 = BatchNorm()

		# Extra affine transformation for dimensionality sync.
		self.dense1 = layers.Dense(hp.embed_size // 2) # (N, T_y, E/2)

		# Highway Nets.
		self.highwaynet = []
		for i in range(4):
			self.highwaynet.append(
				HighwayNet(num_units=hp.embed_size // 2)
			) # (N, T_y, E/2)

		# Bidirectional GRU.
		self.gru = GRULayer(hp.embed_size // 2, bidirection=True) # (N, T_y, E)

		# Output.
		self.dense2 = layers.Dense(1 + hp.n_fft // 2)


	def call(self, inputs, training):
		inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, self.hp.n_mels])
		conv1d_bank_out = self.conv1d_banks(inputs, training=training)
		max_pool_out = self.max_pool(conv1d_bank_out)

		# Conv1D projection outputs.
		conv1d_out1 = self.conv1d_proj1(max_pool_out)
		conv1d_bn_out1 = self.batch_norm1(conv1d_out1)
		conv1d_out2 = self.conv1d_proj2(conv1d_bn_out1)
		conv1d_bn_out2 = self.batch_norm2(conv1d_out2)

		dense1_out = self.dense1(conv1d_bn_out2)

		# Highwaynet outputs.
		highway_out = dense1_out
		for net in self.highwaynet:
			highway_out = net(highway_out)

		# Bidirectional GRU output.
		gru_out = self.gru(highway_out)

		# Outputs => (N, T_y, 1 + n_fft // 2).
		outputs = self.dense2(gru_out)

		return outputs


class Tacotron(tf.keras.Model):
	def __init__(self, hp, max_mel_len, **kwargs):
		super(Tacotron, self).__init__()

		self.encoder = Encoder(hp)
		self.decoder = Decoder(hp, max_mel_len)
		self.postnet = PostNet(hp)


	def call(self, inputs, training=None):
		texts, mels, mags = inputs

		encoder_outputs = self.encoder(texts, training=training)
		mel_outputs = self.decoder(encoder_outputs + mels, training=training)
		mag_outputs = self.postnet(mels)

		return mel_outputs, mag_outputs


	@tf.function
	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			decoder_inputs = tf.concat(
				(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
			)
			decoder_inputs = decoder_inputs[:, :, -hp.n_mels:]

			# Feed forward in training mode.
			#y_hat, z_hat = self(
			#	(texts, mels, mags), training=True
			#)
			encoder_outputs = self.encoder(texts)
			self.decoder.attention_mech.setup_memory(encoder_outputs)
			decoder_outputs = self.decoder(decoder_inputs)

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
data, num_batch = get_batch()

#'''
model.build(input_shape=[
	(None, None,), 
	(None, None, hp.n_mels * hp.r), 
	(None, None, hp.n_fft // 2 + 1)
	]
)

model.summary()
model.fit(data, epochs=5)

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