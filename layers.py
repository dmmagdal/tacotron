# layers.py
# author: Diego Magdaleno
# OOP implementation of the modules.py and networks.py files in the
# Tensorflow implementation of dc_tts.
# Python 3.7
# Tensorflow 2.4.0


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class EmbeddingLayer(layers.Layer):
	def __init__(self, vocab_size, num_units, **kwargs):
		super(EmbeddingLayer, self).__init__()

		self.vocab_size = vocab_size
		self.num_units = num_units

		self.embedding_table = layers.Embedding(vocab_size, num_units)


	def call(self, inputs):
		return self.embedding_table(inputs)


	# Save any special configurations for this layer so that the model
	# can be saved and loaded without an issue. 
	# @param: takes no arguments.
	# @return: returns a copy of the config object for a tensorflow/
	#	keras layer.
	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"vocab_size": self.vocab_size,
						"num_units": self.num_units})
		return config


class BatchNorm(layers.Layer):
	def __init__(self, activation=None, **kwargs):
		super(BatchNorm, self).__init__()

		self.batch_norm = layers.BatchNormalization(
			scale=True, center=True,
		)
		self.activation = activation

		if self.activation is not None:
			self.act = layers.Activation(activation)


	def call(self, inputs, training=None):
		input_shape = inputs.get_shape()
		input_rank = len(input_shape)

		# Use fused batch norm if input_rank in [2, 3, 4] as it is much
		# faster. Pay attention to the fact that fused_batch_norm
		# requires shape to be rank 4 of NHWC.
		if input_rank in [2, 3, 4]:
			if input_rank == 2:
				inputs = tf.expand_dims(inputs, axis=1)
				inputs = tf.expand_dims(inputs, axis=2)
			elif input_rank == 3:
				inputs = tf.expand_dims(inputs, axis=1)

			outputs = self.batch_norm(inputs, training=training)

			# Restore original shape.
			if input_rank == 2:
				outputs = tf.squeeze(outputs, axis=[1, 2])
			elif input_rank == 3:
				outputs = tf.squeeze(outputs, axis=1)
		else: # Fallback to naive batch norm.
			outputs = self.batch_norm(inputs)

		if self.activation is not None:
			outputs = self.act(outputs)

		return outputs


class HighwayNet(layers.Layer):
	def __init__(self, num_units=None, **kwargs):
		super(HighwayNet, self).__init__()

		self.num_units = num_units


	def build(self, input_shape):
		if self.num_units is None:
			self.num_units = input_shape[-1]
		self.H = layers.Dense(self.num_units, activation="relu")
		self.T = layers.Dense(self.num_units, activation="sigmoid",
			bias_initializer=tf.constant_initializer(-1.0)
		)


	def call(self, inputs):
		h_outputs = self.H(inputs)
		t_outputs = self.T(inputs)
		outputs = h_outputs * t_outputs * (1.0 - t_outputs)
		return outputs


	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"num_units": self.num_units})
		return config


class Conv1DLayer(layers.Layer):
	def __init__(self, filters=None, size=1, rate=1, padding="same", 
			use_bias=True, activation=None, **kwargs):
		super(Conv1DLayer, self).__init__()

		self.filters = filters
		self.size = size
		self.rate = rate
		self.padding = padding
		self.use_bias = use_bias
		self.activation = activation
		self.pad_inputs = False
		self.pad_len = 0


	def build(self, input_shape):
		if self.padding.lower() == "causal":
			# Pre-padding for causality.
			self.pad_len = (self.size - 1) * self.rate # padding size.
			self.pad_inputs = True
			self.padding = "valid"

		if self.filters is None:
			self.filters = input_shape[-1]

		self.conv = layers.Conv1D(filters=self.filters, kernel_size=self.size, 
			dilation_rate=self.rate, padding=self.padding, use_bias=self.use_bias,
			activation=self.activation
		)


	def call(self, inputs):
		if self.pad_inputs:
			inputs = tf.pad(inputs, [[0, 0], [self.pad_len, 0], [0, 0]])

		outputs = self.conv(inputs)
		return outputs


	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"filters": self.filters,
						"size": self.size,
						"rate": self.rate,
						"padding": self.padding,
						"use_bias": self.use_bias,
						"activation": self.activation,
						"pad_inputs": self.pad_inputs,
						"pad_len": self.pad_len})
		return config


class Conv1DBanks(layers.Layer):
	def __init__(self, hp, K=16, **kwargs):
		super(Conv1DBanks, self).__init__()

		self.hp = hp
		self.K = K

		self.conv_layers = [Conv1DLayer(hp.embed_size // 2, k) 
			for k in range(1, K + 1)]
		self.batch_norm = BatchNorm(activation="relu")


	def call(self, inputs, training=None):
		outputs = tf.concat(
			tuple(conv(inputs) for conv in self.conv_layers), axis=-1
		)
		outputs = self.batch_norm(outputs, training=training)

		return outputs # (N, T, hp.embed_size // 2 * K)


class GRULayer(layers.Layer):
	def __init__(self, num_units=None, bidirection=False, **kwargs):
		super(GRULayer, self).__init__()

		self.num_units = num_units
		self.bidirection = bidirection


	def build(self, input_shape):
		if self.num_units is None:
			self.num_units = input_shape[-1]

		self.gru_cell = layers.GRUCell(self.num_units)
		self.gru_fw = layers.GRU(self.num_units, return_sequences=True)

		if self.bidirection:
			self.bidirect = layers.Bidirectional(
				self.gru_fw, merge_mode=None
			)
		else:
			self.rnn = layers.RNN(self.gru_cell)


	def call(self, inputs, training):
		if self.bidirection:
			outputs = self.bidirect(inputs)
			return tf.concat(tuple(outputs), 2)
		else:
			return self.rnn(inputs)


class AttentionDecoder(layers.Layer):
	#def __init__(self, memory, num_units=None, **kwargs):
	def __init__(self, num_units=None, **kwargs):
		super(AttentionDecoder, self).__init__()

		#self.memory = memory
		self.num_units = num_units


	def build(self, input_shape):
		if self.num_units is None:
			self.num_units = input_shape[-1]

		self.gru = layers.GRU(self.num_units, return_sequences=True)
		self.att = layers.AdditiveAttention()
		'''
		self.attention_mechanism = tfa.seq2seq.BahdanauAttention(
			self.num_units, self.memory
		)
		'''
		'''
		self.attention_mechanism = tfa.seq2seq.BahdanauAttention(
			self.num_units
		)
		self.decoder_cell = layers.GRUCell(self.num_units)
		self.cell_with_attention = tfa.seq2seq.AttentionWrapper(
			self.decoder_cell, self.attention_mechanism, self.num_units,
			#cell=self.decoder_cell, 
			#attention_mechanism=self.attention_mechanism, 
			#attention_layer_size=self.num_units,
			alignment_history=True
		)
		print(type(self.cell_with_attention))
		#self.rnn_cell = layers.SimpleRNNCell(self.num_units)
		#self.rnn = layers.StackedRNNCells(
		#	[self.cell_with_attention, self.rnn_cell]
		#)
		self.rnn = layers.RNN(self.cell_with_attention) # (N, T', 16)
		#self.rnn = tf.compat.v1.nn.dynamic_rnn(self.cell_with_attention)
		'''


	def call(self, inputs, memory):
		#self.attention_mechanism.setup_memory(memory)
		#return self.rnn(inputs, memory)
		#outputs, state = self.rnn(inputs, memory)
		#outputs, state = self.rnn(inputs)
	
		gru_out = self.gru(inputs)
		outputs = self.att([memory, gru_out])

		return outputs


class Prenet(layers.Layer):
	def __init__(self, hp, num_units=None, **kwargs):
		super(Prenet, self).__init__()

		self.hp = hp
		self.num_units = num_units


	def build(self, input_shape):
		if self.num_units is None:
			self.num_units = [self.hp.embed_size, self.hp.embed_size // 2]

		self.dense1 = layers.Dense(
			units=self.num_units[0], activation="relu"
		)
		self.dropout1 = layers.Dropout(rate=self.hp.dropout_rate) 
		self.dense2 = layers.Dense(
			units=self.num_units[1], activation="relu"
		)
		self.dropout2 = layers.Dropout(rate=self.hp.dropout_rate)


	def call(self, inputs, training=None):
		outputs = self.dense1(inputs)
		outputs = self.dropout1(outputs, training=training)
		outputs = self.dense2(outputs)
		outputs = self.dropout2(outputs, training=training)
		return outputs


class Encoder(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(Encoder, self).__init__()

		self.hp = hp

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
		prenet_out = self.prenet(inputs, training=training)
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


class Decoder1(layers.Layer):
	#def __init__(self, memory, hp, **kwargs):
	def __init__(self, hp, **kwargs):
		super(Decoder1, self).__init__()

		#self.memory = memory
		self.hp = hp

		# Decoder prenet.
		self.prenet = Prenet(hp) # (N, T_y/r, E/2)

		# Attention RNN.
		#self.att_dec = AttentionDecoder(memory, num_units=hp.embed_size) # (N, T_y/r, E)
		self.att_dec = AttentionDecoder(num_units=hp.embed_size) # (N, T_y/r, E)

		# Decoder RNNs.
		self.gru1 = GRULayer(hp.embed_size, bidirection=False) # (N, T_y/r, E)
		self.gru2 = GRULayer(hp.embed_size, bidirection=False) # (N, T_y/r, E)

		# Dense.
		self.dense = layers.Dense(hp.n_mels * hp.r) # Outputs => (N, T_y/r, n_mels*r)


	def call(self, inputs, memory, training=None):
		prenet_out = self.prenet(inputs, training=training)

		#att_out, state = self.att_dec(inputs, memory)
		#att_out, state = self.att_dec(prenet_out, memory)
		#att_out, state = self.att_dec((prenet_out, memory))
		att_out = self.att_dec(prenet_out, memory)

		# For attention monitoring.
		'''
		alignments = tf.transpose(
			state.alignment_history.stack(), [1, 2, 0]
		)
		'''

		decoder_out = att_out
		#gru_out1 = self.gru1(att_out)
		#gru_out2 = self.gru2(gru_out1)
		print("att out shape")
		print(decoder_out.get_shape())
		decoder_out += self.gru1(decoder_out)
		print("gru1 out shape")
		print(decoder_out.get_shape())
		decoder_out += self.gru2(decoder_out)
		print("gru2 out shape")
		print(decoder_out.get_shape())

		#mel_hats = self.dense(gru_out2)
		mel_hats = self.dense(decoder_out)

		return mel_hats#, alignments


class Decoder2(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(Decoder2, self).__init__()

		self.hp = hp

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


	def call(self, inputs, training=None):
		inputs = tf.reshape(inputs, [inputs.get_shape()[0], -1, self.hp.n_mels])
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