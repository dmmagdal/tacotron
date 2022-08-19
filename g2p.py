# g2p.py
# Source: https://fehiepsi.github.io/blog/grapheme-to-phoneme/
# Python 3.7
# Tensorflow 2.4.0


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class Encoder(layers.Layer):
	def __init__(self, vocab_size, embed_dim, hidden_dim, **kwargs):
		super(Encoder, self).__init__()
		self.embedding = layers.Embedding(vocab_size, embed_dim)
		self.lstm = layers.LSTMCell(hidden_dim, input_shape=(embed_dim))
		self.hidden_dim = hidden_dim


	def call(self, x_seq, training=False):
		# Input shape (seq_len, batch_size)
		output = []
		embedding_seq = self.embedding(x_seq) # (seq_len, batch_size, hidden_dim)
		hidden_state = tf.zeros(
			(embedding_seq.get_shape()[1], self.embed_dim), dtype=tf.float32
		)
		context_state = tf.zeros(
			(embedding_seq.get_shape()[1], self.embed_dim), dtype=tf.float32
		)

		for embed in embedding_seq:
			embed = tf.squeeze(embed, 0)
			hidden_state, context_state = self.lstm(
				embed, (hidden_state, context_state)
			)
			output.append(hidden_state)
		return tf.stack(output, 0), hidden_state, context_state


class Attention(layers.Layer):
	def __init__(self, dim, **kwargs):
		super(Attention, self).__init__()
		self.dense = layers.Dense(dim, input_shape=(dim * 2))


	def call(self, x, context=None):
		if context is None:
			return x

		assert x.get_shape()[0] == context.get_shape()[0]
		assert x.get_shape()[1] == context.get_shape()[2]


class Decoder(layers.Layer):
	def __init__(self, vocab_size, embed_dim, hidden_dim, **kwargs):
		super(Decoder, self).__init__()
		self.embedding = layers.Embedding(vocab_size, embed_dim)
		self.lstm = layers.LSTMCell(hidden_dim, input_shape=(embed_dim))
		self.att = Attention(hidden_dim)
		self.dense = layers.Dense(vocab_size, input_shape=(hidden_dim))


	def call(self, x_seq, hidden_state, context_state, context=None):
		output = []