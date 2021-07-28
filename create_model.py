import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from hyperparams import Hyperparams as hp
from layers import *


class Text2Mel(Model):
	def __init__(self, input_hp=None):
		super(Text2Mel, self).__init__()

		self.hp = hp if input_hp is None else input_hp

		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp)
		self.audioDec = AudioDecoder(hp)


	def call(self, inputs, training=False):
		text, mels, pma, s = inputs
		k, v = self.textEnc(text, training=training)
		q = self.audioEnc(s, training=training)
		r, alignments, max_attentions = self.attention((q, k, v), 
			training=training
		)
		y_logits, y = self.audioDec(r, training=training)
		return y, y_logits, alignments, max_attentions

text = tf.keras.Input(shape=(None,), dtype=tf.int32) # (N)
mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # (T/r, n_mels)
pma = tf.keras.Input(shape=(), dtype=tf.int32)
s = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # same shape as mel

text2mel = Text2Mel(inputs=[text, mels, pma, s], outputs=[y, y_logits, alignments, max_attentions])
text2mel.compile(optimizer="adam")
text2mel.summary()