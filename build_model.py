# build_model.py
# author: Diego Magdaleno
# Simple program that quickly builds a dc_tts model.


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from hyperparams import Hyperparams as hp
from layers import *
from utils import *
from data_load import load_vocab, get_batch


from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution

disable_eager_execution()
enable_eager_execution()


#training = True


# Inputs to the model.
# text = tf.keras.Input(shape=(None,None), dtype=tf.int32) # (B, N)
# mels = tf.keras.Input(shape=(None,None, 80), dtype=tf.float32) # (B, T/r, n_mels)
# pma = tf.keras.Input(shape=(None,), dtype=tf.int32) #(B)
text = tf.keras.Input(shape=(None,), dtype=tf.int32, batch_size=hp.B,
	name="text"
) # (N)
mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32, batch_size=hp.B,
	name="mels"
) # (T/r, n_mels)
#pma = tf.keras.Input(shape=(), dtype=tf.int32)
s = tf.concat((tf.zeros_like(mels[:, :1, :]), mels[:, :-1, :]), 1) # Same shape as mels
pma = tf.zeros(shape=(hp.B,), dtype=tf.int32)

# Model layers
textEnc = TextEncoder(hp)
audioEnc = AudioEncoder(hp)
#attention = Attention(hp, monotonic_attention=(not training), 
#	prev_max_attention=pma)
#attention = Attention(hp, prev_max_attention=pma)
attention = Attention(hp)
audioDec = AudioDecoder(hp)
ssrn = SSRN(hp)


# Model pass through.
'''
k, v = textEnc(text, training=training)
q = audioEnc(s, training=training)
r, alignments, max_attentions = attention((q, k, v), training=training)
y_logits, y = audioDec(r, training=training)
z_logits, z = ssrn(y, training=training)
'''
s = tf.keras.Input(shape=(None, 80), dtype=tf.float32, batch_size=hp.B,
	name="s"
)
#print(mels.get_shape())
#print(s.get_shape())
#print(text.get_shape())
#exit()
k, v = textEnc(text, training=False)
q = audioEnc(s, training=False)
r, alignments, max_attentions = attention((q, k, v), pma, training=False)
y_logits, y = audioDec(r, training=False)
z_logits, z = ssrn(y, training=False)

# Compile whole model (Text2Mel + SSRN).
model = Model(inputs=[text, mels, s], 
	outputs=[z, z_logits, y, y_logits, alignments, max_attentions], 
	name="dc_tts"
)
model.compile(optimizer="adam")
model.summary()

print("\n"*3)

# Compile Text2Mel.
k, v = textEnc(text, training=False)
q = audioEnc(s, training=False)
r, alignments, max_attentions = attention((q, k, v), pma, training=False)
y_logits, y = audioDec(r, training=False)
model = Model(inputs=[text, s], outputs=[y, y_logits, alignments, max_attentions], 
	name="text2mel"
)
model.compile(optimizer="adam")
model.summary()

print("\n"*3)

# Compile SSRN.
z_logits, z = ssrn(mels, training=False)
model = Model(inputs=[mels], outputs=[z, z_logits], name="ssrn")
model.compile(optimizer="adam")
model.summary()